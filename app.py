from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from datetime import datetime, timedelta
import json
import dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
import requests
import re
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
dotenv.load_dotenv()

app = Flask(__name__)
CORS(app)

# 设置API密钥
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")

# 初始化模型和嵌入
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embedding_model = OpenAIEmbeddings()

# 获取高德地图API密钥
AMAP_KEY = os.getenv("AMAP_KEY", "b3cb1e6090a59518bc141503765464d7")


# 创建自定义工具
@tool
def get_yantai_weather(days: int = 3):
    """获取烟台多日天气预报"""
    try:
        # 使用高德地图API获取天气预报（多日）
        response = requests.get(
            "https://restapi.amap.com/v3/weather/weatherInfo",
            params={
                "key": AMAP_KEY,
                "city": "烟台",
                "extensions": "all",  # 改为all获取预报
                "output": "JSON"
            },
            timeout=10
        )

        logger.info(f"高德天气API响应状态码: {response.status_code}")

        if response.status_code != 200:
            logger.warning("高德天气API调用失败，使用模拟数据")
            return get_fallback_weather(days)

        data = response.json()
        logger.info(f"高德天气API返回JSON: {data}")

        if data.get('status') != '1':
            logger.warning("高德天气API返回错误状态，使用模拟数据")
            return get_fallback_weather(days)

        # 解析预报数据
        forecasts = data.get('forecasts', [])
        if not forecasts:
            logger.warning("高德天气API返回无预报数据，使用模拟数据")
            return get_fallback_weather(days)

        casts = forecasts[0].get('casts', [])

        # 限制返回天数
        casts = casts[:days]

        weather_data = []
        for i, cast in enumerate(casts):
            date = cast.get('date', f'第{i + 1}天')
            day_weather = cast.get('dayweather', '晴')
            night_weather = cast.get('nightweather', '晴')
            day_temp = cast.get('daytemp', '20')
            night_temp = cast.get('nighttemp', '15')

            # 统一温度格式
            temperature_str = f"{night_temp}~{day_temp}°C"
            try:
                avg_temp = (int(day_temp) + int(night_temp)) / 2
            except (ValueError, TypeError):
                avg_temp = 20

            daily_weather = {
                "date": date,
                "weather": f"{day_weather}转{night_weather}" if day_weather != night_weather else day_weather,
                "temperature": temperature_str,
                "temperature_avg": avg_temp,
                "day_weather": day_weather,
                "night_weather": night_weather,
                "day_temp": day_temp,
                "night_temp": night_temp,
                "wind": f"{cast.get('daywind', '东北')}风{cast.get('daypower', '3-4')}级",
                "clothing": get_clothing_suggestion(avg_temp)
            }
            weather_data.append(daily_weather)

        return {
            "city": "烟台",
            "days": days,
            "data": weather_data
        }

    except Exception as e:
        logger.error(f"天气服务异常: {e}")
        return get_fallback_weather(days)


def get_fallback_weather(days):
    """备用天气数据"""
    import random
    weather_types = ["晴", "多云", "阴", "小雨"]
    weather_data = []

    base_date = datetime.now()

    for i in range(days):
        weather = random.choice(weather_types)
        day_temp = random.randint(18, 25)
        night_temp = random.randint(10, 15)
        avg_temp = (day_temp + night_temp) / 2
        date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")

        daily_weather = {
            "date": date_str,
            "weather": weather,
            "temperature": f"{night_temp}~{day_temp}°C",
            "temperature_avg": avg_temp,
            "wind": "东北风3-4级",
            "clothing": get_clothing_suggestion(avg_temp)
        }
        weather_data.append(daily_weather)

    return {
        "city": "烟台",
        "days": days,
        "data": weather_data,
        "note": "模拟数据"
    }


def get_clothing_suggestion(temperature):
    """根据温度提供穿衣建议"""
    try:
        if isinstance(temperature, str):
            try:
                temp = int(float(temperature))
            except ValueError:
                temp_str = ''.join(filter(str.isdigit, temperature)) or '20'
                temp = int(temp_str)
        else:
            temp = int(temperature)

        if temp >= 25:
            return "建议穿短袖、薄外套"
        elif temp >= 18:
            return "建议穿长袖衬衫、薄外套"
        elif temp >= 10:
            return "建议穿毛衣、厚外套"
        else:
            return "建议穿厚外套、保暖衣物"
    except (ValueError, TypeError):
        return "建议穿长袖衬衫、薄外套"


# 在 get_ticket_info 函数中增强搜索逻辑
@tool
def get_ticket_info(attraction: str):
    """获取景点门票信息 - 使用高德地图POI搜索"""
    try:
        # 尝试多种搜索关键词
        search_keywords = [attraction, f"{attraction}景区", f"{attraction}公园", f"烟台{attraction}"]

        for keyword in search_keywords:
            response = requests.get(
                "https://restapi.amap.com/v3/place/text",
                params={
                    "key": AMAP_KEY,
                    "keywords": keyword,
                    "city": "烟台",
                    "types": "风景名胜",
                    "output": "JSON"
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                pois = data.get('pois', [])
                if pois:  # 如果找到结果就使用
                    logger.info(f"使用关键词 '{keyword}' 成功找到景点信息")
                    break
            logger.info(f"使用关键词 '{keyword}' 未找到结果")

        logger.info(f"高德POI API响应状态码: {response.status_code}")

        # 检查响应状态
        if response.status_code != 200:
            logger.warning("高德POI API调用失败，使用模拟数据")
            return get_fallback_ticket_info(attraction)

        data = response.json()
        logger.info(f"高德POI API返回JSON: {data}")

        # 检查API返回状态
        if data.get('status') != '1':
            logger.warning("高德POI API返回错误状态，使用模拟数据")
            return get_fallback_ticket_info(attraction)

        # 解析真实API数据
        pois = data.get('pois', [])
        if pois:
            # 获取第一个匹配的景点
            poi = pois[0]
            biz_ext = poi.get('biz_ext', {})  # 获取扩展商务信息

            return {
                "name": poi.get('name', attraction),
                "price": biz_ext.get('cost', '需现场查询') or '需现场查询',
                "opening_hours": biz_ext.get('opentime2', '需现场查询') or '需现场查询',
                "address": poi.get('address', '未知'),
                "tel": poi.get('tel', '未知'),
                "rating": biz_ext.get('rating', '暂无评分')
            }
        else:
            return get_fallback_ticket_info(attraction)

    except Exception as e:
        logger.error(f"景点信息获取异常: {e}")
        return get_fallback_ticket_info(attraction)


def get_fallback_ticket_info(attraction):
    """备用门票信息"""
    ticket_info = {
        "烟台山": {"price": "30元", "opening_hours": "8:00-17:00", "address": "烟台市芝罘区历新路7号"},
        "蓬莱阁": {"price": "100元", "opening_hours": "7:30-17:00", "address": "烟台市蓬莱区北关路1号"},
        "长岛": {"price": "120元", "opening_hours": "8:00-16:30", "address": "烟台市长岛县"},
        "金沙滩": {"price": "免费", "opening_hours": "全天", "address": "烟台市开发区海滨路"},
        "烟台博物馆": {"price": "免费", "opening_hours": "9:00-17:00", "address": "烟台市芝罘区南大街61号"},
        "张裕酒文化博物馆": {"price": "60元", "opening_hours": "8:30-17:00", "address": "烟台市芝罘区大马路56号"}
    }
    return ticket_info.get(attraction, {"price": "未知", "opening_hours": "未知", "address": "未知"})


@tool
def get_transport_info(origin: str, destination: str):
    """获取交通信息 - 优先使用高德地图API，再使用本地知识库"""
    try:
        # 首先尝试使用高德地图API
        if origin and destination:
            # 确保地址包含城市信息
            formatted_origin = origin
            formatted_destination = destination

            if "烟台" not in origin:
                formatted_origin = f"烟台{origin}" if not origin.startswith("烟台") else origin
            if "烟台" not in destination:
                formatted_destination = f"烟台{destination}" if not destination.startswith("烟台") else destination

            # 使用高德地图API获取路径规划信息
            response = requests.get(
                "https://restapi.amap.com/v3/direction/transit/integrated",
                params={
                    "key": AMAP_KEY,
                    "origin": formatted_origin,
                    "destination": formatted_destination,
                    "city": "烟台",
                    "output": "JSON"
                },
                timeout=10
            )

            logger.info(f"高德路径规划API响应状态码: {response.status_code}")
            logger.info(f"请求参数: origin={formatted_origin}, destination={formatted_destination}")

            if response.status_code == 200:
                data = response.json()
                logger.info(f"高德路径规划API返回JSON: {data}")

                if data.get('status') == '1':
                    # 解析真实API数据
                    route = data.get('route', {})
                    if route:
                        transits = route.get('transits', [])
                        if transits and len(transits) > 0:
                            # 获取第一个推荐路线
                            transit = transits[0]
                            cost = transit.get('cost', '')
                            duration = int(transit.get('duration', 0)) // 60 if transit.get('duration', 0) else 0

                            # 构建详细的交通信息
                            transport_info = {
                                "type": "公共交通",
                                "price": f"{cost}元" if cost and cost != '0' and cost != '[]' else "",
                                "duration": f"{duration}分钟" if duration > 0 else "",
                                "distance": f"{transit.get('distance', 0) / 1000:.1f}公里" if transit.get('distance',
                                                                                                          0) > 0 else "",
                                "departure_time": "",
                                "arrival_time": "",
                                "route_details": []
                            }

                            # 添加路线详情
                            segments = transit.get('segments', [])
                            if segments:
                                for segment in segments:
                                    if 'walking' in segment:
                                        walking = segment['walking']
                                        if walking.get('distance', 0) > 0 or walking.get('duration', 0) > 0:
                                            transport_info['route_details'].append({
                                                "type": "步行",
                                                "distance": f"{walking.get('distance', 0)}米" if walking.get('distance',
                                                                                                             0) > 0 else "",
                                                "duration": f"{int(walking.get('duration', 0)) // 60}分钟" if walking.get(
                                                    'duration', 0) > 0 else ""
                                            })
                                    elif 'bus' in segment:
                                        bus = segment['bus']
                                        buses = bus.get('buslines', [])
                                        if buses and len(buses) > 0:
                                            busline = buses[0]
                                            transport_info['route_details'].append({
                                                "type": "公交",
                                                "name": busline.get('name', ''),
                                                "departure_stop": busline.get('departure_stop', {}).get('name', ''),
                                                "arrival_stop": busline.get('arrival_stop', {}).get('name', ''),
                                                "distance": f"{busline.get('distance', 0)}米" if busline.get('distance',
                                                                                                             0) > 0 else "",
                                                "duration": f"{int(busline.get('duration', 0)) // 60}分钟" if busline.get(
                                                    'duration', 0) > 0 else ""
                                            })

                            # 只有当有实际数据时才返回
                            if any([transport_info["price"], transport_info["duration"], transport_info["distance"]] or
                                   transport_info['route_details']):
                                logger.info(f"成功解析交通信息: {transport_info}")
                                return transport_info

        # 如果高德地图API无法获取有效数据，尝试使用本地知识库
        local_result = get_local_transport_info(origin, destination)
        if local_result and any([local_result.get("price", ""), local_result.get("duration", ""),
                                 local_result.get("distance", "")] or local_result.get('route_details', [])):
            return local_result

        # 如果都没有数据，返回备用数据
        fallback_result = get_fallback_transport_info(origin, destination)
        # 确保备用数据中不包含"需现场查询"等无用信息
        if fallback_result.get("price") == "需现场查询":
            fallback_result["price"] = ""
        if fallback_result.get("duration") == "需现场查询":
            fallback_result["duration"] = ""
        if fallback_result.get("distance") == "未知距离":
            fallback_result["distance"] = ""

        return fallback_result

    except Exception as e:
        logger.error(f"交通信息获取异常: {e}")
        # 出现异常时也尝试使用本地知识库
        try:
            local_result = get_local_transport_info(origin, destination)
            if local_result:
                return local_result
        except:
            pass
        return get_fallback_transport_info(origin, destination)


def get_fallback_transport_info(origin, destination):
    """备用交通信息 - 基于常见景点提供准确数据"""
    # 常见景点间的交通信息
    common_routes = {
        ("烟台站", "烟台山"): {
            "type": "公交",
            "price": "2元",
            "duration": "20分钟",
            "distance": "5公里",
            "route_details": [
                {
                    "type": "公交",
                    "name": "3路/17路",
                    "description": "从烟台站乘坐3路或17路公交车直达烟台山",
                    "departure_stop": "烟台站",
                    "arrival_stop": "烟台山",
                    "distance": "5公里",
                    "duration": "20分钟"
                }
            ]
        },
        ("烟台站", "蓬莱阁"): {
            "type": "旅游专线",
            "price": "25元",
            "duration": "1小时",
            "distance": "70公里",
            "route_details": [
                {
                    "type": "公交",
                    "name": "蓬莱阁旅游专线",
                    "description": "从烟台站乘坐蓬莱阁旅游专线大巴直达蓬莱阁",
                    "departure_stop": "烟台站",
                    "arrival_stop": "蓬莱阁",
                    "distance": "70公里",
                    "duration": "1小时"
                }
            ]
        },
        ("烟台山", "金沙滩"): {
            "type": "公交",
            "price": "2元",
            "duration": "30分钟",
            "distance": "15公里",
            "route_details": [
                {
                    "type": "公交",
                    "name": "21路",
                    "description": "从烟台山乘坐21路公交车前往金沙滩",
                    "departure_stop": "烟台山",
                    "arrival_stop": "金沙滩",
                    "distance": "15公里",
                    "duration": "30分钟"
                }
            ]
        },
        ("烟台市区", "长岛"): {
            "type": "轮渡+公交",
            "price": "80元",
            "duration": "2小时",
            "distance": "60公里",
            "route_details": [
                {
                    "type": "公交",
                    "name": "长途客车",
                    "description": "从烟台市区乘车到蓬莱",
                    "departure_stop": "烟台市区",
                    "arrival_stop": "蓬莱",
                    "distance": "50公里",
                    "duration": "1小时"
                },
                {
                    "type": "轮渡",
                    "name": "蓬长轮渡",
                    "description": "从蓬莱乘船到长岛",
                    "departure_stop": "蓬莱",
                    "arrival_stop": "长岛",
                    "distance": "10公里",
                    "duration": "30分钟"
                }
            ]
        }
    }

    # 尝试匹配常见路线（更灵活的匹配方式）
    for route, info in common_routes.items():
        origin_match = False
        dest_match = False

        # 检查起点是否匹配
        for route_origin in route[0].split('/'):
            if route_origin in origin or origin in route_origin:
                origin_match = True
                break

        # 检查终点是否匹配
        for route_dest in route[1].split('/'):
            if route_dest in destination or destination in route_dest:
                dest_match = True
                break

        if origin_match and dest_match:
            logger.info(f"匹配到预设路线: {route[0]} -> {route[1]}")
            return info

    # 默认返回
    return {
        "type": "公共交通",
        "price": "",
        "duration": "",
        "distance": "",
        "departure_time": "",
        "arrival_time": "",
        "route_details": [
            {
                "type": "公交/出租车",
                "description": "建议使用导航App查询具体路线",
                "name": "",
                "departure_stop": "",
                "arrival_stop": "",
                "distance": "",
                "duration": ""
            }
        ]
    }



def get_local_transport_info(origin: str, destination: str):
    """从本地知识库获取交通信息"""
    try:
        # 使用向量数据库搜索相关信息
        if vectorstore and retriever:
            # 构造查询语句
            query = f"烟台{origin}到{destination}交通方式 公交路线"

            # 搜索相关文档
            docs = retriever.get_relevant_documents(query)

            if docs:
                # 解析文档内容，提取交通信息
                for doc in docs:
                    content = doc.page_content
                    # 解析交通信息
                    transport_info = parse_transport_from_text(content)
                    if transport_info:
                        return transport_info

        return None
    except Exception as e:
        logger.error(f"从本地知识库获取交通信息异常: {e}")
        return None


def parse_transport_from_text(text: str):
    """从文本中解析交通信息"""
    try:
        # 这里需要根据实际文档格式实现解析逻辑
        lines = text.split('\n')
        transport_info = {
            "type": "公共交通",
            "route_details": []
        }

        for line in lines:
            if "价格" in line or "费用" in line:
                price_match = re.search(r'(\d+\.?\d*)元', line)
                if price_match:
                    transport_info["price"] = f"{price_match.group(1)}元"
            elif "时间" in line or "耗时" in line:
                time_match = re.search(r'(\d+\.?\d*)分钟', line)
                if time_match:
                    transport_info["duration"] = f"{time_match.group(1)}分钟"
            elif "距离" in line:
                distance_match = re.search(r'(\d+\.?\d*)公里', line)
                if distance_match:
                    transport_info["distance"] = f"{distance_match.group(1)}公里"
            elif "路线" in line or "公交" in line:
                # 解析公交路线详情
                route_detail = {"type": "公交", "description": line.strip()}
                transport_info["route_details"].append(route_detail)

        # 如果解析到了必要信息，返回结果
        if transport_info["route_details"] or any([transport_info.get("price"), transport_info.get("duration"), transport_info.get("distance")]):
            # 移除所有"需现场查询"占位符
            if transport_info.get("price") == "需现场查询":
                transport_info["price"] = ""
            if transport_info.get("duration") == "需现场查询":
                transport_info["duration"] = ""
            if transport_info.get("distance") == "未知距离":
                transport_info["distance"] = ""
            if transport_info.get("departure_time") == "需现场查询":
                transport_info["departure_time"] = ""
            if transport_info.get("arrival_time") == "需现场查询":
                transport_info["arrival_time"] = ""
            return transport_info

        return None
    except Exception as e:
        logger.error(f"解析交通信息文本异常: {e}")
        return None


@tool
def get_accommodation_info(location: str, budget: int = None):
    """获取住宿信息 - 使用高德地图POI搜索，支持预算筛选"""
    try:
        logger.info(f"开始查询住宿信息: location={location}, budget={budget}")

        # 构建查询参数
        params = {
            "key": AMAP_KEY,
            "keywords": location,
            "city": "烟台",
            "types": "住宿服务",
            "output": "JSON"
        }

        # 使用高德地图API搜索酒店信息
        response = requests.get(
            "https://restapi.amap.com/v3/place/text",
            params=params,
            timeout=10
        )

        logger.info(f"高德酒店API响应状态码: {response.status_code}")

        # 检查响应状态
        if response.status_code != 200:
            logger.warning("高德酒店API调用失败，使用模拟数据")
            return get_fallback_accommodation_info(location, budget)

        data = response.json()
        logger.info(f"高德酒店API返回JSON: {data}")

        # 检查API返回状态
        if data.get('status') != '1':
            logger.warning("高德酒店API返回错误状态，使用模拟数据")
            return get_fallback_accommodation_info(location, budget)

        # 解析真实API数据
        pois = data.get('pois', [])
        logger.info(f"找到 {len(pois)} 个住宿点")

        if pois:
            accommodations = []
            # 增加返回的酒店数量到30个，提高选择范围
            for poi in pois[:30]:
                biz_ext = poi.get('biz_ext', {})

                # 改进价格信息处理逻辑
                price_info = biz_ext.get('cost')
                # 如果cost字段没有有效信息，尝试其他可能的价格字段
                if not price_info or price_info in ["[]", ""]:
                    # 尝试从deepinfo中获取价格信息
                    deep_info = poi.get('deepinfo', {})
                    if deep_info:
                        # 检查是否有价格相关信息
                        price_fields = ['price', 'avg_price', 'min_price', 'cost']
                        for field in price_fields:
                            if deep_info.get(field):
                                price_info = deep_info.get(field)
                                break

                # 如果仍然没有价格信息，尝试从其他字段获取
                if not price_info or price_info in ["[]", ""]:
                    # 检查是否有其他可能包含价格的字段
                    if biz_ext.get('price'):
                        price_info = biz_ext.get('price')

                # 如果仍然没有价格信息，设置为"需现场查询"
                if not price_info or price_info in ["[]", ""]:
                    price_info = "需现场查询"

                accommodation = {
                    "hotel": poi.get('name', location),
                    "price": price_info,
                    "rating": biz_ext.get('rating', '暂无评分'),
                    "address": poi.get('address', '未知'),
                    "contact": poi.get('tel', '未知'),
                    "distance": poi.get('distance', '未知距离')
                }
                accommodations.append(accommodation)
                logger.info(f"添加住宿信息: {accommodation}")

            # 如果指定了预算，筛选符合预算的住宿
            if budget:
                filtered_accommodations = []
                for acc in accommodations:
                    try:
                        # 尝试解析价格信息
                        price_str = str(acc["price"])
                        if price_str and price_str not in ["需现场查询", "[]", ""]:
                            # 提取价格数字
                            price_numbers = re.findall(r'\d+', price_str)
                            if price_numbers:
                                price = int(price_numbers[0])
                                # 给预算增加10%的宽容度，避免过于严格
                                if price <= budget * 1.1:
                                    acc["parsed_price"] = price  # 添加解析后的价格用于排序
                                    filtered_accommodations.append(acc)
                                    logger.info(f"符合预算的住宿: {acc['hotel']} - {price}元")
                            else:
                                # 如果无法提取数字但包含价格信息，仍然保留
                                filtered_accommodations.append(acc)
                        else:
                            # 无法查询价格的也保留，供用户现场确认
                            filtered_accommodations.append(acc)
                    except Exception as e:
                        logger.warning(f"解析价格时出错: {e}")
                        # 出错时仍然保留该住宿
                        filtered_accommodations.append(acc)

                # 按价格排序，价格低的在前面
                filtered_accommodations.sort(key=lambda x: x.get("parsed_price", float('inf')))

                logger.info(f"预算筛选后找到 {len(filtered_accommodations)} 个住宿选项")

                # 返回最多30个符合预算的住宿选项
                if filtered_accommodations:
                    result = filtered_accommodations[:30]
                    logger.info(f"返回符合预算的住宿信息: {result}")
                    return result
                else:
                    # 如果没有完全符合预算的，返回价格最接近的5个选项
                    accommodations.sort(key=lambda x: x.get("parsed_price", float('inf')))
                    result = accommodations[:5]
                    logger.info(f"返回最接近预算的住宿信息: {result}")
                    return result
            else:
                # 没有预算限制时，按评分排序
                def get_rating_value(rating_str):
                    try:
                        return float(rating_str.replace("星", ""))
                    except:
                        return 0

                accommodations.sort(key=lambda x: get_rating_value(x["rating"]), reverse=True)
                result = accommodations[:30]
                logger.info(f"无预算限制时返回评分最高的住宿信息: {result}")
                return result
        else:
            result = get_fallback_accommodation_info(location, budget)
            logger.info(f"使用备用住宿数据: {result}")
            return result

    except Exception as e:
        logger.error(f"住宿信息获取异常: {e}")
        fallback_result = get_fallback_accommodation_info(location, budget)
        logger.info(f"返回备用住宿数据: {fallback_result}")
        return fallback_result


def get_fallback_accommodation_info(location, budget=None):
    """备用住宿信息，支持预算筛选"""
    # 根据预算提供不同的住宿选项
    if budget and budget <= 150:
        # 经济型住宿（预算150元以下）
        budget_options = [
            {
                "hotel": "烟台如家快捷酒店",
                "price": "120-150元/晚",
                "rating": "4.0星",
                "address": "烟台市芝罘区南大街",
                "contact": "0535-1234567"
            },
            {
                "hotel": "烟台7天连锁酒店",
                "price": "100-130元/晚",
                "rating": "3.8星",
                "address": "烟台市莱山区迎春大街",
                "contact": "0535-2345678"
            },
            {
                "hotel": "烟台汉庭酒店",
                "price": "130-160元/晚",
                "rating": "4.2星",
                "address": "烟台市芝罘区北大街",
                "contact": "0535-3456789"
            }
        ]
        return budget_options
    elif budget and budget <= 300:
        # 中档住宿（预算150-300元）
        mid_options = [
            {
                "hotel": "烟台海景花园酒店",
                "price": "200-300元/晚",
                "rating": "4.3星",
                "address": "烟台市芝罘区海滨路",
                "contact": "0535-4567890"
            },
            {
                "hotel": "烟台华泰酒店",
                "price": "180-280元/晚",
                "rating": "4.1星",
                "address": "烟台市莱山区观海路",
                "contact": "0535-5678901"
            }
        ]
        return mid_options
    else:
        # 默认高档住宿
        return {
            "hotel": "烟台国际大酒店",
            "price": "300-500元/晚",
            "rating": "4.5星",
            "address": "烟台市芝罘区",
            "contact": "0535-1234567"
        }


# 加载和处理文档
def load_documents():
    """加载烟台旅游相关文档"""
    try:
        documents = []
        data_files = [
            "./data/attractions.txt",
            "./data/transport.txt",
            "./data/accommodation.txt",
            "./data/food.txt",
            "./data/activities.txt"
        ]

        for file_path in data_files:
            if os.path.exists(file_path):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
                logger.info(f"成功加载文档: {file_path}")
            else:
                logger.warning(f"文档不存在: {file_path}")

        return documents
    except Exception as e:
        logger.error(f"加载文档时出错: {e}")
        return []


# 创建向量数据库
def create_vectorstore():
    """创建向量数据库"""
    documents = load_documents()
    if not documents:
        logger.error("没有可用的文档数据")
        return None

    # 分割文档
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(documents)

    # 创建向量数据库
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    logger.info("向量数据库创建成功")
    return vectorstore


# 初始化向量数据库
vectorstore = create_vectorstore()
if vectorstore:
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="yantai_travel_search",
        description="搜索烟台旅游相关信息，包括景点介绍、历史文化、特色美食等",
    )
else:
    retriever_tool = None
    logger.warning("向量数据库初始化失败")

# 创建工具列表
tools = [get_yantai_weather, get_ticket_info, get_transport_info, get_accommodation_info]
if retriever_tool:
    tools.append(retriever_tool)

# 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的烟台旅游规划助手。请根据用户需求制定详细的旅游计划。

重要规则：
1. 对于天气信息，只需在行程概述中简要说明总体天气情况，不需要重复每天的详细天气
2. 行程安排要合理，考虑景点之间的距离和开放时间
3. 提供实用的门票、交通、住宿信息
4. 避免重复相同的信息
5. 以清晰的结构化格式输出
6. 特别注意用户的住宿预算要求，推荐符合预算的具体住宿选项
7. 对于住宿推荐，需要提供酒店名称、价格范围、评分、地址和联系电话
8. 当用户有预算限制时，至少提供3个符合预算的住宿选项，并按价格从低到高排序
9. 如果没有完全符合预算的住宿，推荐价格最接近预算的选项，并明确说明价格差异

输出结构：
## 总体天气情况
[简要说明总体天气趋势]

## 行程概述
[总体行程安排]

## 每日详细安排
### 第1天 [日期]
- **上午**: [活动安排]
- **下午**: [活动安排] 
- **晚上**: [活动安排]
- **住宿**: [推荐住宿，包括酒店名称、价格、地址和联系电话]
- **餐饮**: [推荐美食]

## 费用估算
[门票、交通、住宿等费用的详细估算，需分别列出各项费用]

## 实用贴士
[注意事项和建议]"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 创建Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 创建带历史记录的Agent
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


agent_with_chat_history = RunnableWithMessageHistory(
    runnable=agent_executor,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


def extract_days_from_query(user_input):
    """从用户输入中提取天数信息"""
    day_match = re.search(r'(\d+)[天日]', user_input)
    if day_match:
        return int(day_match.group(1))
    return 3  # 默认3天


def extract_budget_from_query(user_input):
    """从用户输入中提取预算信息"""
    # 支持多种预算表达方式
    patterns = [
        r'住宿预算(?:只有|为)?(\d+)元',
        r'预算(?:只有|为)?(\d+)元',
        r'总共(\d+)元',
        r'预算(\d+)块',
        r'花(\d+)元',
        r'不超过(\d+)元'
    ]

    for pattern in patterns:
        budget_match = re.search(pattern, user_input)
        if budget_match:
            return int(budget_match.group(1))
    return None


# 生成旅游计划
def generate_travel_plan(user_input: str, session_id: str = "default"):
    """生成旅游计划"""
    try:
        # 提取天数信息
        days = extract_days_from_query(user_input)

        # 提取预算信息
        budget = extract_budget_from_query(user_input)

        # 先获取天气信息
        weather_info = get_yantai_weather.invoke({"days": days})

        # 构建天气摘要
        weather_summary = "天气预报信息：\n"
        for daily in weather_info.get('data', []):
            weather_summary += f"- {daily['date']}: {daily['weather']}, 温度{daily['temperature']}, {daily['clothing']}\n"

        # 如果有预算信息，添加到增强查询中
        if budget:
            enhanced_input = f"{user_input}\n\n住宿预算：{budget}元/晚\n\n{weather_summary}"
        else:
            enhanced_input = f"{user_input}\n\n{weather_summary}"

        logger.info(f"生成行程计划，天数: {days}, 预算: {budget}, 会话ID: {session_id}")

        response = agent_with_chat_history.invoke(
            {"input": enhanced_input},
            config={"configurable": {"session_id": session_id}},
        )

        logger.info("行程计划生成成功")
        return response

    except Exception as e:
        logger.error(f"生成计划时出错: {str(e)}")
        return {"output": f"生成计划时出错: {str(e)}"}


# API路由
@app.route('/api/travel-plan', methods=['POST'])
def create_travel_plan():
    """创建旅游计划API"""
    try:
        data = request.json
        user_input = data.get('query', '')
        session_id = data.get('session_id', str(uuid.uuid4()))

        if not user_input:
            return jsonify({
                'success': False,
                'error': '请输入旅游需求'
            }), 400

        result = generate_travel_plan(user_input, session_id)
        return jsonify({
            'success': True,
            'data': result['output'],
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"创建旅游计划API错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """聊天接口"""
    try:
        # 检查请求数据是否存在
        if not request.json:
            return jsonify({
                'success': False,
                'error': '请求数据格式错误'
            }), 400

        data = request.json
        user_input = data.get('message', '').strip() if data.get('message') else ''
        session_id = data.get('session_id', str(uuid.uuid4()))

        # 更严格的参数验证
        if not user_input or len(user_input) == 0:
            return jsonify({
                'success': False,
                'error': '请输入消息内容'
            }), 400

        if not session_id or len(session_id) == 0:
            return jsonify({
                'success': False,
                'error': '会话ID无效'
            }), 400

        result = agent_with_chat_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        return jsonify({
            'success': True,
            'data': result['output'],
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"聊天API错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/weather', methods=['GET'])
def get_weather():
    """获取天气信息"""
    try:
        days = request.args.get('days', 3, type=int)
        weather_info = get_yantai_weather.invoke({"days": days})
        return jsonify({
            'success': True,
            'data': weather_info
        })
    except Exception as e:
        logger.error(f"天气API错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ticket', methods=['GET'])
def get_ticket():
    """获取门票信息"""
    try:
        attraction = request.args.get('attraction', '')
        if not attraction:
            return jsonify({
                'success': False,
                'error': '请提供景点名称'
            }), 400

        ticket_info = get_ticket_info.invoke({"attraction": attraction})
        return jsonify({
            'success': True,
            'data': ticket_info
        })
    except Exception as e:
        logger.error(f"门票API错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/transport', methods=['GET'])
def get_transport():
    """获取交通信息"""
    try:
        origin = request.args.get('origin', '')
        destination = request.args.get('destination', '')

        if not origin or not destination:
            return jsonify({
                'success': False,
                'error': '请提供起点和终点'
            }), 400

        transport_info = get_transport_info.invoke({
            "origin": origin,
            "destination": destination
        })
        return jsonify({
            'success': True,
            'data': transport_info
        })
    except Exception as e:
        logger.error(f"交通API错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/accommodation', methods=['GET'])
def get_accommodation():
    """获取住宿信息"""
    try:
        location = request.args.get('location', '烟台')
        budget = request.args.get('budget', type=int)
        accommodation_info = get_accommodation_info.invoke({"location": location, "budget": budget})
        return jsonify({
            'success': True,
            'data': accommodation_info
        })
    except Exception as e:
        logger.error(f"住宿API错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'service': 'Yantai Tourism AI',
        'timestamp': datetime.now().isoformat()
    })


# 启动应用
if __name__ == '__main__':
    logger.info("正在启动烟台文旅AI服务...")

    if vectorstore is None:
        logger.warning("向量数据库初始化失败，将仅使用工具功能")
    else:
        logger.info("向量数据库初始化成功")

    logger.info("Flask应用启动成功，端口: 5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
