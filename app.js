// app.js
App({
  globalData: {
    // 根据实际部署情况修改地址
    // 开发环境使用本地IP，生产环境使用域名
    apiUrl: "http://localhost:5000/api", // 开发时替换为实际IP
    sessionId: null
  },
  
  onLaunch: function() {
    // 应用启动时初始化
    console.log('烟台旅游助手小程序启动');
    // 初始化会话ID
    this.globalData.sessionId = 'session_' + Date.now();
  }
})
