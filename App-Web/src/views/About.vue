<template>
  <div class="about-container">
    <div class="header-section">
      <h1>
        <span class="material-icons">info</span>
        关于
      </h1>
    </div>

    <div class="content-section">
      <section class="intro-section">
        <h2>系统介绍</h2>
        <p>
          PCB缺陷检测系统是一个基于人工智能的电路板质量检测平台。
          通过先进的计算机视觉技术，能够自动识别和标记PCB上的各种缺陷，
          提高生产效率，确保产品质量。
        </p>
      </section>

      <div class="features-grid">
        <div class="feature-card">
          <span class="material-icons feature-icon">precision_manufacturing</span>
          <h3>智能检测</h3>
          <p>基于深度学习的AI模型，准确识别PCB上的各类缺陷</p>
        </div>

        <div class="feature-card">
          <span class="material-icons feature-icon">speed</span>
          <h3>高效处理</h3>
          <p>快速分析处理，几秒钟内获得检测结果</p>
        </div>

        <div class="feature-card">
          <span class="material-icons feature-icon">analytics</span>
          <h3>详细分析</h3>
          <p>提供缺陷位置、类型、置信度等详细信息</p>
        </div>

        <div class="feature-card">
          <span class="material-icons feature-icon">history</span>
          <h3>历史记录</h3>
          <p>完整的检测历史，支持查询和回溯</p>
        </div>
      </div>

      <section class="tech-section">
        <h2>技术特点</h2>
        <div class="tech-list">
          <div class="tech-item">
            <span class="material-icons">computer</span>
            <div>
              <strong>前端技术</strong>
              <p>Vue 3 + TypeScript + Ionic，现代化跨平台应用</p>
            </div>
          </div>
          
          <div class="tech-item">
            <span class="material-icons">memory</span>
            <div>
              <strong>AI模型</strong>
              <p>基于YOLO模型，高精度高效率缺陷识别</p>
            </div>
          </div>
          
          <div class="tech-item">
            <span class="material-icons">api</span>
            <div>
              <strong>API接口</strong>
              <p>RESTful API设计，支持图片上传和结果查询</p>
            </div>
          </div>
          
          <div class="tech-item">
            <span class="material-icons">storage</span>
            <div>
              <strong>数据存储</strong>
              <p>使用PostgreSQL数据库，支持历史数据查询</p>
            </div>
          </div>
        </div>
      </section>

      <section class="usage-section">
        <h2>使用说明</h2>
        <div class="usage-steps">
          <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
              <h4>上传图片</h4>
              <p>在首页点击或拖拽上传PCB图片</p>
            </div>
          </div>
          
          <div class="step">
            <div class="step-number">2</div>
            <div class="step-content">
              <h4>AI分析</h4>
              <p>系统自动分析图片，检测潜在缺陷</p>
            </div>
          </div>
          
          <div class="step">
            <div class="step-number">3</div>
            <div class="step-content">
              <h4>查看结果</h4>
              <p>获得详细的检测报告和缺陷标注</p>
            </div>
          </div>
          
          <div class="step">
            <div class="step-number">4</div>
            <div class="step-content">
              <h4>历史查询</h4>
              <p>在历史记录页面查看所有检测记录</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.header-section {
  text-align: center;
}

.about-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 3rem 1rem;
  min-height: calc(100vh - 3em);
  padding-bottom: 4rem;
}


.content-section {
  display: flex;
  flex-direction: column;
  gap: 3rem;
}

.intro-section {
  text-align: center;
}

.intro-section h2 {
  color: var(--text-color);
  margin-bottom: 1rem;
}

.intro-section p {
  color: var(--text-secondary);
  font-size: 1.1rem;
  line-height: 1.6;
  max-width: 800px;
  margin: 0 auto;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
}

.feature-card {
  background: var(--card-bg);
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
  transition: all 0.3s ease;
}

.feature-icon {
  font-size: 3rem;
  color: var(--primary-color);
  margin-bottom: 1rem;
  display: block;
}

.feature-card h3 {
  color: var(--text-color);
  margin: 0 0 1rem 0;
  font-size: 1.3rem;
}

.feature-card p {
  color: var(--text-secondary);
  line-height: 1.5;
  margin: 0;
}

.tech-section h2 {
  color: var(--text-color);
  margin-bottom: 2rem;
  text-align: center;
}

.tech-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
}

.tech-item {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  padding: 1.5rem;
  background: var(--card-bg);
  border-radius: 8px;
}

.tech-item .material-icons {
  font-size: 2rem;
  color: var(--primary-color);
  margin-top: 0.2rem;
  flex-shrink: 0;
}

.tech-item strong {
  color: var(--text-color);
  display: block;
  margin-bottom: 0.5rem;
  font-size: 1.1rem;
}

.tech-item p {
  color: var(--text-secondary);
  margin: 0;
  line-height: 1.5;
}

.usage-section h2 {
  color: var(--text-color);
  margin-bottom: 2rem;
  text-align: center;
}

.usage-steps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
}

@media (max-width: 768px) {
  .usage-steps {
    grid-template-columns: 1fr;
  }
}

.step {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  gap: 2rem;
  padding: 1.5rem;
  background: var(--card-bg);
  border-radius: 12px;
}

.step-number {
  width: 2.5rem;
  height: 2.5rem;
  background: var(--primary-color);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 1.1rem;
  flex-shrink: 0;
}

.step-content h4 {
  color: var(--text-color);
  margin: 0 0 0.5rem 0;
  font-size: 1.1rem;
  width: 100%;
}

.step-content p {
  color: var(--text-secondary);
  margin: 0;
  line-height: 1.5;
}
</style>
