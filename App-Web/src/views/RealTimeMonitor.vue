<template>
  <div class="realtime-container">
    <div class="header-section">
      <h1>
        <span class="material-icons">monitor_heart</span>
        实时监测
      </h1>
      <!-- 安卓端悬浮返回按钮，仅显示箭头 -->
      <button class="back-btn-float" @click="$router.back()">
        <span class="material-icons back-arrow">arrow_back</span>
      </button>
      <p class="subtitle">基于WebSocket的移动端实时监控解决方案</p>
    </div>
    <div class="main-div">
      <div class="cards-div">
        <div class="card">
          <h2><span class="material-icons">image</span> 实时检测图像</h2>
          <div class="image-container">
            <img v-if="currentImage && !imageError" :src="currentImage" :alt="`检测图像 #${pcbCount}`" @error="onImageError" @load="onImageLoad" />
            <div v-else-if="imageError" class="placeholder error">图片加载异常</div>
            <div v-else class="placeholder">等待连接服务器获取实时图像...</div>
          </div>
          <div class="review-btn-section">
            <button class="review-btn" @click="handleFullModelReview" :disabled="!currentImage || imageError || !imageLoaded">调用全参数模型复查</button>
          </div>
          <!-- 复查结果弹窗，内容与Home页面一致，复用DetectionResult组件 -->
          <div v-if="showReviewResult" class="review-modal">
            <div class="review-modal-content">
              <button class="review-modal-close" @click="closeReviewResult">
                <span class="material-icons">close</span>
              </button>
              <DetectionResult v-if="reviewResult" :result="reviewResult" :imageUrl="reviewResult.imageUrl || ''" />
              <div v-if="reviewLoading" class="review-loading">正在复查，请稍候...</div>
            </div>
          </div>
        </div>
        <div class="card">
          <h2><span class="material-icons">info</span> 系统状态</h2>
          <div class="status-indicator">
            <span
              class="status-dot"
              :style="statusDotStyle"
              :class="{ pulse: isConnecting }"
            ></span>
            <span>{{ statusText }}</span>
          </div>
          <div class="stats-grid">
            <div class="stat-box">
              <div class="stat-value">{{ totalFlaws }}</div>
              <div class="stat-label">总缺陷数</div>
            </div>
            <div class="stat-box">
              <div class="stat-value">{{ newFlaws }}</div>
              <div class="stat-label">新发现缺陷</div>
            </div>
            <div class="stat-box">
              <div class="stat-value">{{ pcbCount }}</div>
              <div class="stat-label">检测开发板</div>
            </div>
          </div>
          <div class="uptime-section">
            <div class="uptime-label">系统运行时间:</div>
            <div class="uptime-value">{{ uptime }}</div>
          </div>
          <div class="ws-edit-section">
            <input
              v-model="wsUrlInput"
              :disabled="isConnected || isConnecting"
              class="ws-url-input"
              placeholder="输入WebSocket地址"
            />
            <button
              class="ws-url-icon-btn"
              :disabled="isConnected || isConnecting || !wsUrlInput"
              @click="changeWsUrl"
              title="修改连接"
            >
              <span class="material-icons">edit</span>
            </button>
          </div>
          <div class="controls">
            <button @click="toggleConnection">
              <span class="material-icons">{{ isConnected ? 'link_off' : (isConnecting ? 'link_off' : 'link') }}</span>
              {{ isConnected ? '断开连接' : (isConnecting ? '取消连接' : '连接服务器') }}
            </button>
            <button class="secondary clear-btn-fix" @click="clearData">
              <span class="material-icons">delete</span>
              <span class="clear-text">清除数据</span>
            </button>
          </div>
          <div class="connection-info">
            <div v-for="log in connectionLogs" :key="log.timestamp">{{ log.text }}</div>
          </div>        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import DetectionResult from '../components/DetectionResult.vue';
import { useNotificationStore } from '../stores/notification';

// 使用通知Store
const notificationStore = useNotificationStore();

// 状态

const isConnected = ref(false);
const isConnecting = ref(false);
const isDarkTheme = ref(false);
const totalFlaws = ref(0); // 累加所有板子的缺陷数
const newFlaws = ref(0);   // 当前板子的缺陷数，直到下一个板子数据到来
const pcbCount = ref(0);   // 监测到的开发板数
const uptime = ref('00:00:00');
const currentImage = ref('');
const qualityText = ref('-');
const qualityClass = ref('');
const showReviewResult = ref(false);
const reviewResult = ref<any>(null);
const reviewLoading = ref(false);
const imageError = ref(false);
const imageLoaded = ref(false);
const connectionLogs = ref<{text:string,timestamp:number}[]>([{text:'正在初始化 WebSocket 连接...',timestamp:Date.now()}]);
// 移除自定义通知系统，使用notificationStore
// 主色/副色边框切换
function resetImageAndQuality() {
  currentImage.value = '';
  imageError.value = false;
  imageLoaded.value = false;
  qualityText.value = '-';
  qualityClass.value = '';
}
let uptimeSeconds = 0;
let uptimeTimer: any = null;
let ws: WebSocket | null = null;
 // let lastEventTime: string | null = null; // 已废弃，不再使用
let lastBoardId: string | null = null; // 用于区分不同开发板
const wsUrlInput = ref('ws://10.21.207.212:5002/6874f19632771f177b4c0504_60372b766c9b6ad4');
let wsUrl = wsUrlInput.value;

const statusText = computed(() => {
  if (isConnecting.value) return '连接中...';
  if (isConnected.value) return '已连接 - 实时监控中';
  return '连接已断开';
});

const statusDotStyle = computed(() => {
  if (isConnecting.value) {
    return { backgroundColor: '#FFC107' };
  } else if (isConnected.value) {
    return { backgroundColor: '#4CAF50' };
  } else {
    return { backgroundColor: '#F44336' };
  }
});




function toggleConnection() {
  if (isConnecting.value) {
    isConnecting.value = false;
    isConnected.value = false;
    if (ws) ws.close();
    logConnection('连接已取消');
    notificationStore.showNotification('连接已取消', 'info');
    return;
  }
  if (isConnected.value) {
    isConnected.value = false;
    if (ws) ws.close();
    logConnection('手动断开连接');
    // 不在这里弹窗，onclose里弹一次即可
    stopUptime();
    resetImageAndQuality();
    return;
  }
  connectWebSocket();
}


function connectWebSocket() {
  isConnecting.value = true;
  logConnection(`正在连接到 WebSocket 服务器: ${wsUrl}`);
  ws = new WebSocket(wsUrl);
  ws.onopen = () => {
    isConnecting.value = false;
    isConnected.value = true;
    logConnection('连接成功');
    notificationStore.showNotification('已连接到实时检测服务器', 'check');
    startUptime();
  };
  ws.onclose = () => {
    isConnected.value = false;
    isConnecting.value = false;
    logConnection('连接已关闭');
    notificationStore.showNotification('WebSocket 连接已断开', 'info');
    stopUptime();
    resetImageAndQuality();
  };
  ws.onerror = () => {
    logConnection('WebSocket 发生错误');
    notificationStore.showNotification('WebSocket 发生错误', 'error');
  };
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      // 兼容后端示例数据结构
      let properties = data;
      if (data.result && data.result.shadow && data.result.shadow[0] && data.result.shadow[0].reported && data.result.shadow[0].reported.properties) {
        properties = data.result.shadow[0].reported.properties;
      }
      // 处理数据
      updateFromProperties(properties);
    } catch (err) {
      logConnection('数据解析失败: ' + String(err));
    }
  };
}

function changeWsUrl() {
  if (isConnected.value || isConnecting.value) return;
  wsUrl = wsUrlInput.value;
  logConnection('已修改 WebSocket 地址: ' + wsUrl);
  notificationStore.showNotification('WebSocket 地址已修改', 'info');
}

function startUptime() {
  uptimeSeconds = 0;
  updateUptimeDisplay();
  if (uptimeTimer) clearInterval(uptimeTimer);
  uptimeTimer = setInterval(() => {
    uptimeSeconds++;
    updateUptimeDisplay();
  }, 1000);
}
function stopUptime() {
  if (uptimeTimer) clearInterval(uptimeTimer);
}
function updateUptimeDisplay() {
  const h = String(Math.floor(uptimeSeconds / 3600)).padStart(2, '0');
  const m = String(Math.floor((uptimeSeconds % 3600) / 60)).padStart(2, '0');
  const s = String(uptimeSeconds % 60).padStart(2, '0');
  uptime.value = `${h}:${m}:${s}`;
}

function onImageError() {
  imageError.value = true;
  imageLoaded.value = false;
}
function onImageLoad() {
  imageError.value = false;
  imageLoaded.value = true;
}

import { usePCBStore } from '../stores/pcb';
const pcbStore = usePCBStore();

async function handleFullModelReview() {
  if (!currentImage.value || imageError.value) {
    notificationStore.showNotification('当前图片无法获取，无法复查', 'error');
    return;
  }
  reviewLoading.value = true;
  showReviewResult.value = false;
  try {
    // 通过fetch获取图片blob
    const response = await fetch(currentImage.value);
    if (!response.ok) throw new Error('图片下载失败');
    const blob = await response.blob();
    const file = new File([blob], 'realtime.jpg', { type: blob.type });
    await pcbStore.uploadImage(file);
    if (pcbStore.currentResult) {
      reviewResult.value = { ...pcbStore.currentResult, imageUrl: pcbStore.currentResult.annotated_image_url };
      showReviewResult.value = true;
    } else {
      notificationStore.showNotification('未获取到检测结果', 'error');
    }
  } catch (e) {
    notificationStore.showNotification('无法获取实时图片或上传失败', 'error');
  } finally {
    reviewLoading.value = false;
  }
}

function closeReviewResult() {
  showReviewResult.value = false;
}

function updateFromProperties(props: any) {
  // 假设每个开发板有唯一标识符，优先用 board_id/device_id，否则用 event_time 区分
  const boardId = props.board_id || props.device_id || props.event_time || Math.random().toString();

  // 总缺陷数：累加每个板子的 total_flaws_detected
  if (typeof props.total_flaws_detected === 'number') {
    // 如果是新板子（或新数据），累加
    if (boardId !== lastBoardId) {
      totalFlaws.value += props.total_flaws_detected;
      pcbCount.value += 1;
      newFlaws.value = props.total_flaws_detected;
      lastBoardId = boardId;
      notificationStore.showNotification(`检测到 ${props.total_flaws_detected} 个缺陷`, 'warning');
      // 新监测到异常开发板（有缺陷）弹窗
      if (props.total_flaws_detected > 0) {
        notificationStore.showNotification(`发现异常开发板：${props.total_flaws_detected} 个缺陷`, 'error');
      }
    }
    // 如果是同一板子重复上报，不做处理
  }
  // 实时图像：每次都更新
  if (props.streaming_url) {
    currentImage.value = props.streaming_url;
    imageError.value = false;
    imageLoaded.value = false;
  }
  // 质量指示（已废弃）
  // updateQualityIndicator(props.total_flaws_detected || 0);
}
// 检测质量相关逻辑已废弃
function clearData() {
  totalFlaws.value = 0;
  newFlaws.value = 0;
  pcbCount.value = 0;
  lastBoardId = null;
  resetImageAndQuality();
  logConnection('已清除所有数据');
  notificationStore.showNotification('数据已重置', 'info');
}
function logConnection(msg: string) {
  connectionLogs.value.unshift({ text: `[${new Date().toLocaleTimeString()}] ${msg}`, timestamp: Date.now() });
  if (connectionLogs.value.length > 8) connectionLogs.value.pop();
}
onMounted(() => {
  document.body.classList.toggle('dark-theme', isDarkTheme.value);
});
</script>

<style scoped>
.review-btn-section {
  display: flex;
  justify-content: center;
  margin-top: 1.2rem;
}
.review-btn {
  background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
  color: #fff;
  border: none;
  border-radius: 30px;
  padding: 0.8rem 1.5rem;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(74,0,224,0.08);
  transition: all 0.2s;
}
.review-btn:disabled {
  background: #ccc;
  color: #fff;
  cursor: not-allowed;
}
.review-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0,0,0,0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
}
.review-modal-content {
  background: var(--card-bg);
  border-radius: 12px;
  padding: 2rem 1.5rem;
  max-width: 90vw;
  min-width: 320px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.18);
  position: relative;
  text-align: center;
}
.review-modal-close {
  position: absolute;
  top: 12px;
  right: 12px;
  background: none;
  border: none;
  font-size: 1.5rem;
  color: var(--primary-color);
  cursor: pointer;
}
.review-loading {
  font-size: 1.1rem;
  color: var(--primary-color);
  margin: 2rem 0;
}
.realtime-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1rem;
  min-height: calc(100vh - 3em);
  padding-bottom: 4rem;
}

.header-section {
  text-align: center;
  padding: 1rem 0;
  margin-bottom: 1rem;
  width: 100%;
  position: relative;
}

.header-section h1 {
  font-size: 2rem;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.header-section .material-icons {
  font-size: 2.2rem;
  color: var(--primary-color);
}

.back-btn {
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  background: #fff;
  color: var(--primary-color);
  border: 1px solid var(--primary-color);
  border-radius: 30px;
  padding: 0.5rem 1.2rem;
  font-size: 1rem;
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 2px 8px rgba(74,0,224,0.08);
}
.back-btn:hover {
  background: #f5f5f5;
  color: var(--primary-color);
}
.back-arrow {
  color: var(--primary-color) !important;
  transition: none;
}

.status-dot {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-right: 8px;
  vertical-align: middle;
  transition: background 0.3s;
}
.pulse {
  animation: pulse 1.5s infinite;
}
.clear-btn-fix {
  display: flex;
  align-items: center;
  gap: 6px;
  justify-content: center;
}
.clear-btn-fix .material-icons {
  font-size: 1.2em;
  vertical-align: middle;
}
.clear-text {
  display: inline-block;
  vertical-align: middle;
}

.subtitle {
  font-size: 1rem;
  color: var(--text-secondary);
  max-width: 100%;
  margin: 0 auto;
  padding: 0 1rem;
}

.main-div {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.cards-div {
  width: 98%;
  max-width: 600px;
  display: flex;
  flex-direction: column;
}

.card {
  /* 使用全局 .card 样式，避免局部覆盖 */
  width: 100%;
  box-sizing: border-box;
}
.card h2 {
  font-size: 1.3rem;
  margin-bottom: 1.2rem;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--primary-color);
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 0.8rem;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 1.2rem;
}
.indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: var(--success-color);
}
.indicator.connecting {
  background-color: var(--warning-color);
  animation: pulse 1.5s infinite;
}
.indicator.disconnected {
  background-color: var(--error-color);
}
.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.8rem;
  margin-bottom: 1.2rem;
}
.stat-box {
  background: rgba(74, 0, 224, 0.08);
  border-radius: 10px;
  padding: 1.2rem 0.7rem;
  text-align: center;
  transition: var(--transition);
}
.stat-box:hover {
  background: rgba(74, 0, 224, 0.15);
}
.stat-value {
  font-size: 1.4rem;
  font-weight: bold;
  margin: 0.2rem 0;
  color: var(--primary-color);
}
.stat-label {
  font-size: 0.85rem;
  color: var(--text-secondary);
}
.image-container {
  position: relative;
  height: 250px;
  border-radius: 10px;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.05);
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border-color);
  margin-bottom: 1rem;
}
.image-container img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  transition: transform 0.3s ease;
}
.image-container img:hover {
  transform: scale(1.03);
}
.placeholder {
  color: var(--text-secondary);
  font-size: 1rem;
  text-align: center;
  padding: 20px;
}
.placeholder.error {
  color: var(--error-color);
}
.controls {
  display: flex;
  justify-content: space-between;
  gap: 0.8rem;
  margin-top: 1rem;
  flex-wrap: wrap;
}
.controls button {
  font-weight: 500;
  flex: 1;
  min-width: 140px;
  justify-content: center;
}
.controls button:not(.secondary) {
  background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
  color: #fff;
  border: none;
  padding: 0.8rem 1.2rem;
  border-radius: 30px;
  font-size: 1rem;
  cursor: pointer;
  transition: var(--transition);
  display: flex;
  align-items: center;
  gap: 8px;
}
.controls button:not(.secondary):hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 10px rgba(100, 108, 255, 0.18);
}
.controls button.secondary {
  background: var(--surface-color);
  color: var(--primary-color);
  border: 1px solid var(--primary-color);
}

.connection-info {
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.03);
  border-radius: 8px;
  font-family: monospace;
  font-size: 0.85rem;
  overflow: auto;
  max-height: 150px;
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
}
.uptime-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.8rem;
  background: rgba(0, 0, 0, 0.03);
  border-radius: 8px;
  margin-top: 1rem;
  border: 1px solid var(--border-color);
  font-size: 1rem;
}
.ws-edit-section {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 1rem 0 0.5rem 0;
  max-width: 340px;
}
.ws-url-input {
  flex: 1;
  padding: 0.5rem 0.5rem;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  font-size: 1rem;
}
.ws-url-btn {
  padding: 0.5rem 1.2rem;
  border-radius: 6px;
  background: var(--primary-color);
  color: #fff;
  border: none;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.2s;
}
.ws-url-btn:disabled {
  background: #ccc;
  color: #fff;
  cursor: not-allowed;
}
.uptime-label { color: var(--text-secondary); }
.uptime-value { font-weight: bold; color: var(--primary-color); }
.quality-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-top: 1rem;
  padding: 0.8rem;
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.03);
  border: 1px solid var(--border-color);
}
.quality-value { font-weight: bold; font-size: 1.2rem; }
.quality-excellent { color: var(--success-color); }
.quality-good { color: #8BC34A; }
.quality-fair { color: var(--warning-color); }
.quality-poor { color: var(--error-color); }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
@media (max-width: 480px) {
  .realtime-container { padding: 0.5rem; }
  .header-section { padding: 0.8rem 0; }
  .header-section h1 { font-size: 1.5rem; }
  .header-section .material-icons { font-size: 1.7rem; }
  .card { padding: 1.2rem; }
  .image-container { height: 180px; }
  .stat-value { font-size: 1.1rem; }
  button { padding: 0.7rem 1rem; font-size: 0.9rem; min-width: 120px; }
  .stats-grid { gap: 0.5rem; }
  .stat-box { padding: 1.2rem 0.7rem; }
}
@media (max-width: 360px) {
  .stat-value { font-size: 1rem; }
  .image-container { height: 120px; }
  .subtitle { font-size: 0.9rem; }
  .card h2 { font-size: 1.1rem; }
  button { min-width: 100%; }
}
@media (max-width: 600px) {
  .back-btn { position: static; transform: none; margin-bottom: 1rem; }
}

/* 安卓端悬浮返回按钮样式 */
/* 保证返回按钮为正圆 */
.back-btn-float {
  position: fixed;
  top: 20px;
  left: 20px;
  z-index: 2100;
  background: var(--card-bg);
  color: var(--primary-color);
  border: 1.5px solid var(--primary-color);
  border-radius: 50%;
  width: 44px !important;
  height: 44px !important;
  aspect-ratio: 1/1;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(74,0,224,0.08);
  cursor: pointer;
  transition: all 0.2s;
  font-size: 1.5rem;
  padding: 0;
  min-width: 0;
  min-height: 0;
}
.back-btn-float .material-icons {
  font-size: 2rem;
  color: var(--primary-color);
}
.back-btn-float:active {
  background: #f5f5f5;
}

/* 隐藏原返回按钮样式 */
.back-btn {
  display: none !important;
}

/* 修改连接按钮为图标按钮 */
/* 保证修改连接按钮为正圆 */
.ws-url-icon-btn {
  background: var(--primary-color);
  color: #fff;
  border: none;
  border-radius: 50%;
  width: 35px !important;
  height: 35px !important;
  aspect-ratio: 1/1;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.3rem;
  cursor: pointer;
  transition: background 0.2s;
  margin-left: 4px;
  padding: 0;
  min-width: 0;
  min-height: 0;
}
.ws-url-icon-btn:disabled {
  background: #ccc;
  color: #fff;
  cursor: not-allowed;
}
.ws-url-icon-btn .material-icons {
  font-size: 1.3rem;
}
</style>
