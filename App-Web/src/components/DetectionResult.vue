<template>
  <div style="width: 100%;">
    <div class="detection-result">
      <div class="result-header">
        <h3>
          <span class="material-icons">analytics</span>
          检测结果
        </h3>
      </div>

      <div class="result-content">
        <!-- 图片显示区域 -->
        <div class="image-section">
          <div class="image-container" @touchstart="handleTouchStart" @touchmove="handleTouchMove"
            @touchend="handleTouchEnd" @mousedown="handleMouseDown" @mousemove="handleMouseMove"
            @mouseup="handleMouseUp" @mouseleave="handleMouseUp" @wheel="handleWheel" @dblclick="resetZoom">
            <img :src="imageUrl" alt="检测图片" class="result-image" :style="imageStyle">
          </div>
          <div class="image-info">
            <p class="image-note">
              <span class="material-icons">info</span>
              图片已自动标注检测到的缺陷位置
            </p>
          </div>
        </div>

        <!-- 检测统计 -->
        <div class="summary-section">
          <div class="summary-card">
            <div class="summary-item">
              <span class="material-icons summary-icon" :class="result.summary.total_defects ? 'defect' : 'good'">
                {{ result.summary.total_defects ? 'error' : 'check' }}
              </span>
              <div>
                <span class="summary-number">{{ result.summary.total_defects }}</span>
                个缺陷
              </div>
            </div>
          </div>

          <!-- 缺陷类型分布 -->
          <div v-if="defectTypes.length > 0" class="defect-types">
            <h4>缺陷类型分布</h4>
            <div class="defect-list">
              <div v-for="[type, count] in defectTypes" :key="type" class="defect-item">
                <span class="defect-name">{{ type }}</span>
                <span class="defect-count">{{ count }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- 详细检测结果 -->
        <div class="details-section">
          <h4>详细检测结果</h4>
          <div v-if="result.detection_results.length === 0" class="no-defects">
            <span class="material-icons">check_circle</span>
            <p>未检测到缺陷</p>
          </div>
          <div v-else class="detection-list">
            <div v-for="(detection, index) in result.detection_results" :key="index" class="detection-item"
              @click="focusOnDetection(detection.bbox)">
              <div class="detection-header">
                <span class="detection-class">{{ detection.class }}</span>
                <span class="detection-confidence">
                  置信度: {{ (detection.confidence * 100).toFixed(1) }}%
                </span>
              </div>
              <div class="detection-details">
                <div class="detection-bbox">
                  <span class="material-icons">crop_free</span>
                  <span>
                    位置 [{{detection.bbox.map(v => Math.round(v)).join(', ')}}]
                  </span>
                </div>
                <div class="confidence-bar">
                  <div class="confidence-fill" :style="{ width: (detection.confidence * 100) + '%' }"></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 时间戳 -->
        <div class="timestamp">
          <span class="material-icons">schedule</span>
          检测时间: {{ formatTimestamp(result.timestamp) }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import type { InferenceResult } from '../services/pcbApi';
import { ref } from 'vue';

interface Props {
  result: InferenceResult;
  imageUrl: string;
}

const props = defineProps<Props>();

// 图片缩放相关状态
const scale = ref(1);
const translateX = ref(0);
const translateY = ref(0);
const isZooming = ref(false);

// 触摸事件相关
let initialDistance = 0;
let initialScale = 1;
let initialTranslateX = 0;
let initialTranslateY = 0;
let lastTouchX = 0;
let lastTouchY = 0;
let lastTouchTime = 0;
let touchCount = 0;

// 鼠标事件相关
let isDragging = false;
let lastMouseX = 0;
let lastMouseY = 0;

const imageStyle = computed(() => ({
  transform: `scale(${scale.value}) translate(${translateX.value}px, ${translateY.value}px)`,
}));

// 计算两点间距离
const getDistance = (touch1: Touch, touch2: Touch) => {
  const dx = touch1.clientX - touch2.clientX;
  const dy = touch1.clientY - touch2.clientY;
  return Math.sqrt(dx * dx + dy * dy);
};

// 触摸开始
const handleTouchStart = (e: TouchEvent) => {
  e.preventDefault();

  const currentTime = Date.now();

  // 检测双击
  if (e.touches.length === 1) {
    touchCount++;
    if (touchCount === 1) {
      setTimeout(() => {
        touchCount = 0;
      }, 300);
    } else if (touchCount === 2 && currentTime - lastTouchTime < 300) {
      // 双击重置
      resetZoom();
      isZooming.value = true; // 防止手指松开时移动图片
      setTimeout(() => {
        isZooming.value = false;
      }, 300);
      touchCount = 0;
      return;
    }
    lastTouchTime = currentTime;
  }

  if (e.touches.length === 2) {
    // 双指缩放
    isZooming.value = true;
    initialDistance = getDistance(e.touches[0], e.touches[1]);
    initialScale = scale.value;
    // 记录当前位置，避免缩放时跳动
    initialTranslateX = translateX.value;
    initialTranslateY = translateY.value;
  } else if (e.touches.length === 1) {
    // 单指拖拽（允许在任何缩放状态下拖拽）
    lastTouchX = e.touches[0].clientX;
    lastTouchY = e.touches[0].clientY;
    initialTranslateX = translateX.value;
    initialTranslateY = translateY.value;
  }
};

// 触摸移动
const handleTouchMove = (e: TouchEvent) => {
  e.preventDefault();

  if (e.touches.length === 2 && isZooming.value) {
    // 双指缩放
    const currentDistance = getDistance(e.touches[0], e.touches[1]);
    const scaleChange = currentDistance / initialDistance;
    const newScale = Math.max(0.5, initialScale * scaleChange);
    scale.value = newScale;
  } else if (e.touches.length === 1 && !isZooming.value) {
    // 单指拖拽（仅在非缩放状态下才能拖拽，防止双指松开时差导致的跳动）
    const deltaX = e.touches[0].clientX - lastTouchX;
    const deltaY = e.touches[0].clientY - lastTouchY;
    translateX.value = initialTranslateX + deltaX / scale.value;
    translateY.value = initialTranslateY + deltaY / scale.value;
  }
};

// 触摸结束
const handleTouchEnd = (e: TouchEvent) => {
  // 重置缩放状态，但不重置位置，防止双指松开时差导致的图片跳动
  if (e.touches.length === 0) {
    isZooming.value = false;
  }
};

// 鼠标滚轮缩放（PC端）
const handleWheel = (e: WheelEvent) => {
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  scale.value = Math.max(0.5, scale.value * delta);
};

// 重置缩放
const resetZoom = () => {
  const image = document.querySelector('.result-image') as HTMLImageElement;
  if (!image) return;
  image.style.transition = 'transform 0.3s ease';
  scale.value = 1;
  translateX.value = 0;
  translateY.value = 0;
  setTimeout(() => {
    image.style.transition = 'none';
  }, 300);
};

// 聚焦到指定的检测区域
const focusOnDetection = (bbox: number[]) => {
  // bbox 格式：[x1, y1, x2, y2]
  const [x1, y1, x2, y2] = bbox;

  // 计算检测区域的中心点和大小
  const centerX = (x1 + x2) / 2;
  const centerY = (y1 + y2) / 2;
  const bboxWidth = Math.abs(x2 - x1);
  const bboxHeight = Math.abs(y2 - y1);
  // 获取图片容器元素
  const container = document.querySelector('.image-container') as HTMLElement;
  const image = document.querySelector('.result-image') as HTMLImageElement;
  if (!container || !image) return;

  // 获取容器的实际尺寸
  const containerRect = container.getBoundingClientRect();

  // 计算图片在容器中的显示尺寸（考虑 object-fit: contain）
  const containerWidth = containerRect.width;
  const containerHeight = containerRect.height;

  // 计算图片在原始尺寸下的中心点
  const imageNaturalWidth = image.naturalWidth || 1;
  const imageNaturalHeight = image.naturalHeight || 1;

  // 计算合适的缩放比例，让检测区域占视窗的30%左右
  const targetSize = Math.min(imageNaturalWidth, imageNaturalHeight) * 0.3;
  const bboxSize = Math.max(bboxWidth, bboxHeight);
  let targetScale = targetSize / bboxSize;

  // 计算图片显示尺寸
  const imageAspectRatio = imageNaturalWidth / imageNaturalHeight;
  const containerAspectRatio = containerWidth / containerHeight;

  let scaleFactor;
  if (imageAspectRatio > containerAspectRatio) {
    // 图片更宽，以容器宽度为准
    scaleFactor = containerWidth / imageNaturalWidth;
  } else {
    // 图片更高，以容器高度为准
    scaleFactor = containerHeight / imageNaturalHeight;
  }

  // 计算需要的偏移量（以显示坐标系为准）
  const offsetX = (centerX - imageNaturalWidth / 2) * scaleFactor;
  const offsetY = (centerY - imageNaturalHeight / 2) * scaleFactor

  image.style.transition = 'transform 1.0s ease';
  // 设置平移与缩放
  container.scrollIntoView({ behavior: 'smooth', block: 'center' });
  translateX.value = -offsetX;
  translateY.value = -offsetY;
  scale.value = targetScale;
  setTimeout(() => {
    image.style.transition = 'none'; // 禁用过渡效果，防止后续操作影响
  }, 1000);
};

// 鼠标按下
const handleMouseDown = (e: MouseEvent) => {
  isDragging = true;
  lastMouseX = e.clientX;
  lastMouseY = e.clientY;
  initialTranslateX = translateX.value;
  initialTranslateY = translateY.value;
  e.preventDefault();
};

// 鼠标移动
const handleMouseMove = (e: MouseEvent) => {
  if (isDragging) {
    const deltaX = e.clientX - lastMouseX;
    const deltaY = e.clientY - lastMouseY;
    translateX.value = initialTranslateX + deltaX / scale.value;
    translateY.value = initialTranslateY + deltaY / scale.value;
    e.preventDefault();
  }
};

// 鼠标松开
const handleMouseUp = () => {
  if (isDragging) {
    isDragging = false;
    // 移除边界限制，允许图片自由移动
  }
};

const defectTypes = computed(() => {
  return Object.entries(props.result.summary.defect_types);
});

const formatTimestamp = (timestamp: string) => {
  return new Date(timestamp).toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
};
</script>

<style scoped>
.detection-result {
  background: var(--surface-color);
  border-radius: 12px;
  padding: 3%;
  max-width: 1000px;
  margin: auto;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  border-bottom: 1px solid var(--border-color);
}

.result-header h3 {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0.2em;
  color: var(--text-color);
}

.clear-result-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: transparent;
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.clear-result-btn:active {
  background: var(--surface-hover-bg);
  color: var(--text-color);
}

.result-content {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.5rem;
}

@media (min-width: 768px) {
  .result-content {
    grid-template-columns: 2fr 1fr;
  }
}

.image-section {
  order: 1;
}

.image-container {
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  min-height: 300px;
  overflow: hidden;
  touch-action: none;
  /* 禁用默认触摸行为 */
  user-select: none;
  background-color: var(--background-color);
  border-radius: 8px;
}

.result-image {
  max-width: 100%;
  max-height: 500px;
  object-fit: contain;
  cursor: grab;
  transform-origin: center center;
  display: block;
}

.result-image:active {
  cursor: grabbing;
}

.image-info {
  margin-top: 1rem;
  text-align: center;
}

.image-note {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  color: var(--text-secondary);
  font-size: 0.9rem;
  margin: 0;
  padding: 0.5rem;
  background: var(--background-color);
  border-radius: 6px;
  border: 1px solid var(--border-color);
}

.image-note .material-icons {
  font-size: 1rem;
  color: var(--primary-color);
}

.summary-section {
  order: 2;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.summary-card {
  background: var(--background-color);
  border-radius: 8px;
  padding: 1rem;
  border: 1px solid var(--border-color);
}

.summary-item {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.summary-icon {
  font-size: 2rem;
}

.summary-number {
  font-size: 1.8rem;
  font-weight: bold;
  color: var(--text-color);
}

.defect-types h4 {
  margin: 0 0 0.5rem 0;
  color: var(--text-color);
}

.defect-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.defect-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem;
  background: var(--background-color);
  border-radius: 6px;
  border: 1px solid var(--border-color);
}

.defect-name {
  font-weight: 500;
  color: var(--text-color);
}

.defect-count {
  background: var(--primary-color);
  color: white;
  padding: 0.2rem 0.5rem;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: bold;
}

.details-section {
  order: 3;
  grid-column: 1 / -1;
}

.details-section h4 {
  margin: 0 0 1rem 0;
  color: var(--text-color);
}

.no-defects {
  text-align: center;
  padding: 2rem;
  color: var(--text-secondary);
}

.no-defects .material-icons {
  font-size: 3rem;
  color: #27ae60;
  margin-bottom: 0.5rem;
  display: block;
}

.detection-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.detection-item {
  padding: 0.8rem;
  background: var(--background-color);
  border-radius: 8px;
  transition: all 0.3s ease;
  cursor: pointer;
}

.detection-item:active {
  background: var(--surface-hover-bg);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.detection-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.6rem;
}

.detection-class {
  font-weight: bold;
  color: var(--text-color);
  background: var(--primary-color);
  color: white;
  padding: 0.1rem 0.8rem;
  border-radius: 1em;
  font-size: 0.9rem;
}

.detection-confidence {
  color: var(--text-secondary);
  font-size: 0.9rem;
  font-weight: 500;
}

.detection-details {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.detection-bbox {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
  color: var(--text-secondary);
  font-family: monospace;
}

.detection-bbox .material-icons {
  font-size: 1rem;
  color: var(--primary-color);
}

.confidence-bar {
  width: 100%;
  height: 6px;
  background: var(--border-color);
  border-radius: 3px;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
  border-radius: 3px;
  transition: width 0.3s ease;
}

.timestamp {
  order: 4;
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--text-secondary);
  font-size: 0.9rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border-color);
}

@media (max-width: 767px) {
  .summary-section {
    order: 2;
  }

  .details-section {
    order: 3;
  }

  .timestamp {
    order: 4;
  }
}

@media (min-width: 768px) {
  .detection-list {
    flex-direction: row;
    flex-wrap: wrap;
    align-items: stretch;
    justify-content: center;
  }

  .detection-item {
    flex: 1 1 300px;
    min-width: 300px;
  }
}
</style>
