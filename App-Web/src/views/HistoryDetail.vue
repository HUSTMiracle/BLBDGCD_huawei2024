<template>
  <div class="history-detail-container">
    <!-- 返回按钮 -->
    <div class="back-section">
      <button @click="goBack" class="back-btn">
        <span class="material-icons">arrow_back</span>
        返回列表
      </button>
    </div>

    <!-- 加载状态 -->
    <div v-if="pcbStore.isLoading" class="loading-message">
      <span class="material-icons spinning">hourglass_empty</span>
      <span>加载详情中...</span>
    </div>

    <!-- 详情内容 -->
    <div v-else-if="pcbStore.selectedHistory" class="detail-content">
      <div class="detail-header">
        <h1>
          <span class="material-icons">image</span>
          {{ pcbStore.selectedHistory.image_id }}
        </h1>
        <div class="timestamp">
          <span class="material-icons">schedule</span>
          {{ formatTimestamp(pcbStore.selectedHistory.timestamp) }}
        </div>
      </div>

      <!-- 使用检测结果组件 -->
      <DetectionResult :result="pcbStore.selectedHistory" :imageUrl="imageUrl"/>
    </div>

    <!-- 未找到记录 -->
    <div v-else class="not-found">
      <span class="material-icons">search_off</span>
      <h3>未找到该检测记录</h3>
      <p>该记录可能已被删除或不存在</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, computed } from 'vue';
import { useRouter } from 'vue-router';
import { usePCBStore } from '../stores/pcb';
import DetectionResult from '../components/DetectionResult.vue';

interface Props {
  imageId: string;
}

const props = defineProps<Props>();
const router = useRouter();
const pcbStore = usePCBStore();

const imageUrl = computed(() => {
  if (pcbStore.selectedHistory) {
    // 使用标注后的图片URL - 通过/images/路径访问
    return pcbStore.selectedHistory.annotated_image_url;
  }
  return '';
});

const loadHistoryDetail = async () => {
  try {
    await pcbStore.loadHistoryDetail(decodeURIComponent(props.imageId));
  } catch (error) {
    console.error('加载历史记录详情失败:', error);
  }
};

const goBack = () => {
  pcbStore.clearSelectedHistory();
  router.push('/history');
};

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

onMounted(() => {
  loadHistoryDetail();
});
</script>

<style scoped>
.history-detail-container {
  margin: 0 auto;
  padding: 2rem 3%;
}

.error-message {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: #fee2e2;
  color: #dc2626;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 2rem;
  border: 1px solid #fecaca;
}

.loading-message {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 4rem;
  color: var(--text-secondary);
  font-size: 1.1rem;
}

.detail-content {
  margin-bottom: 2rem;
}

.detail-header {
  text-align: center;
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border-color);
}

.detail-header h1 {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin: 3em 0 1rem 0;
  color: var(--text-color);
  font-size: 2rem;
  font-weight: 600;
  word-break: break-all;
}

@media (max-width: 768px) {
  .detail-header h1 {
    font-size: 1.5rem;
    flex-direction: column;
    gap: 0.3rem;
  }
}

.detail-header h1 .material-icons {
  font-size: 2rem;
  color: var(--primary-color);
  flex-shrink: 0;
}

@media (max-width: 768px) {
  .detail-header h1 .material-icons {
    font-size: 1.5rem;
  }
}

.timestamp {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  color: var(--text-secondary);
  font-size: 1rem;
}

.not-found {
  text-align: center;
  padding: 4rem 2rem;
  color: var(--text-secondary);
}

.not-found .material-icons {
  font-size: 4rem;
  color: var(--text-secondary);
  margin-bottom: 1rem;
  display: block;
}

.not-found h3 {
  margin: 0 0 0.5rem 0;
  color: var(--text-color);
}

.not-found p {
  margin: 0 0 2rem 0;
  max-width: 400px;
  margin-left: auto;
  margin-right: auto;
}

.spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }

  to {
    transform: rotate(360deg);
  }
}
</style>
