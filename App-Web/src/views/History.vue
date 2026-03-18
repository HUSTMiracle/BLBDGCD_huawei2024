<template>
  <div class="history-container">
    <div class="header-section">
      <h1>
        <span class="material-icons">history</span>
        检测历史
      </h1>
      <p class="subtitle">查看历史检测记录和结果</p>
    </div>

    <!-- 筛选控件 -->
    <div class="filter-section card">
      <div class="filter-item">
        <label for="date-filter">日期</label>
        <input id="date-filter" type="date" v-model="selectedDate" @change="applyDateFilter" class="date-input">
      </div>
      <button @click="clearDateFilter">
        <span class="iconspan">clear</span>
        清除
      </button>
    </div>

    <!-- 加载状态 -->
    <div v-if="pcbStore.isLoading" class="loading-message">
      <span class="material-icons spinning">hourglass_empty</span>
      <span>加载中...</span>
    </div>

    <!-- 历史记录列表 -->
    <div v-else-if="pcbStore.hasHistories" class="history-list">
      <div v-for="history in pcbStore.histories" :key="history.image_id" class="history-item"
        @click="viewDetail(history.image_id)">
        <div class="history-header">
          <div class="history-title">
            <span class="material-icons">image</span>
            {{ history.image_id }}
          </div>
          <div class="history-time">
            {{ formatTimestamp(history.image_id) }}
          </div>
        </div>

        <div class="history-summary">
          <div class="summary-item">
            <span class="material-icons" :class="history.detected_flaws? 'defect':'good'">{{ history.detected_flaws ? 'error' : 'check' }}</span>
            <span>缺陷数: {{ history.detected_flaws }}</span>
          </div>
          <span class="view-detail">
            <span class="material-icons">arrow_forward</span>
            查看详情
          </span>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div v-else class="empty-state">
      <span class="material-icons">inbox</span>
      <h3>暂无历史记录</h3>
      <p>还没有检测记录，去首页上传图片开始检测吧！</p>
    </div>

    <!-- 分页控件 -->
    <div v-if="pcbStore.hasHistories" class="pagination">
      <button @click="pcbStore.prevPage()" :disabled="pcbStore.currentPage <= 1" class="page-btn icons md-18">
        chevron_left
      </button>
      <span>
        第 {{ pcbStore.currentPage }} 页
      </span>
      <button @click="pcbStore.nextPage()" class="page-btn icons md-18">
        chevron_right
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { usePCBStore } from '../stores/pcb';

const router = useRouter();
const pcbStore = usePCBStore();

const selectedDate = ref<string>('');

const loadHistories = async () => {
  try {
    await pcbStore.loadHistories({
      page: 1,
      limit: 10,
      date: selectedDate.value || undefined,
    });
  } catch (error) {
    console.error('加载历史记录失败:', error);
  }
};

const applyDateFilter = async () => {
  pcbStore.setDateFilter(selectedDate.value);
  await loadHistories();
};

const clearDateFilter = async () => {
  selectedDate.value = '';
  pcbStore.setDateFilter('');
  await loadHistories();
};

const viewDetail = (imageId: string) => {
  router.push(`/history/${encodeURIComponent(imageId)}`);
};

const formatTimestamp = (imageId: string) => {
  try {
    // 从image_id提取日期时间信息，格式: 20250715-202541.jpg
    const match = imageId.match(/(\d{8})-(\d{6})/);
    if (match) {
      const dateStr = match[1]; // 20250715
      const timeStr = match[2]; // 202541

      const year = dateStr.substring(0, 4);
      const month = dateStr.substring(4, 6);
      const day = dateStr.substring(6, 8);
      const hour = timeStr.substring(0, 2);
      const minute = timeStr.substring(2, 4);
      const second = timeStr.substring(4, 6);

      const date = new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}`);

      return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    }
  } catch (error) {
    console.error('解析日期失败:', error);
  }

  // 如果解析失败，返回原始的image_id
  return imageId;
};

onMounted(() => {
  loadHistories();
});
</script>

<style scoped>
.history-container {
  margin: 0 auto;
  padding: 3rem 1rem;
}

.subtitle {
  color: var(--text-secondary);
  font-size: 1.1rem;
  margin: 0;
  max-width: 500px;
  margin: 0 auto;
}

.date-input{
  width: 140px;
}

.filter-section {
  display: flex;
  align-items: center;
  margin-bottom: 2rem;
  padding: 1rem;
  flex-wrap: wrap;
}

.filter-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.filter-item label {
  color: var(--text-color);
  font-weight: 500;
  white-space: nowrap;
}


.clear-filter-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: transparent;
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.3s ease;
  white-space: nowrap;
}

.clear-filter-btn:active {
  background: var(--surface-hover-bg);
  color: var(--text-color);
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
  padding: 3rem;
  color: var(--text-secondary);
  font-size: 1.1rem;
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  align-items: center;
}

.history-item {
  background: var(--surface-color);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 0.5rem 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  width: 90%;
  max-width: 600px;
}

.history-item:active {
  background: var(--surface-hover-bg);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

@media (max-width: 768px) {
  .history-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }
}

.history-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  color: var(--text-color);
  font-size: 1.1rem;
}

.history-time {
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.history-summary {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  gap: 0.5rem;
}

.summary-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--text-color);
}


.defect-types {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-left: 1.7rem;
}

.defect-tag {
  background: var(--primary-color);
  color: white;
  padding: 0.2rem 0.5rem;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: 500;
}

.history-actions {
  display: flex;
  justify-content: flex-end;
}

.view-detail {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  color: var(--primary-color);
  font-weight: 500;
  font-size: 0.9rem;
}

.empty-state {
  text-align: center;
  padding: 4rem 2rem;
  color: var(--text-secondary);
}

.empty-state .material-icons {
  font-size: 4rem;
  color: var(--text-secondary);
  margin-bottom: 1rem;
  display: block;
}

.empty-state h3 {
  margin: 0 0 0.5rem 0;
  color: var(--text-color);
}

.empty-state p {
  margin: 0 0 2rem 0;
  max-width: 400px;
  margin-left: auto;
  margin-right: auto;
}

.start-detection-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 1rem 2rem;
  background: var(--primary-color);
  color: white;
  text-decoration: none;
  border-radius: 8px;
  font-weight: 600;
  transition: all 0.3s ease;
}

.start-detection-btn:active {
  background: var(--primary-hover);
  transform: translateY(-2px);
}

.pagination {
  display: flex;
  justify-content: center;
  gap: 1rem;
  align-items: center;
  margin-top: 1rem;
  border-top: 1px solid var(--border-color);
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
