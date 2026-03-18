import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { pcbApiService, type InferenceResult, type HistoryItem } from '../services/pcbApi';
import { useNotificationStore } from './notification';

export const usePCBStore = defineStore('pcb', () => {
  const notificationStore = useNotificationStore();
  
  // 状态
  const isLoading = ref(false);
  const currentResult = ref<InferenceResult | null>(null);
  const histories = ref<HistoryItem[]>([]);
  const selectedHistory = ref<InferenceResult | null>(null);
  
  // 分页状态
  const currentPage = ref(1);
  const itemsPerPage = ref(10);
  
  // 筛选状态
  const selectedDate = ref<string>('');

  // 计算属性
  const hasResult = computed(() => currentResult.value !== null);
  const hasHistories = computed(() => histories.value.length > 0);
  const defectTypes = computed(() => {
    if (!currentResult.value) return [];
    return Object.entries(currentResult.value.summary.defect_types);
  });
  // 操作方法
  const uploadImage = async (file: File) => {
    try {
      isLoading.value = true;
      console.log('开始上传图片:', file.name);
      
      currentResult.value = await pcbApiService.uploadImageForInference(file);
      console.log('检测结果:', currentResult.value);
      notificationStore.showNotification('图片检测完成', 'check');
      
    } catch (err) {
      console.error('上传图片失败:', err);
      const errorMessage = err instanceof Error ? err.message : '上传失败';
      notificationStore.showNotification(errorMessage, 'error');
      throw err;
    } finally {
      isLoading.value = false;
      console.log('上传流程完成，loading状态:', isLoading.value);
    }
  };
  const loadHistories = async (params?: {
    page?: number;
    limit?: number;
    date?: string;
  }) => {
    try {
      isLoading.value = true;
      if (params?.date) {
        params.date = params.date.replace(/-/g, '');
      }
      const response = await pcbApiService.getAllHistories(params);
      
      if (response.success) {
        histories.value = response.result;
      } else {
        throw new Error('获取历史记录失败');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '加载历史记录失败';
      notificationStore.showNotification(errorMessage, 'error');
      throw err;
    } finally {
      isLoading.value = false;
    }
  };

  const loadHistoryDetail = async (imageId: string) => {
    try {
      isLoading.value = true;
      selectedHistory.value = await pcbApiService.getOneHistory(imageId);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '加载历史记录详情失败';
      notificationStore.showNotification(errorMessage, 'error');
      throw err;
    } finally {
      isLoading.value = false;
    }
  };
  const clearCurrentResult = () => {
    currentResult.value = null;
  };

  const clearSelectedHistory = () => {
    selectedHistory.value = null;
  };

  const setDateFilter = (date: string) => {
    selectedDate.value = date;
  };

  const nextPage = async () => {
    currentPage.value++;
    try {
      await loadHistories({
        page: currentPage.value,
        limit: itemsPerPage.value,
        date: selectedDate.value || undefined,
      });

      if (histories.value.length === 0) {
        prevPage();
        notificationStore.showNotification('没有更多结果', 'info');
      }
    } catch (err) {
      prevPage();
      throw err;
    }
  };

  const prevPage = () => {
    if (currentPage.value > 1) {
      currentPage.value--;
      loadHistories({
        page: currentPage.value,
        limit: itemsPerPage.value,
        date: selectedDate.value || undefined,
      });
    }
  };
  return {
    // 状态
    isLoading,
    currentResult,
    histories,
    selectedHistory,
    currentPage,
    itemsPerPage,
    selectedDate,
    
    // 计算属性
    hasResult,
    hasHistories,
    defectTypes,
    
    // 方法
    uploadImage,
    loadHistories,
    loadHistoryDetail,
    clearCurrentResult,
    clearSelectedHistory,
    setDateFilter,
    nextPage,
    prevPage,
  };
});
