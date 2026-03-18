<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { usePCBStore } from '../stores/pcb';
import CameraUpload from '../components/CameraUpload.vue';
import DetectionResult from '../components/DetectionResult.vue';

const pcbStore = usePCBStore()

const uploadedImageUrl = ref<string>('')

const handleImageUpload = async (file: File) => {
  try {
    // 创建图片预览URL
    if (uploadedImageUrl.value) {
      URL.revokeObjectURL(uploadedImageUrl.value)
    }
    uploadedImageUrl.value = URL.createObjectURL(file)

    // 上传图片进行检测
    await pcbStore.uploadImage(file)

    // 检测成功后，清理blob URL并使用标注后的图片URL
    if (pcbStore.currentResult) {
      if (uploadedImageUrl.value.startsWith('blob:')) {
        URL.revokeObjectURL(uploadedImageUrl.value)
      }
      uploadedImageUrl.value = pcbStore.currentResult.annotated_image_url
    }
  } catch (error) {
    console.error('上传失败:', error)
    // 清理预览URL
    if (uploadedImageUrl.value && uploadedImageUrl.value.startsWith('blob:')) {
      URL.revokeObjectURL(uploadedImageUrl.value)
    }
    uploadedImageUrl.value = ''
  }
}

const handleClearResult = () => {
  pcbStore.clearCurrentResult();
  if (uploadedImageUrl.value) {
    // 只有当URL是blob时才需要释放
    if (uploadedImageUrl.value.startsWith('blob:')) {
      URL.revokeObjectURL(uploadedImageUrl.value);
    }
    uploadedImageUrl.value = '';
  }
};

onMounted(()=>{
  handleClearResult();
})
</script>

<template>
  <div class="home-container">
    <div class="header-section">
      <h1>
        <span class="material-icons">precision_manufacturing</span>
        PCB 缺陷检测
      </h1>
      <p class="subtitle">拍照或上传PCB图片</p>
      <p class="subtitle">智能检测电路板缺陷</p>
    </div>

    <!-- 上传区域 -->
    <div v-if="!pcbStore.hasResult" class="upload-section">
      <button class="realtime-btn" @click="$router.push('/realtime')">
        <span class="material-icons">monitor_heart</span>
        实时监测
      </button>
      <CameraUpload :isUploading="pcbStore.isLoading" @upload="handleImageUpload" />
    </div>

    <!-- 检测结果 -->
    <div class="back-section" v-if="pcbStore.hasResult && pcbStore.currentResult">
      <button @click="handleClearResult" class="back-btn">
        <span class="material-icons">arrow_back</span>
        返回主页
      </button>
    </div>
    <div v-if="pcbStore.hasResult && pcbStore.currentResult" class="result-section">
      <DetectionResult :result="pcbStore.currentResult" :imageUrl="uploadedImageUrl" />
    </div> <!-- 加载状态 -->
    <div v-if="pcbStore.isLoading" class="loading-overlay">
      <div class="loading-content">
        <span class="material-icons spinning">hourglass_empty</span>
        <p>正在分析PCB图片...</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.home-container {
  margin: 0 auto;
  padding: 3rem 1rem;
  padding-bottom: 0rem;
}


.subtitle {
  color: var(--text-secondary);
  font-size: 1.1rem;
  margin: 0;
  max-width: 500px;
  margin: 0 auto;
}

.error-message {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  background: #fee2e2;
  color: #dc2626;
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 2rem;
  border: 1px solid #fecaca;
}

.error-message .material-icons {
  font-size: 1.5rem;
  margin-top: 0.2rem;
  flex-shrink: 0;
}

.error-content {
  flex: 1;
}

.error-content strong {
  display: block;
  margin-bottom: 0.5rem;
  font-size: 1.1rem;
}

.error-content p {
  margin: 0 0 1rem 0;
  line-height: 1.5;
}

.upload-section {
  margin-bottom: 2rem;
}

.result-section {
  margin-bottom: 2rem;
}

.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

.loading-content {
  background: var(--surface-color);
  padding: 2rem;
  border-radius: 12px;
  text-align: center;
  max-width: 300px;
  margin: 0 1rem;
}

.loading-content .material-icons {
  font-size: 3rem;
  color: var(--primary-color);
  margin-bottom: 1rem;
  display: block;
}

.loading-content p {
  margin: 0;
  color: var(--text-color);
  font-size: 1.1rem;
}

.spinning {
  animation: spin 1s linear infinite;
}

.realtime-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
  color: #fff;
  border: none;
  border-radius: 30px;
  padding: 0.9rem 1.5rem;
  font-size: 1.1rem;
  font-weight: 500;
  margin: 0 auto 1.5rem auto;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(74,0,224,0.08);
  transition: all 0.2s;
  width: 100%;
  max-width: 500px;
}
.realtime-btn .material-icons {
  font-size: 1.5em;
}
.realtime-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(74,0,224,0.18);
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
