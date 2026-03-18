<template>
  <div class="camera-upload">    <!-- 相机/相册选择按钮 -->
    <div class="camera-controls" v-if="!previewUrl">
      <button 
        v-if="isNative" 
        @click="takeFromCamera" 
        :disabled="isUploading"
        class="camera-btn"
      >
        <span class="material-icons">camera_alt</span>
        拍照检测
      </button>
      
      <button 
        v-if="isNative" 
        @click="selectFromGallery" 
        :disabled="isUploading"
        class="gallery-btn"
      >
        <span class="material-icons">photo_library</span>
        从相册选择
      </button>

      <!-- 传统文件上传作为备选 -->
      <div v-if="isNative" class="divider">
        <span>或</span>
      </div>

      <div class="upload-area" @click="triggerFileInput" @drop="handleDrop" @dragover.prevent @dragenter.prevent>
        <input 
          ref="fileInput" 
          type="file" 
          accept="image/jpeg,image/jpg,image/png,image/bmp" 
          @change="handleFileSelect" 
          style="display: none;"
        >
        
        <div class="upload-placeholder">
          <span class="material-icons upload-icon">cloud_upload</span>
          <p>点击或拖拽图片到此处</p>
          <p class="upload-hint">支持 JPG、PNG、BMP 格式，最大20MB</p>
        </div>
      </div>
    </div>

    <!-- 图片预览 -->
    <div v-if="previewUrl" class="preview-section">
      <div class="preview-container">
        <img :src="previewUrl" alt="预览图片" class="preview-image">
        <button @click="clearImage" class="clear-btn">
          <span class="material-icons">close</span>
        </button>
      </div>
      
      <button 
        @click="uploadImage" 
        :disabled="isUploading"
        class="upload-btn"
      >
        <span v-if="isUploading" class="material-icons spinning">hourglass_empty</span>
        <span v-else class="material-icons">send</span>
        {{ isUploading ? '检测中...' : '开始检测' }}
      </button>    </div>

    <!-- 移除了本地错误提示，现在使用统一的notification系统 -->
  </div>
</template>

<script setup lang="ts">
import { ref, onUnmounted, computed } from 'vue'
import { CameraService } from '../services/cameraService'
import { useNotificationStore } from '../stores/notification'

const notificationStore = useNotificationStore()

interface Props {
  isUploading?: boolean
}

interface Emits {
  (e: 'upload', file: File): void
}

const { isUploading = false } = defineProps<Props>()
const emit = defineEmits<Emits>()

const fileInput = ref<HTMLInputElement>()
const selectedFile = ref<File | null>(null)
const previewUrl = ref<string>('')

// 检查是否在原生环境中运行
const isNative = computed(() => CameraService.isNative())

const showError = (message: string) => {
  notificationStore.showNotification(message, 'error')
}

const takeFromCamera = async () => {
  try {
    console.log('开始拍照流程...')
    
    // 在Web环境中，相机功能可能不可用
    if (!CameraService.isNative()) {
      showError('拍照功能仅在移动设备中可用，请选择文件上传')
      return
    }

    // 检查权限
    console.log('检查相机权限...')
    const hasPermission = await CameraService.checkPermissions()
    if (!hasPermission) {
      console.log('权限不足，请求权限...')
      const granted = await CameraService.requestPermissions()
      if (!granted) {
        showError('需要相机权限才能使用拍照功能，请在设备设置中开启权限')
        return
      }
    }    console.log('权限检查通过，开始拍照...')
    const photo = await CameraService.takeFromCamera({
      quality: 90,
      allowEditing: false // Android下建议关闭编辑功能以提高兼容性
    })
    
    console.log('拍照成功，转换文件...', photo)
    const file = await CameraService.photoToFile(photo, `camera_${Date.now()}.jpg`)
    console.log('文件转换成功:', file)
    setSelectedFile(file)
  } catch (error: any) {
    console.error('拍照失败:', error)
    showError(error.message || '拍照失败，请重试')
  }
}

const selectFromGallery = async () => {
  try {
    console.log('开始相册选择流程...')
    
    // 在Web环境中，相册功能可能不可用
    if (!CameraService.isNative()) {
      showError('相册功能仅在移动设备中可用，请选择文件上传')
      return
    }

    // 检查权限
    const hasPermission = await CameraService.checkPermissions()
    if (!hasPermission) {
      console.log('权限不足，请求权限...')
      const granted = await CameraService.requestPermissions()
      if (!granted) {
        showError('需要相册权限才能选择图片，请在设备设置中开启权限')
        return
      }
    }    console.log('权限检查通过，打开相册...')
    const photo = await CameraService.selectFromGallery({
      quality: 90,
      allowEditing: false // Android下建议关闭编辑功能以提高兼容性
    })
    
    console.log('图片选择成功，转换文件...', photo)
    const file = await CameraService.photoToFile(photo, `gallery_${Date.now()}.jpg`)
    console.log('文件转换成功:', file)
    setSelectedFile(file)
  } catch (error: any) {
    console.error('选择图片失败:', error)
    showError(error.message || '选择图片失败，请重试')
  }
}

const triggerFileInput = () => {
  fileInput.value?.click()
}

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    if (validateFile(file)) {
      setSelectedFile(file)
    }
  }
}

const handleDrop = (event: DragEvent) => {
  event.preventDefault()
  const file = event.dataTransfer?.files[0]
  if (file && validateFile(file)) {
    setSelectedFile(file)
  }
}

const validateFile = (file: File): boolean => {
  // 检查文件类型
  const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp']
  if (!allowedTypes.includes(file.type)) {
    notificationStore.showNotification('请选择支持的图片格式：JPG、PNG、BMP', 'error')
    return false
  }
  
  // 检查文件大小 (最大20MB)
  const maxSize = 20 * 1024 * 1024
  if (file.size > maxSize) {
    notificationStore.showNotification('图片文件大小不能超过20MB', 'error')
    return false
  }
  
  return true
}

const setSelectedFile = (file: File) => {
  selectedFile.value = file
  
  // 创建预览URL
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
  }
  previewUrl.value = URL.createObjectURL(file)
}

const clearImage = () => {
  selectedFile.value = null
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = ''
  }
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

const uploadImage = () => {
  if (selectedFile.value) {
    emit('upload', selectedFile.value)
  }
}

// 清理函数
const cleanup = () => {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
  }
}

// 在组件卸载时清理
onUnmounted(cleanup)
</script>

<style scoped>
.camera-upload {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  max-width: 500px;
  margin: 0 auto;
  padding: 1rem;
}

.camera-controls {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.camera-btn, .gallery-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 1rem 2rem;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  width: 100%;
}

.camera-btn {
  background: var(--primary-color);
  color: white;
}


.gallery-btn {
  background: transparent;
  color: var(--primary-color);
  border: 2px solid var(--primary-color);
}

.camera-btn:disabled, .gallery-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.divider {
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 1rem 0;
  position: relative;
}

.divider::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 0;
  right: 0;
  height: 1px;
  background: var(--border-color);
}

.divider span {
  background: var(--background-color);
  padding: 0 1rem;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.upload-area {
  border: 2px dashed var(--border-color);
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  min-height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--surface-color);
}

.upload-area:active {
  border-color: var(--primary-color);
  background: var(--surface-hover-bg);
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  color: var(--text-secondary);
}

.upload-icon {
  font-size: 2rem;
  color: var(--primary-color);
}

.upload-hint {
  font-size: 0.8rem;
  opacity: 0.7;
  margin: 0;
}

.preview-section {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.preview-container {
  position: relative;
  width: 100%;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  background-color: var(--card-bg);
}

.preview-image {
  width: 100%;
  height: auto;
  max-height: 400px;
  object-fit: contain;
  display: block;
}

.clear-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s ease;
}

.clear-btn:active {
  background: rgba(0, 0, 0, 0.8);
}

.clear-btn .material-icons {
  font-size: 18px;
}

.upload-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 1rem 2rem;
  background: var(--primary-color);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  width: 100%;
}

.upload-btn:active:not(:disabled) {
  background: var(--primary-hover);
  transform: translateY(-2px);
}

.upload-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: none;
}

.spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* 移动端优化 */
@media (max-width: 768px) {
  .camera-upload {
    padding: 0.5rem;
    max-width: 100%;
  }
  
  .upload-area {
    padding: 1.5rem;
    min-height: 100px;
  }
  
  .preview-image {
    max-height: 300px;
  }
  
  .camera-btn, .gallery-btn {
    padding: 0.8rem 1.5rem;
    font-size: 1rem;
  }
}
</style>
