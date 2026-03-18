<script setup lang="ts">
import { ref, watch, onMounted } from 'vue';
import { useThemeStore } from '../stores/theme';
import { useNotificationStore } from '../stores/notification';
import ThemeDisplay from '../components/ThemeDisplay.vue';

const themeStore = useThemeStore();
const notificationStore = useNotificationStore();

const isMstChecked = ref(themeStore.isAutoDark);
const apiUrl = ref('');

// 根据平台获取默认API地址
const getDefaultApiUrl = (): string => {
  return 'http://localhost:5001/api/v1';
};

// 从localStorage获取自定义API地址，如果没有则使用默认值
const getStoredApiUrl = () => {
  return localStorage.getItem('custom_api_url') || getDefaultApiUrl();
};

onMounted(() => {
  apiUrl.value = getStoredApiUrl();
});

function mstChecked() {
  if (isMstChecked.value === true) {
    themeStore.useDeviceTheme();
  } else {
    themeStore.setLightDark(themeStore.isDark ? 'dark' : 'light');
  }
}

function saveApiUrl() {
  if (apiUrl.value.trim()) {
    // 确保URL格式正确
    let url = apiUrl.value.trim();
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      url = 'http://' + url;
    }

    // 移除末尾的斜杠
    url = url.replace(/\/$/, '');

    localStorage.setItem('custom_api_url', url);
    notificationStore.showNotification('API地址已保存，请刷新页面使设置生效', 'check');
  }
}

function resetApiUrl() {
  const defaultUrl = getDefaultApiUrl();
  apiUrl.value = defaultUrl;
  localStorage.setItem('custom_api_url', defaultUrl);
  notificationStore.showNotification('API地址已重置为默认值', 'check');
}

function clearLocalStorage() {
  localStorage.clear();
  window.location.reload();
}

watch(() => themeStore.isAutoDark, (newValue) => {
  isMstChecked.value = newValue;
});
</script>

<template>
  <div class="settings-container">

    <div class="header-section">
      <h1>
        <span class="material-icons">settings</span>
        设置
      </h1>
    </div>
    <div class="main-div">
      <div class="cards-div">
        <div class="card">
          <h2>API 配置</h2>
          <div style="display: flex; flex-direction: column; gap: 0.5rem; width: 100%; max-width: 400px;">
            <label>服务器地址:</label>
            <input v-model="apiUrl" type="text" placeholder="例: http://192.168.1.100:5001/api/v1" />
            <div>
              <button @click="saveApiUrl">
                <span class="iconspan">save</span>
                保存
              </button>
              <button @click="resetApiUrl">
                <span class="iconspan">refresh</span>
                重置
              </button>
            </div>
            <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0;">
              * 修改后需要刷新页面才能生效。Android设备请使用实际IP地址而非localhost
            </p>
          </div>
        </div>

        <div class="card">
          <h2>主题设置</h2>
          <ThemeDisplay v-for="theme in themeStore.colorThemes" :key="theme.name" :theme="theme"
            @click="themeStore.setColorTheme(theme.name)"
            :class="{ 'selected': theme.name === themeStore.currentColorTheme }" />

          <button @click="themeStore.toggleLightDark" class="icons">
            {{ themeStore.isDark ? 'light_mode' : 'dark_mode' }}</button>
          <label class="checkbox-label">
            <input type="checkbox" v-model="isMstChecked" @change="mstChecked" />
            跟随系统主题
          </label>
        </div>

        <div class="card">
          <h2>本地数据</h2>
          <button @click="clearLocalStorage"><span class="iconspan">delete_forever</span>清除本地数据</button>
          <p style="font-size: 0.8rem; color: var(--text-secondary);">*这会删除所有本地存储的数据，包括您的偏好设置和API设置，且无法撤销。</p>
        </div>
      </div>
    </div>
  </div>
</template>


<style scoped>
.header-section {
  margin-bottom: 0rem;
}

.card {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.cards-div {
  max-width: 600px;
  width: 90%;
}

.checkbox-label {
  display: flex;
  align-items: center;
  margin-top: 10px;
  cursor: pointer;
}

.checkbox-label input[type="checkbox"] {
  margin-right: 8px;
  width: 18px;
  height: 18px;
  cursor: pointer;
}

/* 自定义checkbox样式，确保在所有浏览器上都能正确显示 */
.checkbox-label input[type="checkbox"] {
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  width: 18px;
  height: 18px;
  border: 2px solid var(--border-color);
  border-radius: 3px;
  background-color: var(--background-color);
  margin-right: 8px;
  position: relative;
  cursor: pointer;
  transition: all 0.3s ease;
}

.checkbox-label input[type="checkbox"]:checked {
  background-color: var(--primary-color);
  border-color: var(--primary-color);
}

.checkbox-label input[type="checkbox"]:checked::after {
  content: '✓';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: white;
  font-size: 12px;
  font-weight: bold;
}

.checkbox-label input[type="checkbox"]:active {
  border-color: var(--primary-color);
}

.settings-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 3rem 0;
}

h1 .material-icons {
  font-size: 2.5rem;
  color: var(--primary-color);
}
</style>
