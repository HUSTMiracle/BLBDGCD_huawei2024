<script lang="ts" setup>
import { useNotificationStore } from '../stores/notification';

const nStore = useNotificationStore();

</script>

<template>
  <div class="noticontainer">
    <div v-for="noti in nStore.notifications" class="noti-bubble">
      <div class="noti-content">
        <span class="noti-type icons" :class="noti.type">{{ noti.type }}</span>
        <span class="noti-text">{{ noti.text }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.noticontainer {
  position: fixed;
  bottom: 50px;
  right: 10px;
  z-index: 3000;
  text-align: left;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}

@media (max-width: 768px) {
  .noticontainer {
    bottom: 50px;
    left: 0;
    right: 0;
    margin-left: auto;
    margin-right: auto;
    align-items: center;
  }
}

.check {
  color: #4CAF50;
}

.error {
  color: #F44336;
}

.warning {
  color: #FF9800;
}

.info {
  color: var(--primary-color);
}

.noti-bubble {
  border-radius: 8px;
  padding: 10px;
  margin-bottom: 10px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  width: fit-content;
  background-color: var(--surface-alt);
  color: var(--background-color);
  max-width: 90vw;
}

.noti-content {
  display: flex;
  align-items: center;
  gap: 5px;
}
</style>