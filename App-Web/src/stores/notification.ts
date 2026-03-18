import { defineStore } from "pinia";
import { ref } from "vue";

interface Notification {
  text: string;
  type: "info" | "check" | "error" | "warning";
  id: number;
}

export const useNotificationStore = defineStore("notification", () => {
  const notifications = ref<Notification[]>([]);
  let nextId:number = 1;

  function showNotification(text: string, type: "info" | "check" | "error" | "warning") {
    const notification: Notification = {
      text,
      type,
      id: nextId++,
    };
    notifications.value.push(notification);
    setTimeout(() => {
      notifications.value = notifications.value.filter(n => n.id !== notification.id);
    }, 3000); // Remove notification after 3 seconds
  }

  return {
    notifications,
    showNotification,
  };
});
