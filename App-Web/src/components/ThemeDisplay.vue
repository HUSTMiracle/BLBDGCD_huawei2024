<script setup lang="ts">
interface Theme {
  name: string;
  lightPrimary: string;
  darkPrimary: string;
  lightSecondary: string;
  darkSecondary: string;
}

defineProps<{
  theme: Theme,
}>();

</script>

<template>
  <div class="theme-display">
    <p>{{ theme.name }}</p>
    <div class="color-box" :style="{ backgroundColor: theme.lightPrimary }"/>
    <div class="color-box" :style="{ backgroundColor: theme.lightSecondary }"/>
    <div class="color-box" :style="{ backgroundColor: theme.darkPrimary }"/>
    <div class="color-box" :style="{ backgroundColor: theme.darkSecondary }"/>
  </div>
</template>

<style scoped>
.theme-display {
  background-color: var(--surface-color);
  border: 2px solid var(--surface-color);
  border-radius: 1000px;
  padding: 5px 8px;
  margin: 5px;
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 5px;
  height: 1.6em;
  cursor: pointer;
  width: fit-content;
}

.selected{
  border-color: var(--primary-color);
}

.color-box {
  width: 1.5em;
  height: 1.5em;
  border-radius: 50%;
}

p{
  width: 70px;
}
</style>
