<script setup>
import { computed } from 'vue';
import { useThemeStore } from '../stores/theme';

const themeStore = useThemeStore();

</script>

<template>
  <nav>
    <ul class="icon-nav">
      <li><router-link to="/" class="icons">camera_alt</router-link></li>
      <li><router-link to="/history" class="icons">history</router-link></li>
      <li><router-link to="/settings" class="icons">settings</router-link></li>
      <li><router-link to="/about" class="icons">info</router-link></li>
    </ul>
  </nav>
</template>

<style scoped>
nav {
  background-color: var(--surface-color);
  color: var(--text-color);
  height: 3em;
  transition: background-color 0.3s ease, color 0.3s ease;
  position: fixed;
  bottom: 0;
  left: 0;
  width: 100%;
  z-index: 1000; /* Ensure nav is above other content */
}

ul {
  list-style: none;
  margin: 0;
  padding: 0;
  height: 100%;
  display: flex;
  justify-content: space-around;
  align-items: center;
}

li {
  margin: 0 10px;
}

.nav-right>ul > li {
  margin: 0 2px;
}

a, router-link {
  color: var(--text-color);
  text-decoration: none;
  font-weight: bold;
  padding: 0.5rem 1rem;
  border-radius: 10px;
  transition: background-color 0.3s ease, color 0.3s ease, opacity 0.3s ease;
  cursor: pointer;
}

a:active, .router-link-active {
  background-color: var(--surface-hover-bg);
}

.logo {
  font-size: 1.5em;
  font-weight: bold;
  color: var(--primary-color);
  text-align: left;
  margin-left: 20px;
  width: 150px;
}

.nav-right {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  width: 150px;
  margin-right: 10px;
}

.router-link-active {
  background-color: var(--primary-color);
  color: white;
}

.router-link-active:active {
  background-color: var(--primary-color);
  opacity: 0.8;
}


</style>