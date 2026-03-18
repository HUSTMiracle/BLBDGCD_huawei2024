<script setup lang="ts">
import Nav from './components/Nav.vue';
import NotiBubble from './components/NotiBubbles.vue';

</script>

<template>
  <Nav />
  <NotiBubble />
  <div class="router-div">
    <router-view />
  </div>
</template>

<style scoped>
.router-div {
  height: calc(100vh - 3em);
  margin-bottom: 3em;
  width: 100%;
  overflow: auto;
}


button {
  margin: 20px;
  padding: 10px 20px;
  cursor: pointer;
}
</style>

