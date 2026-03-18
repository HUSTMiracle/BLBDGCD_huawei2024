import { createRouter, createWebHistory } from 'vue-router'

import Home from '../views/Home.vue'
import About from '../views/About.vue'
import Settings from '../views/Settings.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/history',
    name: 'History',
    component: () => import('../views/History.vue')
  },
  {
    path: '/history/:imageId',
    name: 'HistoryDetail',
    component: () => import('../views/HistoryDetail.vue'),
    props: true
  },
  {
    path: '/about',
    name: 'About',
    component: About
  },
  {
    path: '/settings',
    name: 'Settings',
    component: Settings
  },
  {
    path: '/realtime',
    name: 'RealTimeMonitor',
    component: () => import('../views/RealTimeMonitor.vue')
  }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes
})

export default router
