import { defineStore } from 'pinia'
import { ref } from 'vue'

interface Theme{
  name: string;
  lightPrimary: string;
  lightSecondary: string;
  darkPrimary: string;
  darkSecondary: string;
}

export const useThemeStore = defineStore('theme', () => {
  const isDark = ref(true)
  const isAutoDark = ref(true)

  const colorThemes: Theme[] = [
    {
      name: 'Default',
      lightPrimary: '#646cff',
      lightSecondary: '#f1a127',
      darkPrimary: '#646cff',
      darkSecondary: '#c78c33',
    },
    {
      name: 'Forest',
      lightPrimary: '#4caf50',
      lightSecondary: '#afcc47ff',
      darkPrimary: '#388e3c',
      darkSecondary: '#95aa37ff'
    },
    {
      name: 'Ocean',
      lightPrimary: '#2196f3',
      lightSecondary: '#39c7b4',
      darkPrimary: '#1976d2',
      darkSecondary: '#26b9a1'
    }
  ];
  const currentColorTheme = ref(localStorage.getItem('colorTheme') || 'Default');

  const setLightDark = (theme: 'light' | 'dark', force: boolean = true) => {
    if (force) isAutoDark.value = false;
    isDark.value = theme === 'dark'
    document.documentElement.setAttribute('data-theme', theme)
    document.documentElement.style.setProperty('--primary-color', isDark.value ? colorThemes.find(t => t.name === currentColorTheme.value)?.darkPrimary || '#646cff' : colorThemes.find(t => t.name === currentColorTheme.value)?.lightPrimary || '#646cff')
    document.documentElement.style.setProperty('--secondary-color', isDark.value ? colorThemes.find(t => t.name === currentColorTheme.value)?.darkSecondary || '#f1a127' : colorThemes.find(t => t.name === currentColorTheme.value)?.lightSecondary || '#f1a127')
    if(!isAutoDark.value) localStorage.setItem('theme', theme);
  }

  const toggleLightDark = () => {
    const newTheme = isDark.value ? 'light' : 'dark';
    setLightDark(newTheme)
  }

  const initializeTheme = () => {
    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | null
    
    if (savedTheme) {
      setLightDark(savedTheme, true)
    } else {
      // Detect system preference
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      setLightDark(prefersDark ? 'dark' : 'light', false);
    }
  }
  initializeTheme();

  const useDeviceTheme = () => {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    if (prefersDark) {
      setLightDark('dark',false)
    } else {
      setLightDark('light', false)
    }
    isAutoDark.value = true;
    localStorage.removeItem('theme')
  }

  const setColorTheme = (themeName: string) => {
    const theme = colorThemes.find(t => t.name === themeName)
    if (theme) {
      currentColorTheme.value = themeName
      document.documentElement.style.setProperty('--primary-color', isDark.value ? theme.darkPrimary : theme.lightPrimary)
      document.documentElement.style.setProperty('--secondary-color', isDark.value ? theme.darkSecondary : theme.lightSecondary)
    } else {
      console.warn(`Theme ${themeName} not found`)
    }
    localStorage.setItem('colorTheme', themeName)
  }

  return {
    isDark,
    setLightDark,
    toggleLightDark,
    colorThemes,
    setColorTheme,
    currentColorTheme,
    useDeviceTheme,
    isAutoDark
  }
})
