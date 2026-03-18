// API Configuration
const DEFAULT_API_URL = 'http://localhost:5001/api/v1';

// 获取API基础URL，优先使用localStorage中的自定义地址
export const getApiBaseUrl = (): string => {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('custom_api_url') || DEFAULT_API_URL;
  }
  return DEFAULT_API_URL;
};

export const API_BASE_URL = getApiBaseUrl();

export const API_ENDPOINTS = {
  INFERENCE: '/inference',
  GET_ALL_HISTORIES: '/get_all_histories',
  GET_ONE_HISTORY: '/get_one_history'
};