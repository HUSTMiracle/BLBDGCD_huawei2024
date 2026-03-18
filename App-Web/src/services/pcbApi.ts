import { getApiBaseUrl, API_ENDPOINTS } from '../config';

// 服务器返回的推理结果结构
export interface ServerInferenceResponse {
  success: boolean;
  result: {
    detection_boxes?: number[][];  // [[y1,x1,y2,x2], ...]
    detection_classes?: string[];  // 类别名称数组
    detection_scores?: number[];   // 置信度数组
  };
  filename: string;
  source: string;
  error_code?: number;
  error_msg?: string;
}

// 转换后的推理结果结构
export interface InferenceResult {
  image_id: string;
  detection_results: Array<{
    class: string;
    confidence: number;
    bbox: [number, number, number, number];
  }>;
  summary: {
    total_defects: number;
    defect_types: Record<string, number>;
  };
  image_path: string;
  timestamp: string;
  annotated_image_url: string; // 标注后的图片URL
}

// 历史记录项目结构
export interface HistoryItem {
  image_id: string;
  detected_flaws: number;
}

// 历史记录详情结构
export interface HistoryDetail {
  image_id: string;
  inference_result: ServerInferenceResponse;
}

export interface HistoriesResponse {
  success: boolean;
  result: HistoryItem[];
}

export interface HistoryDetailResponse {
  success: boolean;
  result: HistoryDetail;
  error_code?: number;
  error_msg?: string;
}

class PCBApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = getApiBaseUrl();
  }

  // 更新API基础URL
  updateBaseUrl() {
    this.baseUrl = getApiBaseUrl();
  }
  // 上传图片进行检测
  async uploadImageForInference(file: File): Promise<InferenceResult> {
    try {
      // 每次请求都获取最新的API地址
      this.baseUrl = getApiBaseUrl();
      
      const formData = new FormData();
      formData.append('file', file);

      console.log('Making request to:', `${this.baseUrl}${API_ENDPOINTS.INFERENCE}`);

      const response = await fetch(`${this.baseUrl}${API_ENDPOINTS.INFERENCE}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // 尝试解析错误信息
        let errorMsg = `HTTP ${response.status}: ${response.statusText}`;
        try {
          const errorText = await response.text();
          if (errorText.includes('cv2.error')) {
            errorMsg = '图片格式不支持或图片文件损坏，请尝试其他图片';
          } else if (errorText.includes('error')) {
            errorMsg = '服务器处理图片时发生错误，请检查图片格式';
          }
        } catch (e) {
          // 忽略解析错误的错误
        }
        throw new Error(errorMsg);
      }

      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        throw new Error('服务器返回了非JSON格式的响应，可能是服务器内部错误');
      }

      const serverResponse: ServerInferenceResponse = await response.json();
      
      if (!serverResponse.success) {
        throw new Error(serverResponse.error_msg || '检测失败');
      }

      // 转换服务器响应为前端需要的格式
      return this.transformInferenceResult(serverResponse);
    } catch (error) {
      console.error('API request failed:', error);
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('无法连接到服务器，请确认后端服务是否正常运行');
      }
      throw error;
    }
  }

  // 转换服务器响应格式
  private transformInferenceResult(serverResponse: ServerInferenceResponse): InferenceResult {
    const detectionBoxes = serverResponse.result.detection_boxes || [];
    const detectionClasses = serverResponse.result.detection_classes || [];
    const detectionScores = serverResponse.result.detection_scores || [];
    
    // 计算缺陷统计
    const defectTypes: Record<string, number> = {};
    detectionClasses.forEach(className => {
      defectTypes[className] = (defectTypes[className] || 0) + 1;
    });

    // 转换检测结果格式
    const detection_results = detectionBoxes.map((box, index) => ({
      class: detectionClasses[index] || 'Unknown',
      confidence: detectionScores[index] || 0,
      bbox: [box[1], box[0], box[3], box[2]] as [number, number, number, number] // 转换 [y1,x1,y2,x2] 到 [x1,y1,x2,y2]
    }));

    return {
      image_id: serverResponse.filename,
      detection_results,
      summary: {
        total_defects: detectionBoxes.length,
        defect_types: defectTypes
      },
      image_path: `uploads/${serverResponse.filename}`,
      timestamp: new Date().toISOString(),
      annotated_image_url: `${this.baseUrl.replace('/api/v1', '')}/images/${serverResponse.filename}`
    };
  }
  // 获取历史记录
  async getAllHistories(params?: {
    page?: number;
    limit?: number;
    date?: string;
  }): Promise<HistoriesResponse> {
    // 每次请求都获取最新的API地址
    this.baseUrl = getApiBaseUrl();
    
    const queryParams = new URLSearchParams();
    
    if (params?.page) queryParams.append('page', params.page.toString());
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    if (params?.date) queryParams.append('date', params.date);

    const url = `${this.baseUrl}${API_ENDPOINTS.GET_ALL_HISTORIES}${
      queryParams.toString() ? `?${queryParams.toString()}` : ''
    }`;

    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  // 获取单个历史记录详情
  async getOneHistory(imageId: string): Promise<InferenceResult> {
    const response = await fetch(
      `${this.baseUrl}${API_ENDPOINTS.GET_ONE_HISTORY}?image_id=${encodeURIComponent(imageId)}`
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const serverResponse: HistoryDetailResponse = await response.json();
    
    if (!serverResponse.success) {
      throw new Error(serverResponse.error_msg || '获取历史记录失败');
    }

    // 转换历史记录详情为标准格式
    return this.transformInferenceResult(serverResponse.result.inference_result);
  }

  // 获取标注后的图片URL
  getAnnotatedImageUrl(filename: string): string {
    return `${this.baseUrl.replace('/api/v1', '')}/images/${filename}`;
  }
}

export const pcbApiService = new PCBApiService();
