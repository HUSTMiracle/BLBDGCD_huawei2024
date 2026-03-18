import { Camera, CameraResultType, CameraSource } from '@capacitor/camera'
import type { Photo } from '@capacitor/camera'
import { Filesystem, Directory } from '@capacitor/filesystem'
import { Capacitor } from '@capacitor/core'

export interface CameraOptions {
  source?: CameraSource
  quality?: number
  allowEditing?: boolean
  resultType?: CameraResultType
}

export class CameraService {  /**
   * 拍照或从相册选择图片
   */  static async takePicture(options: CameraOptions = {}): Promise<Photo> {
    const isAndroid = Capacitor.getPlatform() === 'android'
    
    // Android平台使用Base64模式更可靠
    const defaultOptions = {
      quality: 90,
      allowEditing: false,
      resultType: isAndroid ? CameraResultType.Base64 : CameraResultType.Uri,
      source: CameraSource.Prompt,
      ...options
    }

    try {
      console.log('拍照选项:', defaultOptions)
      console.log('平台信息:', {
        platform: Capacitor.getPlatform(),
        isNative: Capacitor.isNativePlatform(),
        isAndroid: isAndroid
      })
      
      const image = await Camera.getPhoto(defaultOptions)
      console.log('拍照结果:', {
        hasWebPath: !!image.webPath,
        hasBase64: !!image.base64String,
        format: image.format,
        webPath: image.webPath ? image.webPath.substring(0, 50) + '...' : 'none',
        base64Length: image.base64String ? image.base64String.length : 0
      })
      
      // 验证返回的图片数据
      if (!image.webPath && !image.base64String) {
        console.error('相机返回了无效的图片数据:', JSON.stringify(image, null, 2))
        throw new Error('相机未返回有效的图片数据，请重试')
      }
      
      // Android平台必须有base64数据
      if (isAndroid && !image.base64String) {
        console.error('Android平台没有返回base64数据')
        throw new Error('Android平台图片数据格式错误，请重试')
      }
      
      return image
    } catch (error) {
      console.error('拍照失败:', error)
      // 改善错误信息
      if (error && typeof error === 'object' && 'message' in error) {
        const errorMessage = (error as any).message
        if (errorMessage.includes('User cancelled photos app')) {
          throw new Error('用户取消了操作')
        } else if (errorMessage.includes('Permission denied')) {
          throw new Error('没有相机或存储权限，请在设置中开启')
        } else if (errorMessage.includes('Camera not available')) {
          throw new Error('相机不可用，请检查设备')
        }
      }
      throw error instanceof Error ? error : new Error('拍照或选择图片失败')
    }
  }

  /**
   * 从相机拍照
   */
  static async takeFromCamera(options: CameraOptions = {}): Promise<Photo> {
    return this.takePicture({
      ...options,
      source: CameraSource.Camera
    })
  }

  /**
   * 从相册选择图片
   */
  static async selectFromGallery(options: CameraOptions = {}): Promise<Photo> {
    return this.takePicture({
      ...options,
      source: CameraSource.Photos
    })
  }  /**
   * 将Photo对象转换为File对象
   */  static async photoToFile(photo: Photo, fileName?: string): Promise<File> {
    try {
      const isAndroid = Capacitor.getPlatform() === 'android'
      
      console.log('开始转换Photo到File:', {
        hasWebPath: !!photo.webPath,
        hasBase64: !!photo.base64String,
        format: photo.format,
        platform: Capacitor.getPlatform(),
        isAndroid: isAndroid
      })
      
      let blob: Blob

      // Android平台优先使用base64，其他平台优先使用webPath
      if (isAndroid && photo.base64String) {
        console.log('Android平台：使用base64String转换')
        try {
          // 处理base64数据，可能包含或不包含data:image前缀
          let base64Data = photo.base64String
          if (base64Data.startsWith('data:')) {
            // 移除data:image/jpeg;base64,前缀
            base64Data = base64Data.split(',')[1]
          }
          
          const binaryString = atob(base64Data)
          const bytes = new Uint8Array(binaryString.length)
          for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i)
          }
          blob = new Blob([bytes], { type: 'image/jpeg' })
          console.log('Android base64转换成功，blob大小:', blob.size)
        } catch (base64Error) {
          console.error('Android base64转换失败:', base64Error)
          throw new Error('Android平台图片数据处理失败')
        }
      } else if (!isAndroid && photo.webPath) {
        console.log('非Android平台：使用webPath转换:', photo.webPath)
        const response = await fetch(photo.webPath)
        if (!response.ok) {
          throw new Error(`无法获取图片数据: ${response.status} ${response.statusText}`)
        }
        blob = await response.blob()
        console.log('webPath转换成功，blob大小:', blob.size)
      } else if (photo.base64String) {
        console.log('备选方案：使用base64String转换')
        try {
          // 处理base64数据，可能包含或不包含data:image前缀
          let base64Data = photo.base64String
          if (base64Data.startsWith('data:')) {
            // 移除data:image/jpeg;base64,前缀
            base64Data = base64Data.split(',')[1]
          }
          
          const binaryString = atob(base64Data)
          const bytes = new Uint8Array(binaryString.length)
          for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i)
          }
          blob = new Blob([bytes], { type: 'image/jpeg' })
          console.log('base64转换成功，blob大小:', blob.size)
        } catch (base64Error) {
          console.error('base64转换失败:', base64Error)
          if (photo.webPath) {
            console.log('fallback到webPath转换')
            const response = await fetch(photo.webPath)
            if (!response.ok) {
              throw new Error('无法获取图片数据')
            }
            blob = await response.blob()
          } else {
            throw new Error('base64转换失败且没有webPath备选方案')
          }
        }
      } else if (photo.webPath) {
        console.log('最后备选：使用webPath转换:', photo.webPath)
        const response = await fetch(photo.webPath)
        if (!response.ok) {
          throw new Error(`无法获取图片数据: ${response.status} ${response.statusText}`)
        }
        blob = await response.blob()
        console.log('webPath转换成功，blob大小:', blob.size)
      } else {
        // 最后的尝试：检查是否有其他可用的数据
        console.error('Photo对象详细信息:', JSON.stringify(photo, null, 2))
        throw new Error('无效的图片数据：没有webPath或base64String')
      }

      // 验证blob是否有效
      if (!blob || blob.size === 0) {
        throw new Error('转换后的图片数据为空')
      }

      const file = new File([blob], fileName || `photo_${Date.now()}.jpg`, {
        type: blob.type || 'image/jpeg'
      })

      console.log('图片转换成功:', {
        fileName: file.name,
        size: file.size,
        type: file.type
      })
      
      return file
    } catch (error) {
      console.error('转换图片失败:', error)
      throw new Error('图片处理失败: ' + (error instanceof Error ? error.message : '未知错误'))
    }
  }

  /**
   * 保存图片到设备存储
   */
  static async savePhoto(photo: Photo, fileName?: string): Promise<string> {
    try {
      if (!photo.base64String) {
        throw new Error('图片数据无效')
      }

      const savedFile = await Filesystem.writeFile({
        path: fileName || `photo_${Date.now()}.jpg`,
        data: photo.base64String,
        directory: Directory.Data
      })

      return savedFile.uri
    } catch (error) {
      console.error('保存图片失败:', error)
      throw new Error('保存图片失败')
    }
  }
  /**
   * 检查相机权限
   */
  static async checkPermissions(): Promise<boolean> {
    try {
      if (!Capacitor.isNativePlatform()) {
        console.log('Web环境，跳过权限检查')
        return true // Web端不需要权限检查
      }
      
      console.log('检查相机权限...')
      const permissions = await Camera.checkPermissions()
      const hasCameraPermission = permissions.camera === 'granted'
      const hasPhotosPermission = permissions.photos === 'granted'
      
      console.log('权限检查结果:', { 
        camera: permissions.camera, 
        photos: permissions.photos 
      })
      
      return hasCameraPermission && hasPhotosPermission
    } catch (error) {
      console.error('检查权限失败:', error)
      return false
    }
  }

  /**
   * 请求相机权限
   */
  static async requestPermissions(): Promise<boolean> {
    try {
      if (!Capacitor.isNativePlatform()) {
        console.log('Web环境，跳过权限请求')
        return true // Web端不需要权限请求
      }
      
      console.log('请求相机权限...')
      const permissions = await Camera.requestPermissions()
      const hasCameraPermission = permissions.camera === 'granted'
      const hasPhotosPermission = permissions.photos === 'granted'
      
      console.log('权限请求结果:', { 
        camera: permissions.camera, 
        photos: permissions.photos 
      })
      
      return hasCameraPermission && hasPhotosPermission
    } catch (error) {
      console.error('请求权限失败:', error)
      return false
    }
  }

  /**
   * 检查是否在原生环境中运行
   */
  static isNative(): boolean {
    return Capacitor.isNativePlatform()
  }
}
