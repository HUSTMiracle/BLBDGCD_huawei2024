import cv2
import time
import os
import ctypes

l_user_id = None
l_channel = 1
user_name = 'admin'
user_password = 'IA5cWJEY'
camera_ip = '202.114.213.56'
port = 8000
NET_DVR_GET_FOCUSMODECFG = 3305
NET_DVR_SET_FOCUSMODECFG = 3306

# print(os.getcwd())

class NET_DVR_JPEGPARA(ctypes.Structure):
    _fields_ = [
        ("wPicSize", ctypes.c_ushort),
        ("wPicQuality", ctypes.c_ushort)]

class LPNET_DVR_DEVICEINFO_V30(ctypes.Structure):
    _fields_ = [
        ("sSerialNumber", ctypes.c_byte * 48),
        ("byAlarmInPortNum", ctypes.c_byte),
        ("byAlarmOutPortNum", ctypes.c_byte),
        ("byDiskNum", ctypes.c_byte),
        ("byDVRType", ctypes.c_byte),
        ("byChanNum", ctypes.c_byte),
        ("byStartChan", ctypes.c_byte),
        ("byAudioChanNum", ctypes.c_byte),
        ("byIPChanNum", ctypes.c_byte),
        ("byZeroChanNum", ctypes.c_byte),
        ("byMainProto", ctypes.c_byte),
        ("bySubProto", ctypes.c_byte),
        ("bySupport", ctypes.c_byte),
        ("bySupport1", ctypes.c_byte),
        ("bySupport2", ctypes.c_byte),
        ("wDevType", ctypes.c_uint16),
        ("bySupport3", ctypes.c_byte),
        ("byMultiStreamProto", ctypes.c_byte),
        ("byStartDChan", ctypes.c_byte),
        ("byStartDTalkChan", ctypes.c_byte),
        ("byHighDChanNum", ctypes.c_byte),
        ("bySupport4", ctypes.c_byte),
        ("byLanguageType", ctypes.c_byte),
        ("byVoiceInChanNum", ctypes.c_byte),
        ("byStartVoiceInChanNo", ctypes.c_byte),
        ("byRes3", ctypes.c_byte * 2),
        ("byMirrorChanNum", ctypes.c_byte),
        ("wStartMirrorChanNo", ctypes.c_uint16),
        ("byRes2", ctypes.c_byte * 2)]


def load_so(so_dir):
    # print('切换工作目录和库文件所在路径：', so_dir)
    # os.chdir(so_dir)
    lib_hc_net_sdk = ctypes.cdll.LoadLibrary(so_dir + "/libhcnetsdk.so")
    ok = lib_hc_net_sdk.NET_DVR_Init()
    print("NET_DVR_Init: ", ok)
    out_device_info = LPNET_DVR_DEVICEINFO_V30()
    l_user_id = lib_hc_net_sdk.NET_DVR_Login_V30(bytes(camera_ip, 'ascii'),
                                                 port,
                                                 bytes(user_name, 'ascii'),
                                                 bytes(user_password, 'ascii'),
                                                 ctypes.byref(out_device_info))
    print('out_device_info', out_device_info)
    if l_user_id == -1:  # 登录失败
        error_num = lib_hc_net_sdk.NET_DVR_GetLastError()
        if error_num == 7:
            print("连接设备失败设备不在线或网络原因引起的连接超时等")
        print('err_num', error_num)
        res = lib_hc_net_sdk.NET_DVR_Cleanup()
        print('NET_DVR_Cleanup', res)
        return
    else:
        print('login success')
    print("NET_DVR_Login_V30:", l_user_id)

    # 拍照
    name = "/home/orangepi/huawei-cloud-orangepi-controller/assets/test.jpg"
    obj = NET_DVR_JPEGPARA()
    obj.wPicQuality = 0
    obj.wPicSize = 6
    res = lib_hc_net_sdk.NET_DVR_CaptureJPEGPicture(l_user_id, l_channel, ctypes.byref(obj), bytes(name, 'utf-8'))
    if not res:
        print('fail')
    else:
        print('success %s ' % name)

    # 释放资源
    res = lib_hc_net_sdk.NET_DVR_Logout(l_user_id)
    if not res:
        print('退出失败', lib_hc_net_sdk.NET_DVR_GetLastError())
    res = lib_hc_net_sdk.NET_DVR_Cleanup()
    print('NET_DVR_Cleanup', res)
    print('NET_DVR_Logout', res)
    return name

class VideoStreamHandler:
    def __init__(self, save_path, capture_interval, camera_source):
        """
        初始化视频流处理类
        :param save_path: 保存视频帧的路径
        :param capture_interval: 保存帧的时间间隔（秒）
        :param camera_source: 摄像头源，可以是设备路径如 /dev/video0
        """
        self.save_path = save_path
        self.capture_interval = capture_interval
        self.camera_source = camera_source
        self.cap = None
        self.running = False

    def start(self):
        """
        启动视频流处理
        """
        # self.camera_source = "rtsp://10.12.168.10:8554/camera"
        # self.cap = cv2.VideoCapture(self.camera_source)
        # if not self.cap.isOpened():
        #     print(f"无法打开摄像头: {self.camera_source}")
        #     return

        # self.running = True
        # print(f"视频流已启动，摄像头: {self.camera_source}")

        # # last_capture_time = time.time()

        # # while self.running:
        # ret, frame = self.cap.read()
        # if not ret:
        #     print("无法读取视频帧")

        # file_count = 0
        # for root,dirs,files in os.walk(self.save_path):
            
        #     file_count += len(files)

        # current_time = time.time()

        # # if current_time - last_capture_time >= self.capture_interval:
        # #     timestamp = int(current_time)
        # frame_save_path = f"{self.save_path}/frame_{0}.jpg"
        # cv2.imwrite(frame_save_path, frame)
        # print(f"视频帧已保存: {frame_save_path}")
        # # 保存了一次后停止
        # self.stop()
        

        # self.cap.release()
        # return frame_save_path
        name = load_so('/home/orangepi/huawei-cloud-orangepi-controller/src/HCNetPythonSDK-Linux/hklib')
        return name

    def stop(self):
        """
        停止视频流处理
        """
        self.running = False
        print("视频流已停止")


# 使用示例
video_handler = VideoStreamHandler(
    save_path='/home/orangepi/huawei-cloud-orangepi-controller/assets',
    capture_interval=3,
    camera_source="/dev/video0"
)
video_handler.start()
