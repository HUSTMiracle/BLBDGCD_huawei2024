import cv2
import time
import os
import ctypes
from hik_camera import HikCamera
from .convertpixel import picture

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
        self.save_path = "/home/orangepi/huawei-cloud-orangepi-controller/test_onnx.bmp"
        picture(self.save_path)
        return self.save_path

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
