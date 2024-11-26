import threading

import cv2
import subprocess as sp

class StreamingThread(threading.Thread):
    def __init__(self,stream_type,push_url,camera_path):
        threading.Thread.__init__(self)
        self.stream_type = stream_type
        self.camera_path = camera_path
        self.push_url = push_url
        self.get_url = "rtsp://admin:IA5cWJEY@202.114.213.56:554/streaming/channels/101/?transportmode=unicast"
    
    def run(self):

        # ffmpeg command
        if self.stream_type == "rtmp":
            command = ['ffmpeg',
                    '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', "{}x{}".format(width, height),
                    '-r', str(fps),
                    '-i', '-',
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'ultrafast',
                    '-f', 'flv',
                    self.push_url]
        else:
            command = [
                        'ffmpeg',
                        '-i', 'rtsp://admin:IA5cWJEY@202.114.213.56:554/streaming/channels/101/?transportmode=unicast',
                        '-r','15',
                        '-c', 'copy',
                        '-f', 'rtsp',
                        'rtsp://10.12.168.10:8554/camera'
                        ]
        # 管道配置
        p = sp.Popen(command, stdin=sp.PIPE)
