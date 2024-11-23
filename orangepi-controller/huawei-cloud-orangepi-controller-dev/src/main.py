import sys
import dotenv
from pathlib import Path
import time
from streaming.streaming import StreamingThread
import http
from flask import Flask, redirect, url_for, request, send_from_directory
import os

# Add src to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from src.config.param import START, OUTPUT_PIN
from src.controller import Controller

def main() -> None:
    '''
    Main function
    '''
    # Streaming instance
    streaming_thread = StreamingThread(
        stream_type="rtsp",
        push_url="rtsp://10.12.168.10:8554/camera",
        camera_path="/dev/video0",
    )
    streaming_thread.start()
    print("Streaming started")
    
    # file_server_thread = file_server()
    # file_server_thread.start()

    # iot config
    dotenv.load_dotenv(verbose=True)
    time.sleep(5)
    # Controller instance
    controller = Controller(
        OUTPUT_PIN,
        "114",
        "514",
        "rtsp://10.12.168.10:8554/camera",
        "http://202.114.213.224:8010/",
        "/test_onnx.jpg"
        
        )
    controller(START)



if __name__ == "__main__":
    main()
