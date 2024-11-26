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

    # iot config
    dotenv.load_dotenv(verbose=True)

    controller = Controller(
        OUTPUT_PIN,
        "114",
        "514",
        "rtsp://10.12.168.10:8554/camera",
        "http://202.114.213.225:8010/",
        "/test_onnx.bmp"
        
        )
    controller(START)



if __name__ == "__main__":
    main()
