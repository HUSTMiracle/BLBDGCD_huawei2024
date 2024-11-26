import logging
import threading
import time
import os
from random import randint
from src.controller.videohandle import VideoStreamHandler

from src.controller.Yolov9.test import test as yolov9_infer

from src.config.param import START, RUNNING, YOLOV9_DETECTING, SORTING, STOPPED
#from src.controller.deepstream.deepstream import DeepStreamPipeline

from src.iot_util.util import property_uploader

class Controller(object):
    '''
    Controller class to control the GPIO pins of the Raspberry Pi
    Singleton class
    '''
    def __init__(self, output_pin: int , device_path : str , model_path : str, streaming_url : str, host : str, path: str) -> None:
        # define states
        self.belt_moving = False
        self.total_flaws_detected = 1
        self.host = host
        self.path = path
        self.streaming_url = self.host + str(randint(1000,10000)) + self.path
        self.same_flaw_flag = False
        #define IoT service
        self.property_uploader = property_uploader.PropertyUploader()

        #set IoT thread
        self.propThread = threading.Thread(target=self.batch_upload_property)
        self.propThread.start()
        self.img_path = ""

        self.output_pin = output_pin

        # set pin mode
        os.system(f"gpio mode {self.output_pin} out")


        logging.basicConfig(level=logging.INFO)

        # TODO:perfectly match interval time with the belt.
        self.video_handler = VideoStreamHandler(
            save_path= path,
            capture_interval=3,
            camera_source=streaming_url
        )

    def batch_upload_property(self):
        while True:
            property = {
                "total_flaws_detected": self.total_flaws_detected,
                "belt_moving": self.belt_moving,
                "streaming_url": self.streaming_url
            }
            self.property_uploader.upload_properties("get_deviceinfo",property)
            time.sleep(2)

    @classmethod
    def __new__(cls, *args, **kwargs) -> object:
        if not hasattr(cls, 'instance'):
            cls.instance = super(Controller, cls).__new__(cls)
        return cls.instance

    def __call__(self, current_state: str) -> None:
        '''
        State machine to control the orangepi
        '''
        while True:
            try:
                print(current_state)
                match current_state:
                    case "START":
                        logging.info("Initial state, press Ctrl+C to exit the program")
                        self.conveyor_belts_stopped()
                        current_state = RUNNING
                        time.sleep(0.5)

                    case "RUNNING":
                        logging.info("Conveyor belts running and deepstream processing")
                        self.conveyor_belts_running()
                        current_state = YOLOV9_DETECTING
                        time.sleep(2)

                    case "YOLOV9_DETECTING":
                        logging.info("YOLOv9 object detection")
                        self.conveyor_belts_stopped()
#=================================Yolov9 infer===============================
                        # capture a frame
                        self.img_path = self.video_handler.start()
                        out_path,detected_num = yolov9_infer(self.img_path)
                        if detected_num > 0:
                            if self.same_flaw_flag == False:
                                self.total_flaws_detected += 1
                                self.same_flaw_flag = True
                            logging.info(f"Total flaws detected: {self.total_flaws_detected}")
                            current_state = YOLOV9_DETECTING
                            self.streaming_url = self.host + str(randint(1000,10000)) + self.path
                        else:
                            logging.info("No flaws detected")
                            current_state = RUNNING
#=================================Yolov9 inference==============================
                    # TODO:sorting automatically.

                    case "STOPPED":
                        logging.info("Conveyor belts stopped")
                        self.conveyor_belts_stopped()

                        # Waiting for manual input to start the conveyor belts or exit the program
                        input(f'Input "{START}" to start the conveyor belts or press Ctrl+C to exit the program: ')
                        current_state = START
                        time.sleep(5)

            except KeyboardInterrupt:
                logging.info("Keyboard interrupt, exiting the program")
                self.release_resources()
                break

            except Exception as e:
                logging.error(f"Error: {e}")
                current_state = "STOPPED"

    def conveyor_belts_running(self) -> None:
        #GPIO.output(self.output_pin, GPIO.HIGH)
        os.system(f"gpio write {self.output_pin} 1")
        self.belt_moving = True

    def conveyor_belts_stopped(self) -> None:
        #GPIO.output(self.output_pin, GPIO.LOW)
        os.system(f"gpio write {self.output_pin} 0")
        self.belt_moving = False
            
    def release_resources(self) -> None:
        #GPIO.cleanup()
        os.system(f"gpio write {self.output_pin} 0")
        logging.info("GPIO cleanup")
