import os
import util
from flask import request
import logging
import sys
import io

import util.draw_boxes
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np

import traceback
from PIL import Image

import time
import util.inference_token
import util.inference_ak_sk
import util.inference_local_v9
import json

from backend_model.postgres import PostgresModel

#from src.yolo_v9.utils.plots import Annotator, colors, save_one_box

class InferenceController:
    def __init__(self) -> None:
        self.local_model_busy = False
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    def ctrl_inference(self):
        os.makedirs(f"{os.getcwd()}/uploads", exist_ok=True)
        # Get the file from the request
        file = request.files['file']
        filetype = file.filename.split(".")[-1]
        filename = time.strftime("%Y%m%d-%H%M%S.") + filetype
        path = f"{os.getcwd()}/uploads/{filename}"
        file.save(path)
        result = None
        source = None

        #优先使用modelarts推理
        self.logger.info("using modelarts")
        result = util.inference_token.InferenceToken().infer(path)
        source = "ModelArts"
        if result.get("error_code"):
            self.logger.warn("modelarts inference failed, using edge server")
            if self.local_model_busy:
                self.logger.info("local model is busy, return error")
                result = {
                    "success": False,
                    "result": result,
                    "filename": f"{filename}",
                    "source": source
                }
                return result
            # 使用本地模型推理
            self.local_model_busy = True
            self.logger.info("start edge inference")
            try :
                print("loading model" + os.environ["MODEL_PATH"])
                local_model = util.inference_local_v9.InferenceLocalV9(os.environ["MODEL_PATH"])
                result = local_model.infer(path)
            except Exception as e:
                self.logger.error(e)
                result = {
                    "error_code": 500,
                    "error_msg": f"Internal Server Error, {e}"
                }
                self.local_model_busy = False
            self.logger.info("finished edge inference")
            self.local_model_busy = False
            source = "Edge Server"
        result = {
       "success": True,
            "result": result,
            "filename": f"{filename}",
            "source": source
            }
        PostgresModel().insert_inference_result(filename,result)
        # 将检测结果标注在图上
        if result["success"]:
            util.draw_boxes.draw_boxes(path, result["result"])
        return result
    


    def ctrl_inference_binary(self):
        os.makedirs(f"{os.getcwd()}/uploads", exist_ok=True)
        arg_filename = request.args.get("filename")

        self.logger.info(f"Received request: Method={request.method}, Content-Type={request.content_type}")
        self.logger.info(f"Request headers: {request.headers}")

        # 获取字符串形式的图像数据
        image_data_str = request.data.decode('utf-8')
        self.logger.info(f"Received data: {image_data_str[:100]}...")  # 记录前100个字符

        # 将字符串转换回字节数组
        image_data = bytes(map(int, image_data_str.split(',')))
        
        self.logger.info(f"Converted data length: {len(image_data)} bytes")
        self.logger.info(f"First few bytes: {image_data[:20].hex()}")

        # 尝试检测图像格式
        image = Image.open(io.BytesIO(image_data))
        self.logger.info(f"Detected image format: {image.format}")
        self.logger.info(f"Created image with size: {image.size}")

        # 保存图像
        file_ext = arg_filename.split(".")[-1]
        filename = time.strftime("%Y%m%d-%H%M%S.") +  file_ext
        path = f"{os.getcwd()}/uploads/{filename}"
        image.save(path)
        self.logger.info("Image saved successfully")

        result = None
        source = None

        #优先使用modelarts推理
        self.logger.info("using modelarts")
        result = util.inference_token.InferenceToken().infer(path)
        source = "ModelArts"
        if result.get("error_code"):
            self.logger.warn("modelarts inference failed, using edge server")
            if self.local_model_busy:
                self.logger.info("local model is busy, return error")
                result = {
                    "success": False,
                    "result": result,
                    "filename": f"{filename}",
                    "source": source
                }
                return result
            # 使用本地模型推理
            self.local_model_busy = True
            self.logger.info("start edge inference")
            try :
                print("loading model" + os.environ["MODEL_PATH"])
                local_model = util.inference_local_v9.InferenceLocalV9(os.environ["MODEL_PATH"])
                result = local_model.infer(path)
            except Exception as e:
                self.logger.error(e)
                result = {
                    "error_code": 500,
                    "error_msg": f"Internal Server Error, {e}"
                }
                self.local_model_busy = False
            self.logger.info("finished edge inference")
            self.local_model_busy = False
            source = "Edge Server"
        result = {
            "success": True,
            "result": result,
            "filename": f"{filename}",
            "source": source
            }
        PostgresModel().insert_inference_result(filename,result)
        # 将检测结果标注在图上
        if result["success"]:
            util.draw_boxes.draw_boxes(path, result["result"])
        return result


        
        
    
    def ctrl_inference_test(self):
        os.makedirs(f"{os.getcwd()}/uploads", exist_ok=True)
        # Get the file string from the request
        fileStr = request.form["file"]
        result = {
            "filestr": fileStr
        }
        return result
    
    def ctrl_inference_old(self):
        os.makedirs(f"{os.getcwd()}/uploads", exist_ok=True)
        # Get the file from the request
        file = request.files['file']
        filename = time.strftime("%Y%m%d-%H%M%S") + ".png"
        path = f"{os.getcwd()}/uploads/{filename}"
        file.save(path)
        result = None
        source = None
        if self.local_model_busy:
            self.logger.info("local model is busy, using modelarts")
            result = util.inference_token.InferenceToken().infer(path)
            source = "ModelArts"
        else:
            self.local_model_busy = True
            self.logger.info("start edge inference")
            result = self.model.infer(path)
            self.logger.info("finished edge inference")
            self.local_model_busy = False
            source = "Edge Server"
        if result.get("error_code"):
            result = {
                "success": False,
                "result": result,
                "filename": f"{filename}",
                "source": source
            }
            return result
        result = {
            "success": True,
            "result": result,
            "filename": f"{filename}",
            "source": source
            }
        PostgresModel().insert_inference_result(filename,result)
        return result
    def ctrl_get_all_histories(self):
        page = request.args.get("page")
        limit = request.args.get("limit")
        date_string = str(request.args.get("date"))
        result = PostgresModel().get_all_histories(page,limit,date_string)
        resp = []
        for row in result:
            try:
                resp.append({
                    "image_id": row[0],
                    "detected_flaws": json.loads(row[1])["result"]["detection_boxes"].__len__()
                    
                })
            except:
                resp.append({
                    "image_id": row[0],
                    "detected_flaws": 0
                })
        return {
            "success": True,
            "result": resp
        }
    def ctrl_get_one_history_result(self):
        image_id = request.args.get("image_id")
        result = PostgresModel().get_inference_json(image_id)
        return {
            "success": True,
            "result": {
                "image_id": image_id,
                "inference_result": result
            }
        }
    
