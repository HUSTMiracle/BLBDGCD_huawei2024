import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.getcwd())
from yolo_v9.model import Yolov9_on_pt, Yolov9_on_onnx

class InferenceLocalV9:
    def __init__(self,model_path) -> None:
        self.model = Yolov9_on_onnx(onnx_model=model_path.replace('.pt', '.onnx'), conf_thres=0.15, iou_thres=0.45, device='cpu')
    def infer(self,file_path):
        # Send request.
        data = self.model.inference(file_path)

        result = {}
        detection_classes = []
        detection_boxes = []
        detection_scores = []               

        for pred in data:
            classes, _, x1, y1, x2, y2, conf = pred            

            detection_classes.append(classes)
            boxes = [y1,x1,y2,x2]
            detection_boxes.append(boxes)
            detection_scores.append(conf)

        result['detection_classes'] = detection_classes
        result['detection_boxes'] = detection_boxes
        result['detection_scores'] = detection_scores
            
        print('result:',result)    
        return result