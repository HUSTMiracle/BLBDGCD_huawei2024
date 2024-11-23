from collections import defaultdict

from yolo_v9.model import Yolov9_on_pt, Yolov9_on_onnx
from PIL import Image, ImageEnhance
from model_service.pytorch_model_service import PTServingBaseService # type: ignore

class yolov9_detection(PTServingBaseService):
    def __init__(self, model_name, model_path):
        # TODO: Change the model to Yolov9_on_pt if you want to use the PyTorch model
        #self.model = Yolov9_on_pt(model_path, conf_thres=0.3, iou_thres=0.01, device='cpu', imgsz=[2048, 2048])
        self.model = Yolov9_on_onnx(onnx_model=model_path.replace('.pt', '.onnx'), conf_thres=0.3, iou_thres=0.01, device='cpu',imgsz=[3072,3072])

        self.capture = "test.png"

    def _preprocess(self, data):
        for _, v in data.items():
            for _, file_content in v.items():
                with open(self.capture, 'wb') as f:
                    file_content_bytes = file_content.read()
                    f.write(file_content_bytes)

                # img = Image.open(self.capture)
                # enhancer = ImageEnhance.Brightness(img)
                # img = enhancer.enhance(1.7)

                # enhancer = ImageEnhance.Contrast(img)
                # img = enhancer.enhance(2)

                # img.save(self.capture)
        return "ok"
    
    def _inference(self, data):
        # TODO: Choose segmentation or not, onnx不建议切割
        pred_result = self.model.inference(self.capture)
        # pred_result = self.model.inference(self.capture, segment=True, row_num=4, col_num=6)
        return pred_result

    def _postprocess(self, data):
        result = {}
        detection_classes = []
        detection_boxes = []
        detection_scores = []
        
        count = defaultdict(int)

        for pred in data:
            classes, _, x1, y1, x2, y2, conf = pred
            count[classes] += 1


        max_num = 0
        class_name = []
        single_class = []
        for key, value in count.items():
            if value > max_num and value < 15:
                class_name.append(key)
            if value == 1:
                single_class.append(key)

        for pred in data:
            classes, _, x1, y1, x2, y2, conf = pred

            if classes not in class_name:
                continue


            

            detection_classes.append(classes)
            boxes = [y1,x1,y2,x2]
            detection_boxes.append(boxes)
            detection_scores.append(conf)

        result['detection_classes'] = detection_classes
        result['detection_boxes'] = detection_boxes
        result['detection_scores'] = detection_scores
            
        print('result:',result)    
        return result
