from pathlib import Path
import re
import psutil
import torch
import sys
import time
import os
import numpy as np
import onnxruntime as ort
from openvino.runtime import Core

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
print(ROOT)
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class_name = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
CLASSES =[ 'mouse', 'open', 'short', 'spur', 'spurious']

from yolo_v9.models.common import DetectMultiBackend
from yolo_v9.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolo_v9.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolo_v9.utils.plots import Annotator, colors, save_one_box
from yolo_v9.utils.torch_utils import select_device, smart_inference_mode

# 切割图片
class SegmentImg(object):
    def __init__(self, row_num: int, col_num: int, img_path, save_path=None):
        self.img_path = img_path
        self.save_path = save_path if save_path is not None else "temp/"
        self.row_num = row_num
        self.col_num = col_num

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    # 以 row_num * col_num 的格式切割
    def segment_img(self) -> list:
        img = cv2.imread(self.img_path)
        img_h, img_w = img.shape[:2]
        w = int (img_w / self.col_num)
        h = int (img_h / self.row_num)

        # 切割图片
        img_list = []
        img_path_list = []
        for i in range(self.row_num):
            for j in range(self.col_num):
                x = j * w
                y = i * h
                img_list.append(img[y:y+h, x:x+w])
                img_path_list.append(self.save_path + f'_{i}_{j}_{self.row_num}_{self.col_num}.png')
                if self.save_path is not None:
                    cv2.imwrite(self.save_path + f'_{i}_{j}_{self.row_num}_{self.col_num}.png', img_list[-1])

        return img_path_list, img_list, h, w
    
    # 根据切割后的 x1，y1 坐标，恢复原坐标 x0, y0
    def restore_xy(self, x1, y1, i, j):
        img = cv2.imread(self.img_path)
        img_h, img_w = img.shape[:2]
        w = int (img_w / self.col_num)
        h = int (img_h / self.row_num)
        x0 = j * w + x1
        y0 = i * h + y1
        return x0, y0


class Yolov9_on_pt(object):
    def __init__(self, weights, conf_thres=0.1, iou_thres=0.45, device='cpu', dnn=False, half=False, imgsz=(3072, 3072)):
        self.weights=weights  # self.model path or triton URL
        self.imgsz=imgsz  # inference size (height, width)
        self.conf_thres=conf_thres  # confidence threshold
        self.iou_thres=iou_thres  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device=''  # cuda self.device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=False  # show results
        self.save_txt=False  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False # self.augmented inference
        self.visualize=False  # self.visualize features
        self.update=False  # self.update all self.models
        self.project=ROOT / 'runs/detect'  # save results to self.project/self.name
        self.name='exp'  # save results to self.project/self.name
        self.exist_ok=False  # existing self.project/self.name ok, do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 self.half-precision inference
        self.dnn=False  # use OpenCV self.dnn for ONNX inference
        self.vid_stride=1  # video frame-rate stride

        # Load self.model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

    def inference(self, source, segment=False, save_path=None, row_num=4, col_num=6):
        source = str(source)

        pred_result = []
        seg1 = None

        if segment:
            seg1 = SegmentImg(row_num, col_num, source, save_path)
            source, _, h, w = seg1.segment_img()
            self.imgsz = (h, w)
            self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        
        # Dataloader
        bs = 1  # batch_size
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.self.device)
                im = im.self.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred = self.model(im, augment=self.augment, visualize=self.visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    for x1, y1, x2, y2, conf, cls in reversed(det):
                        c = int(cls)

                        if segment:
                            # 正则解析文件名，获取切割后的坐标
                            i_index, j_index, _, _ = re.findall(r'_(\d+)_(\d+)_(\d+)_(\d+)', p.stem)[0]
                            x0_1, y0_1 = seg1.restore_xy(float(x1), float(y1), int(i_index), int(j_index))
                            x0_2, y0_2 = seg1.restore_xy(float(x2), float(y2), int(i_index), int(j_index))
                            x1, y1, x2, y2 = x0_1, y0_1, x0_2, y0_2

                        det_result = class_name[c], c, float(x1), float(y1), float(x2), float(y2), float(conf)
                        pred_result.append(det_result)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)

        if self.update:
            strip_optimizer(self.weights[0])  # self.update self.model (to fix SourceChangeWarning)

        return pred_result

class OpenvinoInference(object):
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        ie = Core()
        self.model_onnx = ie.read_model(model=self.onnx_path)
        print(self.model_onnx)
        self.compiled_model_onnx = ie.compile_model(model=self.model_onnx, device_name="CPU")
        self.output_layer_onnx = self.compiled_model_onnx.output(0)

    def predict(self, datas):
        predict_data = self.compiled_model_onnx([datas])[self.output_layer_onnx]
        return predict_data

class Yolov9_on_onnx(object):
    def __init__(self, onnx_model, conf_thres=0.1, iou_thres=0.45, device='cpu', dnn=False, half=False, infer_tool='openvino', imgsz=(3072, 3072)):
        # self.weights=weights  # self.model path or triton URL
        self.imgsz=imgsz  # inference size (height, width)
        self.conf_thres=conf_thres  # confidence threshold
        self.iou_thres=iou_thres  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device=device  # cuda self.device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=False  # show results
        self.save_txt=False  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False # self.augmented inference
        self.visualize=False  # self.visualize features
        self.update=False  # self.update all self.models
        self.project=ROOT / 'runs/detect'  # save results to self.project/self.name
        self.name='exp'  # save results to self.project/self.name
        self.exist_ok=False  # existing self.project/self.name ok, do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 self.half-precision inference
        self.dnn=False  # use OpenCV self.dnn for ONNX inference
        self.vid_stride=1  # video frame-rate stride
        self.fp16=half  # self.half precision
        
        self.onnx_model = onnx_model
        self.infer_tool = infer_tool
        imgsz = self.imgsz
        self.stride=32
        self.names={0: 'mouse', 1: 'open', 2: 'short', 3: 'spur', 4: 'spurious'}
        self.pt = True
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        if self.infer_tool == 'openvino':
            #构建openvino推理引擎
            self.openvino = OpenvinoInference(onnx_model)
            self.ndtype = np.single
        else:
            #构建onnxruntime推理引擎
            self.ort_session = ort.InferenceSession(onnx_model,
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                                if ort.get_device() == 'GPU' else ['CPUExecutionProvider'])

            #Numpy dtype : support both FP32 and FP16 onnx model
            self.ndtype = np.half if self.ort_session.get_inputs()[0].type == 'tensor(float16)' else np.single
       
        self.classes = CLASSES  # 加载模型类别
        self.model_height, self.model_width = imgsz[0], imgsz[1]  # 图像resize大小
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))  # 为每个类别生成调色板
    
    # 绑定核心
    def set_cpu_affinity(self, cpu_list):
        p=psutil.Process()
        p.cpu_affinity(cpu_list)

    def inference(self, source, segment=False, save_path=None, row_num=4, col_num=6):
        pred_result = []

        self.set_cpu_affinity([0,1,2,3,4,5,6,7])

        seg1 = None
        img_list = None

        if segment:
            seg1 = SegmentImg(row_num, col_num, source, save_path)
            _, img_list, h, w = seg1.segment_img()
            self.imgsz = (h, w)
            self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        else:
            img_list = [cv2.imread(source)]

        for idx, im0 in enumerate(img_list):
            start_time = time.time()
            #前处理Pre - process
            im, ratio, (pad_w, pad_h) = self.preprocess(im0)

            #推理 inference
            if self.infer_tool == 'openvino':
                preds = self.openvino.predict(im)
            else:
                preds = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: im})[0]
            pred=torch.tensor(preds)
            
            
            #后处理Post - process
            boxes = self.postprocess(preds,
                            im0=im0,
                            ratio=ratio,
                            pad_w=pad_w,
                            pad_h=pad_h,
                            conf_threshold=self.conf_thres,
                            iou_threshold=self.iou_thres,
                            )
            
            for box in boxes:
                x1, y1, x2, y2, conf, class_id = box

                if segment:
                    j_index = idx % col_num
                    i_index = idx // col_num
                    x0_1, y0_1 = seg1.restore_xy(float(x1), float(y1), i_index, j_index)
                    x0_2, y0_2 = seg1.restore_xy(float(x2), float(y2), i_index, j_index)
                    x1, y1, x2, y2 = x0_1, y0_1, x0_2, y0_2

                det_result = class_name[int(class_id)], int(class_id), float(x1), float(y1), float(x2), float(y2), float(conf)
                pred_result.append(det_result)

            end_time = time.time()

            LOGGER.info(f"image {idx + 1}/{len(img_list)} processed: {end_time - start_time:.2f}s")
        return pred_result
    
    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """
        #Resize and pad input image using letterbox()(Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 填充

        # TODO
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #Transforms : HWC to CHW->BGR to RGB->div(255)->contiguous->add axis(optional)
        img = np.ascontiguousarray(np.einsum('HWC->CHW', img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)
    
    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.

        Returns:
            boxes (List): list of bounding boxes.
        """
        x = preds  # outputs: predictions (1, 84, 8400)
        #Transpose the first output : (Batch_size, xywh_conf_cls, Num_anchors)->(Batch_size, Num_anchors, xywh_conf_cls)
        x = np.einsum('bcn->bnc', x)  # (1, 8400, 84)

        #Predictions filtering by conf - threshold
        x = x[np.amax(x[..., 4:], axis=-1) > conf_threshold]

        #Create a new matrix which merge these(box, score, cls) into one
        #For more details about `numpy.c_()`: https: // numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:], axis=-1), np.argmax(x[..., 4:], axis=-1)]

        #NMS filtering
        #经过NMS后的值, np.array([[x, y, w, h, conf, cls], ...]), shape = (-1, 4 + 1 + 1)
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        #重新缩放边界框，为画图做准备
        if len(x) > 0:
        #Bounding boxes format change : cxcywh->xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            #Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            #Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            return x[..., :6]  # boxes
        else:
            return []
