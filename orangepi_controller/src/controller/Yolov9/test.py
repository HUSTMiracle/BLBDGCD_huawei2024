from collections import defaultdict
import os
import sys
from PIL import Image, ImageEnhance
from pathlib import Path
import numpy as np
import torch
from .draw_boxes import draw_boxes
 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolo_v9.model import Yolov9_on_onnx, Yolov9_on_pt
from yolo_v9.utils.metrics import box_iou
from yolo_v9.utils.general import LOGGER

pt_model_path = None
onnx_model_path = "/home/orangepi/huawei-cloud-orangepi-controller/src/controller/Yolov9/weights/best.onnx"
model = None

if onnx_model_path is None:
    LOGGER.info("ONNX model file not found")
else:
    LOGGER.info(f"ONNX model file found: {onnx_model_path}")

if onnx_model_path is not None:
    model = Yolov9_on_onnx(onnx_model=onnx_model_path, conf_thres=0.15, iou_thres=0.45, device='cpu')
elif pt_model_path is not None:
    model = Yolov9_on_pt(pt_model_path, conf_thres=0.15, iou_thres=0.45, device='cpu')

class MeanAveragePrecison:
    def __init__(self, device="cpu"):
        """
        计算mAP: mAP@0.5; mAP @0.5:0.95; mAP @0.75
        """
        self.iouv = torch.linspace(0.5, 0.95, 10, device=device)  # 不同的IoU置信度 @0.5:0.95
        self.niou = self.iouv.numel()  # IoU置信度数量
        self.stats = []  # 存储预测结果
        self.device = device
    def process_batch(self, detections, labels):
        """
        预测结果匹配(TP/FP统计)
        :param detections:(array[N,6]) x1,y1,x2,y2,conf,class (原图绝对坐标)
        :param labels:(array[M,5]) class,x1,y1,x2,y2 (原图绝对坐标)
        """
        # 每一个预测结果在不同IoU下的预测结果匹配
        correct = np.zeros((detections.shape[0], self.niou)).astype(bool)
        if detections is None:
            self.stats.append((correct, *torch.zeros((2, 0), device=self.device), labels[:, 0]))
        else:
        # 计算标签与所有预测结果之间的IoU
            iou = box_iou(labels[:, 1:], detections[:, :4])
            # 计算每一个预测结果可能对应的实际标签
            correct_class = labels[:, 0:1] == detections[:, 5]
            for i in range(self.niou):  # 在不同IoU置信度下的预测结果匹配结果
                # 根据IoU置信度和类别对应得到预测结果与实际标签的对应关系
                x = torch.where((iou >= self.iouv[i]) & correct_class)
                # 若存在和实际标签相匹配的预测结果
                if x[0].shape[0]:  # x[0]:存在为True的索引(实际结果索引), x[1]当前所有True的索引(预测结果索引)
                    # [label, detect, iou]
                    matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
                    if x[0].shape[0] > 1:  # 存在多个与目标对应的预测结果
                        matches = matches[matches[:, 2].argsort()[::-1]]  # 根据IoU从高到低排序 [实际结果索引,预测结果索引,结果IoU]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 每一个预测结果保留一个和实际结果的对应
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 每一个实际结果和一个预测结果对应
                    correct[matches[:, 1].astype(int), i] = True  # 表面当前预测结果在当前IoU下实现了目标的预测
            # 预测结果在不同IoU是否预测正确, 预测置信度, 预测类别, 实际类别
            self.stats.append((torch.tensor(correct, dtype=torch.bool, device=self.device), detections[:, 4], detections[:, 5], labels[:, 0]))

    def calculate_ap_per_class(self, save_dir='.', names=(), eps=1e-16):
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        # tp:所有预测结果在不同IoU下的预测结果 [n, 10]
        # conf: 所有预测结果的置信度
        # pred_cls: 所有预测结果得到的类别
        # target_cls: 所有图片上的实际类别
        tp, conf, pred_cls, target_cls = stats[0], stats[1], stats[2], stats[3]
        # 根据类别置信度从大到小排序
        i = np.argsort(-conf)  # 根据置信度从大到小排序
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # 得到所有类别及其对应数量(目标类别数)
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes
 
        # ap: 每一个类别在不同IoU置信度下的AP, p:每一个类别的P曲线(不同类别置信度), r:每一个类别的R(不同类别置信度)
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):  # 对每一个类别进行P,R计算
            i = pred_cls == c
            n_l = nt[ci]  # number of labels 该类别的实际数量(正样本数量)
            n_p = i.sum()  # number of predictions 预测结果数量
            if n_p == 0 or n_l == 0:
                continue

            # cumsum：轴向的累加和, 计算当前类别在不同的类别置信度下的P,R
            fpc = (1 - tp[i]).cumsum(0)  # FP累加和(预测为负样本且实际为负样本)
            tpc = tp[i].cumsum(0)  # TP累加和(预测为正样本且实际为正样本)
            # 召回率计算(不同的类别置信度下)
            recall = tpc / (n_l + eps)
            # 精确率计算(不同的类别置信度下)
            precision = tpc / (tpc + fpc)


            # 计算不同类别置信度下的AP(根据P-R曲线计算)
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = self.compute_ap(recall[:, j], precision[:, j])
        # 所有类别的ap值 @0.5:0.95
        return ap


    def compute_ap(self, recall, precision):
        # 增加初始值(P=1.0 R=0.0) 和 末尾值(P=0.0, R=1.0)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope np.maximun.accumulate
        # (返回一个数组,该数组中每个元素都是该位置及之前的元素的最大值)
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # 计算P-R曲线面积
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':  # 插值积分求面积
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO))
            # 积分(求曲线面积)
            ap = np.trapz(np.interp(x, mrec, mpre), x)
        elif method == 'continuous':  # 不插值直接求矩阵面积
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

def load_txt(txt_path, img_size=(2048, 3072)):
    '''
    读取txt文件
    :param txt_path: txt文件路径
    :return: (array[N,5]) class,x1,y1,x2,y2
    '''
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        labels = []
        # 读取每一行的标签信息
        for line in lines:
            line = line.strip().split()
            cls = int(line[0])
            x, y, w, h = map(float, line[1:])
            x1 = round((x - w / 2) * img_size[1])
            y1 = round((y - h / 2) * img_size[0])
            x2 = round((x + w / 2) * img_size[1])
            y2 = round((y + h / 2) * img_size[0])

            labels.append([cls, x1, y1, x2, y2])

    return labels

def test(img_path):
    score = []

    pred_results = model.inference(img_path)
    
    result = {}
    detection_classes = []
    detection_boxes = []
    detection_scores = []
    detections = []
    count = defaultdict(int)

    for pred in pred_results:
        classes, _, x1, y1, x2, y2, conf = pred
        count[classes] += 1

    max_num = 0
    class_name = None
    for key, value in count.items():
        if value > max_num:
            max_num = value
            class_name = key

    for pred in pred_results:
        classes, c, x1, y1, x2, y2, conf = pred

        if classes != class_name:
            continue

        detections.append([x1, y1, x2, y2, conf, c])

        detection_classes.append(classes)
        boxes = [y1,x1,y2,x2]
        detection_boxes.append(boxes)
        detection_scores.append(conf)

    result['detection_classes'] = detection_classes
    result['detection_boxes'] = detection_boxes
    result['detection_scores'] = detection_scores
    out_path = "/home/orangepi/huawei-cloud-orangepi-controller/test_onnx.bmp"
    draw_boxes(img_path, out_path, result)
    return out_path, len(detection_boxes)

def main():
    img_path = "/home/orangepi/huawei-cloud-orangepi-controller/src/controller/test_onnx.bmp"
    outpt,leng = test(img_path)
    print(leng)

if __name__ == '__main__':
    main()