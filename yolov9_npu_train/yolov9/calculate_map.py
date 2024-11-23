import os
import numpy as np

def load_detection_results(file_path):
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            class_id, x, y, w, h = map(float, line.strip().split())
            results.append([class_id, x, y, w, h])
    return results

def calculate_iou(box1, box2):
    x1_min = box1[1] - box1[3] / 2
    x1_max = box1[1] + box1[3] / 2
    y1_min = box1[2] - box1[4] / 2
    y1_max = box1[2] + box1[4] / 2
    
    x2_min = box2[1] - box2[3] / 2
    x2_max = box2[1] + box2[3] / 2
    y2_min = box2[2] - box2[4] / 2
    y2_max = box2[2] + box2[4] / 2
    
    intersect_x_min = max(x1_min, x2_min)
    intersect_x_max = min(x1_max, x2_max)
    intersect_y_min = max(y1_min, y2_min)
    intersect_y_max = min(y1_max, y2_max)
    
    intersect_area = max(0, intersect_x_max - intersect_x_min) * max(0, intersect_y_max - intersect_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou

def calculate_map(detection1, detection2, iou_thresholds):
    aps = []
    for iou_threshold in iou_thresholds:
        ious = np.zeros((len(detection1), len(detection2)))
        for i, det1 in enumerate(detection1):
            for j, det2 in enumerate(detection2):
                if det1[0] == det2[0]:  # Check if class is the same
                    ious[i, j] = calculate_iou(det1, det2)
        
        tp = (ious > iou_threshold).sum()
        fp = len(detection2) - tp
        fn = len(detection1) - tp
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        
        aps.append(precision)
    
    map_50_95 = np.mean(aps)
    return aps, map_50_95

def main():
    folder1 = '/home/tr/huawei_cloud_competition/data/test/labels'
    folder2 = '/home/tr/huawei_cloud_competition/data/test/txt'
    
    files = os.listdir(folder1)
    files = [f for f in files if f.endswith('.txt')]
    files.sort()  # Sort files by name
    
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    results = {}
    
    for file in files:
        detection1 = load_detection_results(os.path.join(folder1, file))
        detection2 = load_detection_results(os.path.join(folder2, file))
        
        aps, map_50_95 = calculate_map(detection1, detection2, iou_thresholds)
        results[file] = {'aps': aps, 'map_50_95': map_50_95}
    
    for file in sorted(results.keys()):  # Ensure output is sorted by filename
        metrics = results[file]
        print(f'File: {file}')
        for i, iou_threshold in enumerate(iou_thresholds):
            print(f'mAP@{int(iou_threshold*100)}: {metrics["aps"][i]}')
        print(f'mAP@50:95: {metrics["map_50_95"]}')
        print('')

if __name__ == '__main__':
    main()