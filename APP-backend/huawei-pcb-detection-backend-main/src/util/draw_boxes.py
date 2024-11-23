import os
import cv2


def draw_boxes(img_path, detection_results, output_dir="uploads"):

    img = cv2.imread(img_path)

    result_index = 0
    for result_boxes in detection_results["detection_boxes"]:            
        y1,x1,y2,x2 = result_boxes
        # 绘制矩形框
        color = (0, 255, 0)  # BGR颜色，绿色
        thickness = 1
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        trimmed_score_str = str(int(detection_results["detection_scores"][result_index]*10000)/100) + "%"
        # 可以在此处添加文本标签，如类别名
        cv2.putText(img, "Class: " + str(detection_results["detection_classes"][result_index]) + "; Score:" + trimmed_score_str, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        result_index += 1
    # 保存或显示带有矩形框的图片
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(img_path, img)