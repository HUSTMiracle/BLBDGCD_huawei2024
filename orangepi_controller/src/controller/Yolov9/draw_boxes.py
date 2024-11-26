import cv2


def draw_boxes(img_path,out_path,detection_results):

    img = cv2.imread(img_path)

    result_index = 0
    for result_boxes in detection_results["detection_boxes"]:            
        y1,x1,y2,x2 = result_boxes
        # 绘制矩形框
        color = (0, 0, 255)  # BGR颜色，红色
        thickness = 2
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # 基于框，画一个更大的圆形
        center = (int((x1+x2)/2), int((y1+y2)/2))
        radius = 30
        cv2.circle(img, center, radius, color, thickness*3)
        

        trimmed_score_str = str(int(detection_results["detection_scores"][result_index]*10000)/100) + "%"
        
        # 添加文本标签，如类别名, 置信度;并给文字添加底色
        cv2.putText(img, "Class: " + str(detection_results["detection_classes"][result_index]) + "; Conf:" + trimmed_score_str, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)

        
        result_index += 1
    # 保存或显示带有矩形框的图片
    cv2.imwrite(out_path, img)