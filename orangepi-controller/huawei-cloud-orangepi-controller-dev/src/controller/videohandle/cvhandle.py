import cv2
import numpy as np
import threading


class VideoHandler:
    def __init__(self, video_source='/dev/video0', save_path='pcb_image.png', area_threshold=2000, min_contour_area=500,
                 stability_frames=5, size_change_threshold=0.01):
        self.video_source = video_source
        self.save_path = save_path
        self.area_threshold = area_threshold
        self.min_contour_area = min_contour_area
        self.cap = None
        self.running = False
        self.thread = None
        self.stability_frames = stability_frames  # 稳定帧数
        self.size_change_threshold = size_change_threshold  # 大小变化阈值
        self.prev_area = None
        self.stable_count = 0

    def start(self):
        if self.running:
            print("Video processing is already running.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._process_video)
        self.thread.start()

    def stop(self):
        if not self.running:
            print("Video processing is not running.")
            return
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def _process_video(self):
        self.cap = cv2.VideoCapture(self.video_source)
        background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 前景检测
            fg_mask = background_subtractor.apply(frame)

            # 去除噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

            # 轮廓检测
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # 忽略小轮廓
                if cv2.contourArea(contour) < self.min_contour_area:
                    continue

                # 获取包围轮廓的矩形框
                x, y, w, h = cv2.boundingRect(contour)

                # 计算当前面积
                current_area = w * h

                # 如果当前面积大于面积阈值，开始检测是否稳定
                if current_area > self.area_threshold:
                    if self.prev_area is not None:
                        # 计算面积变化率
                        area_change = abs(current_area - self.prev_area) / self.prev_area

                        if area_change < self.size_change_threshold:
                            self.stable_count += 1
                        else:
                            self.stable_count = 0

                    # 如果连续多帧面积变化较小，认为PCB板稳定，保存图像
                    if self.stable_count >= self.stability_frames:
                        pcb_image = frame[y:y + h, x:x + w]
                        cv2.imwrite(self.save_path, pcb_image)
                        print(f"PCB image saved to {self.save_path}")
                        self.running = False  # 保存后停止处理
                        break

                    self.prev_area = current_area

        self.cap.release()


# 使用示例
if __name__ == "__main__":
    handler = VideoHandler(video_source='/dev/video0', save_path='pcb_image.png', stability_frames=5,
                           size_change_threshold=0.01)

    # 直接启动
    handler.start()

    # 在外部通过其他控制逻辑调用stop()，这里模拟运行过程中稳定时自动停止
    handler.stop()

