"""ROI drawing and image coordinate mapping for Qt."""
import cv2
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import QLabel, QMessageBox

class ROIDisplayLabel(QLabel):
    roi_selected = pyqtSignal(tuple)  # 信号：发出ROI矩形(x1, y1, x2, y2)
    prompt_point_selected = pyqtSignal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setMouseTracking(True)

        # ROI绘制状态
        self.drawing_roi = False
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.roi_fixed = False  # 新增：标记ROI是否已固定

        # 当前显示的图像
        self.current_pixmap = None
        self.original_image = None
        self.prompt_mode = False
        self.segmentation_mask = None
        self.prompt_points = []

    def mousePressEvent(self, event):
        if self.prompt_mode and event.button() in (Qt.LeftButton, Qt.RightButton):
            img_pos = self.map_to_image(event.pos())
            if img_pos is not None:
                self.prompt_point_selected.emit(
                    img_pos[0], img_pos[1], 1 if event.button() == Qt.LeftButton else 0
                )
            return
        # 如果ROI已固定，不响应鼠标点击
        if self.roi_fixed:
            return

        if not self.drawing_roi:
            return

        if event.button() == Qt.LeftButton:
            pos = event.pos()
            img_pos = self.map_to_image(pos)
            if img_pos:
                self.roi_start = img_pos
                self.roi_end = img_pos
                self.roi_rect = None

    def mouseMoveEvent(self, event):
        # 如果ROI已固定，不响应鼠标移动
        if self.roi_fixed:
            return

        if not self.drawing_roi or self.roi_start is None:
            return

        pos = event.pos()
        img_pos = self.map_to_image(pos)
        if img_pos:
            self.roi_end = img_pos
            self.update_display()

    def mouseReleaseEvent(self, event):
        # 如果ROI已固定，不响应鼠标释放
        if self.roi_fixed:
            return

        if not self.drawing_roi or self.roi_start is None:
            return

        if event.button() == Qt.LeftButton:
            pos = event.pos()
            img_pos = self.map_to_image(pos)
            if img_pos:
                self.roi_end = img_pos

                x1 = min(self.roi_start[0], self.roi_end[0])
                y1 = min(self.roi_start[1], self.roi_end[1])
                x2 = max(self.roi_start[0], self.roi_end[0])
                y2 = max(self.roi_start[1], self.roi_end[1])

                if x2 - x1 > 10 and y2 - y1 > 10:
                    self.roi_rect = (x1, y1, x2, y2)
                    self.roi_fixed = True  # 标记ROI已固定
                    self.roi_selected.emit(self.roi_rect)

                    print(f"ROI选择完成: ({x1}, {y1}) - ({x2}, {y2})")

                    self.drawing_roi = False
                    self.roi_start = None
                    self.roi_end = None

                    self.update_display()
                else:
                    QMessageBox.warning(self, "警告", "ROI区域太小，请选择更大的区域")
                    self.roi_start = None
                    self.roi_end = None
                    self.update_display()

    def map_to_image(self, pos):
        """将控件坐标映射到图像坐标"""
        if self.current_pixmap is None:
            return None

        # 获取当前显示的pixmap
        pixmap = self.pixmap()
        if pixmap.isNull():
            return None
        pixmap_rect = pixmap.rect()

        # 关键：计算从显示坐标到原始图像坐标的映射
        # 当前显示图像的尺寸
        disp_width = pixmap_rect.width()
        disp_height = pixmap_rect.height()

        # 原始图像尺寸
        if self.original_image is None:
            return None
        img_h, img_w = self.original_image.shape[:2]

        # 计算缩放比例（显示图像/原始图像）
        scale_x = disp_width / img_w if img_w > 0 else 1
        scale_y = disp_height / img_h if img_h > 0 else 1

        # 由于Qt保持宽高比缩放，取相同的缩放比例（取最小值）
        scale = min(scale_x, scale_y)

        # 计算显示图像在QLabel中的偏移
        offset_x = (self.width() - disp_width) / 2
        offset_y = (self.height() - disp_height) / 2

        if not (offset_x <= pos.x() < offset_x + disp_width and
                offset_y <= pos.y() < offset_y + disp_height):
            return None

        # 将控件坐标转换为显示图像坐标
        disp_x = pos.x() - offset_x
        disp_y = pos.y() - offset_y

        # 将显示图像坐标转换为原始图像坐标
        img_x = int(disp_x / scale)
        img_y = int(disp_y / scale)

        # 限制在图像范围内
        img_x = max(0, min(img_x, img_w - 1))
        img_y = max(0, min(img_y, img_h - 1))

        return (img_x, img_y)

    def update_display(self):
        """更新显示，绘制ROI矩形"""
        if self.current_pixmap is None:
            return

        # 获取原始图像的显示版本
        if self.original_image is None:
            return

        # 重新创建显示图像
        img_h, img_w = self.original_image.shape[:2]

        # 将OpenCV图像转换为Qt格式
        display_image = self.original_image.copy()
        if self.segmentation_mask is not None:
            mask = self.segmentation_mask
            if mask.shape == display_image.shape[:2]:
                tinted = display_image.copy()
                tinted[mask] = (0, 0, 255)
                display_image = cv2.addWeighted(display_image, 0.65, tinted, 0.35, 0)
                contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display_image, contours, -1, (255, 0, 0), 2)
        for index, (x, y, label) in enumerate(self.prompt_points, start=1):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(display_image, (x, y), 6, color, -1)
            cv2.putText(display_image, str(index), (x + 9, y + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if len(display_image.shape) == 3:
            rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb.data, img_w, img_h, rgb.strides[0], QImage.Format_RGB888)
        else:
            q_img = QImage(display_image.data, img_w, img_h,
                           display_image.strides[0], QImage.Format_Grayscale8)

        # 创建QPixmap
        pixmap = QPixmap.fromImage(q_img)

        # 缩放以适合标签（保持宽高比）
        self.current_pixmap = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # 创建用于绘制的pixmap副本
        draw_pixmap = self.current_pixmap.copy()
        painter = QPainter(draw_pixmap)

        # 如果正在绘制ROI，绘制临时矩形
        if self.drawing_roi and self.roi_start and self.roi_end:
            pen = QPen(QColor(0, 255, 0), 2)  # 绿色，2像素宽
            painter.setPen(pen)

            # 将原始图像坐标映射到显示坐标
            scale = self.current_pixmap.width() / img_w

            start_x = int(self.roi_start[0] * scale)
            start_y = int(self.roi_start[1] * scale)
            end_x = int(self.roi_end[0] * scale)
            end_y = int(self.roi_end[1] * scale)

            # 绘制矩形
            painter.drawRect(start_x, start_y, end_x - start_x, end_y - start_y)

            # 添加文本
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(start_x + 5, start_y + 20, "拖动鼠标选择ROI区域")

        # 如果有已确定的ROI，绘制固定矩形
        elif self.roi_rect:
            x1, y1, x2, y2 = self.roi_rect

            # 将原始图像坐标映射到显示坐标
            scale = self.current_pixmap.width() / img_w

            disp_x1 = int(x1 * scale)
            disp_y1 = int(y1 * scale)
            disp_x2 = int(x2 * scale)
            disp_y2 = int(y2 * scale)

            # 绘制矩形
            pen = QPen(QColor(255, 0, 0), 2)  # 红色，2像素宽
            painter.setPen(pen)
            painter.drawRect(disp_x1, disp_y1, disp_x2 - disp_x1, disp_y2 - disp_y1)

            # 添加文本
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(disp_x1 + 5, disp_y1 + 20, f"ROI: {x2 - x1}x{y2 - y1}")

        painter.end()

        # 更新显示
        self.setPixmap(draw_pixmap)

    def set_image(self, cv_img):
        """设置OpenCV图像"""
        if cv_img is None:
            return

        self.original_image = cv_img.copy()

        # 将OpenCV图像转换为Qt格式
        if len(cv_img.shape) == 3:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        else:
            h, w = cv_img.shape
            q_img = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)

        # 创建QPixmap
        pixmap = QPixmap.fromImage(q_img)

        # 缩放以适合标签
        self.current_pixmap = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # 更新显示
        self.update_display()

    def start_roi_selection(self):
        """开始ROI选择模式"""
        self.drawing_roi = True
        self.roi_fixed = False  # 重置固定标记
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.update_display()

    def set_prompt_mode(self, enabled: bool):
        self.prompt_mode = enabled
        if enabled:
            self.drawing_roi = False
            self.roi_fixed = False
        self.update_display()

    def set_segmentation_preview(self, mask, points):
        self.segmentation_mask = mask
        self.prompt_points = list(points)
        self.update_display()

    def clear_roi(self):
        """清除ROI"""
        self.roi_rect = None
        self.roi_fixed = False  # 重置固定标记
        self.update_display()
