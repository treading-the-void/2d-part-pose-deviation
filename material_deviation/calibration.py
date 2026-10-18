"""Read RGB camera calibration and compute planar pixel scale."""
import cv2
import numpy as np
from pathlib import Path

class CalibrationMixin:
    def load_camera_params(self, filepath):
        """加载相机标定参数 - 使用OpenCV的FileStorage"""
        try:
            # 使用OpenCV的FileStorage读取YAML文件
            # OpenCV's Windows FileStorage path handling may fail on Chinese paths.
            # Reading with pathlib and parsing from memory keeps the original YAML format.
            content = Path(filepath).read_text(encoding="utf-8")
            fs = cv2.FileStorage(content, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)

            if not fs.isOpened():
                print(f"无法打开相机参数文件: {filepath}")
                return False

            # 读取相机内参矩阵
            K_node = fs.getNode("K")
            if not K_node.empty():
                self.camera_matrix = K_node.mat()
                print(f"相机内参矩阵 K:\n{self.camera_matrix}")

                # 提取焦距和主点
                self.fx = self.camera_matrix[0, 0]
                self.fy = self.camera_matrix[1, 1]
                self.cx = self.camera_matrix[0, 2]
                self.cy = self.camera_matrix[1, 2]

                print(f"焦距: fx={self.fx:.2f}, fy={self.fy:.2f}")
                print(f"主点: cx={self.cx:.2f}, cy={self.cy:.2f}")

            # 读取畸变系数
            D_node = fs.getNode("D")
            if not D_node.empty():
                self.dist_coeffs = D_node.mat()
                print(f"畸变系数 D:\n{self.dist_coeffs}")

            # 读取图像尺寸（如果有）
            image_width_node = fs.getNode("image_width")
            image_height_node = fs.getNode("image_height")
            if not image_width_node.empty() and not image_height_node.empty():
                self.image_width = int(image_width_node.real())
                self.image_height = int(image_height_node.real())
                print(f"标定图像尺寸: {self.image_width} x {self.image_height}")

            fs.release()

            # 计算像素到物理的转换因子
            if self.camera_matrix is not None and self.working_distance > 0:
                # 在距离Z处，1像素对应的物理尺寸
                self.pixel_to_mm_x = (self.working_distance / self.fx) * 1000  # 转换为毫米
                self.pixel_to_mm_y = (self.working_distance / self.fy) * 1000  # 转换为毫米

                print(f"像素到物理转换因子: {self.pixel_to_mm_x:.4f} mm/像素(X), {self.pixel_to_mm_y:.4f} mm/像素(Y)")
                print(f"工作距离: {self.working_distance} 米")

                # 计算在1.3米处的视野范围
                if hasattr(self, 'image_width') and hasattr(self, 'image_height'):
                    fov_x = 2 * np.arctan(self.image_width / (2 * self.fx))
                    fov_y = 2 * np.arctan(self.image_height / (2 * self.fy))
                    fov_x_deg = np.degrees(fov_x)
                    fov_y_deg = np.degrees(fov_y)

                    # 计算在1.3米处的实际视野尺寸
                    real_width = 2 * self.working_distance * np.tan(fov_x / 2) * 1000  # 毫米
                    real_height = 2 * self.working_distance * np.tan(fov_y / 2) * 1000  # 毫米

                    print(f"视野角度: {fov_x_deg:.1f}° x {fov_y_deg:.1f}°")
                    print(f"视野尺寸: {real_width:.1f}mm x {real_height:.1f}mm")

            return True

        except Exception as e:
            print(f"加载相机参数失败: {e}")
            import traceback
            traceback.print_exc()
            return False
