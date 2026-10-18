"""Template ROI capture, feature extraction, save and load."""
import os, time, json, traceback
import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog

class TemplateWorkflowMixin:
    def start_template_extraction(self):
        """Choose a template ROI and collect the configured number of frames."""
        self.leave_sam_mode()
        self.template_reference_frame = None
        self.template_reference_roi = None
        if not self.feature_extractor.available:
            QMessageBox.warning(self, "错误", "特征提取器不可用，请检查相关依赖")
            return

        self.status_label.setText(f"请在图像上拖动鼠标选择ROI区域，然后采集{self.capture_target}帧")
        self.roi_label.start_roi_selection()
        self.clear_button.setEnabled(True)

        # 设置多帧捕获标志
        self.multi_frame_mode = True
        self.refresh_action_states()
    def on_roi_selected(self, roi_rect):
        """ROI选择完成后的处理"""
        self.current_roi = roi_rect
        x1, y1, x2, y2 = roi_rect
        width = x2 - x1
        height = y2 - y1

        if self.multi_frame_mode:
            # 多帧模式：提取稳定特征点
            self.status_label.setText(f"正在连续捕获{self.capture_target}帧以提取稳定特征点...")
            QApplication.processEvents()

            # 捕获配置的帧数
            captured_frames = []
            for i in range(self.capture_target):
                # 尝试从相机获取帧
                frame = self.cam.get_frame(timeout=0.1)
                if frame is not None:
                    captured_frames.append(frame.copy())
                    self.status_label.setText(f"正在捕获第 {i + 1}/{self.capture_target} 帧...")
                    QApplication.processEvents()
                    time.sleep(0.05)  # 短暂延迟，确保帧之间的变化
                else:
                    self.status_label.setText(f"警告: 第 {i + 1} 帧获取失败")

            if len(captured_frames) > 0:
                self.process_multi_frame_extraction(captured_frames, roi_rect)
            else:
                self.status_label.setText("错误: 未捕获到任何帧")
                self.multi_frame_mode = False

            # 重置多帧模式
            self.multi_frame_mode = False
            self.refresh_action_states()
        else:
            # 单帧模式：原有逻辑
            self.status_label.setText(f"ROI选择完成: ({x1}, {y1}) - ({x2}, {y2}), 尺寸: {width}x{height}")
            self.save_button.setEnabled(True)
            self.preview_features()
    def process_multi_frame_extraction(self, frames, roi_rect):
        """处理多帧特征点提取"""
        if not frames or len(frames) == 0:
            QMessageBox.warning(self, "错误", "未捕获到任何帧")
            return

        print(f"开始处理 {len(frames)} 帧的稳定特征点提取")

        # 显示捕获进度
        self.status_label.setText(f"正在处理 {len(frames)} 帧图像...")
        QApplication.processEvents()

        # 方法1: 使用多帧平均法提取特征点
        # 首先提取每一帧的特征点，然后合并
        all_features = []

        for i, frame in enumerate(frames):
            print(f"处理第 {i + 1} 帧...")

            # 提取单帧特征点
            features = self.feature_extractor.extract_features(frame, roi_rect)
            if features and len(features) > 0:
                all_features.append(features)

        if len(all_features) == 0:
            QMessageBox.warning(self, "错误", "未提取到任何特征点")
            return

        # 合并特征点（简单方法：使用第一帧的特征点，但筛选那些在多帧中都出现的特征点）
        merged_features = self.merge_features_across_frames(all_features, roi_rect, frames[0])

        # 显示特征点数量
        self.feature_label.setText(f"稳定特征点: {len(merged_features)} 个")

        # 保存特征点到变量，供保存使用
        self.stable_features = merged_features
        self.template_reference_frame = frames[0].copy()
        self.template_reference_roi = roi_rect

        # 在最后一帧上绘制特征点
        display_frame = frames[-1].copy()

        # 绘制ROI矩形
        x1, y1, x2, y2 = roi_rect
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 绘制特征点
        for feature in merged_features[:100]:  # 只绘制前100个点
            x, y = int(feature['x']), int(feature['y'])
            radius = int(5 * feature.get('response', 0.5) / 100) + 2
            cv2.circle(display_frame, (x, y), radius, (0, 0, 255), 1)
            cv2.circle(display_frame, (x, y), 2, (0, 255, 0), -1)

        self.roi_label.set_image(display_frame)

        # 启用保存按钮
        self.save_button.setEnabled(True)

        self.status_label.setText(f"稳定提取完成: 从 {len(frames)} 帧中获得 {len(merged_features)} 个特征点")
        self.refresh_action_states()
    def merge_features_across_frames(self, all_features_list, roi_rect, reference_frame):
        """合并多帧中的特征点"""
        if not all_features_list or len(all_features_list) == 0:
            return []

        print(f"合并 {len(all_features_list)} 帧的特征点...")

        # 方法1: 取所有帧中特征点的并集
        # 简单的实现：使用第一帧的特征点作为基础
        if len(all_features_list) == 1:
            return all_features_list[0]

        # 使用第一帧的特征点
        base_features = all_features_list[0]

        # 如果需要更复杂的合并，可以使用以下方法：
        # 1. 特征点聚类（根据位置和描述子相似度）
        # 2. 选择在多帧中重复出现的特征点
        # 3. 对描述子进行平均

        # 这里使用一个简化方法：只保留第一帧的特征点
        # 实际应用中，您可能需要实现更复杂的合并逻辑

        # 标记重复出现的特征点（基于位置近似）
        stable_features = []
        position_tolerance = 5.0  # 像素容差

        for i, feature in enumerate(base_features):
            x = feature['x']
            y = feature['y']

            # 检查这个特征点在其他帧中是否也出现
            appears_in_other_frames = 0

            for other_features in all_features_list[1:]:
                for other_feature in other_features:
                    other_x = other_feature['x']
                    other_y = other_feature['y']

                    # 计算距离
                    distance = np.sqrt((x - other_x) ** 2 + (y - other_y) ** 2)

                    if distance < position_tolerance:
                        appears_in_other_frames += 1
                        break  # 找到近似匹配，跳出内层循环

            # 如果这个特征点在大多数帧中都出现，认为是稳定的
            if appears_in_other_frames >= len(all_features_list) // 2:
                stable_features.append(feature)

        print(f"从 {len(base_features)} 个特征点中筛选出 {len(stable_features)} 个稳定特征点")

        return stable_features
    def popup_feature_view(self, cv_img, roi_rect, features):
        """弹窗仅显示 ROI 区域（含特征点）"""
        if cv_img is None or not features:
            return

        x1, y1, x2, y2 = roi_rect
        roi_img = cv_img[y1:y2, x1:x2]  # 1. 先裁出 ROI
        h, w = roi_img.shape[:2]

        vis = roi_img.copy()

        # 3. 把特征点坐标转成 ROI 相对坐标再画
        for f in features:
            fx, fy = int(f['x']), int(f['y'])
            # 减 ROI 左上角即为相对坐标
            rx = fx - x1
            ry = fy - y1
            # 半径基于响应值
            r = int(5 * f.get('response', 0.5) / 100) + 2
            cv2.circle(vis, (rx, ry), r, (0, 0, 255), 1)
            cv2.circle(vis, (rx, ry), 2, (0, 255, 0), -1)

        # 4. 仅显示 ROI 弹窗
        cv2.imshow('SuperPoint 特征点预览（ROI）', vis)
        cv2.waitKey(0)
        cv2.destroyWindow('SuperPoint 特征点预览（ROI）')
    def preview_features(self):
        """预览特征点提取结果"""
        if self.current_frame is None or self.current_roi is None:
            return

        # 提取特征点
        features = self.feature_extractor.extract_features(self.current_frame, self.current_roi)

        if features is None:
            self.feature_label.setText("特征点: 提取失败")
            return

        if len(features) == 0:
            self.feature_label.setText("特征点: 0 (未检测到特征点)")
            return

        # 显示特征点数量
        feature_type = "SuperPoint" if hasattr(self.feature_extractor, 'superpoint') else "ORB(替代)"
        self.feature_label.setText(f"特征点: {len(features)} ({feature_type}特征)")

        # 在图像上绘制特征点
        frame_copy = self.current_frame.copy()

        # 绘制ROI矩形
        x1, y1, x2, y2 = self.current_roi
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 绘制特征点（大小表示响应值强度）
        for feature in features:
            x, y = int(feature['x']), int(feature['y'])
            # 半径基于响应值
            radius = int(5 * feature.get('response', 0.5) / 100) + 2
            cv2.circle(frame_copy, (x, y), radius, (0, 0, 255), 1)
            cv2.circle(frame_copy, (x, y), 2, (0, 255, 0), -1)

            # 绘制特征点方向
            if 'angle' in feature and feature['angle'] > 0:
                angle = feature['angle'] * np.pi / 180.0
                length = 10
                x2_dir = int(x + length * np.cos(angle))
                y2_dir = int(y + length * np.sin(angle))
                cv2.line(frame_copy, (x, y), (x2_dir, y2_dir), (255, 0, 0), 1)

        # 更新显示
        self.roi_label.set_image(frame_copy)
        # 把带特征点的整图弹窗显示
        self.popup_feature_view(self.current_frame, self.current_roi, features)
    def real_save_template(self):
        """保存模板"""
        reference_frame = None
        reference_roi = None
        reference_mask = None
        if getattr(self, 'sam_template_features', None):
            features = self.sam_template_features
            source = "SAM分割"
            reference_frame = self.sam_frame
            reference_mask = self.sam_mask
            default_name = os.path.join(
                str(self.app_settings.templates_dir),
                f"template_sam_{time.strftime('%Y%m%d_%H%M%S')}.json",
            )
        # 优先使用多帧提取的特征点
        elif hasattr(self, 'stable_features') and self.stable_features:
            features = self.stable_features
            source = "多帧稳定提取"
            reference_frame = self.template_reference_frame
            reference_roi = self.template_reference_roi
            default_name = os.path.join(str(self.app_settings.templates_dir), f"template_stable_{time.strftime('%Y%m%d_%H%M%S')}.json")
        elif self.current_roi is not None and self.current_frame is not None:
            features = self.feature_extractor.extract_features(self.current_frame, self.current_roi)
            source = "单帧提取"
            reference_frame = self.current_frame
            reference_roi = self.current_roi
            default_name = os.path.join(str(self.app_settings.templates_dir), "template.json")
        else:
            QMessageBox.warning(self, "提示", "没有可保存的 ROI")
            return

        if not features:
            QMessageBox.warning(self, "提示", "未提取到任何特征点")
            return

        if reference_mask is None and reference_frame is not None and reference_roi is not None:
            reference_mask = np.zeros(reference_frame.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = reference_roi
            reference_mask[max(0, y1):min(y2, reference_mask.shape[0]),
                           max(0, x1):min(x2, reference_mask.shape[1])] = 255

        # 选择保存路径
        path, _ = QFileDialog.getSaveFileName(
            self, "保存模板", default_name, "JSON (*.json)"
        )

        if not path:
            return

        # 保存模板
        ok = self.feature_extractor.save_template(
            features, path, reference_frame=reference_frame, reference_mask=reference_mask
        )
        if ok:
            self.status_label.setText(f"已保存 {len(features)} 个{source}特征点到 {os.path.basename(path)}")
            self.save_button.setEnabled(False)
            self.sam_save_button.setEnabled(False)

            # 清除稳定提取的特征点
            if hasattr(self, 'stable_features'):
                delattr(self, 'stable_features')
            if getattr(self, 'sam_template_features', None):
                self.sam_template_features = None
                self.sam_template_saved = True
            if not self.sam_mode:
                self.current_roi = None
                self.roi_label.clear_roi()
            self.refresh_action_states()
    def load_template(self):
        """加载模板文件 - 使用异步处理避免UI冻结"""
        try:
            # 显示等待状态
            self.status_label.setText("正在加载模板...")
            QApplication.processEvents()  # 让UI更新

            # 选择模板文件
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择模板文件", str(self.app_settings.templates_dir), "JSON文件 (*.json)"
            )

            if not file_path:
                self.status_label.setText("就绪 - 未选择文件")
                return

            print(f"开始加载模板文件: {file_path}")

            # 使用定时器延迟执行加载，避免阻塞UI
            QTimer.singleShot(100, lambda: self._async_load_template(file_path))

        except Exception as e:
            self.status_label.setText("加载模板时发生错误")
            print(f"加载模板初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
    def _async_load_template(self, file_path):
        """异步加载模板"""
        try:
            print(f"异步加载模板: {file_path}")

            # 检查文件是否存在
            if not os.path.exists(file_path):
                self.status_label.setText(f"模板文件不存在: {os.path.basename(file_path)}")
                return

            # 检查文件大小
            file_size = os.path.getsize(file_path)
            print(f"模板文件大小: {file_size / 1024:.2f} KB")

            if file_size > 10 * 1024 * 1024:  # 10MB
                reply = QMessageBox.question(
                    self, "确认",
                    f"模板文件较大 ({file_size / 1024 / 1024:.2f} MB)，加载可能需要较长时间，是否继续？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    self.status_label.setText("已取消加载大文件")
                    return

            # 显示加载进度
            self.status_label.setText(f"正在加载模板: {os.path.basename(file_path)}...")
            QApplication.processEvents()

            # 加载模板
            template_data = self.feature_extractor.load_template(file_path)

            if template_data:
                self.loaded_template = template_data
                self.template_filename = os.path.basename(file_path)
                self.comparison_enabled = False
                self.comparison_button.setText("偏差对比")

                # 更新UI信息
                num_features = template_data['metadata']['num_features']
                feature_type = template_data['metadata']['feature_type']

                print(f"模板加载成功: {num_features}个{feature_type}特征点")

                # 在主线程中显示消息
                QTimer.singleShot(0, lambda: self._show_template_loaded_message(
                    num_features, feature_type, file_path
                ))

                # 如果之前没有匹配ROI，使用整个图像
                if self.match_roi is None:
                    self.match_roi = (0, 0, self.cfg.width, self.cfg.height)
                self.refresh_action_states()

                print("模板加载完成")
            else:
                QTimer.singleShot(0, lambda: (
                    QMessageBox.warning(self, "警告", "加载模板失败或模板文件为空"),
                    self.status_label.setText("模板加载失败")
                ))
                print("模板加载失败")

        except MemoryError:
            QTimer.singleShot(0, lambda: (
                QMessageBox.critical(self, "内存不足", "加载模板时内存不足，请尝试加载较小的模板文件"),
                self.status_label.setText("内存不足，加载失败")
            ))
            print("内存不足错误")

        except json.JSONDecodeError as e:
            QTimer.singleShot(0, lambda: (
                QMessageBox.critical(self, "文件格式错误", f"模板文件格式错误:\n{str(e)}"),
                self.status_label.setText("文件格式错误")
            ))
            print(f"JSON解析错误: {e}")

        except Exception as e:
            QTimer.singleShot(0, lambda: (
                QMessageBox.critical(self, "错误", f"加载模板失败:\n{str(e)}"),
                self.status_label.setText("加载失败")
            ))
            print(f"加载模板失败: {str(e)}")
            import traceback
            traceback.print_exc()
    def _show_template_loaded_message(self, num_features, feature_type, file_path):
        """在主线程中显示模板加载成功的消息"""
        QMessageBox.information(self, "成功",
                                f"成功加载模板\n"
                                f"文件: {os.path.basename(file_path)}\n"
                                f"特征点数量: {num_features}\n"
                                f"特征类型: {feature_type}")
        self.status_label.setText(f"模板加载成功: {num_features}个{feature_type}特征点")
        self.feature_label.setText(f"加载的模板: {num_features}个特征点")
