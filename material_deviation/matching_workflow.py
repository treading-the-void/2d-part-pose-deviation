"""Multi-frame material deviation matching and result display."""
import os, time, json, traceback
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox

class MatchingWorkflowMixin:
    def toggle_matching(self):
        """切换匹配模式"""
        if self.loaded_template is None:
            QMessageBox.warning(self, "警告", "请先加载模板文件")
            return

        if not self.matching_enabled:
            self.leave_sam_mode()
            # 开始多帧匹配模式
            self.start_multi_frame_matching()
        else:
            # 停止匹配，恢复实时画面
            self.stop_matching()
    def start_multi_frame_matching(self):
        """开始多帧匹配模式"""
        self.matching_enabled = True
        self.multi_frame_matching = True
        self.matching_processed = False
        self.matching_frames_collected = []
        self.matching_results = []
        self.collected_frame_count = 0

        self.match_button.setText("处理中...")
        self.match_button.setEnabled(False)  # 处理期间禁用按钮
        self.status_label.setText(f"正在收集{self.matching_target_frames}帧进行匹配分析...")
        self.match_label.setText(f"匹配: 收集帧中... 0/{self.matching_target_frames}")

        # 清除之前的匹配结果
        self.final_matching_result = None
        self.refresh_action_states()

        print(f"开始多帧匹配模式，将收集{self.matching_target_frames}帧进行分析")
    def stop_matching(self):
        """停止匹配，恢复实时画面"""
        self.matching_enabled = False
        self.multi_frame_matching = False
        self.matching_processed = False

        # 重置缓存标记
        if hasattr(self, '_final_result_displayed'):
            delattr(self, '_final_result_displayed')
        if hasattr(self, 'best_match_for_display'):
            delattr(self, 'best_match_for_display')
        if hasattr(self, 'best_match_frame'):
            delattr(self, 'best_match_frame')

        self.match_button.setText("开始匹配")
        self.match_button.setEnabled(True)
        self.status_label.setText("匹配已停止")
        self.match_label.setText("匹配: 未开始")

        # 清除收集的数据
        self.matching_frames_collected = []
        self.matching_results = []
        self.final_matching_result = None
        self.refresh_action_states()

        print("停止匹配，恢复实时画面")
    def collect_frame_for_matching(self, frame):
        """收集帧用于匹配"""
        if self.collected_frame_count < self.matching_target_frames:
            # 保存帧
            self.matching_frames_collected.append(frame.copy())
            self.collected_frame_count += 1

            # 更新状态显示
            self.match_label.setText(f"匹配: 收集帧中... {self.collected_frame_count}/{self.matching_target_frames}")

            # 在图像上显示进度
            display_frame = self._comparison_frame(frame)
            cv2.putText(display_frame, f"Collecting frames: {self.collected_frame_count}/{self.matching_target_frames}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 如果设置了匹配ROI，绘制它
            if self.match_roi:
                x1, y1, x2, y2 = self.match_roi
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            self.roi_label.set_image(display_frame)

            # 收集完成，开始处理
            if self.collected_frame_count >= self.matching_target_frames:
                print("帧收集完成，开始处理...")
                self.process_multi_frame_matching()
        else:
            # 不应该到达这里
            pass
    def process_multi_frame_matching(self):
        """处理收集的多帧进行匹配"""
        if not self.matching_frames_collected or len(self.matching_frames_collected) == 0:
            self.status_label.setText("错误: 未收集到帧")
            self.stop_matching()
            return

        print(f"开始处理 {len(self.matching_frames_collected)} 帧的匹配...")
        self.status_label.setText(f"正在处理 {len(self.matching_frames_collected)} 帧...")
        QApplication.processEvents()

        # 处理每一帧
        self.matching_results = []
        self.best_match_for_display = None  # 新增：用于显示的最佳匹配结果
        max_matches = 0  # 新增：记录最大匹配点数

        for i, frame in enumerate(self.matching_frames_collected):
            print(f"处理第 {i + 1}/{len(self.matching_frames_collected)} 帧...")
            self.status_label.setText(f"匹配处理中: {i + 1}/{len(self.matching_frames_collected)}")
            QApplication.processEvents()

            # 进行特征匹配
            match_result = self.feature_extractor.match_features(
                frame, self.loaded_template, self.match_roi
            )

            if match_result and match_result['success']:
                # 收集成功的匹配结果
                result_data = {
                    'rotation': match_result['rotation'],
                    'offset': match_result['offset'],
                    'num_matches': match_result['num_matches'],
                    'frame_index': i,
                    'match_result': match_result  # 保存完整的匹配结果用于显示
                }
                self.matching_results.append(result_data)
                print(f"  第 {i + 1} 帧匹配成功: {match_result['num_matches']} 个匹配点")

                # 保存匹配点最多的结果用于显示
                if match_result['num_matches'] > max_matches:
                    max_matches = match_result['num_matches']
                    self.best_match_for_display = match_result
                    self.best_match_frame = frame.copy()  # 保存对应的帧
            else:
                print(f"  第 {i + 1} 帧匹配失败")

        # 计算综合结果
        self.compute_final_matching_result()

        # 更新UI状态
        self.multi_frame_matching = False
        self.matching_processed = True

        self.match_button.setText("停止匹配")
        self.match_button.setEnabled(True)
        self.refresh_action_states()

        print("多帧匹配处理完成")
    def compute_final_matching_result(self):
        """计算最终的匹配结果"""
        if not self.matching_results or len(self.matching_results) == 0:
            self.status_label.setText("错误: 没有成功的匹配结果")
            self.final_matching_result = None
            return

        # 提取旋转角度和偏移
        rotations = [r['rotation'] for r in self.matching_results]
        offsets_x = [r['offset'][0] for r in self.matching_results]
        offsets_y = [r['offset'][1] for r in self.matching_results]
        num_matches_list = [r['num_matches'] for r in self.matching_results]

        # 计算中值（避免异常值影响）
        final_rotation = np.median(rotations)
        final_offset_x = np.median(offsets_x)
        final_offset_y = np.median(offsets_y)

        # 计算平均值（用于显示统计信息）
        avg_rotation = np.mean(rotations)
        avg_offset_x = np.mean(offsets_x)
        avg_offset_y = np.mean(offsets_y)

        # 计算标准差
        std_rotation = np.std(rotations)
        std_offset_x = np.std(offsets_x)
        std_offset_y = np.std(offsets_y)

        # 找到匹配点最多的结果
        best_match_idx = np.argmax(num_matches_list)
        best_match = self.matching_results[best_match_idx]

        # 使用最后一帧进行显示
        display_frame = self.matching_frames_collected[-1].copy()

        # 计算物理尺寸偏移
        physical_offset_x_mm = None
        physical_offset_y_mm = None

        if hasattr(self, 'pixel_to_mm_x') and hasattr(self, 'pixel_to_mm_y'):
            # 将像素偏移转换为毫米
            physical_offset_x_mm = final_offset_x * self.pixel_to_mm_x
            physical_offset_y_mm = final_offset_y * self.pixel_to_mm_y

            # 计算物理偏移的统计量 - 这里直接使用offsets_x和offsets_y，它们已经是浮点数列表
            # 修正：不需要再使用offset[0]，因为offsets_x已经是浮点数列表
            phys_offsets_x_mm = [ox * self.pixel_to_mm_x for ox in offsets_x]  # 直接乘以转换因子
            phys_offsets_y_mm = [oy * self.pixel_to_mm_y for oy in offsets_y]  # 直接乘以转换因子

            phys_std_x = np.std(phys_offsets_x_mm)
            phys_std_y = np.std(phys_offsets_y_mm)
            phys_conf_x = 1.96 * phys_std_x / np.sqrt(len(phys_offsets_x_mm)) if len(phys_offsets_x_mm) > 0 else 0
            phys_conf_y = 1.96 * phys_std_y / np.sqrt(len(phys_offsets_y_mm)) if len(phys_offsets_y_mm) > 0 else 0
        else:
            phys_std_x = phys_std_y = phys_conf_x = phys_conf_y = 0

        # 创建最终结果
        self.final_matching_result = {
            'rotation': float(final_rotation),
            'offset': (float(final_offset_x), float(final_offset_y)),
            'avg_rotation': float(avg_rotation),
            'avg_offset': (float(avg_offset_x), float(avg_offset_y)),
            'std_rotation': float(std_rotation),
            'std_offset': (float(std_offset_x), float(std_offset_y)),
            'physical_offset_mm': (float(physical_offset_x_mm) if physical_offset_x_mm is not None else None,
                                   float(physical_offset_y_mm) if physical_offset_y_mm is not None else None),
            'physical_std_mm': (float(phys_std_x), float(phys_std_y)),
            'num_success_frames': len(self.matching_results),
            'total_frames': len(self.matching_frames_collected),
            'display_frame': display_frame
        }

        self.save_matching_result()

        # 更新状态显示
        success_rate = (len(self.matching_results) / len(self.matching_frames_collected)) * 100
        self.status_label.setText(
            f"匹配完成: {len(self.matching_results)}/{len(self.matching_frames_collected)} 帧成功 ({success_rate:.1f}%)")

        print(f"最终结果: 旋转={final_rotation:.2f}°, 偏移=({final_offset_x:.1f}, {final_offset_y:.1f}), 物理偏移=({physical_offset_x_mm}, {physical_offset_y_mm})")
    def display_final_result(self):
        """显示最终匹配结果（静止画面）"""
        if not self.final_matching_result:
            return

        # 使用保存的显示帧
        if hasattr(self, 'best_match_frame'):
            frame = self.best_match_frame.copy()
        else:
            frame = self.final_matching_result['display_frame'].copy()
        frame = self._comparison_frame(frame)

        # 获取统计结果
        rotation = self.final_matching_result['rotation']
        offset = self.final_matching_result['offset']
        num_success = self.final_matching_result['num_success_frames']
        total_frames = self.final_matching_result['total_frames']
        physical_offset = self.final_matching_result['physical_offset_mm']
        use_mm = (self.display_unit == "mm" and physical_offset[0] is not None
                  and physical_offset[1] is not None)
        shown_offset = physical_offset if use_mm else offset
        unit = "mm" if use_mm else "px"

        # 绘制匹配ROI区域（如果设置了）
        if self.match_roi:
            x1, y1, x2, y2 = self.match_roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # ===== 使用保存的最佳匹配结果进行绘制，而不是重新匹配 =====
        if hasattr(self, 'best_match_for_display') and self.best_match_for_display:
            match_result = self.best_match_for_display
            # 绘制匹配点对连线（黄色线条）
            matches = match_result['matches']
            template_kp = match_result['template_keypoints']
            scene_kp = match_result['scene_keypoints']

            # 限制绘制的匹配线数量，避免图像过密
            max_display = min(25, len(matches))
            if len(matches) > max_display:
                import random
                random.seed(42)  # 固定随机种子，确保每次绘制相同的匹配点
                display_matches = random.sample(matches, max_display)
            else:
                display_matches = matches

            for match in display_matches:
                template_pt = tuple(map(int, template_kp[match.queryIdx].pt))
                scene_pt = tuple(map(int, scene_kp[match.trainIdx].pt))

                # 绘制黄色连线（类似txt文件）
                cv2.line(frame, template_pt, scene_pt, (0, 255, 255), 1)
                # 绘制红色场景点
                cv2.circle(frame, scene_pt, 4, (0, 0, 255), 1)
                cv2.circle(frame, scene_pt, 2, (0, 0, 255), -1)

        # 在图像上显示统计信息
        info_y = 30
        cv2.putText(frame, "Multi-Frame Matching Result (Static)",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Rotation: {rotation:.2f} deg",
                    (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Offset: X={shown_offset[0]:.1f}, Y={shown_offset[1]:.1f} {unit}",
                    (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Success: {num_success}/{total_frames} frames",
                    (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 绘制中心坐标系
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        axis_length = 50

        # X轴 (红色)
        cv2.arrowedLine(frame, (center_x, center_y),
                        (center_x + axis_length, center_y), (0, 0, 255), 2)
        # Y轴 (绿色)
        cv2.arrowedLine(frame, (center_x, center_y),
                        (center_x, center_y + axis_length), (0, 255, 0), 2)

        # 绘制偏移向量（黄色箭头）
        if abs(offset[0]) > 5 or abs(offset[1]) > 5:
            scaled_dx = int(offset[0] / 5)
            scaled_dy = int(offset[1] / 5)
            cv2.arrowedLine(frame, (center_x, center_y),
                            (center_x + scaled_dx, center_y + scaled_dy),
                            (255, 255, 0), 2, tipLength=0.3)

        # 更新显示
        self.roi_label.set_image(frame)

        # 更新标签
        self.match_label.setText(
            f"匹配完成: {num_success}/{total_frames}帧 | 旋转: {rotation:.2f}° | "
            f"{'实际距离' if use_mm else '像素偏移'}: "
            f"({shown_offset[0]:.1f}, {shown_offset[1]:.1f}) {unit}"
        )

        # 标记最终结果已显示，防止重复绘制
        self._final_result_displayed = True

        self._final_result_displayed = True
    def set_matching_roi(self):
        """设置匹配时的ROI区域"""
        if self.current_frame is None:
            return

        self.status_label.setText("请在图像上拖动鼠标选择匹配ROI区域")
        self.selecting_match_roi = True
        self.refresh_action_states()
        self.roi_label.start_roi_selection()
        # 临时存储之前的ROI状态
        self.temp_roi_callback = self.on_roi_selected
        # 修改回调函数
        self.roi_label.roi_selected.disconnect()
        self.roi_label.roi_selected.connect(self.on_matching_roi_selected)
    def on_matching_roi_selected(self, roi_rect):
        """匹配ROI选择完成后的处理"""
        self.match_roi = roi_rect
        self.selecting_match_roi = False
        x1, y1, x2, y2 = roi_rect

        # 显示ROI信息
        width = x2 - x1
        height = y2 - y1
        self.status_label.setText(f"匹配ROI设置完成: ({x1}, {y1}) - ({x2}, {y2}), 尺寸: {width}x{height}")

        # 恢复原来的回调函数
        self.roi_label.roi_selected.disconnect()
        self.roi_label.roi_selected.connect(self.on_roi_selected)
        self.refresh_action_states()

        # 如果匹配模式已启动，立即开始匹配
        if self.matching_enabled:
            self.status_label.setText("匹配ROI已设置，正在实时匹配...")
    def perform_matching(self, frame):
        """执行模板匹配"""
        if self.loaded_template is None:
            return

        try:
            # 进行特征匹配
            match_result = self.feature_extractor.match_features(
                frame, self.loaded_template, self.match_roi
            )

            if match_result and match_result['success']:
                # 获取匹配结果
                rotation = match_result['rotation']
                offset = match_result['offset']
                num_matches = match_result['num_matches']

                # 更新匹配信息标签
                self.match_label.setText(
                    f"匹配: {num_matches}个匹配点 | 旋转: {rotation:.2f}° | 偏移: ({offset[0]:.1f}, {offset[1]:.1f})"
                )

                # 在图像上绘制匹配结果
                display_frame = self.draw_matching_result(frame, match_result)

                # 更新显示
                self.roi_label.set_image(display_frame)
            else:
                # 匹配失败
                self.match_label.setText("匹配: 失败 - 未找到足够匹配点")

                # 绘制匹配ROI（如果设置了）
                display_frame = frame.copy()
                if self.match_roi:
                    x1, y1, x2, y2 = self.match_roi
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(display_frame, "Match ROI", (x1 + 5, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                self.roi_label.set_image(display_frame)

        except Exception as e:
            print(f"匹配过程中出错: {e}")
            self.match_label.setText("匹配: 错误")
    def draw_matching_result(self, frame, match_result):
        """在图像上绘制匹配结果"""
        display_frame = frame.copy()

        # 绘制匹配ROI区域（如果设置了）
        if self.match_roi:
            x1, y1, x2, y2 = self.match_roi
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # 获取匹配信息
        matches = match_result['matches']
        template_kp = match_result['template_keypoints']
        scene_kp = match_result['scene_keypoints']
        rotation = match_result['rotation']
        offset = match_result['offset']
        num_matches = match_result['num_matches']

        # 绘制匹配点对
        if len(matches) > 0:
            # 随机选择最多20个匹配点对进行绘制，避免图像过于拥挤
            max_display = min(20, len(matches))
            if len(matches) > max_display:
                import random
                display_matches = random.sample(matches, max_display)
            else:
                display_matches = matches

            for match in display_matches:
                template_pt = tuple(map(int, template_kp[match.queryIdx].pt))
                scene_pt = tuple(map(int, scene_kp[match.trainIdx].pt))

                # 绘制线连接匹配点
                cv2.line(display_frame, template_pt, scene_pt, (0, 255, 255), 1)
                # 绘制场景点（红色）
                cv2.circle(display_frame, scene_pt, 5, (0, 0, 255), 1)
                cv2.circle(display_frame, scene_pt, 2, (0, 0, 255), -1)

        # 在图像上显示匹配信息
        info_y = 30
        cv2.putText(display_frame, f"Matches: {num_matches}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Rotation: {rotation:.2f} deg", (10, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Offset: X={offset[0]:.1f}, Y={offset[1]:.1f}", (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 在图像中心绘制坐标系
        center_x = display_frame.shape[1] // 2
        center_y = display_frame.shape[0] // 2

        # 绘制坐标系
        axis_length = 50
        # X轴 (红色)
        cv2.arrowedLine(display_frame, (center_x, center_y),
                        (center_x + axis_length, center_y), (0, 0, 255), 2)
        # Y轴 (绿色)
        cv2.arrowedLine(display_frame, (center_x, center_y),
                        (center_x, center_y + axis_length), (0, 255, 0), 2)

        # 绘制偏移向量
        if abs(offset[0]) > 10 or abs(offset[1]) > 10:  # 只显示明显的偏移
            cv2.arrowedLine(display_frame, (center_x, center_y),
                            (center_x + int(offset[0] / 10), center_y + int(offset[1] / 10)),
                            (255, 255, 0), 2, tipLength=0.3)

        return display_frame
    def save_matching_result(self):
        """保存匹配结果为JSON文件"""
        if not self.final_matching_result:
            QMessageBox.warning(self, "警告", "没有匹配结果可以保存")
            return

        try:
            # 创建保存目录
            results_dir = str(self.app_settings.results_dir)
            os.makedirs(results_dir, exist_ok=True)

            # 生成文件名：material_时间戳
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"material_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)

            # 准备保存的数据
            save_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "filename": filename,
                "template_file": self.template_filename if hasattr(self, 'template_filename') else "",

                # 匹配结果
                "rotation_deg": float(self.final_matching_result['rotation']),
                "pixel_offset": [
                    float(self.final_matching_result['offset'][0]),
                    float(self.final_matching_result['offset'][1])
                ],

                # 物理偏移（如果有）
                "physical_offset_mm": None,
                "physical_offset_std_mm": None,
                "physical_offset_conf_mm": None,

                # 统计信息
                "num_success_frames": int(self.final_matching_result['num_success_frames']),
                "total_frames": int(self.final_matching_result['total_frames']),

                # 标准差和置信区间
                "rotation_std_deg": float(self.final_matching_result['std_rotation']),
                "pixel_offset_std": [
                    float(self.final_matching_result['std_offset'][0]),
                    float(self.final_matching_result['std_offset'][1])
                ],

                # 相机信息（如果有）
                "camera_params_loaded": hasattr(self, 'pixel_to_mm_x') and hasattr(self, 'pixel_to_mm_y'),
                "working_distance_m": getattr(self, 'working_distance', 0),
                "pixel_to_mm_factor": None
            }

            # 添加物理偏移信息（如果可用）
            if self.final_matching_result['physical_offset_mm'][0] is not None:
                save_data["physical_offset_mm"] = [
                    float(self.final_matching_result['physical_offset_mm'][0]),
                    float(self.final_matching_result['physical_offset_mm'][1])
                ]
                save_data["physical_offset_std_mm"] = [
                    float(self.final_matching_result['physical_std_mm'][0]),
                    float(self.final_matching_result['physical_std_mm'][1])
                ]

            # 添加像素到物理转换因子（如果可用）
            if hasattr(self, 'pixel_to_mm_x') and hasattr(self, 'pixel_to_mm_y'):
                save_data["pixel_to_mm_factor"] = [
                    float(self.pixel_to_mm_x),
                    float(self.pixel_to_mm_y)
                ]

            # 保存为JSON文件
            temp_filepath = filepath + ".tmp"
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            # 替换原文件
            if os.path.exists(temp_filepath):
                os.replace(temp_filepath, filepath)

                # 更新状态显示
                self.status_label.setText(f"匹配结果已保存: {filename}")

                # 显示成功消息
                QMessageBox.information(self, "保存成功",
                                        f"匹配结果已保存为:\n{filename}\n\n"
                                        f"旋转: {save_data['rotation_deg']:.2f}°\n"
                                        f"像素偏移: ({save_data['pixel_offset'][0]:.1f}, {save_data['pixel_offset'][1]:.1f})\n")

                print(f"匹配结果已保存到: {filepath}")
                return True
            else:
                QMessageBox.warning(self, "保存失败", "临时文件未创建")
                return False

        except Exception as e:
            error_msg = f"保存匹配结果失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "保存失败", error_msg)
            return False
