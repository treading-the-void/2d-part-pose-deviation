"""Freeze a frame, collect SAM prompts, and build a masked SuperPoint template."""

from PyQt5.QtWidgets import QApplication, QMessageBox

from .sam_segmentation import SamSegmenter


class SamTemplateWorkflowMixin:
    def start_sam_template(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "没有画面", "请先等待相机画面")
            return
        if self.matching_enabled:
            self.stop_matching()
        self.sam_freezing = True
        self.refresh_action_states()
        self.sam_frame = self.current_frame.copy()
        self.sam_points = []
        self.sam_mask = None
        self.sam_template_features = None
        self.sam_template_saved = False
        self.current_roi = None
        self.clear_button.setEnabled(False)
        if hasattr(self, 'stable_features'):
            delattr(self, 'stable_features')
        self.save_button.setEnabled(False)
        self.sam_save_button.setEnabled(False)
        self.undo_sam_button.setEnabled(False)
        self.clear_sam_button.setEnabled(False)
        self.roi_label.clear_roi()
        self.roi_label.set_image(self.sam_frame)
        self.roi_label.set_prompt_mode(True)
        self._show_sam_preview()
        self.status_label.setText("SAM 模板：左键标记板件，右键标记背景；可撤销或清空")
        QApplication.processEvents()
        try:
            if self.sam_segmenter is None:
                self.sam_segmenter = SamSegmenter(
                    self.app_settings.sam_weights_file,
                    self.app_settings.sam_model_type,
                    self.app_settings.sam_device,
                )
            self.sam_segmenter.set_image(self.sam_frame)
            self.sam_mode = True
            self.sam_freezing = False
            self.refresh_action_states()
            self.status_label.setText("画面已定格：左键前景，右键背景；每次点击更新 SAM 分割")
        except Exception as exc:
            self.sam_mode = False
            self.sam_freezing = False
            self.roi_label.set_prompt_mode(False)
            self.sam_save_button.setEnabled(False)
            self.refresh_action_states()
            self.status_label.setText(f"SAM 启动失败：{exc}")
            QMessageBox.warning(self, "SAM 不可用", str(exc))

    def on_sam_point(self, x: int, y: int, label: int):
        if not self.sam_mode:
            return
        self.sam_points.append((x, y, label))
        self._update_sam_mask()

    def undo_sam_point(self):
        if self.sam_mode and self.sam_points:
            self.sam_points.pop()
            self._update_sam_mask()

    def clear_sam_points(self):
        if self.sam_mode:
            self.sam_points.clear()
            self._update_sam_mask()

    def _update_sam_mask(self):
        self.sam_mask = None
        self.sam_template_features = None
        self.sam_template_saved = False
        self.sam_save_button.setEnabled(False)
        self.undo_sam_button.setEnabled(bool(self.sam_points))
        self.clear_sam_button.setEnabled(bool(self.sam_points))
        self.refresh_action_states()
        self._show_sam_preview()
        if not any(label == 1 for _, _, label in self.sam_points):
            self.status_label.setText("请在目标板件上添加至少一个前景点")
            return
        self.status_label.setText("SAM 正在分割...")
        QApplication.processEvents()
        try:
            mask, score = self.sam_segmenter.predict(self.sam_points)
            self.sam_mask = mask
            self.sam_save_button.setEnabled(True)
            self.refresh_action_states()
            self._show_sam_preview()
            self.status_label.setText(f"SAM 分割完成，分数 {score:.3f}；检查掩码后点击“SAM模板保存”")
        except Exception as exc:
            self.status_label.setText(f"SAM 分割失败：{exc}")

    def _show_sam_preview(self):
        self.roi_label.set_segmentation_preview(self.sam_mask, self.sam_points)

    def extract_sam_template(self):
        if not self.sam_mode or self.sam_frame is None or self.sam_mask is None:
            QMessageBox.warning(self, "没有掩码", "先点击“SAM 点选模板”并标记板件")
            return
        self.status_label.setText("正在提取掩码内的 SuperPoint 特征...")
        QApplication.processEvents()
        # Extract on the whole frozen frame, then filter points by the mask.
        # Cropping/blackening the image changes SuperPoint descriptors.
        features = self.feature_extractor.extract_features(self.sam_frame)
        if not features:
            self.status_label.setText("没有检测到 SuperPoint 特征")
            return
        height, width = self.sam_mask.shape
        selected = [feature for feature in features
                    if 0 <= int(feature["x"]) < width
                    and 0 <= int(feature["y"]) < height
                    and self.sam_mask[int(feature["y"]), int(feature["x"])]]
        if not selected:
            self.status_label.setText("掩码内没有特征点，请调整前景/背景点")
            return
        self.sam_template_features = selected
        self.feature_label.setText(f"SAM 掩码内特征点：{len(selected)} 个")
        self.sam_save_button.setEnabled(True)
        self.status_label.setText(f"已提取 {len(selected)} 个特征点；正在准备保存模板")

    def save_sam_template(self):
        if not self.sam_mode or self.sam_mask is None:
            QMessageBox.warning(self, "没有掩码", "先点击“SAM模板提取”并标记板件")
            return
        if not self.sam_template_features:
            self.extract_sam_template()
        if self.sam_template_features:
            self._safe_save()

    def leave_sam_mode(self):
        if not self.sam_mode:
            return
        self.sam_mode = False
        self.sam_freezing = False
        self.roi_label.set_prompt_mode(False)
        self.roi_label.set_segmentation_preview(None, [])
        self.sam_frame = None
        self.sam_points = []
        self.sam_mask = None
        self.sam_template_features = None
        self.sam_template_saved = False
        self.save_button.setEnabled(False)
        self.sam_save_button.setEnabled(False)
        self.undo_sam_button.setEnabled(False)
        self.clear_sam_button.setEnabled(False)
        if self.current_frame is not None:
            self.roi_label.set_image(self.current_frame)
        self.status_label.setText("已恢复实时画面")
        self.refresh_action_states()
