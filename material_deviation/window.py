"""PyQt5 desktop window for template capture and deviation measurement."""
import os, time, traceback
import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QMessageBox, QFrame)
from .camera import UsbCameraThread
from .roi import ROIDisplayLabel
from .settings import CameraConfig
from .features import SuperPointFeatureExtractor
from .calibration import CalibrationMixin
from .template_workflow import TemplateWorkflowMixin
from .matching_workflow import MatchingWorkflowMixin
from .sam_workflow import SamTemplateWorkflowMixin
from .comparison import draw_reference_overlay

class MaterialDeviationWindow(CalibrationMixin, TemplateWorkflowMixin, MatchingWorkflowMixin,
                              SamTemplateWorkflowMixin, QWidget):
    def __init__(self, cfg: CameraConfig, weights_path=None, cuda=False, app_settings=None):
        super().__init__()
        self.cfg = cfg
        self.app_settings = app_settings
        self.cam = UsbCameraThread(cfg)

        # SuperPoint特征提取器 - 传入权重文件路径
        self.feature_extractor = SuperPointFeatureExtractor(
            weights_path=weights_path,
            cuda=cuda
        )

        # ROI区域
        self.current_roi = None
        self.current_frame = None

        # 模板数据
        self.loaded_template = None
        self.template_filename = ""
        self.sam_mode = False
        self.sam_freezing = False
        self.sam_frame = None
        self.sam_points = []
        self.sam_mask = None
        self.sam_template_features = None
        self.sam_template_saved = False
        self.sam_segmenter = None
        self.template_reference_frame = None
        self.template_reference_roi = None
        self.comparison_enabled = False
        self.display_unit = "px"

        # 匹配状态
        self.matching_enabled = False
        self.match_roi = None  # 匹配时的ROI区域
        self.selecting_match_roi = False

        self._init_ui()
        self._init_timer()

        # 连接信号
        self.roi_label.roi_selected.connect(self.on_roi_selected)
        self.roi_label.prompt_point_selected.connect(self.on_sam_point)

        # 添加稳定提取相关变量
        self.multi_frame_mode = False
        self.captured_frames = []
        self.capture_count = 0
        self.capture_target = app_settings.template_frames if app_settings else 20

        # 匹配状态
        self.matching_enabled = False
        self.match_roi = None  # 匹配时的ROI区域

        # 多帧匹配相关变量
        self.multi_frame_matching = False  # 是否正在多帧匹配模式
        self.matching_frames_collected = []  # 收集的帧
        self.matching_results = []  # 收集的匹配结果
        self.matching_target_frames = app_settings.match_frames if app_settings else 15
        self.matching_processed = False  # 是否已处理完多帧
        self.final_matching_result = None  # 最终的匹配结果
        self.collected_frame_count = 0  # 已收集的帧数

        # 相机标定参数
        self.camera_matrix = None
        self.dist_coeffs = None
        self.working_distance = app_settings.working_distance_m if app_settings else 1.38

        # 加载相机标定参数
        camera_params_path = str(app_settings.calibration_file) if app_settings else None
        if camera_params_path and os.path.exists(camera_params_path):
            self.load_camera_params(camera_params_path)

        # 检查特征提取器状态
        if not self.feature_extractor.available:
            print("警告: 特征提取器初始化失败，可能无法使用模板提取功能")
        self.refresh_action_states()
    def _init_ui(self):
        if hasattr(self.feature_extractor, 'use_orb') and self.feature_extractor.use_orb:
            title = 'USB 相机模板提取与匹配 (ORB特征 - SuperPoint不可用)'
        else:
            title = 'USB 相机模板提取与匹配 (SuperPoint特征)'

        self.setWindowTitle(title)
        self.resize(1400, 900)

        # 创建自定义的ROI显示标签
        self.roi_label = ROIDisplayLabel()

        # 三栏操作区：模板制作 | 匹配 | 查看与窗口操作
        controls = QHBoxLayout()
        controls.setSpacing(0)
        controls.addStretch(1)
        left = QVBoxLayout()
        left_top = QHBoxLayout()
        left_bottom = QHBoxLayout()
        middle = QVBoxLayout()
        middle_top = QHBoxLayout()
        middle_bottom = QHBoxLayout()
        right = QVBoxLayout()
        right_top = QHBoxLayout()
        right_bottom = QHBoxLayout()
        for row in (left_top, left_bottom, middle_top, middle_bottom,
                    right_top, right_bottom):
            row.setSpacing(12)
        for column in (left, middle, right):
            column.setSpacing(12)

        self.template_button = QPushButton("手动模板提取")
        self.template_button.clicked.connect(self.start_template_extraction)

        self.save_button = QPushButton("手动模板保存")
        self.save_button.clicked.connect(self._safe_save)
        self.save_button.setEnabled(False)

        self.clear_button = QPushButton("清除ROI")
        self.clear_button.clicked.connect(self.clear_roi)
        self.clear_button.setEnabled(False)

        self.sam_button = QPushButton("SAM模板提取")
        self.sam_button.clicked.connect(self.start_sam_template)
        self.undo_sam_button = QPushButton("撤销点")
        self.undo_sam_button.clicked.connect(self.undo_sam_point)
        self.undo_sam_button.setEnabled(False)
        self.clear_sam_button = QPushButton("清空点")
        self.clear_sam_button.clicked.connect(self.clear_sam_points)
        self.clear_sam_button.setEnabled(False)
        self.sam_save_button = QPushButton("SAM模板保存")
        self.sam_save_button.clicked.connect(self.save_sam_template)
        self.sam_save_button.setEnabled(False)

        self.load_button = QPushButton("加载模板")
        self.load_button.clicked.connect(self.load_template)

        self.match_button = QPushButton("开始匹配")
        self.match_button.clicked.connect(self.toggle_matching)

        self.set_roi_button = QPushButton("设置匹配ROI")
        self.set_roi_button.clicked.connect(self.set_matching_roi)
        self.set_roi_button.setEnabled(False)

        self.restore_button = QPushButton("恢复画面")
        self.restore_button.clicked.connect(self.restore_live_view)

        self.unit_button = QPushButton("单位切换")
        self.unit_button.clicked.connect(self.toggle_display_unit)
        self.unit_button.setToolTip("切换匹配结果的像素偏移与毫米偏移")

        self.comparison_button = QPushButton("偏差对比")
        self.comparison_button.clicked.connect(self.toggle_comparison)

        self.help_button = QPushButton("使用说明")
        self.help_button.clicked.connect(self.show_usage)

        self.exit_button = QPushButton("退出")
        self.exit_button.clicked.connect(self.close)

        left_buttons = (self.template_button, self.clear_button, self.save_button,
                        self.sam_button, self.undo_sam_button, self.clear_sam_button,
                        self.sam_save_button)
        middle_buttons = (self.load_button, self.set_roi_button,
                          self.match_button, self.restore_button)
        right_buttons = (self.unit_button, self.comparison_button,
                         self.help_button, self.exit_button)
        for button in left_buttons:
            button.setFixedSize(124, 40)
        for button in middle_buttons:
            button.setFixedSize(108, 40)
        for button in right_buttons:
            button.setFixedSize(100, 40)
        self.setStyleSheet(
            "QPushButton:disabled { background-color: #d5d5d5; color: #777; "
            "border: 1px solid #aaa; }"
        )

        for layout, buttons in (
            (left_top, (self.template_button, self.clear_button, self.save_button)),
            (left_bottom, (self.sam_button, self.undo_sam_button,
                           self.clear_sam_button, self.sam_save_button)),
            (middle_top, (self.load_button, self.set_roi_button)),
            (middle_bottom, (self.match_button, self.restore_button)),
            (right_top, (self.unit_button, self.comparison_button)),
            (right_bottom, (self.help_button, self.exit_button)),
        ):
            for button in buttons:
                layout.addWidget(button)
            layout.addStretch(1)

        for column, top, bottom in ((left, left_top, left_bottom),
                                    (middle, middle_top, middle_bottom),
                                    (right, right_top, right_bottom)):
            column.addLayout(top)
            column.addLayout(bottom)

        self.control_separators = []
        for index, column in enumerate((left, middle, right)):
            if index:
                controls.addSpacing(28)
                separator = QFrame()
                separator.setFrameShape(QFrame.VLine)
                separator.setFixedWidth(1)
                separator.setStyleSheet("background-color: #777;")
                self.control_separators.append(separator)
                controls.addWidget(separator)
                controls.addSpacing(28)
            controls.addLayout(column)
        controls.addStretch(1)

        # 状态标签
        self.status_label = QLabel("就绪 - 使用SuperPoint特征")
        self.status_label.setStyleSheet("color: white; background-color: #333333; padding: 5px;")

        # 特征点信息标签
        self.feature_label = QLabel("特征点: 0")
        self.feature_label.setStyleSheet("color: white; background-color: #555555; padding: 5px;")

        # 匹配信息标签
        self.match_label = QLabel("匹配: 未开始")
        self.match_label.setStyleSheet("color: white; background-color: #555555; padding: 5px;")

        # 主布局
        v = QVBoxLayout(self)
        v.addWidget(self.roi_label, stretch=1)
        v.addWidget(self.status_label)
        v.addWidget(self.feature_label)
        v.addWidget(self.match_label)
        v.addLayout(controls)
    def refresh_action_states(self):
        """Keep unavailable steps grey and prevent overlapping workflows."""
        frame_ready = self.current_frame is not None
        sam_active = self.sam_mode or self.sam_freezing
        matching_active = self.matching_enabled or self.multi_frame_matching or self.matching_processed
        manual_active = (self.multi_frame_mode or self.current_roi is not None
                         or bool(getattr(self, "stable_features", None)))
        idle = not (sam_active or matching_active or manual_active or self.selecting_match_roi)
        template_ready = self.loaded_template is not None
        reference_ready = template_ready and self.loaded_template.get("reference_visual") is not None
        physical = ((self.final_matching_result or {}).get("physical_offset_mm")
                    if self.matching_processed else None)

        self.template_button.setEnabled(frame_ready and idle)
        self.sam_button.setEnabled(frame_ready and idle)
        self.clear_button.setEnabled(manual_active and not sam_active and not matching_active)
        self.save_button.setEnabled(
            (bool(getattr(self, "stable_features", None)) or self.current_roi is not None)
            and not self.multi_frame_mode and not sam_active and not matching_active
        )
        self.undo_sam_button.setEnabled(sam_active and bool(self.sam_points))
        self.clear_sam_button.setEnabled(sam_active and bool(self.sam_points))
        self.sam_save_button.setEnabled(sam_active and self.sam_mask is not None
                                        and not self.sam_template_saved)
        self.load_button.setEnabled(idle)
        self.set_roi_button.setEnabled(frame_ready and template_ready and idle)
        self.match_button.setEnabled(
            (frame_ready and template_ready and idle)
            or (matching_active and not self.multi_frame_matching)
        )
        self.restore_button.setEnabled(not idle and not self.sam_freezing)
        self.unit_button.setEnabled(bool(physical and all(value is not None for value in physical)))
        self.comparison_button.setEnabled(reference_ready and not sam_active and not manual_active)
    def _safe_save(self):
        """保险版本：任何异常都打印，不让 Qt 直接杀进程"""
        try:
            self.real_save_template()  # 真正的保存逻辑放这里
        except Exception as e:
            print('保存模板时异常:', e)
            traceback.print_exc()
            QMessageBox.critical(self, "保存失败", str(e))
    def _init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        if self.cam.start_grab():
            self.timer.start(30)  # 30ms，约33FPS
            self.status_label.setText("相机启动成功 - 使用SuperPoint特征")
        else:
            self.status_label.setText("相机启动失败")
    def _update_frame(self):
        if self.sam_mode or self.sam_freezing:
            return
        frame = self.cam.get_frame(timeout=0.01)
        if frame is None:
            return

        # 显示实际获取到的分辨率
        h, w = frame.shape[:2]
        if not hasattr(self, '_last_resolution') or self._last_resolution != (w, h):
            self._last_resolution = (w, h)
            print(f"实际获取帧分辨率: {w}x{h}")

        self.current_frame = frame.copy()
        if not hasattr(self, "_actions_frame_ready"):
            self._actions_frame_ready = True
            self.refresh_action_states()

        # 判断匹配状态
        if self.multi_frame_matching:
            # 多帧匹配模式：收集帧
            self.collect_frame_for_matching(frame)
        elif self.matching_processed and self.final_matching_result and not hasattr(self, '_final_result_displayed'):
            # 已经处理完多帧，但还没有显示最终结果（静止画面）
            self.display_final_result()
        elif self.matching_enabled and self.loaded_template is not None:
            # 实时匹配模式（旧逻辑，现在应该不会进入这里）
            # self.perform_matching(frame)
            pass
        else:
            # 正常显示
            self.roi_label.set_image(self._comparison_frame(frame))
    def _comparison_frame(self, frame):
        reference = (self.loaded_template or {}).get("reference_visual")
        if self.comparison_enabled and reference is not None:
            opacity = self.app_settings.comparison_opacity if self.app_settings else 0.35
            return draw_reference_overlay(frame, reference, opacity)
        return frame
    def toggle_comparison(self):
        if self.loaded_template is None:
            QMessageBox.information(self, "偏差对比", "请先加载模板")
            return
        if self.loaded_template.get("reference_visual") is None:
            QMessageBox.information(self, "偏差对比", "此模板没有参考图，请重新提取并保存模板")
            return
        self.comparison_enabled = not self.comparison_enabled
        self.comparison_button.setText("关闭对比" if self.comparison_enabled else "偏差对比")
        if self.matching_processed and self.final_matching_result:
            if hasattr(self, '_final_result_displayed'):
                delattr(self, '_final_result_displayed')
            self.display_final_result()
        elif self.current_frame is not None and not self.sam_mode:
            self.roi_label.set_image(self._comparison_frame(self.current_frame))
        self.refresh_action_states()
    def toggle_display_unit(self):
        if self.display_unit == "px":
            if not (hasattr(self, "pixel_to_mm_x") and hasattr(self, "pixel_to_mm_y")):
                QMessageBox.information(self, "单位切换", "没有可用的相机标定参数，当前只能显示像素偏移")
                return
            self.display_unit = "mm"
        else:
            self.display_unit = "px"
        self.status_label.setText(f"当前偏移单位：{'毫米' if self.display_unit == 'mm' else '像素'}")
        if self.matching_processed and self.final_matching_result:
            self.display_final_result()
    def restore_live_view(self):
        if self.selecting_match_roi:
            self.roi_label.roi_selected.disconnect()
            self.roi_label.roi_selected.connect(self.on_roi_selected)
        if self.sam_mode:
            self.leave_sam_mode()
        if self.matching_enabled or self.matching_processed or self.multi_frame_matching:
            self.stop_matching()
        self.multi_frame_mode = False
        self.current_roi = None
        self.selecting_match_roi = False
        self.template_reference_frame = None
        self.template_reference_roi = None
        if hasattr(self, "stable_features"):
            delattr(self, "stable_features")
        self.roi_label.drawing_roi = False
        self.roi_label.clear_roi()
        self.clear_button.setEnabled(False)
        self.save_button.setEnabled(False)
        if self.current_frame is not None:
            self.roi_label.set_image(self._comparison_frame(self.current_frame))
        self.status_label.setText("已恢复实时画面")
        self.refresh_action_states()
    def show_usage(self):
        QMessageBox.information(
            self, "使用说明",
            "模板提取方式分为两种，手动框选ROI后保存模板，或用SAM左键点目标、右键点背景，再保存模板。\n\n"
            "开始匹配前要加载模板，可设置匹配ROI，以减小计算量。开始匹配后画面定格，并显示匹配结果，点击恢复画面可退出定格。\n\n"
            "单位切换在像素与毫米之间切换，毫米需要有效相机标定。偏差对比会在原位置叠加半透明模板。",
        )
    def clear_roi(self):
        """清除ROI"""
        self.current_roi = None
        self.multi_frame_mode = False
        self.template_reference_frame = None
        self.template_reference_roi = None
        if hasattr(self, "stable_features"):
            delattr(self, "stable_features")
        self.roi_label.drawing_roi = False
        self.roi_label.clear_roi()
        self.save_button.setEnabled(False)
        self.feature_label.setText("特征点: 0")
        self.status_label.setText("已清除ROI")

        # 恢复原始图像显示
        if self.current_frame is not None:
            self.roi_label.set_image(self.current_frame)
        self.refresh_action_states()
    def closeEvent(self, event):
        self.cam.stop_grab()
        event.accept()
