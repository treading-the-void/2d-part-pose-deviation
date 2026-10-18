"""Point-prompted SAM segmentation used while creating a template."""

from pathlib import Path

import cv2
import numpy as np


class SamSegmenter:
    def __init__(self, checkpoint: str | Path, model_type: str = "vit_b", device: str = "auto"):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM 权重不存在：{checkpoint}")
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError("缺少 SAM 依赖；请安装项目的 sam 可选依赖") from exc
        if model_type not in sam_model_registry:
            raise ValueError(f"不支持的 SAM 模型类型：{model_type}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        model.to(device=device)
        self.predictor = SamPredictor(model)
        self.image_shape = None
        self.device = device

    def set_image(self, bgr_image: np.ndarray) -> None:
        if bgr_image is None or bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            raise ValueError("SAM 需要 BGR 彩色图像")
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(np.ascontiguousarray(rgb))
        self.image_shape = bgr_image.shape[:2]

    def predict(self, points: list[tuple[int, int, int]]) -> tuple[np.ndarray, float]:
        if self.image_shape is None:
            raise RuntimeError("请先为 SAM 设置图像")
        if not points or not any(label == 1 for _, _, label in points):
            raise ValueError("请至少添加一个前景点")
        height, width = self.image_shape
        coords = np.asarray([(x, y) for x, y, _ in points], dtype=np.float32)
        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
        labels = np.asarray([label for _, _, label in points], dtype=np.int32)
        masks, scores, _ = self.predictor.predict(
            point_coords=coords, point_labels=labels, multimask_output=True
        )
        best = int(np.argmax(scores))
        mask = np.asarray(masks[best], dtype=bool)
        if mask.shape != (height, width) or not mask.any():
            raise RuntimeError("SAM 没有生成有效掩码")
        return mask, float(scores[best])
