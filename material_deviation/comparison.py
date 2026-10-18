"""Embed and draw a cropped reference image for visual deviation comparison."""

import base64

import cv2
import numpy as np


def encode_reference_visual(frame: np.ndarray, mask: np.ndarray) -> dict:
    """Store a masked BGR reference crop as a portable alpha PNG in template JSON."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("参考画面必须是 BGR 彩色图像")
    if mask is None or mask.shape != frame.shape[:2]:
        raise ValueError("参考掩码尺寸与画面不一致")
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("参考掩码为空")
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    rgba = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = (mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        raise OSError("无法编码模板参考图")
    return {"origin": [x1, y1], "png_base64": base64.b64encode(encoded).decode("ascii")}


def decode_reference_visual(data: dict | None) -> dict | None:
    if not data:
        return None
    origin = data.get("origin")
    if not isinstance(origin, list) or len(origin) != 2:
        raise ValueError("模板参考图缺少原点坐标")
    encoded = base64.b64decode(data["png_base64"], validate=True)
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("模板参考图不是四通道 PNG")
    return {"origin": (int(origin[0]), int(origin[1])), "bgra": image}


def draw_reference_overlay(frame: np.ndarray, reference: dict, opacity: float) -> np.ndarray:
    """Blend the original template pixels at their original image coordinates."""
    result = frame.copy()
    x0, y0 = reference["origin"]
    bgra = reference["bgra"]
    image_h, image_w = result.shape[:2]
    crop_h, crop_w = bgra.shape[:2]
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(image_w, x0 + crop_w), min(image_h, y0 + crop_h)
    if x2 <= x1 or y2 <= y1:
        return result
    crop = bgra[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    alpha = (crop[:, :, 3:4].astype(np.float32) / 255.0) * float(np.clip(opacity, 0, 1))
    target = result[y1:y2, x1:x2].astype(np.float32)
    result[y1:y2, x1:x2] = np.clip(
        target * (1 - alpha) + crop[:, :, :3].astype(np.float32) * alpha,
        0, 255,
    ).astype(np.uint8)
    contours, _ = cv2.findContours(crop[:, :, 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result[y1:y2, x1:x2], contours, -1, (255, 255, 0), 2)
    return result
