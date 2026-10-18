"""User settings and paths resolved relative to the JSON config file."""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2


BACKENDS = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}


@dataclass(frozen=True)
class CameraConfig:
    video_id: int = 2
    width: int = 2560
    height: int = 1440
    fps: int = 30
    exposure: float = -3
    gain: float = 10
    backend: int = cv2.CAP_DSHOW
    retry_times: int = 5
    retry_interval: float = 1.0
    thread_sleep: float = 0.001
    queue_max: int = 5


@dataclass(frozen=True)
class AppSettings:
    camera: CameraConfig
    calibration_file: Path
    weights_file: Path
    sam_weights_file: Path
    sam_model_type: str
    sam_device: str
    comparison_opacity: float
    templates_dir: Path
    results_dir: Path
    working_distance_m: float
    template_frames: int
    match_frames: int
    cuda: bool


def load_settings(path: str | Path) -> AppSettings:
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    camera_data = data.get("camera", {}).copy()
    backend_name = camera_data.get("backend", "dshow").lower()
    if backend_name not in BACKENDS:
        raise ValueError(f"Unsupported camera backend: {backend_name}")
    camera_data["backend"] = BACKENDS[backend_name]
    base = path.parent

    def resolve(key: str, default: str) -> Path:
        candidate = Path(data.get(key, default))
        return candidate if candidate.is_absolute() else (base / candidate).resolve()

    comparison_opacity = float(data.get("comparison_opacity", 0.35))
    if not 0.0 < comparison_opacity <= 1.0:
        raise ValueError("comparison_opacity must be between 0 and 1")
    return AppSettings(
        camera=CameraConfig(**camera_data),
        calibration_file=resolve("calibration_file", "rgb_calibration.yml"),
        weights_file=resolve("weights_file", "../models/superpoint_v1.pth"),
        sam_weights_file=resolve("sam_weights_file", "../models/sam_vit_b_01ec64.pth"),
        sam_model_type=str(data.get("sam_model_type", "vit_b")),
        sam_device=str(data.get("sam_device", "auto")),
        comparison_opacity=comparison_opacity,
        templates_dir=resolve("templates_dir", "../templates"),
        results_dir=resolve("results_dir", "../matching_results"),
        working_distance_m=float(data.get("working_distance_m", 1.38)),
        template_frames=int(data.get("template_frames", 20)),
        match_frames=int(data.get("match_frames", 15)),
        cuda=bool(data.get("cuda", False)),
    )
