"""Capture chessboard frames and write an OpenCV RGB camera calibration YAML.

This script is independent of the PyQt UI and material_deviation package.
Run ``python scripts/calibrate_rgb_camera.py --help`` from the project root.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from glob import glob
import json
from pathlib import Path
import sys

import cv2
import numpy as np


BACKENDS = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}


def camera_options(config_path: Path, camera_id: int | None) -> dict:
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    camera = settings.get("camera", {})
    backend_name = str(camera.get("backend", "dshow")).lower()
    if backend_name not in BACKENDS:
        raise ValueError(f"Unsupported camera backend: {backend_name}")
    return {
        "id": int(camera_id if camera_id is not None else camera.get("video_id", 2)),
        "backend": BACKENDS[backend_name],
        "width": int(camera.get("width", 2560)),
        "height": int(camera.get("height", 1440)),
        "fps": float(camera.get("fps", 30)),
        "exposure": camera.get("exposure"),
        "gain": camera.get("gain"),
    }


def capture_images(directory: Path, config_path: Path, camera_id: int | None,
                   pattern_size: tuple[int, int]) -> str:
    options = camera_options(config_path, camera_id)
    directory.mkdir(parents=True, exist_ok=False)
    camera = cv2.VideoCapture(options["id"], options["backend"])
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Cannot open RGB camera {options['id']}")
    try:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, options["width"])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, options["height"])
        camera.set(cv2.CAP_PROP_FPS, options["fps"])
        if options["exposure"] is not None:
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            camera.set(cv2.CAP_PROP_EXPOSURE, float(options["exposure"]))
        if options["gain"] is not None:
            camera.set(cv2.CAP_PROP_GAIN, float(options["gain"]))
        print(f"Camera {options['id']}: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
              f"{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print("Move the chessboard across the view and tilt it. Space: save valid frame; Q/Esc: finish.")
        saved = 0
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera frame read failed")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size)
            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, pattern_size, corners, found)
            cv2.putText(preview, f"Corners: {'OK' if found else 'not found'} | saved: {saved}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow("RGB calibration - Space: save, Q: finish", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if not found:
                    print("Chessboard corners not found; frame skipped.")
                    continue
                saved += 1
                path = directory / f"chessboard_{saved:04d}.png"
                encoded, data = cv2.imencode(".png", frame)
                if not encoded:
                    raise OSError(f"Cannot encode {path}")
                path.write_bytes(data.tobytes())
                print(f"Saved {path}")
        if saved < 3:
            raise ValueError(f"Need at least 3 valid images; captured {saved}")
        return str(directory / "chessboard_*.png")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def calibrate(images: str, pattern_size: tuple[int, int], square_mm: float):
    columns, rows = pattern_size
    object_grid = np.zeros((columns * rows, 3), np.float32)
    object_grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_mm
    objects, image_points = [], []
    size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    paths = sorted(glob(images))
    if not paths:
        raise ValueError(f"No images match: {images}")
    for filename in paths:
        raw = np.fromfile(filename, dtype=np.uint8)
        gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) if raw.size else None
        if gray is None:
            print(f"Skipped unreadable image: {filename}")
            continue
        current_size = (gray.shape[1], gray.shape[0])
        if size is not None and current_size != size:
            raise ValueError(f"Image size mismatch: {filename}: {current_size} != {size}")
        size = current_size
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            print(f"No chessboard corners: {filename}")
            continue
        objects.append(object_grid.copy())
        image_points.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
    if len(objects) < 3 or size is None:
        raise ValueError(f"Need at least 3 valid chessboard images; found {len(objects)}")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        objects, image_points, size, None, None
    )
    return float(rms), matrix, distortion, size, len(objects)


def save_yaml(path: Path, matrix: np.ndarray, distortion: np.ndarray,
              size: tuple[int, int], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; choose another --output or add --overwrite")
    # FileStorage memory mode avoids OpenCV's Windows path issues with Chinese names.
    storage = cv2.FileStorage("calibration.yml", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY)
    if not storage.isOpened():
        raise OSError("Cannot create OpenCV YAML in memory")
    storage.write("K", matrix)
    storage.write("D", distortion)
    storage.write("image_width", size[0])
    storage.write("image_height", size[1])
    content = storage.releaseAndGetString()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture or load RGB chessboard images and generate rgb_calibration.yml"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config.example.json"),
                        help="Camera settings JSON for live capture (default: configs/config.example.json)")
    parser.add_argument("--camera-id", type=int, help="Override camera.video_id in the config")
    parser.add_argument("--images", help="Glob of existing chessboard images; skips camera capture")
    parser.add_argument("--capture-dir", type=Path,
                        help="New directory for captured images (default: dated folder under captures/)")
    parser.add_argument("--columns", type=int, required=True,
                        help="Number of INTERNAL chessboard corners across")
    parser.add_argument("--rows", type=int, required=True,
                        help="Number of INTERNAL chessboard corners down")
    parser.add_argument("--square-mm", type=float, required=True,
                        help="Measured chessboard square side length in millimetres")
    parser.add_argument("--output", type=Path, default=Path("configs/rgb_calibration.yml"),
                        help="Output YAML (default: configs/rgb_calibration.yml)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing output YAML")
    args = parser.parse_args()
    if args.columns < 2 or args.rows < 2 or args.square_mm <= 0:
        parser.error("columns and rows must be >= 2; square-mm must be positive")
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} exists; choose another --output or add --overwrite")
    pattern_size = (args.columns, args.rows)
    try:
        if args.images:
            images = args.images
        else:
            directory = args.capture_dir or Path("captures") / datetime.now().strftime(
                "calibration_%Y%m%d_%H%M%S"
            )
            images = capture_images(directory, args.config, args.camera_id, pattern_size)
        rms, matrix, distortion, size, count = calibrate(
            images, pattern_size, args.square_mm
        )
        save_yaml(args.output, matrix, distortion, size, args.overwrite)
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        parser.exit(1, f"Calibration failed: {exc}\n")
    print(f"Saved {args.output} | valid images: {count} | size: {size[0]}x{size[1]} | RMS: {rms:.3f} px")
    print("Set calibration_file in configs/config.json to this YAML; set working_distance_m separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
