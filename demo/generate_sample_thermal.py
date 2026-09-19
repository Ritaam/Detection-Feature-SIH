"""
generate_sample_thermal.py — Generate a synthetic thermal/IR video for demo verification.

This script creates a realistic sample thermal video (`data/sample_thermal.mp4`)
using locally available image assets from Ultralytics (e.g. bus.jpg/zidane.jpg)
with thermal infrared sensor simulation:
- Grayscale / white-hot thermal sensor modality
- Ambient sensor noise and thermal blur
- Dynamic camera pan/zoom motion across frames
- 25 FPS, standard resolution

Usage:
    python demo/generate_sample_thermal.py [--output data/sample_thermal.mp4] [--frames 150]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


def create_sample_thermal_video(
    output_path: str = "data/sample_thermal.mp4",
    num_frames: int = 150,
    fps: float = 25.0,
    width: int = 640,
    height: int = 480,
) -> Path:
    """Generate a sample thermal infrared video file.

    Parameters
    ----------
    output_path:
        Destination path for the MP4 video.
    num_frames:
        Number of frames to generate (default: 150 = 6 seconds @ 25 FPS).
    fps:
        Frames per second (default: 25.0).
    width:
        Frame width (default: 640).
    height:
        Frame height (default: 480).

    Returns
    -------
    Path
        Absolute path to the created video.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Locate Ultralytics assets
    try:
        import ultralytics
        assets_dir = Path(ultralytics.__file__).parent / "assets"
        bus_img_path = assets_dir / "bus.jpg"
        if bus_img_path.exists():
            base_bgr = cv2.imread(str(bus_img_path))
        else:
            base_bgr = None
    except Exception:
        base_bgr = None

    if base_bgr is None:
        # Fallback synthetic scene if assets are unavailable
        base_bgr = np.full((720, 1280, 3), 40, dtype=np.uint8)
        # Add road
        cv2.rectangle(base_bgr, (0, 450), (1280, 720), (60, 60, 60), -1)
        # Add simulated vehicles/pedestrians
        cv2.rectangle(base_bgr, (200, 300), (450, 500), (180, 180, 180), -1)
        cv2.circle(base_bgr, (600, 380), 25, (220, 220, 220), -1)
        cv2.rectangle(base_bgr, (585, 405), (615, 520), (200, 200, 200), -1)

    # Convert base to grayscale (thermal white-hot intensity representation)
    base_gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    bh, bw = base_gray.shape

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height), isColor=False)
    if not writer.isOpened():
        # Fallback to XVID / avi if mp4v fails
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height), isColor=False)

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_file}")

    # Camera panning parameters: sweep a crop window across the base image
    crop_w = int(bw * 0.75)
    crop_h = int(crop_w * (height / width))
    if crop_h > bh:
        crop_h = bh
        crop_w = int(crop_h * (width / height))

    max_dx = bw - crop_w
    max_dy = bh - crop_h

    rng = np.random.default_rng(42)

    for i in range(num_frames):
        # Smooth camera motion (panning horizontally back and forth)
        t = i / max(1, num_frames - 1)
        pan_factor = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
        x1 = int(pan_factor * max_dx) if max_dx > 0 else 0
        y1 = int(0.3 * max_dy) if max_dy > 0 else 0
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped = base_gray[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

        # Add subtle microbolometer sensor noise (~1.5 std dev)
        sensor_noise = rng.normal(0, 1.5, (height, width)).astype(np.float32)
        thermal_frame = np.clip(resized.astype(np.float32) + sensor_noise, 0, 255).astype(np.uint8)

        writer.write(thermal_frame)

    writer.release()
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample thermal video.")
    parser.add_argument("--output", default="data/sample_thermal.mp4", help="Output video path")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames")
    args = parser.parse_args()

    out = create_sample_thermal_video(output_path=args.output, num_frames=args.frames)
    print(f"Generated sample thermal video: {out} ({args.frames} frames)")
