#!/usr/bin/env python3
"""
Advanced Detection Demo (Thermal-Ready & Highly Optimized)

Loads an image, video, or live webcam, runs YOLO detection, and draws bounding boxes.
Features:
- Threaded I/O for webcams to unblock GPU inference (Production-grade FPS).
- Specialized `--thermal` mode for heat-mapping and pixel intensity (pseudo-heat) calculation.
- Zero-allocation drawing pipeline to minimize CPU/RAM overhead.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running from project root: python scripts/test_detection.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection import Detection, Detector
from src.detection.exceptions import DetectionError, InvalidFrameError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("detection_demo")

# ---------------------------------------------------------------------------
# Class colour palette (BGR)
# ---------------------------------------------------------------------------

_CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "person":     (0,   255,  0),    # green
    "car":        (255, 128,  0),    # orange
    "motorcycle": (0,   200, 255),   # cyan
    "bus":        (128,   0, 255),   # purple
    "truck":      (0,    0,  255),   # red
}
_DEFAULT_COLOUR = (200, 200, 200)


# ---------------------------------------------------------------------------
# Optimization: Threaded Video Capture (for Live Feeds)
# ---------------------------------------------------------------------------

class ThreadedCamera:
    """
    Dedicated background thread for camera I/O.
    Prevents the GPU/CPU inference step from waiting on physical camera hardware.
    """
    def __init__(self, src: int | str):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self) -> ThreadedCamera:
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self) -> None:
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            return self.ret, self.frame

    def release(self) -> None:
        self.stopped = True
        self.cap.release()

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def get(self, propId: int) -> float:
        return self.cap.get(propId)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_detections(
    frame: np.ndarray, 
    detections: list[Detection], 
    is_thermal: bool = False
) -> np.ndarray:
    """Optimized drawing function with zero wasted memory allocations."""
    
    # 1. Optimize Memory & Conversions
    if is_thermal:
        # Only convert if it's actually a 3-channel image
        is_color = len(frame.shape) == 3 and frame.shape[2] == 3
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if is_color else frame
        
        # applyColorMap creates a new array automatically, no need to copy
        vis = cv2.applyColorMap(gray_frame, cv2.COLORMAP_INFERNO)
    else:
        # Draw directly on the original frame in memory to save CPU/RAM overhead
        vis = frame 

    # 2. Draw Detections
    for det in detections:
        # Clamp coordinates to frame boundaries to prevent ROI extraction crashes
        x1, y1 = max(0, int(det.bbox[0])), max(0, int(det.bbox[1]))
        x2, y2 = min(frame.shape[1], int(det.bbox[2])), min(frame.shape[0], int(det.bbox[3]))
        
        colour = _CLASS_COLOURS.get(det.class_name, _DEFAULT_COLOUR)
        label = f"{det.class_name} {det.confidence:.2f}"

        # Only calculate ROI and pseudo-heat if thermal mode is explicitly ON
        if is_thermal and (x2 > x1) and (y2 > y1):
            max_intensity = np.max(gray_frame[y1:y2, x1:x2])
            label += f" | Heat: {(max_intensity / 255.0) * 100:.0f}%"

        # Bounding Box
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
        
        # Label Background
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1 - 5, lh + 5)
        cv2.rectangle(vis, (x1, label_y - lh - baseline), (x1 + lw, label_y), colour, -1)
        
        # Label Text
        cv2.putText(
            vis, label,
            (x1, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

    # 3. Stats Overlay (with black outline for visibility on bright/thermal backgrounds)
    stats = f"Detections: {len(detections)}"
    cv2.putText(vis, stats, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(vis, stats, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    return vis


# ---------------------------------------------------------------------------
# Main Execution Blocks
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO Detection Demo — Optical & Thermal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", "-s", required=True, help="Image path, video path, or webcam index (e.g. 0).")
    parser.add_argument("--model", "-m", default="yolo11n.pt", help="Path to YOLO .pt weights.")
    parser.add_argument("--conf", "-c", type=float, default=0.5, help="Minimum confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--device", "-d", default="auto", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps'.")
    parser.add_argument("--output", "-o", default=None, help="Optional path to save the annotated output.")
    parser.add_argument("--no-display", action="store_true", help="Do not open a GUI window (for headless servers).")
    parser.add_argument("--thermal", action="store_true", help="Enable thermal colormapping and intensity metrics.")
    return parser.parse_args()


def run_on_image(detector: Detector, args: argparse.Namespace) -> None:
    frame = cv2.imread(args.source)
    if frame is None:
        logger.error("Cannot read image: %r", args.source)
        sys.exit(1)

    t0 = time.perf_counter()
    detections = detector.detect(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info("Image: %r | %d detection(s) in %.1f ms", args.source, len(detections), elapsed_ms)
    for d in detections:
        logger.info("  %s", d)

    vis = draw_detections(frame, detections, is_thermal=args.thermal)

    if args.output:
        cv2.imwrite(args.output, vis)
        logger.info("Saved annotated image to %r", args.output)

    if not args.no_display:
        cv2.imshow("Detection Demo — press any key to close", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video(detector: Detector, args: argparse.Namespace, source_id: str | int, is_live: bool) -> None:
    # Use Threaded IO for live webcams to maximize FPS; use sequential IO for files so no frames are dropped
    if is_live:
        logger.info("Starting threaded camera capture for live feed...")
        cap = ThreadedCamera(source_id).start()
    else:
        cap = cv2.VideoCapture(source_id)

    if not cap.isOpened():
        logger.error("Cannot open video source: %r", source_id)
        sys.exit(1)

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        logger.info("Writing output video to %r", args.output)

    frame_count = 0
    total_ms = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            t0 = time.perf_counter()
            try:
                detections = detector.detect(frame)
            except InvalidFrameError as exc:
                logger.warning("Skipping invalid frame %d: %s", frame_count, exc)
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_ms += elapsed_ms
            frame_count += 1

            vis = draw_detections(frame, detections, is_thermal=args.thermal)

            # FPS overlay
            fps_display = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0
            cv2.putText(vis, f"FPS: {fps_display:.1f}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if writer:
                writer.write(vis)

            if not args.no_display:
                cv2.imshow("Detection Demo — press Q to quit", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    if frame_count > 0:
        avg_ms = total_ms / frame_count
        logger.info("Processed %d frame(s) | avg inference: %.1f ms (%.1f FPS)", frame_count, avg_ms, 1000.0 / avg_ms)


def main() -> None:
    args = parse_args()
    logger.info("Initialising detector (model=%r, conf=%.2f, thermal=%s) …", args.model, args.conf, args.thermal)

    try:
        detector = Detector(
            model_path=args.model,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
        )
    except DetectionError as exc:
        logger.error("Failed to initialise detector: %s", exc)
        sys.exit(1)

    logger.info("Detector ready on device: %s", detector.device)

    source = args.source
    is_webcam = source.isdigit()

    if is_webcam or source.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        run_on_video(detector, args, int(source) if is_webcam else source, is_live=is_webcam)
    else:
        run_on_image(detector, args)


if __name__ == "__main__":
    main()