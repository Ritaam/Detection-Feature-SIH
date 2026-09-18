#!/usr/bin/env python3
"""
Detection demo script.

Loads an image or opens a webcam, runs YOLO11n detection, draws bounding
boxes for supported classes, and either displays the result or saves it.

This script is for VISUAL VERIFICATION ONLY.
It does not implement: tracking, alerts, events, ANPR, or any other module.

Usage
-----
Single image::

    python scripts/test_detection.py --source path/to/image.jpg

Webcam (device 0)::

    python scripts/test_detection.py --source 0

Video file::

    python scripts/test_detection.py --source path/to/video.mp4

Save output instead of displaying::

    python scripts/test_detection.py --source image.jpg --output result.jpg

Custom confidence threshold::

    python scripts/test_detection.py --source image.jpg --conf 0.4

Custom model path::

    python scripts/test_detection.py --source image.jpg --model models/yolo11n.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
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
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of *frame*.

    Parameters
    ----------
    frame:
        Source BGR frame (not modified).
    detections:
        List of :class:`~src.detection.models.Detection` objects.

    Returns
    -------
    np.ndarray
        Annotated copy of the frame.
    """
    vis = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        colour = _CLASS_COLOURS.get(det.class_name, _DEFAULT_COLOUR)
        label = f"{det.class_name} {det.confidence:.2f}"

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        # Label background
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1 - 5, lh + 5)
        cv2.rectangle(vis, (x1, label_y - lh - baseline), (x1 + lw, label_y), colour, -1)

        # Label text
        cv2.putText(
            vis, label,
            (x1, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

    # Stats overlay
    stats = f"Detections: {len(detections)}"
    cv2.putText(vis, stats, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO11n detection demo — border surveillance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Image path, video path, or webcam index (e.g. 0).",
    )
    parser.add_argument(
        "--model", "-m",
        default="yolo11n.pt",
        help="Path to YOLO .pt weights (auto-downloads if name only).",
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Minimum confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Optional path to save the annotated image/video.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a GUI window (useful on headless servers).",
    )
    return parser.parse_args()


def _is_webcam(source: str) -> bool:
    try:
        int(source)
        return True
    except ValueError:
        return False


def run_on_image(
    detector: Detector,
    path: str,
    output: str | None,
    no_display: bool,
) -> None:
    frame = cv2.imread(path)
    if frame is None:
        logger.error("Cannot read image: %r", path)
        sys.exit(1)

    t0 = time.perf_counter()
    detections = detector.detect(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Image: %r | %d detection(s) in %.1f ms",
        path, len(detections), elapsed_ms,
    )
    for d in detections:
        logger.info("  %s", d)

    vis = draw_detections(frame, detections)

    if output:
        cv2.imwrite(output, vis)
        logger.info("Saved annotated image to %r", output)

    if not no_display:
        cv2.imshow("Detection Demo — press any key to close", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video(
    detector: Detector,
    source: str | int,
    output: str | None,
    no_display: bool,
) -> None:
    cap = cv2.VideoCapture(source)  # type: ignore[arg-type]
    if not cap.isOpened():
        logger.error("Cannot open video source: %r", source)
        sys.exit(1)

    writer = None
    if output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (w, h))
        logger.info("Writing output video to %r", output)

    frame_count = 0
    total_ms = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
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

            vis = draw_detections(frame, detections)

            # FPS overlay
            fps_display = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0
            cv2.putText(
                vis, f"FPS: {fps_display:.1f}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )

            if writer:
                writer.write(vis)

            if not no_display:
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
        logger.info(
            "Processed %d frame(s) | avg inference: %.1f ms (%.1f FPS)",
            frame_count, avg_ms, 1000.0 / avg_ms,
        )


def main() -> None:
    args = parse_args()

    logger.info("Initialising detector (model=%r, conf=%.2f, device=%s) …",
                args.model, args.conf, args.device)

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
    if _is_webcam(source) or source.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".webm")
    ):
        run_on_video(
            detector,
            int(source) if _is_webcam(source) else source,
            args.output,
            args.no_display,
        )
    else:
        run_on_image(detector, source, args.output, args.no_display)


if __name__ == "__main__":
    main()
