"""
Thermal detection visualization utilities.

This module provides functions to draw thermal detection results on frames
for debugging and visual inspection.

**These functions are NOT part of the inference pipeline.**
The detection engine works without visualization.  These are optional
utilities for development, debugging, and demo purposes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from .models import ThermalDetection

logger = logging.getLogger(__name__)

# Try to import ThermalHotspot; it's optional for visualization
try:
    from .hotspot_segmentation import ThermalHotspot

    _HAS_HOTSPOT = True
except ImportError:
    _HAS_HOTSPOT = False


# ---------------------------------------------------------------------------
# Color palette for different classes
# ---------------------------------------------------------------------------

_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (0, 255, 100),  # bright green
    "car": (255, 165, 0),  # orange
    "truck": (0, 165, 255),  # deep orange
    "bus": (255, 255, 0),  # cyan
    "motorcycle": (255, 0, 255),  # magenta
}

_DEFAULT_COLOR: tuple[int, int, int] = (0, 200, 255)  # amber


def _get_class_color(class_name: str) -> tuple[int, int, int]:
    """Get the display colour for a class name (BGR)."""
    return _CLASS_COLORS.get(class_name, _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Detection visualization
# ---------------------------------------------------------------------------


def draw_thermal_detections(
    frame: np.ndarray,
    detections: list[ThermalDetection],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw thermal detection bounding boxes and labels on a frame.

    Parameters
    ----------
    frame:
        The frame to annotate.  Accepts grayscale (H×W) or
        3-channel (H×W×3).  If grayscale, it will be converted to
        3-channel for coloured annotations.
    detections:
        List of :class:`ThermalDetection` objects to draw.
    thickness:
        Line thickness for bounding boxes.
    font_scale:
        Font scale for label text.

    Returns
    -------
    np.ndarray
        Annotated copy of the frame (H×W×3, uint8).
        The original frame is never modified.
    """
    # Ensure 3-channel for colour drawing
    if frame.ndim == 2:
        canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        canvas = frame.copy()

    if canvas.dtype != np.uint8:
        # Normalize to uint8 for drawing
        v_min, v_max = float(canvas.min()), float(canvas.max())
        if v_max > v_min:
            canvas = (
                (canvas.astype(np.float64) - v_min) / (v_max - v_min) * 255
            ).astype(np.uint8)
        else:
            canvas = np.full_like(canvas, 128, dtype=np.uint8)

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        color = _get_class_color(det.class_name)

        # Draw bounding box
        cv2.rectangle(canvas, pt1, pt2, color, thickness)

        # Draw label with confidence
        label = f"{det.class_name} {det.confidence:.2f}"
        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        label_w, label_h = label_size

        # Label background
        bg_pt1 = (pt1[0], pt1[1] - label_h - baseline - 4)
        bg_pt2 = (pt1[0] + label_w + 4, pt1[1])
        cv2.rectangle(canvas, bg_pt1, bg_pt2, color, cv2.FILLED)

        # Label text (black on coloured background)
        text_pt = (pt1[0] + 2, pt1[1] - baseline - 2)
        cv2.putText(
            canvas,
            label,
            text_pt,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas


# ---------------------------------------------------------------------------
# Hotspot visualization
# ---------------------------------------------------------------------------


def draw_hotspots(
    frame: np.ndarray,
    hotspots: list,
    color: tuple[int, int, int] = (0, 165, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Draw hotspot bounding boxes on a frame.

    Parameters
    ----------
    frame:
        Frame to annotate.  Accepts grayscale or 3-channel.
    hotspots:
        List of :class:`ThermalHotspot` objects.
    color:
        BGR colour for hotspot outlines.
    thickness:
        Line thickness.

    Returns
    -------
    np.ndarray
        Annotated copy (H×W×3, uint8).
    """
    if frame.ndim == 2:
        canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        canvas = frame.copy()

    if canvas.dtype != np.uint8:
        canvas = canvas.astype(np.uint8)

    for hotspot in hotspots:
        x1, y1, x2, y2 = hotspot.bbox
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(canvas, pt1, pt2, color, thickness)

        label = f"hotspot area={hotspot.area:.0f}"
        cv2.putText(
            canvas,
            label,
            (pt1[0], pt1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas


# ---------------------------------------------------------------------------
# Debug frame saving
# ---------------------------------------------------------------------------


def save_debug_frames(
    raw_frame: np.ndarray,
    normalized: np.ndarray,
    preprocessed: np.ndarray,
    detections: list[ThermalDetection],
    output_dir: str = "debug_thermal",
    frame_id: int = 0,
) -> None:
    """Save all pipeline stages for visual inspection.

    This is a **debugging utility** — disabled in production.  It writes
    intermediate pipeline outputs to disk so that thermal preprocessing
    effects can be inspected visually.

    Parameters
    ----------
    raw_frame:
        Original raw thermal frame.
    normalized:
        Normalized uint8 grayscale frame.
    preprocessed:
        Preprocessed 3-channel frame (YOLO input).
    detections:
        Detection results to draw on the annotated frame.
    output_dir:
        Directory to save debug images.
    frame_id:
        Numeric identifier for the frame (used in filenames).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prefix = f"frame_{frame_id:06d}"

    # Save raw (handle different dtypes)
    if raw_frame.ndim == 2:
        raw_save = raw_frame
        if raw_save.dtype != np.uint8:
            v_min, v_max = float(raw_save.min()), float(raw_save.max())
            if v_max > v_min:
                raw_save = (
                    (raw_save.astype(np.float64) - v_min)
                    / (v_max - v_min)
                    * 255
                ).astype(np.uint8)
            else:
                raw_save = np.full_like(raw_save, 128, dtype=np.uint8)
        cv2.imwrite(str(out / f"{prefix}_1_raw.png"), raw_save)
    else:
        cv2.imwrite(str(out / f"{prefix}_1_raw.png"), raw_frame)

    # Save normalized
    cv2.imwrite(str(out / f"{prefix}_2_normalized.png"), normalized)

    # Save preprocessed
    cv2.imwrite(str(out / f"{prefix}_3_preprocessed.png"), preprocessed)

    # Save annotated
    annotated = draw_thermal_detections(preprocessed, detections)
    cv2.imwrite(str(out / f"{prefix}_4_detections.png"), annotated)

    logger.debug("Debug frames saved to %s/%s_*", output_dir, prefix)
