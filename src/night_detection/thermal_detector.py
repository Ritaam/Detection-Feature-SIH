"""
thermal_detector — Main detector for night-time / thermal frames.

Wraps the full preprocessing + segmentation pipeline into a single class
with a clean ``detect(frame)`` interface, mirroring the visible-light
:class:`~src.detection.detector.Detector` API for consistency.

This class is intentionally separate from ``src.detection.Detector`` because:
  - Thermal frames use a different preprocessing pipeline.
  - ``ThermalDetection`` is a different data model (no COCO class IDs).
  - The model (if any) is thermal-specific.

Status: NOT YET IMPLEMENTED — stub only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ThermalDetection:
    """A single detection from the thermal pipeline.

    This is intentionally separate from ``src.detection.models.Detection``
    because thermal detections do not have COCO class IDs.

    Parameters
    ----------
    label:
        Human classification, e.g. ``"human_hotspot"``, ``"vehicle_hotspot"``.
    confidence:
        Confidence score in ``[0, 1]``.  May be 1.0 for threshold-based methods.
    bbox:
        Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
    mean_intensity:
        Mean thermal intensity within the bounding box (0–255).
    """

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    mean_intensity: float


class ThermalDetector:
    """Night-time object detector based on thermal / IR frame analysis.

    Parameters
    ----------
    temperature_threshold:
        Pixel intensity above which regions are considered warm (0–255).
    min_area:
        Minimum hotspot area in pixels.
    confidence_threshold:
        Minimum confidence to include a detection in the output.

    Status: NOT YET IMPLEMENTED.
    """

    def __init__(
        self,
        temperature_threshold: int = 180,
        min_area: int = 200,
        confidence_threshold: float = 0.5,
    ) -> None:
        self._temp_threshold = temperature_threshold
        self._min_area = min_area
        self._conf_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[ThermalDetection]:
        """Run thermal detection on a single frame.

        Parameters
        ----------
        frame:
            Raw thermal frame (any dtype, H×W or H×W×1).

        Returns
        -------
        list[ThermalDetection]
            List of detections.  Empty list if none found.

        Raises
        ------
        NotImplementedError
            Until this method is implemented.
        """
        raise NotImplementedError("ThermalDetector.detect is not yet implemented.")
