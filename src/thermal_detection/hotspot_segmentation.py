"""
Hotspot segmentation — segment heat-signature regions from thermal frames.

This module identifies connected warm regions ("hotspots") in a preprocessed
thermal frame using classical computer-vision techniques (thresholding,
morphology, contour detection).

Important
---------
A hotspot is **NOT** automatically a person or vehicle.  Hot regions may be:

* person
* vehicle
* engine / machinery
* environmental heat source (sun-warmed surface, exhaust, etc.)

Therefore, hotspot segmentation produces **candidate regions**, not
classified objects.  Object classification is the job of the detection
model.

This module is **optional** — the main thermal detector does not depend
on it.  It can be used independently to assist the detection pipeline
or for standalone thermal analysis.

Usage
-----
>>> from src.thermal_detection.hotspot_segmentation import (
...     HotspotSegmenter,
... )
>>> segmenter = HotspotSegmenter(intensity_threshold=180, min_area=200)
>>> hotspots = segmenter.segment(normalized_grayscale_frame)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ThermalHotspot:
    """A single segmented heat region.

    This is NOT a classified detection — it is a candidate region
    that may contain a person, vehicle, or any other heat source.

    Parameters
    ----------
    bbox:
        Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
    area:
        Area of the contour in square pixels (not the bounding-box area).
    mean_intensity:
        Mean pixel intensity within the hotspot bounding-box region (0–255).
    """

    bbox: tuple[float, float, float, float]
    area: float
    mean_intensity: float


class HotspotSegmenter:
    """Segment warm regions from a preprocessed thermal frame.

    Parameters
    ----------
    intensity_threshold:
        Pixel intensity above which regions are considered "warm".
        Range: 0–255.  Applied to a uint8 normalized thermal frame.
    min_area:
        Minimum contour area in pixels.  Contours smaller than this
        are discarded as noise.
    morphology_kernel_size:
        Size of the structuring element for morphological dilation,
        used to merge nearby warm pixels before contour detection.

    Examples
    --------
    >>> segmenter = HotspotSegmenter(intensity_threshold=180, min_area=200)
    >>> hotspots = segmenter.segment(normalized_frame)
    >>> for h in hotspots:
    ...     print(f"Hotspot at {h.bbox}, area={h.area:.0f}")
    """

    def __init__(
        self,
        intensity_threshold: int = 180,
        min_area: int = 200,
        morphology_kernel_size: int = 5,
    ) -> None:
        if not (0 <= intensity_threshold <= 255):
            raise ValueError(
                f"intensity_threshold must be in [0, 255], "
                f"got {intensity_threshold!r}"
            )
        if min_area < 0:
            raise ValueError(
                f"min_area must be non-negative, got {min_area!r}"
            )
        self._threshold = intensity_threshold
        self._min_area = min_area
        self._kernel_size = morphology_kernel_size

    def segment(self, frame: np.ndarray) -> list[ThermalHotspot]:
        """Detect warm regions in a normalized thermal frame.

        Parameters
        ----------
        frame:
            Normalized uint8 single-channel thermal frame (H×W).

        Returns
        -------
        list[ThermalHotspot]
            List of detected heat regions, sorted by area (largest first).
            Empty list if none found.

        Raises
        ------
        ValueError
            If the frame is not 2-D uint8.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy.ndarray.")
        if frame.ndim != 2:
            raise ValueError(
                f"frame must be 2-D (H×W), got ndim={frame.ndim}."
            )
        if frame.dtype != np.uint8:
            raise ValueError(
                f"frame must be uint8, got {frame.dtype!r}."
            )

        # Step 1: Binary threshold — isolate warm regions
        _, binary = cv2.threshold(
            frame, self._threshold, 255, cv2.THRESH_BINARY
        )

        # Step 2: Morphological dilation to merge nearby warm pixels
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._kernel_size, self._kernel_size),
        )
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # Step 3: Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Step 4: Filter by area and build ThermalHotspot list
        hotspots: list[ThermalHotspot] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            x1, y1 = float(x), float(y)
            x2, y2 = float(x + w), float(y + h)

            # Compute mean intensity within bounding box
            roi = frame[y : y + h, x : x + w]
            mean_val = float(roi.mean()) if roi.size > 0 else 0.0

            hotspots.append(
                ThermalHotspot(
                    bbox=(x1, y1, x2, y2),
                    area=float(area),
                    mean_intensity=mean_val,
                )
            )

        # Sort by area, largest first
        hotspots.sort(key=lambda h: h.area, reverse=True)

        logger.debug(
            "HotspotSegmenter found %d hotspot(s) "
            "(threshold=%d, min_area=%d).",
            len(hotspots),
            self._threshold,
            self._min_area,
        )
        return hotspots
