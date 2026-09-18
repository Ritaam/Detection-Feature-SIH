"""
hotspot_segmentation — Segment heat-signature regions from thermal frames.

After preprocessing, this module identifies connected regions ("hotspots")
that correspond to warm objects (humans, vehicles) in the thermal image.

Approach (planned)
------------------
1. Threshold the preprocessed frame to isolate warm regions.
2. Apply morphological operations (dilation, erosion) to merge nearby pixels.
3. Find contours / connected components.
4. Filter by minimum area to discard noise.
5. Return bounding boxes and mask for each hotspot.

Status: NOT YET IMPLEMENTED — stubs only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Hotspot:
    """A single segmented heat region.

    Parameters
    ----------
    bbox:
        Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
    area:
        Area of the hotspot in square pixels.
    mean_intensity:
        Mean pixel intensity within the hotspot region (0–255).
    """

    bbox: tuple[float, float, float, float]
    area: float
    mean_intensity: float


def segment_hotspots(
    frame: np.ndarray,
    temperature_threshold: int = 180,
    min_area: int = 200,
) -> list[Hotspot]:
    """Detect warm regions in a preprocessed thermal frame.

    Parameters
    ----------
    frame:
        Preprocessed uint8 thermal frame (H×W).
    temperature_threshold:
        Pixel intensity above which a region is considered "warm".
        Range: 0–255.
    min_area:
        Minimum contour area in pixels to be considered a valid hotspot.
        Smaller regions are discarded as noise.

    Returns
    -------
    list[Hotspot]
        List of detected heat regions.  Empty list if none found.

    Raises
    ------
    NotImplementedError
        Until this function is implemented.
    """
    raise NotImplementedError("hotspot_segmentation.segment_hotspots is not yet implemented.")
