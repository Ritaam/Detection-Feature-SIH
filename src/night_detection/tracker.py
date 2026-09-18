"""
tracker — Lightweight single-camera tracker for thermal detections.

Tracks ``ThermalDetection`` objects across frames and assigns stable IDs.
Uses a simple IoU-based association (or Kalman filter) — no appearance
embedding needed for thermal modality.

This is intentionally separate from ``src.tracking`` (which tracks
visible-light detections with BoT-SORT / ByteTrack) because:
  - Thermal blobs have no appearance features for ReID.
  - The data model is different (``ThermalDetection`` vs ``Detection``).

Status: NOT YET IMPLEMENTED — stub only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .thermal_detector import ThermalDetection


@dataclass(frozen=True, slots=True)
class TrackedThermalDetection:
    """A ``ThermalDetection`` with an assigned track ID.

    Parameters
    ----------
    track_id:
        Unique integer ID maintained across frames for this object.
    detection:
        The underlying ``ThermalDetection`` for this frame.
    frames_seen:
        Number of consecutive frames this track has been active.
    """

    track_id: int
    detection: ThermalDetection
    frames_seen: int


class ThermalTracker:
    """Associates thermal detections across frames using IoU matching.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU to consider two detections the same object.
    max_lost_frames:
        Number of frames a track can be missing before it is removed.

    Status: NOT YET IMPLEMENTED.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 5,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames

    def update(
        self,
        detections: list[ThermalDetection],
    ) -> list[TrackedThermalDetection]:
        """Update tracker state with detections from the current frame.

        Parameters
        ----------
        detections:
            Output of ``ThermalDetector.detect()`` for the current frame.

        Returns
        -------
        list[TrackedThermalDetection]
            Active tracks with stable IDs.

        Raises
        ------
        NotImplementedError
            Until this method is implemented.
        """
        raise NotImplementedError("ThermalTracker.update is not yet implemented.")
