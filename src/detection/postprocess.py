"""
Post-processing for raw YOLO detections.

This module is the only place in the detection pipeline that:

* filters by confidence threshold
* filters by supported class names
* clamps bounding boxes to frame boundaries
* validates bounding-box geometry (x1 < x2, y1 < y2)
* converts raw YOLO output into :class:`~src.detection.models.Detection`

It is intentionally free of model-specific imports so it can be tested
independently with mock data.

Coordinate convention
---------------------
All bounding boxes are expected and returned in absolute pixel coordinates::

    (x1, y1, x2, y2)

where (0, 0) is the top-left corner of the frame.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from .models import Detection

logger = logging.getLogger(__name__)


class PostProcessor:
    """Converts and filters raw YOLO result boxes into :class:`Detection` objects.

    Parameters
    ----------
    confidence_threshold:
        Minimum confidence ``[0, 1]`` to keep a detection.
    supported_classes:
        Set of class name strings to keep.  Any detection whose
        ``class_name`` is not in this set is silently dropped.

    Notes
    -----
    This class holds no per-frame state; a single instance can safely process
    frames sequentially or be shared across threads if the call-site handles
    its own synchronisation.
    """

    def __init__(
        self,
        confidence_threshold: float,
        supported_classes: frozenset[str],
    ) -> None:
        self._conf_threshold = confidence_threshold
        self._supported_classes = supported_classes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        class_names: list[str],
        frame_height: int,
        frame_width: int,
    ) -> list[Detection]:
        """Convert raw YOLO arrays into a filtered list of :class:`Detection`.

        Parameters
        ----------
        boxes_xyxy:
            Float array of shape ``(N, 4)`` with columns ``[x1, y1, x2, y2]``
            in absolute pixel coordinates.
        confidences:
            Float array of shape ``(N,)`` with confidence scores in ``[0, 1]``.
        class_ids:
            Integer array of shape ``(N,)`` with model class indices.
        class_names:
            Ordered list that maps model class index → class name string.
            e.g. ``["person", "bicycle", "car", ...]`` (COCO order).
        frame_height:
            Height of the source frame in pixels (used for clamping).
        frame_width:
            Width of the source frame in pixels (used for clamping).

        Returns
        -------
        list[Detection]
            Filtered, validated detections.  May be empty.
        """
        if len(boxes_xyxy) == 0:
            return []

        detections: list[Detection] = []

        for box, conf, cls_id in zip(boxes_xyxy, confidences, class_ids):
            conf_float = float(conf)
            cls_int = int(cls_id)

            # ── confidence filter ───────────────────────────────────────────
            if conf_float < self._conf_threshold:
                logger.debug("Skipping detection: confidence %.3f < %.3f", conf_float, self._conf_threshold)
                continue

            # ── class name lookup ───────────────────────────────────────────
            if cls_int < 0 or cls_int >= len(class_names):
                logger.warning("Class id %d out of range for class list (len=%d); skipping.", cls_int, len(class_names))
                continue

            cls_name = class_names[cls_int]

            # ── animal class override ──────────────────────────────────
            animal_classes = {
                "bird", "cat", "dog", "horse", "sheep", "cow", 
                "elephant", "bear", "zebra", "giraffe"
            }
            if cls_name in animal_classes:
                cls_name = "animal"

            # ── supported-class filter ──────────────────────────────────────
            if cls_name not in self._supported_classes:
                logger.debug("Skipping unsupported class: %r", cls_name)
                continue

            # ── bbox clamping & validation ──────────────────────────────────
            bbox = self._clamp_and_validate_bbox(
                box, frame_width, frame_height
            )
            if bbox is None:
                continue

            detections.append(
                Detection(
                    class_id=cls_int,
                    class_name=cls_name,
                    confidence=round(conf_float, 6),
                    bbox=bbox,
                )
            )

        logger.debug("PostProcessor produced %d detection(s).", len(detections))
        return detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clamp_and_validate_bbox(
        self,
        box: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float, float, float] | None:
        """Clamp bbox coordinates to frame bounds and validate geometry.

        Parameters
        ----------
        box:
            1-D array ``[x1, y1, x2, y2]``.
        frame_width:
            Width of the frame in pixels.
        frame_height:
            Height of the frame in pixels.

        Returns
        -------
        tuple or None
            Clamped ``(x1, y1, x2, y2)`` floats, or ``None`` if the box is
            degenerate after clamping.
        """
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

        # Clamp to frame boundaries
        x1 = max(0.0, min(x1, float(frame_width)))
        y1 = max(0.0, min(y1, float(frame_height)))
        x2 = max(0.0, min(x2, float(frame_width)))
        y2 = max(0.0, min(y2, float(frame_height)))

        # Validate geometry after clamping
        if x1 >= x2 or y1 >= y2:
            logger.warning(
                "Degenerate bbox after clamping: (%.1f, %.1f, %.1f, %.1f); skipping.",
                x1, y1, x2, y2,
            )
            return None

        return (x1, y1, x2, y2)
