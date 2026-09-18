"""
Detection data model.

This module defines the canonical ``Detection`` object that every component
in this system exchanges.  It is intentionally minimal: it contains only what
the *detection* step knows — class identity, confidence, and bounding-box
position.

Fields that belong to *other* modules (track_id, event_id, camera_id,
timestamp, alert_id) must **not** be added here.

Bounding-box format
-------------------
``bbox`` is a 4-tuple of floats::

    (x1, y1, x2, y2)

where

* ``x1`` – left edge   (pixels, relative to input frame)
* ``y1`` – top edge    (pixels, relative to input frame)
* ``x2`` – right edge  (pixels, relative to input frame)
* ``y2`` – bottom edge (pixels, relative to input frame)

The origin is the top-left corner of the frame, matching the OpenCV
convention.

Invariants that post-processing guarantees
------------------------------------------
* ``0.0 <= confidence <= 1.0``
* ``x1 < x2`` and ``y1 < y2``
* All coordinates are within ``[0, frame_dimension]``
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detection result.

    This is an immutable value object.  Equality and hashing are based on
    all fields, which makes it safe to use in sets and as dict keys.

    Parameters
    ----------
    class_id:
        Integer class index as defined by the model's class list (e.g. COCO
        class 0 = "person", 2 = "car").
    class_name:
        Human-readable class label (e.g. ``"person"``, ``"car"``).
    confidence:
        Detection confidence in ``[0.0, 1.0]``.
    bbox:
        Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates
        (top-left origin, matching OpenCV).

    Examples
    --------
    >>> d = Detection(
    ...     class_id=0,
    ...     class_name="person",
    ...     confidence=0.94,
    ...     bbox=(120.0, 80.0, 300.0, 500.0),
    ... )
    >>> d.bbox
    (120.0, 80.0, 300.0, 500.0)
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        """Validate field invariants at construction time."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        x1, y1, x2, y2 = self.bbox
        if x1 >= x2:
            raise ValueError(
                f"bbox requires x1 < x2, got x1={x1!r}, x2={x2!r}"
            )
        if y1 >= y2:
            raise ValueError(
                f"bbox requires y1 < y2, got y1={y1!r}, y2={y2!r}"
            )

    @property
    def width(self) -> float:
        """Width of the bounding box in pixels."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Height of the bounding box in pixels."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """Area of the bounding box in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Centre of the bounding box as ``(cx, cy)`` in pixels."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
