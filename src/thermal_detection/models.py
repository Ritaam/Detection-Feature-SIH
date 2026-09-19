"""
Thermal detection data model.

This module defines the canonical ``ThermalDetection`` object that the
thermal detection engine produces.  It is intentionally minimal: it
contains only what the *thermal detection* step knows — class identity,
confidence, bounding-box position, and optional thermal metadata.

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

Temperature metadata
--------------------
``mean_intensity`` and ``temperature_range`` are **optional**.

* ``mean_intensity`` is the mean raw pixel intensity within the bounding box
  of the *normalized* (uint8) thermal frame.  It is NOT a temperature.
* ``temperature_range`` is only populated when the input data genuinely
  provides calibrated radiometric temperature information.

**Do NOT fabricate temperature values from ordinary grayscale intensity.**
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ThermalDetection:
    """A single thermal object detection result.

    This is an immutable value object.  Equality and hashing are based on
    all fields, which makes it safe to use in sets and as dict keys.

    Parameters
    ----------
    bbox:
        Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates
        (top-left origin, matching OpenCV).
    confidence:
        Detection confidence in ``[0.0, 1.0]``.
    class_id:
        Integer class index as defined by the model's class list.
    class_name:
        Human-readable class label (e.g. ``"person"``, ``"car"``).
    mean_intensity:
        Mean normalized pixel intensity within the bounding box (0–255).
        ``None`` if not computed.  This is NOT a temperature value.
    temperature_range:
        ``(min_temp, max_temp)`` in degrees Celsius within the bounding box.
        ``None`` unless the input data provides calibrated radiometric data.

    Examples
    --------
    >>> d = ThermalDetection(
    ...     bbox=(120.0, 80.0, 300.0, 500.0),
    ...     confidence=0.94,
    ...     class_id=0,
    ...     class_name="person",
    ... )
    >>> d.bbox
    (120.0, 80.0, 300.0, 500.0)
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    mean_intensity: float | None = None
    temperature_range: tuple[float, float] | None = None

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
        if self.temperature_range is not None:
            t_min, t_max = self.temperature_range
            if t_min > t_max:
                raise ValueError(
                    f"temperature_range requires min <= max, "
                    f"got ({t_min!r}, {t_max!r})"
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
