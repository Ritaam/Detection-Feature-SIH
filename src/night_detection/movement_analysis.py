"""
movement_analysis — Classify movement patterns from tracked thermal detections.

Analyses the trajectory history of ``TrackedThermalDetection`` objects to
produce ``MovementEvent`` records for suspicious or noteworthy behaviours.

Planned patterns
----------------
* ``fast_movement``    : Speed above configured threshold (intruder running).
* ``loitering``        : Object stationary for more than N frames in an area.
* ``group_formation``  : Multiple tracks converging to same region.
* ``border_crossing``  : Track crosses a configured virtual line.

This module only CLASSIFIES movement — it does not generate alerts.
Alert generation is delegated to ``alert_engine.py``.

Status: NOT YET IMPLEMENTED — stubs only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .tracker import TrackedThermalDetection


class MovementPattern(Enum):
    """Classification of a detected movement pattern."""

    FAST_MOVEMENT = auto()
    LOITERING = auto()
    GROUP_FORMATION = auto()
    BORDER_CROSSING = auto()
    NORMAL = auto()


@dataclass(frozen=True, slots=True)
class MovementEvent:
    """A classified movement event for one or more tracks.

    Parameters
    ----------
    pattern:
        The detected movement pattern type.
    track_ids:
        IDs of the tracks involved in this event.
    frame_index:
        Frame number at which the event was detected.
    description:
        Human-readable summary of the event.
    """

    pattern: MovementPattern
    track_ids: tuple[int, ...]
    frame_index: int
    description: str


class MovementAnalyser:
    """Analyses tracked thermal detections for suspicious movement patterns.

    Parameters
    ----------
    loiter_threshold_frames:
        Number of frames an object must remain stationary to trigger loitering.
    speed_threshold_px_per_frame:
        Pixel displacement per frame above which movement is classified as fast.

    Status: NOT YET IMPLEMENTED.
    """

    def __init__(
        self,
        loiter_threshold_frames: int = 60,
        speed_threshold_px_per_frame: float = 15.0,
    ) -> None:
        self._loiter_threshold = loiter_threshold_frames
        self._speed_threshold = speed_threshold_px_per_frame

    def analyse(
        self,
        tracks: list[TrackedThermalDetection],
        frame_index: int,
    ) -> list[MovementEvent]:
        """Analyse current tracks and return movement events.

        Parameters
        ----------
        tracks:
            Active tracks from the current frame.
        frame_index:
            Current frame number (monotonically increasing).

        Returns
        -------
        list[MovementEvent]
            Detected events.  Empty list if all movement is normal.

        Raises
        ------
        NotImplementedError
            Until this method is implemented.
        """
        raise NotImplementedError("MovementAnalyser.analyse is not yet implemented.")
