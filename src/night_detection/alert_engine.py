"""
alert_engine — Generate alert records from movement analysis events.

Consumes ``list[MovementEvent]`` and produces ``list[NightAlert]`` records.

Responsibilities
----------------
* Apply cooldown logic to avoid alert flooding (same pattern in same region).
* Assign alert severity based on movement pattern type.
* Return structured ``NightAlert`` records.

This module does NOT:
  - Send alerts over the network (use ``src.api_client`` for that).
  - Persist alerts to a database.
  - Make any I/O calls.

Status: NOT YET IMPLEMENTED — stubs only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .movement_analysis import MovementEvent, MovementPattern


class AlertSeverity(Enum):
    """Severity level for a night alert."""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


# Default severity mapping per movement pattern
_DEFAULT_SEVERITY: dict[MovementPattern, AlertSeverity] = {
    MovementPattern.FAST_MOVEMENT:   AlertSeverity.HIGH,
    MovementPattern.LOITERING:       AlertSeverity.MEDIUM,
    MovementPattern.GROUP_FORMATION: AlertSeverity.HIGH,
    MovementPattern.BORDER_CROSSING: AlertSeverity.CRITICAL,
    MovementPattern.NORMAL:          AlertSeverity.LOW,
}


@dataclass(frozen=True, slots=True)
class NightAlert:
    """An alert record generated from a night-time movement event.

    Parameters
    ----------
    severity:
        Severity level of the alert.
    pattern:
        The movement pattern that triggered this alert.
    track_ids:
        Track IDs of the objects involved.
    frame_index:
        Frame number when the alert was generated.
    message:
        Human-readable alert message.
    """

    severity: AlertSeverity
    pattern: MovementPattern
    track_ids: tuple[int, ...]
    frame_index: int
    message: str


class AlertEngine:
    """Generates ``NightAlert`` records from ``MovementEvent`` objects.

    Parameters
    ----------
    cooldown_frames:
        Minimum number of frames between repeated alerts for the same pattern.
        Prevents alert flooding.
    severity_overrides:
        Optional dict to override default severity per ``MovementPattern``.

    Status: NOT YET IMPLEMENTED.
    """

    def __init__(
        self,
        cooldown_frames: int = 30,
        severity_overrides: dict[MovementPattern, AlertSeverity] | None = None,
    ) -> None:
        self._cooldown_frames = cooldown_frames
        self._severity_map: dict[MovementPattern, AlertSeverity] = {
            **_DEFAULT_SEVERITY,
            **(severity_overrides or {}),
        }

    def process_events(
        self,
        events: list[MovementEvent],
        frame_index: int,
    ) -> list[NightAlert]:
        """Convert movement events into alert records.

        Parameters
        ----------
        events:
            Output of ``MovementAnalyser.analyse()`` for the current frame.
        frame_index:
            Current frame number (used for cooldown tracking).

        Returns
        -------
        list[NightAlert]
            Generated alerts.  Empty list if no events warrant an alert.

        Raises
        ------
        NotImplementedError
            Until this method is implemented.
        """
        raise NotImplementedError("AlertEngine.process_events is not yet implemented.")
