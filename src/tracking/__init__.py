"""
tracking — Multi-object tracking module.

Planned responsibilities
------------------------
* Consume ``list[Detection]`` from the detection module each frame.
* Assign and maintain stable track IDs across frames using BoT-SORT or ByteTrack.
* Return ``list[TrackedDetection]`` (Detection + track_id + trajectory metadata).
* Manage track lifecycle: new, active, lost, removed.

Pipeline position
-----------------
::

    Detector
        ↓
    list[Detection]
        ↓
    Tracker  (BoT-SORT / ByteTrack)
        ↓
    list[TrackedDetection]
        ↓
    ┌────────┬──────────┬──────────┐
    ANPR   Behaviour  Face       Intrusion

Status: NOT YET IMPLEMENTED.

Sub-modules (planned)
---------------------
tracker.py          : Main Tracker class wrapping BoT-SORT / ByteTrack
models.py           : TrackedDetection dataclass (extends Detection)
config.py           : TrackerConfig dataclass
kalman.py           : Kalman filter utilities (if implementing custom tracker)

Dependencies (planned)
----------------------
* src.detection.models.Detection  (consumed as input)
* boxmot or custom BoT-SORT implementation

This module MUST NOT modify src.detection in any way.
"""

# NOT YET IMPLEMENTED
