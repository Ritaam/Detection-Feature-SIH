"""
analytics — Behaviour and activity analysis module.

Planned responsibilities
------------------------
* Consume ``list[TrackedDetection]`` from the tracking module.
* Detect loitering, crowd formation, direction changes.
* Flag suspicious activity patterns.
* Feed event records to the event-generation layer.

Status: NOT YET IMPLEMENTED.

Dependencies (planned)
----------------------
* src.tracking  → list[TrackedDetection]
* src.detection → Detection  (data model only)

This module must NOT depend on:
* src.detection.detector  (model inference)
* FastAPI / WebSocket
* Database
* Blockchain
"""

# NOT YET IMPLEMENTED
