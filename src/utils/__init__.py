"""
utils — Shared utilities used across all src sub-packages.

Planned responsibilities
------------------------
* Logging configuration helpers.
* Image/frame utility functions (resize, BGR↔RGB, etc.).
* Bounding-box utility functions (IoU, area, overlap).
* Timing / profiling decorators.
* Serialisation helpers (Detection → dict / JSON).

Status: NOT YET IMPLEMENTED.

Sub-modules (planned)
---------------------
logging.py          : Project-wide logger factory
image.py            : Frame manipulation helpers
bbox.py             : Bounding-box geometry utilities
serialise.py        : Convert dataclass objects to JSON-serialisable dicts
timer.py            : Performance timing helpers

Rules
-----
* utils must NOT import from any other src sub-package to avoid circular deps.
* All functions here must be pure and side-effect free where possible.
"""

# NOT YET IMPLEMENTED
