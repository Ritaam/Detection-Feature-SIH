"""
anpr — Automatic Number Plate Recognition module.

Planned responsibilities
------------------------
* Receive vehicle ``Detection`` objects (car, motorcycle, bus, truck).
* Locate the licence-plate region within the vehicle bounding box.
* Run OCR to extract the plate text.
* Return a structured ``PlateReading`` result (plate text + confidence).

Pipeline position
-----------------
::

    Detector
        ↓
    list[Detection]  (vehicle classes only)
        ↓
    ANPR module
        ↓
    list[PlateReading]

Status: NOT YET IMPLEMENTED.

Sub-modules (planned)
---------------------
plate_detector.py   : Locate plate region (YOLO or dedicated model)
ocr.py              : Extract text from plate region
models.py           : PlateReading dataclass
config.py           : ANPRConfig dataclass

Dependencies (planned)
----------------------
* src.detection.models.Detection  (data model only — NOT the detector itself)
"""

# NOT YET IMPLEMENTED
