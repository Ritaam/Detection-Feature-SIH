"""
night_detection — Thermal / night-time movement detection module.

This package processes thermal or low-light CCTV frames to detect human
movement during nighttime hours, independently of the visible-light detection
pipeline.

Sub-modules
-----------
thermal_preprocessing.py    : Noise reduction, normalisation, contrast enhancement
                              for thermal / IR frames.
hotspot_segmentation.py     : Segment heat-signature regions (blobs) from
                              preprocessed thermal frames.
thermal_detector.py         : Main detector class — wraps preprocessing +
                              segmentation + optional YOLO inference on thermal.
tracker.py                  : Lightweight single-camera tracker for thermal blobs.
movement_analysis.py        : Classify movement patterns (loitering, fast movement,
                              group formation).
alert_engine.py             : Generate alert records from movement analysis results.

Pipeline
--------
::

    Thermal / IR Frame  (np.ndarray)
              ↓
    thermal_preprocessing.py
              ↓
    hotspot_segmentation.py
              ↓
    thermal_detector.py  →  list[ThermalDetection]
              ↓
    tracker.py           →  list[TrackedThermalDetection]
              ↓
    movement_analysis.py →  list[MovementEvent]
              ↓
    alert_engine.py      →  list[NightAlert]

Status: NOT YET IMPLEMENTED.

Design rules
------------
* This module is self-contained — it does NOT depend on src.detection.
* ThermalDetection is a separate dataclass from Detection (different sensor modality).
* alert_engine produces alert records; it does NOT send them over the network.
  That is the responsibility of src.api_client.
"""

# NOT YET IMPLEMENTED
