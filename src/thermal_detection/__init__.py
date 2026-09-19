"""
thermal_detection — Thermal / IR object detection engine.

This package processes thermal or infrared camera frames to detect objects
(persons, vehicles) visible in thermal imagery.  It is self-contained and
independent of the visible-light detection pipeline in ``src.detection``.

Public API
----------
::

    from src.thermal_detection import (
        ThermalDetector,
        ThermalDetection,
        ThermalDetectorConfig,
        BaseThermalDetector,
    )

    config = ThermalDetectorConfig(
        model_path="models/thermal_detector.pt",
        confidence_threshold=0.25,
        device="auto",
    )

    detector = ThermalDetector(config)
    detections = detector.detect(thermal_frame)

Sub-modules
-----------
models.py               : ThermalDetection dataclass
config.py               : ThermalDetectorConfig dataclass
exceptions.py           : Exception hierarchy
base.py                 : BaseThermalDetector abstract base class
thermal_preprocessing.py: Thermal normalization, CLAHE, denoising,
                          grayscale → 3-channel conversion
hotspot_segmentation.py : Optional hotspot region segmentation
postprocessing.py       : Confidence/class filtering, bbox validation
thermal_detector.py     : YOLOThermalDetector / ThermalDetector
visualization.py        : Optional detection/hotspot drawing utilities

Pipeline
--------
::

    Thermal Frame (any dtype, grayscale or single-channel)
          │
          ▼
    Frame Validation
          │
          ▼
    Thermal Preprocessing
    ├── Intensity Normalization (uint8/uint16/float32 → uint8)
    ├── Optional CLAHE Contrast Enhancement
    ├── Optional Gaussian Denoising
    └── Grayscale → 3-Channel BGR Conversion
          │
          ▼
    YOLO Object Detection
          │
          ▼
    Post-Processing
    ├── Confidence Filtering
    ├── Class Filtering
    └── BBox Clamping & Validation
          │
          ▼
    list[ThermalDetection]

Design rules
------------
* This module is self-contained — it does NOT depend on ``src.detection``.
* ``ThermalDetection`` is a separate dataclass from ``Detection``
  (different sensor modality, optional thermal metadata).
* The model is loaded ONCE.  ``detect()`` is called per-frame.
* Temperature metadata is NEVER fabricated from grayscale intensity.
"""

from .base import BaseThermalDetector
from .config import ThermalDetectorConfig
from .models import ThermalDetection
from .thermal_detector import ThermalDetector, YOLOThermalDetector

__all__ = [
    "BaseThermalDetector",
    "ThermalDetection",
    "ThermalDetector",
    "ThermalDetectorConfig",
    "YOLOThermalDetector",
]
