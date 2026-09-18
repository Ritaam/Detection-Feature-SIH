"""
Detection module public API.

Import the three names you need from here rather than from sub-modules::

    from src.detection import Detector, Detection, DetectorConfig

This keeps downstream consumers independent of the internal module layout.
"""

from .config import DetectorConfig
from .detector import Detector
from .exceptions import DetectionError, InferenceError, InvalidFrameError, ModelLoadError
from .models import Detection

__all__ = [
    "Detection",
    "DetectionError",
    "Detector",
    "DetectorConfig",
    "InferenceError",
    "InvalidFrameError",
    "ModelLoadError",
]
