"""
Thermal detection module exceptions.

All thermal-detector-specific errors inherit from ``ThermalDetectionError``
so callers can catch the entire family with a single
``except ThermalDetectionError`` or handle sub-types individually.
"""


class ThermalDetectionError(Exception):
    """Base class for all thermal-detection-module errors."""


class ThermalModelLoadError(ThermalDetectionError):
    """Raised when the thermal model cannot be loaded.

    Typical causes:
    - Model file not found at the given path.
    - Corrupted or incompatible ``.pt`` file.
    - Unsupported Ultralytics version.
    """


class ThermalInvalidFrameError(ThermalDetectionError):
    """Raised when a thermal input frame fails validation.

    Typical causes:
    - Frame is ``None``.
    - Frame is empty (zero-element shape).
    - Frame has unsupported number of dimensions.
    - Frame contains only NaN or Inf values.
    """


class ThermalInferenceError(ThermalDetectionError):
    """Raised when the thermal model fails during inference.

    Typical causes:
    - CUDA out-of-memory.
    - Internal Ultralytics runtime error.
    """


class ThermalPreprocessingError(ThermalDetectionError):
    """Raised when thermal preprocessing encounters an unrecoverable error.

    Typical causes:
    - Input dtype is completely unsupported (e.g. complex numbers).
    - Frame dimensions are incompatible with the pipeline.
    """
