"""
Detection module exceptions.

All detector-specific errors inherit from ``DetectionError`` so callers can
catch the entire detection error family with a single ``except DetectionError``
or handle sub-types individually for finer-grained recovery.
"""


class DetectionError(Exception):
    """Base class for all detection-module errors."""


class ModelLoadError(DetectionError):
    """Raised when the YOLO model cannot be loaded.

    Typical causes:
    - Model file not found at the given path.
    - Corrupted or incompatible ``.pt`` file.
    - Unsupported Ultralytics version.
    """


class InvalidFrameError(DetectionError):
    """Raised when the input frame fails validation.

    Typical causes:
    - Frame is ``None``.
    - Frame is an empty NumPy array (zero-element shape).
    - Frame has wrong number of dimensions (expected H×W×3).
    - Frame has an unsupported ``dtype`` (expected ``uint8``).
    """


class InferenceError(DetectionError):
    """Raised when the model fails during inference.

    Typical causes:
    - CUDA out-of-memory.
    - Internal Ultralytics runtime error.
    """
