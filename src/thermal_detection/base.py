"""
Abstract base class for thermal object detectors.

Purpose
-------
Allow the thermal detection model to be swapped without changing
any downstream code.  The hierarchy is intentionally shallow::

    BaseThermalDetector (ABC)
        └── YOLOThermalDetector

A future custom thermal model (e.g. a thermal-fine-tuned detector) can
inherit from ``BaseThermalDetector`` and be used anywhere the base type
is accepted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .models import ThermalDetection


class BaseThermalDetector(ABC):
    """Abstract interface for all thermal detection backends.

    Subclasses must implement :meth:`detect`, which takes a single
    thermal frame and returns a list of :class:`ThermalDetection` objects.

    The contract is:

    * Accept a NumPy array representing a thermal frame (any dtype,
      grayscale or 3-channel).
    * Return ``list[ThermalDetection]``.  Never return ``None``.
    * Raise :class:`~.exceptions.ThermalInvalidFrameError` on bad input.
    * Raise :class:`~.exceptions.ThermalInferenceError` on model failure.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[ThermalDetection]:
        """Run thermal detection on a single frame.

        Parameters
        ----------
        frame:
            Thermal frame as a NumPy array.  May be grayscale
            (H×W or H×W×1) or 3-channel (H×W×3).  Supported dtypes:
            ``uint8``, ``uint16``, ``float32``.

        Returns
        -------
        list[ThermalDetection]
            Filtered list of detections.  Empty list when nothing is found.
        """
        raise NotImplementedError
