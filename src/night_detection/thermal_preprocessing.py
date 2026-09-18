"""
thermal_preprocessing — Preprocessing pipeline for thermal / IR frames.

Thermal camera frames require different preprocessing compared to visible-light
frames:
  - CLAHE contrast enhancement to improve temperature gradient visibility
  - Gaussian / bilateral smoothing to reduce sensor noise
  - Normalisation to a consistent intensity range

All functions operate on NumPy arrays and are side-effect free.
No model is loaded in this module.

Status: NOT YET IMPLEMENTED — stubs only.
"""

from __future__ import annotations

import numpy as np


def normalise(frame: np.ndarray) -> np.ndarray:
    """Normalise a thermal frame to the range [0, 255] as uint8.

    Parameters
    ----------
    frame:
        Raw thermal frame, any numeric dtype.

    Returns
    -------
    np.ndarray
        Normalised uint8 frame, same H×W shape.

    Raises
    ------
    NotImplementedError
        Until this function is implemented.
    """
    raise NotImplementedError("thermal_preprocessing.normalise is not yet implemented.")


def enhance_contrast(frame: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE contrast enhancement to a thermal frame.

    Parameters
    ----------
    frame:
        Normalised uint8 thermal frame (H×W or H×W×1).
    clip_limit:
        CLAHE clip limit.  Higher values produce stronger contrast.
    tile_grid:
        CLAHE tile grid size.

    Returns
    -------
    np.ndarray
        Contrast-enhanced uint8 frame.

    Raises
    ------
    NotImplementedError
        Until this function is implemented.
    """
    raise NotImplementedError("thermal_preprocessing.enhance_contrast is not yet implemented.")


def denoise(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to reduce thermal sensor noise.

    Parameters
    ----------
    frame:
        uint8 thermal frame.
    kernel_size:
        Gaussian kernel size (must be odd).

    Returns
    -------
    np.ndarray
        Denoised uint8 frame.

    Raises
    ------
    NotImplementedError
        Until this function is implemented.
    """
    raise NotImplementedError("thermal_preprocessing.denoise is not yet implemented.")


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline: normalise → denoise → enhance contrast.

    Parameters
    ----------
    frame:
        Raw thermal frame (any dtype).

    Returns
    -------
    np.ndarray
        Preprocessed uint8 frame, ready for segmentation.

    Raises
    ------
    NotImplementedError
        Until this function is implemented.
    """
    raise NotImplementedError("thermal_preprocessing.preprocess is not yet implemented.")
