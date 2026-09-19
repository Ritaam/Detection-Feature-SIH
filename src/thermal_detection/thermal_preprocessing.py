"""
Thermal frame preprocessing pipeline.

This module converts raw thermal/IR frames into a format suitable for
YOLO inference.  It is the **only** place in the thermal detection pipeline
that performs intensity normalization and format conversion.

Thermal frames differ fundamentally from RGB images:

* They may be single-channel (grayscale) with dtype uint8, uint16, or float32.
* Intensity represents thermal radiation, not visible-light colour.
* The useful dynamic range may be a small fraction of the dtype range.
* Sensor noise is different from visible-light camera noise.

Preprocessing steps
-------------------
1. **Validation** — reject None, empty, invalid-dimension frames.
2. **Ensure 2-D** — squeeze H×W×1 to H×W.
3. **Sanitize** — replace NaN/Inf with 0 in floating-point frames.
4. **Normalize** — map the frame's actual value range to uint8 [0, 255].
5. **Optional CLAHE** — contrast enhancement on the single-channel image.
6. **Optional denoising** — Gaussian blur to reduce sensor noise.
7. **3-channel conversion** — ``cv2.cvtColor(gray, COLOR_GRAY2BGR)``
   so YOLO receives the expected (H, W, 3) uint8 input.

The grayscale → 3-channel conversion is **explicit and deliberate**.
It does NOT mean thermal and RGB images are equivalent.

All functions are side-effect free: they never modify the input array.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import ThermalDetectorConfig
from .exceptions import ThermalInvalidFrameError, ThermalPreprocessingError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported dtypes for thermal frames
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES = (np.uint8, np.uint16, np.float32, np.float64)


# ---------------------------------------------------------------------------
# Standalone preprocessing functions
# ---------------------------------------------------------------------------


def validate_thermal_frame(frame: np.ndarray | None) -> None:
    """Raise :class:`ThermalInvalidFrameError` if *frame* is not usable.

    Accepts:
    - 2-D arrays (H×W) — grayscale thermal
    - 3-D arrays (H×W×1) — single-channel thermal
    - 3-D arrays (H×W×3) — pre-converted or pseudo-colour thermal

    Parameters
    ----------
    frame:
        Value to validate.

    Raises
    ------
    ThermalInvalidFrameError
        With a descriptive message explaining which invariant was violated.
    """
    if frame is None:
        raise ThermalInvalidFrameError(
            "Frame is None; expected a NumPy ndarray."
        )

    if not isinstance(frame, np.ndarray):
        raise ThermalInvalidFrameError(
            f"Frame must be a numpy.ndarray, got {type(frame).__name__!r}."
        )

    if frame.size == 0:
        raise ThermalInvalidFrameError(
            f"Frame is empty (size=0, shape={frame.shape})."
        )

    if frame.ndim not in (2, 3):
        raise ThermalInvalidFrameError(
            f"Frame must be 2-D (H×W) or 3-D (H×W×C), "
            f"got ndim={frame.ndim} (shape={frame.shape})."
        )

    if frame.ndim == 3 and frame.shape[2] not in (1, 3):
        raise ThermalInvalidFrameError(
            f"Frame with 3 dimensions must have 1 or 3 channels, "
            f"got {frame.shape[2]} channels (shape={frame.shape})."
        )

    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ThermalInvalidFrameError(
            f"Frame has zero height or width (shape={frame.shape})."
        )

    if frame.dtype.type not in _SUPPORTED_DTYPES:
        raise ThermalInvalidFrameError(
            f"Frame dtype must be one of uint8, uint16, float32, float64; "
            f"got {frame.dtype!r}."
        )


def normalize_thermal(frame: np.ndarray) -> np.ndarray:
    """Normalize a thermal frame to uint8 [0, 255].

    Handles:

    * **uint8** — returned as-is (after NaN/Inf sanitization if float-like).
    * **uint16** — min-max scaled to [0, 255].
    * **float32 / float64** — finite values are min-max scaled to [0, 255].
      NaN and Inf are replaced with 0 before scaling.
    * **Constant frame** — returned as mid-gray (128) to avoid division by
      zero.  A warning is logged.

    Parameters
    ----------
    frame:
        Single-channel thermal frame (H×W), any supported numeric dtype.

    Returns
    -------
    np.ndarray
        Normalized uint8 frame with the same (H, W) shape.
    """
    # Work on a copy to avoid side effects
    working = frame.copy()

    # --- Sanitize floating-point values ---
    if np.issubdtype(working.dtype, np.floating):
        nan_mask = np.isnan(working)
        inf_mask = np.isinf(working)
        bad_mask = nan_mask | inf_mask
        bad_count = int(np.count_nonzero(bad_mask))
        if bad_count > 0:
            logger.warning(
                "Thermal frame contains %d NaN/Inf pixels; replacing with 0.",
                bad_count,
            )
            working[bad_mask] = 0.0

    # --- Already uint8 → return directly ---
    if working.dtype == np.uint8:
        return working

    # --- Min-max normalization to uint8 ---
    v_min = float(working.min())
    v_max = float(working.max())

    if v_min == v_max:
        logger.warning(
            "Thermal frame has constant value %.4f; returning mid-gray (128).",
            v_min,
        )
        return np.full(working.shape, 128, dtype=np.uint8)

    normalized = (working.astype(np.float64) - v_min) / (v_max - v_min)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def apply_clahe(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE contrast enhancement to a single-channel uint8 frame.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) improves the
    visibility of temperature gradients in thermal images by enhancing
    local contrast without over-amplifying noise.

    Parameters
    ----------
    frame:
        uint8 grayscale frame (H×W).
    clip_limit:
        CLAHE clip limit.  Higher values produce stronger contrast.
    tile_grid:
        CLAHE tile grid size as ``(rows, cols)``.

    Returns
    -------
    np.ndarray
        Contrast-enhanced uint8 frame with the same shape.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=tile_grid
    )
    return clahe.apply(frame)


def apply_denoising(
    frame: np.ndarray, kernel_size: int = 5
) -> np.ndarray:
    """Apply Gaussian blur to reduce thermal sensor noise.

    Parameters
    ----------
    frame:
        uint8 grayscale frame (H×W).
    kernel_size:
        Gaussian kernel size.  Must be odd and positive.

    Returns
    -------
    np.ndarray
        Denoised uint8 frame with the same shape.
    """
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def to_3channel(frame: np.ndarray) -> np.ndarray:
    """Convert a single-channel grayscale frame to 3-channel BGR.

    This conversion is required because YOLO expects (H, W, 3) input.
    It is **explicitly not** a claim that thermal and RGB images are
    equivalent — it is a format conversion for model compatibility.

    Parameters
    ----------
    frame:
        uint8 grayscale frame (H×W).

    Returns
    -------
    np.ndarray
        uint8 BGR frame (H×W×3) where all three channels are identical.
    """
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def _ensure_2d(frame: np.ndarray) -> np.ndarray:
    """Squeeze H×W×1 to H×W; pass through H×W and H×W×3 unchanged.

    If the frame is already 3-channel (H×W×3), convert to single-channel
    grayscale.
    """
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    if frame.ndim == 3 and frame.shape[2] == 3:
        # 3-channel input → convert to grayscale for thermal processing
        if frame.dtype == np.uint8:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # For non-uint8 3-channel, take the mean across channels
        return frame.mean(axis=2).astype(frame.dtype)
    return frame


# ---------------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------------


class ThermalPreprocessor:
    """Configurable thermal frame preprocessing pipeline.

    Produces a 3-channel uint8 frame suitable for YOLO inference, and
    optionally retains intermediate results for debugging.

    Parameters
    ----------
    config:
        :class:`~.config.ThermalDetectorConfig` with preprocessing settings.

    Examples
    --------
    >>> from src.thermal_detection.config import ThermalDetectorConfig
    >>> cfg = ThermalDetectorConfig(model_path="yolo11n.pt")
    >>> preprocessor = ThermalPreprocessor(cfg)
    >>> preprocessed, normalized_gray = preprocessor.preprocess(raw_frame)
    """

    def __init__(self, config: ThermalDetectorConfig) -> None:
        self._config = config

    def preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the full preprocessing pipeline.

        Parameters
        ----------
        frame:
            Raw thermal frame.  Accepts:
            - (H, W) grayscale, any supported dtype
            - (H, W, 1) single-channel
            - (H, W, 3) pre-converted thermal

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(preprocessed_3ch, normalized_1ch)`` where:

            * ``preprocessed_3ch`` is uint8 (H, W, 3), ready for YOLO.
            * ``normalized_1ch`` is uint8 (H, W), the normalized grayscale
              before 3-channel conversion.  Useful for debugging and for
              hotspot segmentation.

        Raises
        ------
        ThermalInvalidFrameError
            If the frame is invalid.
        ThermalPreprocessingError
            If an unrecoverable error occurs during preprocessing.
        """
        validate_thermal_frame(frame)

        try:
            # Step 1: Ensure single-channel
            gray = _ensure_2d(frame)

            # Step 2: Normalize to uint8
            normalized = normalize_thermal(gray)

            # Step 3: Optional CLAHE
            enhanced = normalized
            if self._config.enable_clahe:
                enhanced = apply_clahe(
                    enhanced,
                    clip_limit=self._config.clahe_clip_limit,
                    tile_grid=self._config.clahe_tile_grid,
                )

            # Step 4: Optional denoising
            if self._config.enable_denoising:
                enhanced = apply_denoising(
                    enhanced,
                    kernel_size=self._config.denoise_kernel_size,
                )

            # Step 5: Convert to 3-channel BGR for YOLO
            preprocessed = to_3channel(enhanced)

            return preprocessed, normalized

        except (ThermalInvalidFrameError, ThermalPreprocessingError):
            raise
        except Exception as exc:
            raise ThermalPreprocessingError(
                f"Thermal preprocessing failed: {exc}"
            ) from exc
