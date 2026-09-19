"""
Core thermal detector.

Architecture
------------
::

    Thermal Frame (np.ndarray, any dtype)
          │
          ▼
    ThermalDetector.detect()         ← public API
          │
          ├─ ThermalPreprocessor     ← normalize, enhance, 3-channel convert
          │
          ├─ YOLO inference          ← model loaded ONCE in __init__
          │
          └─ ThermalPostProcessor    ← confidence filter, class filter, bbox clamp
                │
                ▼
          list[ThermalDetection]

The ``ThermalDetector`` class knows nothing about:
    cameras, RTSP, tracking, events, alerts, databases, APIs, blockchain.

It answers exactly one question:
    "What supported objects are present in this thermal frame and where?"

Model
-----
The default model is YOLO11n pretrained on COCO.  The architecture is
designed so that a thermal-fine-tuned model can be dropped in by changing
``model_path`` in the configuration, with zero code changes.

The preprocessing pipeline explicitly converts thermal frames to the
3-channel uint8 format that YOLO expects.  This is a deliberate format
conversion, not a claim that thermal and RGB are equivalent.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import BaseThermalDetector
from .config import ThermalDetectorConfig
from .exceptions import (
    ThermalInferenceError,
    ThermalInvalidFrameError,
    ThermalModelLoadError,
)
from .hotspot_segmentation import HotspotSegmenter, ThermalHotspot
from .models import ThermalDetection
from .postprocessing import ThermalPostProcessor
from .thermal_preprocessing import ThermalPreprocessor, validate_thermal_frame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to the best available device.

    Parameters
    ----------
    device:
        ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:N"``, or ``"mps"``.

    Returns
    -------
    str
        Concrete device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    """
    if device != "auto":
        return device

    try:
        import torch  # type: ignore[import]

        if torch.cuda.is_available():
            resolved = "cuda:0"
            logger.info(
                "Auto device selection: CUDA available → %s", resolved
            )
            return resolved
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            logger.info("Auto device selection: MPS available → mps")
            return "mps"
    except ImportError:
        pass  # torch not available; fall back to CPU

    logger.info("Auto device selection: no GPU detected → cpu")
    return "cpu"


# ---------------------------------------------------------------------------
# YOLO-based thermal detector
# ---------------------------------------------------------------------------


class YOLOThermalDetector(BaseThermalDetector):
    """YOLO-based thermal object detector.

    This class implements :class:`BaseThermalDetector` using Ultralytics YOLO.
    The model is loaded once during ``__init__`` and reused for every call
    to :meth:`detect`.

    Parameters
    ----------
    config:
        :class:`ThermalDetectorConfig` with all settings.

    Examples
    --------
    >>> from src.thermal_detection.config import ThermalDetectorConfig
    >>> config = ThermalDetectorConfig(model_path="yolo11n.pt")
    >>> detector = YOLOThermalDetector(config)
    >>> detections = detector.detect(thermal_frame)
    """

    def __init__(self, config: ThermalDetectorConfig) -> None:
        self._config = config
        self._resolved_device: str = _resolve_device(config.device)
        logger.info(
            "ThermalDetector using device: %s", self._resolved_device
        )

        # Load model once
        self._model: Any = self._load_model()
        self._class_names: list[str] = list(self._model.names.values())

        # Build pipeline components
        self._preprocessor = ThermalPreprocessor(config)
        self._post_processor = ThermalPostProcessor(
            confidence_threshold=config.confidence_threshold,
            allowed_classes=config.allowed_classes,
        )

        # Optional hotspot segmenter
        self._hotspot_segmenter: HotspotSegmenter | None = None
        if config.enable_hotspot_segmentation:
            self._hotspot_segmenter = HotspotSegmenter()

        logger.info(
            "ThermalDetector ready — model: %s | device: %s | "
            "conf: %.2f | iou: %.2f | classes: %s | "
            "CLAHE: %s | denoise: %s | hotspots: %s",
            config.model_path,
            self._resolved_device,
            config.confidence_threshold,
            config.iou_threshold,
            sorted(config.allowed_classes) if config.allowed_classes else "ALL",
            config.enable_clahe,
            config.enable_denoising,
            config.enable_hotspot_segmentation,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[ThermalDetection]:
        """Run thermal detection on a single frame.

        Parameters
        ----------
        frame:
            Thermal frame as a NumPy array.  Accepts:
            - (H, W) grayscale, dtype uint8/uint16/float32
            - (H, W, 1) single-channel
            - (H, W, 3) pre-converted

        Returns
        -------
        list[ThermalDetection]
            Filtered list of detections.  Empty list when nothing is found.
            Never returns ``None``.

        Raises
        ------
        ThermalInvalidFrameError
            If ``frame`` is ``None``, empty, wrong shape, or wrong dtype.
        ThermalInferenceError
            If the model raises an exception during inference.
        """
        # Validate raw input
        validate_thermal_frame(frame)

        # Preprocess: normalize + enhance + convert to 3-channel
        preprocessed, normalized_gray = self._preprocessor.preprocess(frame)

        h, w = preprocessed.shape[:2]

        # Run YOLO inference on the preprocessed 3-channel frame
        try:
            results = self._model.predict(
                source=preprocessed,
                imgsz=self._config.image_size,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                device=self._resolved_device,
                verbose=False,
            )
        except Exception as exc:
            raise ThermalInferenceError(
                f"YOLO inference failed on thermal frame "
                f"(shape={preprocessed.shape}): {exc}"
            ) from exc

        # Extract raw arrays
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        boxes_xyxy: np.ndarray = boxes.xyxy.cpu().numpy()
        confidences: np.ndarray = boxes.conf.cpu().numpy()
        class_ids: np.ndarray = boxes.cls.cpu().numpy().astype(int)

        # Post-process: filter, clamp, validate, construct ThermalDetection
        return self._post_processor.process(
            boxes_xyxy=boxes_xyxy,
            confidences=confidences,
            class_ids=class_ids,
            class_names=self._class_names,
            frame_height=h,
            frame_width=w,
        )

    def detect_with_hotspots(
        self, frame: np.ndarray
    ) -> tuple[list[ThermalDetection], list[ThermalHotspot]]:
        """Run detection and optional hotspot segmentation.

        Parameters
        ----------
        frame:
            Raw thermal frame.

        Returns
        -------
        tuple[list[ThermalDetection], list[ThermalHotspot]]
            ``(detections, hotspots)``.  ``hotspots`` is empty if hotspot
            segmentation is disabled.
        """
        validate_thermal_frame(frame)

        preprocessed, normalized_gray = self._preprocessor.preprocess(frame)

        # Detections
        h, w = preprocessed.shape[:2]
        try:
            results = self._model.predict(
                source=preprocessed,
                imgsz=self._config.image_size,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                device=self._resolved_device,
                verbose=False,
            )
        except Exception as exc:
            raise ThermalInferenceError(
                f"YOLO inference failed: {exc}"
            ) from exc

        detections: list[ThermalDetection] = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            detections = self._post_processor.process(
                boxes_xyxy=boxes.xyxy.cpu().numpy(),
                confidences=boxes.conf.cpu().numpy(),
                class_ids=boxes.cls.cpu().numpy().astype(int),
                class_names=self._class_names,
                frame_height=h,
                frame_width=w,
            )

        # Hotspots
        hotspots: list[ThermalHotspot] = []
        if self._hotspot_segmenter is not None:
            hotspots = self._hotspot_segmenter.segment(normalized_gray)

        return detections, hotspots

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ThermalDetectorConfig:
        """The configuration this detector was initialised with."""
        return self._config

    @property
    def device(self) -> str:
        """The resolved inference device."""
        return self._resolved_device

    @property
    def class_names(self) -> list[str]:
        """Full ordered class-name list from the loaded model."""
        return list(self._class_names)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Load the YOLO model from disk.  Called once during ``__init__``."""
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as exc:
            raise ThermalModelLoadError(
                "The 'ultralytics' package is not installed.  "
                "Run: pip install ultralytics"
            ) from exc

        try:
            model = YOLO(self._config.model_path)
            model.to(self._resolved_device)
            logger.info(
                "Thermal model loaded from %r on device %r.",
                self._config.model_path,
                self._resolved_device,
            )
            return model
        except FileNotFoundError as exc:
            raise ThermalModelLoadError(
                f"Model weights file not found: "
                f"{self._config.model_path!r}"
            ) from exc
        except Exception as exc:
            raise ThermalModelLoadError(
                f"Failed to load thermal model from "
                f"{self._config.model_path!r}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Convenience facade
# ---------------------------------------------------------------------------


class ThermalDetector(YOLOThermalDetector):
    """Convenience class — the default thermal detector.

    This is the recommended public entry point.  It is currently identical
    to :class:`YOLOThermalDetector` but exists so that downstream code
    imports ``ThermalDetector`` rather than a backend-specific class.

    If the backend changes in the future, only this class needs to be
    updated; all call-sites remain unchanged.

    Examples
    --------
    >>> from src.thermal_detection import ThermalDetector, ThermalDetectorConfig
    >>> config = ThermalDetectorConfig(
    ...     model_path="models/thermal_detector.pt",
    ...     confidence_threshold=0.25,
    ...     device="auto",
    ... )
    >>> detector = ThermalDetector(config)
    >>> detections = detector.detect(frame)
    >>> for d in detections:
    ...     print(f"{d.class_name} {d.confidence:.2f} at {d.bbox}")
    """

    pass
