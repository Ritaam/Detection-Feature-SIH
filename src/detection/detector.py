"""
Core detector class.

Architecture
------------
::

    CCTV Frame (np.ndarray)
          │
          ▼
    Detector.detect()          ← public API
          │
          ├─ validate_frame()  ← raises InvalidFrameError on bad input
          │
          ├─ _resolve_device() ← "auto" → "cuda" if available, else "cpu"
          │
          ├─ YOLO11n inference ← model loaded ONCE in __init__
          │
          └─ PostProcessor     ← confidence filter, class filter, bbox clamp
                │
                ▼
          list[Detection]      ← handed to BoT-SORT / downstream modules

The ``Detector`` class knows nothing about:
    cameras, RTSP, tracking, events, alerts, databases, APIs, blockchain.

It answers exactly one question:
    "What supported objects are present in this frame and where?"

Model
-----
YOLO11n pretrained on COCO (Ultralytics).  Weights are downloaded
automatically by the Ultralytics library on first use when
``model_path="yolo11n.pt"`` is passed.  To bundle weights with the project,
download the file and point ``model_path`` to it.

COCO classes used::

    0  person
    2  car
    3  motorcycle
    5  bus
    7  truck
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import DetectorConfig, DEFAULT_SUPPORTED_CLASSES
from .exceptions import InferenceError, InvalidFrameError, ModelLoadError
from .models import Detection
from .postprocess import PostProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frame validation constants
# ---------------------------------------------------------------------------

_EXPECTED_NDIM = 3          # H × W × C
_EXPECTED_CHANNELS = 3      # BGR (OpenCV convention)
_SUPPORTED_DTYPES = (np.uint8,)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class Detector:
    """YOLO11n-based object detector for border-surveillance CCTV frames.

    Load the model once, then call :meth:`detect` for every frame.

    Parameters
    ----------
    model_path:
        Path to YOLO ``.pt`` weights.  Use ``"yolo11n.pt"`` to auto-download
        the official pretrained weights on first use.
    confidence_threshold:
        Minimum confidence ``[0, 1]`` to retain a detection.  Default 0.5.
    iou_threshold:
        NMS IoU threshold passed to YOLO.  Default 0.45.
    device:
        Inference device: ``"auto"`` (default), ``"cpu"``, ``"cuda"``,
        ``"cuda:N"``, or ``"mps"``.

    Examples
    --------
    >>> import cv2
    >>> from src.detection.detector import Detector
    >>>
    >>> detector = Detector(model_path="yolo11n.pt")
    >>> frame = cv2.imread("test.jpg")
    >>> detections = detector.detect(frame)
    >>> for d in detections:
    ...     print(d)
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        supported_classes: frozenset[str] | None = None,
        imgsz: int = 640,
    ) -> None:
        if supported_classes is None:
            supported_classes = DEFAULT_SUPPORTED_CLASSES

        self._config = DetectorConfig(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device,
            supported_classes=supported_classes,
            imgsz=imgsz,
        )

        self._resolved_device: str = _resolve_device(self._config.device)
        logger.info("Detector using device: %s", self._resolved_device)

        self._model: Any = self._load_model()
        self._class_names: list[str] = list(self._model.names.values())

        self._post_processor = PostProcessor(
            confidence_threshold=self._config.confidence_threshold,
            supported_classes=self._config.supported_classes,
        )

        logger.info(
            "Detector ready — model: %s | device: %s | conf: %.2f | iou: %.2f | classes: %s",
            self._config.model_path,
            self._resolved_device,
            self._config.confidence_threshold,
            self._config.iou_threshold,
            sorted(self._config.supported_classes),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame.

        Parameters
        ----------
        frame:
            OpenCV/NumPy image, shape ``(H, W, 3)``, dtype ``uint8``,
            BGR channel order.

        Returns
        -------
        list[Detection]
            Filtered list of :class:`~src.detection.models.Detection` objects.
            Returns an empty list when no supported objects are detected.
            Never returns ``None``.

        Raises
        ------
        InvalidFrameError
            If ``frame`` is ``None``, empty, wrong shape, or wrong dtype.
        InferenceError
            If the model raises an exception during inference.
        """
        _validate_frame(frame)

        h, w = frame.shape[:2]

        try:
            results = self._model.predict(
                source=frame,
                imgsz=self._config.imgsz,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                device=self._resolved_device,
                verbose=False,
            )
        except Exception as exc:
            raise InferenceError(
                f"YOLO inference failed on frame (shape={frame.shape}): {exc}"
            ) from exc

        # Ultralytics returns a list with one Results object per image
        if not results:
            return []

        result = results[0]

        # Extract raw arrays from the Boxes object
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        boxes_xyxy: np.ndarray = boxes.xyxy.cpu().numpy()       # (N, 4)
        confidences: np.ndarray = boxes.conf.cpu().numpy()       # (N,)
        class_ids: np.ndarray = boxes.cls.cpu().numpy().astype(int)  # (N,)

        return self._post_processor.process(
            boxes_xyxy=boxes_xyxy,
            confidences=confidences,
            class_ids=class_ids,
            class_names=self._class_names,
            frame_height=h,
            frame_width=w,
        )

    # ------------------------------------------------------------------
    # Properties (read-only access to config / state)
    # ------------------------------------------------------------------

    @property
    def config(self) -> DetectorConfig:
        """The configuration this detector was initialised with."""
        return self._config

    @property
    def device(self) -> str:
        """The resolved inference device (e.g. ``"cpu"`` or ``"cuda:0"``)."""
        return self._resolved_device

    @property
    def class_names(self) -> list[str]:
        """Full ordered class-name list from the loaded model (COCO order)."""
        return list(self._class_names)  # defensive copy

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Load the YOLO model from disk.  Called once during ``__init__``."""
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as exc:
            raise ModelLoadError(
                "The 'ultralytics' package is not installed.  "
                "Run: pip install ultralytics"
            ) from exc

        try:
            model = YOLO(self._config.model_path)
            # Move model to target device so first inference isn't delayed
            model.to(self._resolved_device)
            logger.info("Model loaded from %r on device %r.", self._config.model_path, self._resolved_device)
            return model
        except FileNotFoundError as exc:
            raise ModelLoadError(
                f"Model weights file not found: {self._config.model_path!r}"
            ) from exc
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model from {self._config.model_path!r}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Module-level helpers
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
            logger.info("Auto device selection: CUDA available → %s", resolved)
            return resolved
        # Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Auto device selection: MPS available → mps")
            return "mps"
    except ImportError:
        pass  # torch not available; fall back to CPU

    logger.info("Auto device selection: no GPU detected → cpu")
    return "cpu"


def _validate_frame(frame: np.ndarray | None) -> None:
    """Raise :class:`InvalidFrameError` if *frame* is not a valid BGR image.

    Parameters
    ----------
    frame:
        Value to validate.

    Raises
    ------
    InvalidFrameError
        With a descriptive message explaining which invariant was violated.
    """
    if frame is None:
        raise InvalidFrameError("Frame is None; expected a NumPy ndarray.")

    if not isinstance(frame, np.ndarray):
        raise InvalidFrameError(
            f"Frame must be a numpy.ndarray, got {type(frame).__name__!r}."
        )

    if frame.size == 0:
        raise InvalidFrameError(
            f"Frame is empty (size=0, shape={frame.shape})."
        )

    if frame.ndim != _EXPECTED_NDIM:
        raise InvalidFrameError(
            f"Frame must be {_EXPECTED_NDIM}-dimensional (H×W×C), "
            f"got ndim={frame.ndim} (shape={frame.shape})."
        )

    if frame.shape[2] != _EXPECTED_CHANNELS:
        raise InvalidFrameError(
            f"Frame must have {_EXPECTED_CHANNELS} channels (BGR), "
            f"got {frame.shape[2]} channels."
        )

    if frame.dtype not in _SUPPORTED_DTYPES:
        raise InvalidFrameError(
            f"Frame dtype must be uint8, got {frame.dtype!r}.  "
            "Convert with: frame = frame.astype(np.uint8)"
        )
