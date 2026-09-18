"""
Detection module configuration.

All tunable values live here.  Nothing else in ``src/detection/`` should
contain magic numbers or inline defaults — always reference this module.

Device selection
----------------
``device`` accepts:
* ``"auto"``  – use CUDA if available, fall back to CPU (default)
* ``"cuda"``  – force CUDA (raises at runtime if unavailable)
* ``"cpu"``   – force CPU
* ``"mps"``   – Apple Silicon GPU (macOS only)

Supported classes
-----------------
``supported_classes`` is a ``frozenset[str]`` of class *names* (not IDs).
The detector maps these names against the model's own class list at runtime,
so class IDs are never hard-coded here.

YOLO11n COCO class names used here (subset relevant to border surveillance):

    person, car, motorcycle, bus, truck
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Default supported class set
# ---------------------------------------------------------------------------

DEFAULT_SUPPORTED_CLASSES: frozenset[str] = frozenset(
    {
        "person",
        "car",
        "motorcycle",
        "bus",
        "truck",
    }
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorConfig:
    """Immutable configuration for :class:`~src.detection.detector.Detector`.

    Parameters
    ----------
    model_path:
        Path to the YOLO ``.pt`` weights file.  Pass ``"yolo11n.pt"`` to let
        Ultralytics download the official pretrained weights automatically on
        first use.
    confidence_threshold:
        Minimum confidence score ``[0, 1]`` to keep a detection.
        Detections below this threshold are discarded in post-processing.
    iou_threshold:
        IoU threshold used for Non-Maximum Suppression inside YOLO.
        Higher values keep more overlapping boxes; lower values suppress them.
    device:
        Inference device.  One of ``"auto"``, ``"cpu"``, ``"cuda"``,
        ``"cuda:0"``, ``"mps"``.  Default ``"auto"`` selects CUDA when
        available, otherwise CPU.
    supported_classes:
        Frozenset of class name strings to keep in the output.  Any YOLO
        detection whose class name is not in this set is silently dropped.
    imgsz:
        Input image size (pixels) passed to YOLO during inference.  YOLO
        resizes internally; the original frame is never modified.
        Must be a multiple of 32.  Default 640 matches YOLO's training size.

    Examples
    --------
    Default config (auto-download YOLO11n weights):

    >>> cfg = DetectorConfig(model_path="yolo11n.pt")

    Custom config with explicit CUDA and lower threshold:

    >>> cfg = DetectorConfig(
    ...     model_path="models/yolo11n.pt",
    ...     confidence_threshold=0.4,
    ...     device="cuda:0",
    ... )
    """

    model_path: str
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "auto"
    supported_classes: frozenset[str] = field(
        default_factory=lambda: DEFAULT_SUPPORTED_CLASSES
    )
    imgsz: int = 640

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not (0.0 < self.confidence_threshold <= 1.0):
            raise ValueError(
                "confidence_threshold must be in (0, 1], "
                f"got {self.confidence_threshold!r}"
            )
        if not (0.0 < self.iou_threshold <= 1.0):
            raise ValueError(
                "iou_threshold must be in (0, 1], "
                f"got {self.iou_threshold!r}"
            )
        valid_devices = {"auto", "cpu", "cuda", "mps"}
        # Accept "cuda:N" variants too
        base_device = self.device.split(":")[0]
        if base_device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices} (or 'cuda:N'), "
                f"got {self.device!r}"
            )
        if self.imgsz % 32 != 0:
            raise ValueError(
                f"imgsz must be a multiple of 32, got {self.imgsz!r}"
            )
        if not self.supported_classes:
            raise ValueError("supported_classes must not be empty")
