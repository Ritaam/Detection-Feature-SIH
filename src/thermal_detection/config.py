"""
Thermal detection module configuration.

All tunable values live here.  Nothing else in ``src/thermal_detection/``
should contain magic numbers or inline defaults — always reference this
module.

Device selection
----------------
``device`` accepts:
* ``"auto"``  – use CUDA if available, fall back to CPU (default)
* ``"cuda"``  – force CUDA (raises at runtime if unavailable)
* ``"cpu"``   – force CPU
* ``"mps"``   – Apple Silicon GPU (macOS only)

Allowed classes
---------------
``allowed_classes`` is a ``frozenset[str]`` of class *names* (not IDs).
The detector maps these names against the model's own class list at runtime,
so class IDs are never hard-coded here.

Set to ``None`` to accept all classes the model can detect.

Preprocessing
-------------
Thermal frames require different preprocessing than RGB frames.
CLAHE contrast enhancement and denoising are configurable here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Default allowed class set — matches COCO labels relevant for surveillance
# ---------------------------------------------------------------------------

DEFAULT_THERMAL_CLASSES: frozenset[str] = frozenset(
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
class ThermalDetectorConfig:
    """Immutable configuration for the thermal detection engine.

    Parameters
    ----------
    model_path:
        Path to the YOLO ``.pt`` weights file.  May be a thermal-fine-tuned
        model or a general COCO model like ``"yolo11n.pt"``.
    confidence_threshold:
        Minimum confidence score ``[0, 1]`` to keep a detection.
    iou_threshold:
        IoU threshold for Non-Maximum Suppression inside YOLO.
    device:
        Inference device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:0"``,
        ``"mps"``.  Default ``"auto"`` selects CUDA when available.
    image_size:
        Input image size (pixels) passed to YOLO during inference.
        Must be a multiple of 32.  Default 640.
    allowed_classes:
        Frozenset of class name strings to keep.  Detections whose class
        name is not in this set are silently dropped.  ``None`` accepts
        all classes the model can detect.
    enable_clahe:
        Whether to apply CLAHE contrast enhancement during preprocessing.
    enable_denoising:
        Whether to apply Gaussian denoising during preprocessing.
    clahe_clip_limit:
        CLAHE clip limit.  Higher values produce stronger contrast.
    clahe_tile_grid:
        CLAHE tile grid size as ``(rows, cols)``.
    denoise_kernel_size:
        Gaussian blur kernel size for denoising.  Must be odd.
    enable_hotspot_segmentation:
        Whether to run optional hotspot segmentation alongside detection.
    debug:
        Enable debug output (saving intermediate pipeline frames).
        Should be ``False`` in production.

    Examples
    --------
    Minimal config with default YOLO11n:

    >>> cfg = ThermalDetectorConfig(model_path="yolo11n.pt")

    Custom thermal model with tuned settings:

    >>> cfg = ThermalDetectorConfig(
    ...     model_path="models/thermal_detector.pt",
    ...     confidence_threshold=0.3,
    ...     device="cuda:0",
    ...     allowed_classes=frozenset({"person", "car"}),
    ... )
    """

    model_path: str
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "auto"
    image_size: int = 640
    allowed_classes: frozenset[str] | None = field(
        default_factory=lambda: DEFAULT_THERMAL_CLASSES
    )
    enable_clahe: bool = True
    enable_denoising: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    denoise_kernel_size: int = 5
    enable_hotspot_segmentation: bool = False
    debug: bool = False

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
        base_device = self.device.split(":")[0]
        if base_device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices} (or 'cuda:N'), "
                f"got {self.device!r}"
            )
        if self.image_size % 32 != 0:
            raise ValueError(
                f"image_size must be a multiple of 32, got {self.image_size!r}"
            )
        if self.denoise_kernel_size % 2 == 0:
            raise ValueError(
                f"denoise_kernel_size must be odd, "
                f"got {self.denoise_kernel_size!r}"
            )
        if self.clahe_clip_limit <= 0.0:
            raise ValueError(
                f"clahe_clip_limit must be positive, "
                f"got {self.clahe_clip_limit!r}"
            )
