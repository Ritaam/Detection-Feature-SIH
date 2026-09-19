"""
Unit tests for the ThermalDetector (end-to-end with mocked YOLO model).

Tests:
- detect() returns list[ThermalDetection] with mocked model
- detect(None) raises ThermalInvalidFrameError
- detect(empty) raises ThermalInvalidFrameError
- Model loaded once across multiple detect() calls
- Invalid model path raises ThermalModelLoadError
- Grayscale input accepted (single-channel)
- 3-channel input accepted
- uint16 input accepted (via preprocessing)
- Config validation
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.thermal_detection.config import ThermalDetectorConfig
from src.thermal_detection.exceptions import (
    ThermalInferenceError,
    ThermalInvalidFrameError,
    ThermalModelLoadError,
)
from src.thermal_detection.models import ThermalDetection
from src.thermal_detection.thermal_detector import (
    ThermalDetector,
    YOLOThermalDetector,
    _resolve_device,
)


# -----------------------------------------------------------------------
# Helpers — mock YOLO model
# -----------------------------------------------------------------------


def _make_mock_boxes(
    xyxy: list[list[float]],
    confs: list[float],
    cls_ids: list[int],
) -> MagicMock:
    """Create a mock Ultralytics Boxes object."""
    boxes = MagicMock()
    boxes.__len__ = MagicMock(return_value=len(xyxy))
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
        xyxy, dtype=np.float32
    )
    boxes.conf.cpu.return_value.numpy.return_value = np.array(
        confs, dtype=np.float32
    )
    boxes.cls.cpu.return_value.numpy.return_value = np.array(
        cls_ids, dtype=np.float32
    )
    return boxes


def _make_mock_result(
    xyxy: list[list[float]] | None = None,
    confs: list[float] | None = None,
    cls_ids: list[int] | None = None,
) -> MagicMock:
    """Create a mock Ultralytics Results object."""
    result = MagicMock()
    if xyxy is None:
        result.boxes = None
    else:
        result.boxes = _make_mock_boxes(xyxy, confs or [], cls_ids or [])
    return result


def _make_mock_yolo(
    class_names: dict[int, str] | None = None,
    result: MagicMock | None = None,
) -> MagicMock:
    """Create a mock YOLO model."""
    model = MagicMock()
    if class_names is None:
        class_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 7: "truck"}
    model.names = class_names
    if result is not None:
        model.predict.return_value = [result]
    else:
        model.predict.return_value = []
    return model


def _create_detector_with_mock(
    mock_model: MagicMock,
    **config_kwargs,
) -> YOLOThermalDetector:
    """Create a YOLOThermalDetector with a mocked YOLO model."""
    defaults = {
        "model_path": "fake_model.pt",
        "confidence_threshold": 0.25,
        "device": "cpu",
    }
    defaults.update(config_kwargs)
    config = ThermalDetectorConfig(**defaults)

    with patch(
        "src.thermal_detection.thermal_detector.YOLOThermalDetector._load_model",
        return_value=mock_model,
    ):
        detector = YOLOThermalDetector(config)
    return detector


# -----------------------------------------------------------------------
# Detection output
# -----------------------------------------------------------------------


class TestDetectorOutput:
    """Test that detect() returns correct output."""

    def test_returns_list_of_thermal_detection(self):
        result = _make_mock_result(
            xyxy=[[50.0, 50.0, 200.0, 300.0]],
            confs=[0.92],
            cls_ids=[0],
        )
        model = _make_mock_yolo(result=result)
        detector = _create_detector_with_mock(model)

        frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        detections = detector.detect(frame)

        assert isinstance(detections, list)
        assert len(detections) == 1
        assert isinstance(detections[0], ThermalDetection)
        assert detections[0].class_name == "person"

    def test_empty_result(self):
        result = _make_mock_result(xyxy=None)
        model = _make_mock_yolo(result=result)
        detector = _create_detector_with_mock(model)

        frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        detections = detector.detect(frame)

        assert detections == []

    def test_no_results(self):
        model = _make_mock_yolo()
        model.predict.return_value = []
        detector = _create_detector_with_mock(model)

        frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        detections = detector.detect(frame)

        assert detections == []

    def test_multiple_detections(self):
        result = _make_mock_result(
            xyxy=[
                [50.0, 50.0, 200.0, 300.0],
                [300.0, 100.0, 500.0, 400.0],
            ],
            confs=[0.92, 0.85],
            cls_ids=[0, 2],
        )
        model = _make_mock_yolo(result=result)
        detector = _create_detector_with_mock(model)

        frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 2
        class_names = {d.class_name for d in detections}
        assert "person" in class_names
        assert "car" in class_names


# -----------------------------------------------------------------------
# Frame validation
# -----------------------------------------------------------------------


class TestDetectorFrameValidation:
    """Test that invalid frames raise ThermalInvalidFrameError."""

    def _make_detector(self) -> YOLOThermalDetector:
        model = _make_mock_yolo()
        return _create_detector_with_mock(model)

    def test_none_frame(self):
        detector = self._make_detector()
        with pytest.raises(ThermalInvalidFrameError):
            detector.detect(None)  # type: ignore[arg-type]

    def test_empty_frame(self):
        detector = self._make_detector()
        with pytest.raises(ThermalInvalidFrameError):
            detector.detect(np.array([], dtype=np.uint8))


# -----------------------------------------------------------------------
# Input dtype support
# -----------------------------------------------------------------------


class TestDetectorInputTypes:
    """Test that the detector accepts various thermal input formats."""

    def _make_detector(self) -> YOLOThermalDetector:
        result = _make_mock_result(
            xyxy=[[50.0, 50.0, 200.0, 300.0]],
            confs=[0.92],
            cls_ids=[0],
        )
        model = _make_mock_yolo(result=result)
        return _create_detector_with_mock(model)

    def test_uint8_grayscale(self):
        detector = self._make_detector()
        frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        detections = detector.detect(frame)
        assert isinstance(detections, list)

    def test_uint16_grayscale(self):
        detector = self._make_detector()
        frame = np.random.randint(0, 65536, (480, 640), dtype=np.uint16)
        detections = detector.detect(frame)
        assert isinstance(detections, list)

    def test_float32_grayscale(self):
        detector = self._make_detector()
        frame = np.random.rand(480, 640).astype(np.float32)
        detections = detector.detect(frame)
        assert isinstance(detections, list)

    def test_single_channel_3d(self):
        detector = self._make_detector()
        frame = np.random.randint(0, 256, (480, 640, 1), dtype=np.uint8)
        detections = detector.detect(frame)
        assert isinstance(detections, list)

    def test_3channel_input(self):
        detector = self._make_detector()
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        assert isinstance(detections, list)


# -----------------------------------------------------------------------
# Model lifecycle
# -----------------------------------------------------------------------


class TestModelLifecycle:
    """Test that the model is loaded once and reused."""

    def test_model_loaded_once(self):
        result = _make_mock_result(xyxy=None)
        model = _make_mock_yolo(result=result)

        with patch(
            "src.thermal_detection.thermal_detector.YOLOThermalDetector._load_model",
            return_value=model,
        ) as mock_load:
            config = ThermalDetectorConfig(
                model_path="fake.pt", device="cpu"
            )
            detector = YOLOThermalDetector(config)

            # Call detect multiple times
            frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            detector.detect(frame)
            detector.detect(frame)
            detector.detect(frame)

            # _load_model should have been called exactly once (in __init__)
            mock_load.assert_called_once()


# -----------------------------------------------------------------------
# Model loading errors
# -----------------------------------------------------------------------


class TestModelLoading:
    """Test model loading error handling."""

    def test_missing_ultralytics_raises(self):
        with patch.dict("sys.modules", {"ultralytics": None}):
            config = ThermalDetectorConfig(
                model_path="fake.pt", device="cpu"
            )
            with pytest.raises(ThermalModelLoadError, match="ultralytics"):
                YOLOThermalDetector(config)

    def test_invalid_model_path_raises(self):
        config = ThermalDetectorConfig(
            model_path="nonexistent_thermal_model_xyz.pt",
            device="cpu",
        )
        with pytest.raises(ThermalModelLoadError):
            YOLOThermalDetector(config)


# -----------------------------------------------------------------------
# Inference error
# -----------------------------------------------------------------------


class TestInferenceError:
    """Test that inference errors are properly wrapped."""

    def test_model_exception_wrapped(self):
        model = _make_mock_yolo()
        model.predict.side_effect = RuntimeError("CUDA OOM")
        detector = _create_detector_with_mock(model)

        frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        with pytest.raises(ThermalInferenceError, match="CUDA OOM"):
            detector.detect(frame)


# -----------------------------------------------------------------------
# Device resolution
# -----------------------------------------------------------------------


class TestDeviceResolution:
    """Test _resolve_device()."""

    def test_cpu_passthrough(self):
        assert _resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self):
        assert _resolve_device("cuda") == "cuda"

    def test_cuda_indexed_passthrough(self):
        assert _resolve_device("cuda:0") == "cuda:0"

    def test_auto_without_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            # When torch is not importable, should fall back to CPU
            result = _resolve_device("auto")
            assert result == "cpu"


# -----------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------


class TestConfigValidation:
    """Test ThermalDetectorConfig validation."""

    def test_invalid_confidence_threshold(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            ThermalDetectorConfig(
                model_path="x.pt", confidence_threshold=0.0
            )

    def test_invalid_iou_threshold(self):
        with pytest.raises(ValueError, match="iou_threshold"):
            ThermalDetectorConfig(
                model_path="x.pt", iou_threshold=1.5
            )

    def test_invalid_device(self):
        with pytest.raises(ValueError, match="device"):
            ThermalDetectorConfig(
                model_path="x.pt", device="tpu"
            )

    def test_invalid_image_size(self):
        with pytest.raises(ValueError, match="image_size"):
            ThermalDetectorConfig(
                model_path="x.pt", image_size=641
            )

    def test_even_denoise_kernel(self):
        with pytest.raises(ValueError, match="denoise_kernel_size"):
            ThermalDetectorConfig(
                model_path="x.pt", denoise_kernel_size=4
            )


# -----------------------------------------------------------------------
# ThermalDetector facade
# -----------------------------------------------------------------------


class TestThermalDetectorFacade:
    """Test ThermalDetector (convenience facade) works identically."""

    def test_facade_is_subclass(self):
        assert issubclass(ThermalDetector, YOLOThermalDetector)

    def test_facade_detect(self):
        result = _make_mock_result(
            xyxy=[[50.0, 50.0, 200.0, 300.0]],
            confs=[0.92],
            cls_ids=[0],
        )
        model = _make_mock_yolo(result=result)

        config = ThermalDetectorConfig(
            model_path="fake.pt", device="cpu"
        )
        with patch(
            "src.thermal_detection.thermal_detector.YOLOThermalDetector._load_model",
            return_value=model,
        ):
            detector = ThermalDetector(config)
            frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
            detections = detector.detect(frame)
            assert isinstance(detections, list)
            assert len(detections) == 1


# -----------------------------------------------------------------------
# Properties
# -----------------------------------------------------------------------


class TestDetectorProperties:
    """Test read-only properties."""

    def test_config_property(self):
        model = _make_mock_yolo()
        detector = _create_detector_with_mock(model)
        assert isinstance(detector.config, ThermalDetectorConfig)

    def test_device_property(self):
        model = _make_mock_yolo()
        detector = _create_detector_with_mock(model, device="cpu")
        assert detector.device == "cpu"

    def test_class_names_property(self):
        model = _make_mock_yolo()
        detector = _create_detector_with_mock(model)
        names = detector.class_names
        assert isinstance(names, list)
        assert "person" in names
