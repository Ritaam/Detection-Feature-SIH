"""
Integration-level tests for :class:`~src.detection.detector.Detector`.

Strategy
--------
Tests that do NOT require a real model use a ``MockModel`` fixture that mimics
the Ultralytics YOLO interface without loading any weights.  This keeps the
test suite fast and runnable on any machine.

Tests that DO require a real model are marked ``@pytest.mark.integration``
and are skipped by default unless ``--run-integration`` is passed to pytest.

Tests:
1.  Valid frame                   → list[Detection]
2.  None frame                    → InvalidFrameError
3.  Empty frame                   → InvalidFrameError
4.  Wrong dtype (float32)         → InvalidFrameError
5.  Wrong number of channels      → InvalidFrameError
6.  Wrong ndim (2D image)         → InvalidFrameError
7.  No objects in frame           → []
8.  Unsupported class filtered    → absent from output
9.  Confidence threshold filters  → low-conf detections absent
10. Bounding box format           → x1<x2, y1<y2
11. Model loaded only once        → same object reference
12. CPU execution                 → works without CUDA
13. Model loading failure         → ModelLoadError
14. Output is list, never None    → isinstance(list)
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.detector import Detector, _validate_frame
from src.detection.exceptions import InvalidFrameError, ModelLoadError
from src.detection.models import Detection


# ---------------------------------------------------------------------------
# Pytest option for integration tests
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that load the real YOLO11n model.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration (requires real model download).",
    )


# ---------------------------------------------------------------------------
# Mock YOLO infrastructure
# ---------------------------------------------------------------------------


COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
}


def _make_mock_result(boxes_xyxy, confs, cls_ids):
    """Build a mock Ultralytics Results object."""
    mock_boxes = MagicMock()
    mock_boxes.__len__ = lambda self: len(boxes_xyxy)

    _xyxy = MagicMock()
    _xyxy.cpu.return_value.numpy.return_value = np.array(boxes_xyxy, dtype=np.float32)

    _conf = MagicMock()
    _conf.cpu.return_value.numpy.return_value = np.array(confs, dtype=np.float32)

    _cls = MagicMock()
    _cls.cpu.return_value.numpy.return_value = np.array(cls_ids, dtype=np.float32)

    mock_boxes.xyxy = _xyxy
    mock_boxes.conf = _conf
    mock_boxes.cls = _cls

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    return mock_result


def _make_mock_model(boxes_xyxy=None, confs=None, cls_ids=None):
    """Return a mock YOLO model that returns controlled detections."""
    if boxes_xyxy is None:
        boxes_xyxy = []
        confs = []
        cls_ids = []

    mock_model = MagicMock()
    mock_model.names = COCO_NAMES
    mock_model.to.return_value = mock_model

    mock_result = _make_mock_result(boxes_xyxy, confs, cls_ids)

    # Empty Boxes object when no detections
    if not boxes_xyxy:
        mock_result.boxes.__len__ = lambda self: 0

    mock_model.predict.return_value = [mock_result]
    return mock_model


def make_detector_with_mock(
    mock_model,
    confidence_threshold=0.5,
    device="cpu",
) -> Detector:
    """Build a Detector with a pre-built mock model, bypassing file loading."""
    with patch("src.detection.detector._resolve_device", return_value="cpu"), \
         patch("ultralytics.YOLO", return_value=mock_model):
        detector = Detector(
            model_path="yolo11n.pt",
            confidence_threshold=confidence_threshold,
            device=device,
        )
    # Inject mock directly so predict() is called on our mock
    detector._model = mock_model
    return detector


def _blank_frame(h=480, w=640) -> np.ndarray:
    """Create a valid blank BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test 1 – Valid frame returns list[Detection]
# ---------------------------------------------------------------------------


class TestValidFrame:
    def test_returns_list(self):
        mock = _make_mock_model(
            boxes_xyxy=[[50, 50, 200, 300]],
            confs=[0.9],
            cls_ids=[0],
        )
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        assert isinstance(result, list)

    def test_returns_detection_objects(self):
        mock = _make_mock_model(
            boxes_xyxy=[[50, 50, 200, 300]],
            confs=[0.9],
            cls_ids=[0],
        )
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        assert all(isinstance(d, Detection) for d in result)

    def test_person_detected(self):
        mock = _make_mock_model(
            boxes_xyxy=[[50, 50, 200, 300]],
            confs=[0.9],
            cls_ids=[0],  # person
        )
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        assert len(result) == 1
        assert result[0].class_name == "person"

    def test_car_detected(self):
        mock = _make_mock_model(
            boxes_xyxy=[[100, 100, 400, 300]],
            confs=[0.88],
            cls_ids=[2],  # car
        )
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        assert len(result) == 1
        assert result[0].class_name == "car"


# ---------------------------------------------------------------------------
# Test 2 – None frame raises InvalidFrameError
# ---------------------------------------------------------------------------


class TestNoneFrame:
    def test_none_raises_invalid_frame_error(self):
        with pytest.raises(InvalidFrameError, match="None"):
            _validate_frame(None)

    def test_none_via_detect(self):
        mock = _make_mock_model()
        detector = make_detector_with_mock(mock)
        with pytest.raises(InvalidFrameError):
            detector.detect(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 3 – Empty frame raises InvalidFrameError
# ---------------------------------------------------------------------------


class TestEmptyFrame:
    def test_empty_array_raises(self):
        with pytest.raises(InvalidFrameError, match="empty"):
            _validate_frame(np.array([]))

    def test_zero_height_raises(self):
        with pytest.raises(InvalidFrameError):
            _validate_frame(np.zeros((0, 640, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Test 4 – Wrong dtype raises InvalidFrameError
# ---------------------------------------------------------------------------


class TestWrongDtype:
    def test_float32_raises(self):
        frame = np.zeros((480, 640, 3), dtype=np.float32)
        with pytest.raises(InvalidFrameError, match="uint8"):
            _validate_frame(frame)

    def test_float64_raises(self):
        frame = np.zeros((480, 640, 3), dtype=np.float64)
        with pytest.raises(InvalidFrameError, match="uint8"):
            _validate_frame(frame)

    def test_int16_raises(self):
        frame = np.zeros((480, 640, 3), dtype=np.int16)
        with pytest.raises(InvalidFrameError, match="uint8"):
            _validate_frame(frame)


# ---------------------------------------------------------------------------
# Test 5 – Wrong number of channels raises InvalidFrameError
# ---------------------------------------------------------------------------


class TestWrongChannels:
    def test_grayscale_raises(self):
        # Only 1 channel
        frame = np.zeros((480, 640, 1), dtype=np.uint8)
        with pytest.raises(InvalidFrameError, match="3 channel"):
            _validate_frame(frame)

    def test_rgba_raises(self):
        frame = np.zeros((480, 640, 4), dtype=np.uint8)
        with pytest.raises(InvalidFrameError, match="3 channel"):
            _validate_frame(frame)


# ---------------------------------------------------------------------------
# Test 6 – Wrong ndim raises InvalidFrameError
# ---------------------------------------------------------------------------


class TestWrongNdim:
    def test_2d_raises(self):
        frame = np.zeros((480, 640), dtype=np.uint8)
        with pytest.raises(InvalidFrameError, match="3-dimensional"):
            _validate_frame(frame)

    def test_4d_raises(self):
        frame = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        with pytest.raises(InvalidFrameError, match="3-dimensional"):
            _validate_frame(frame)


# ---------------------------------------------------------------------------
# Test 7 – No objects detected returns []
# ---------------------------------------------------------------------------


class TestNoObjects:
    def test_empty_boxes_returns_empty_list(self):
        mock = _make_mock_model()  # no detections
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        assert result == []

    def test_returns_list_not_none(self):
        mock = _make_mock_model()
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        assert result is not None


# ---------------------------------------------------------------------------
# Test 8 – Unsupported class is filtered out
# ---------------------------------------------------------------------------


class TestUnsupportedClassFiltering:
    def test_bicycle_not_in_output(self):
        mock = _make_mock_model(
            boxes_xyxy=[[10, 10, 100, 100]],
            confs=[0.95],
            cls_ids=[1],  # bicycle — NOT in supported set
        )
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        assert result == []

    def test_airplane_not_in_output(self):
        mock = _make_mock_model(
            boxes_xyxy=[[10, 10, 100, 100]],
            confs=[0.95],
            cls_ids=[4],  # airplane — NOT in supported set
        )
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        assert result == []

    def test_supported_plus_unsupported(self):
        mock = _make_mock_model(
            boxes_xyxy=[
                [10, 10, 100, 100],   # person
                [200, 10, 400, 200],  # bicycle (unsupported)
            ],
            confs=[0.9, 0.9],
            cls_ids=[0, 1],
        )
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        assert len(result) == 1
        assert result[0].class_name == "person"


# ---------------------------------------------------------------------------
# Test 9 – Confidence threshold filtering
# ---------------------------------------------------------------------------


class TestConfidenceThresholdFiltering:
    def test_below_threshold_dropped(self):
        mock = _make_mock_model(
            boxes_xyxy=[[10, 10, 100, 200]],
            confs=[0.3],  # below 0.5 threshold
            cls_ids=[0],
        )
        detector = make_detector_with_mock(mock, confidence_threshold=0.5)
        result = detector.detect(_blank_frame())
        assert result == []

    def test_above_threshold_kept(self):
        mock = _make_mock_model(
            boxes_xyxy=[[10, 10, 100, 200]],
            confs=[0.8],
            cls_ids=[0],
        )
        detector = make_detector_with_mock(mock, confidence_threshold=0.5)
        result = detector.detect(_blank_frame())
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test 10 – Bounding box format guaranteed
# ---------------------------------------------------------------------------


class TestBoundingBoxFormat:
    def test_x1_less_than_x2(self):
        mock = _make_mock_model(
            boxes_xyxy=[[50, 50, 300, 400]],
            confs=[0.9],
            cls_ids=[0],
        )
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        for d in result:
            x1, _, x2, _ = d.bbox
            assert x1 < x2

    def test_y1_less_than_y2(self):
        mock = _make_mock_model(
            boxes_xyxy=[[50, 50, 300, 400]],
            confs=[0.9],
            cls_ids=[0],
        )
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        for d in result:
            _, y1, _, y2 = d.bbox
            assert y1 < y2

    def test_bbox_is_four_float_tuple(self):
        mock = _make_mock_model(
            boxes_xyxy=[[50, 50, 300, 400]],
            confs=[0.9],
            cls_ids=[0],
        )
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        assert len(result) == 1
        assert isinstance(result[0].bbox, tuple)
        assert len(result[0].bbox) == 4
        assert all(isinstance(v, float) for v in result[0].bbox)


# ---------------------------------------------------------------------------
# Test 11 – Model loaded only once
# ---------------------------------------------------------------------------


class TestModelLoadedOnce:
    def test_same_model_object_used_across_calls(self):
        mock = _make_mock_model(
            boxes_xyxy=[[50, 50, 200, 300]],
            confs=[0.9],
            cls_ids=[0],
        )
        detector = make_detector_with_mock(mock)
        model_id_1 = id(detector._model)
        detector.detect(_blank_frame())
        detector.detect(_blank_frame())
        model_id_2 = id(detector._model)
        assert model_id_1 == model_id_2

    def test_predict_not_reload(self):
        """Model.predict() called per frame, YOLO() constructor called once."""
        mock = _make_mock_model()
        call_count = [0]
        original_init = Detector.__init__

        with patch("src.detection.detector._resolve_device", return_value="cpu"), \
             patch("ultralytics.YOLO", return_value=mock) as mock_yolo_cls:
            detector = Detector(model_path="yolo11n.pt", device="cpu")
            detector._model = mock

        detector.detect(_blank_frame())
        detector.detect(_blank_frame())
        detector.detect(_blank_frame())

        # YOLO() constructor should have been called exactly once (during __init__)
        mock_yolo_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Test 12 – CPU execution
# ---------------------------------------------------------------------------


class TestCPUExecution:
    def test_device_resolved_to_cpu(self):
        mock = _make_mock_model()
        detector = make_detector_with_mock(mock, device="cpu")
        assert detector.device == "cpu"

    def test_detect_on_cpu_succeeds(self):
        mock = _make_mock_model(
            boxes_xyxy=[[10, 10, 100, 200]],
            confs=[0.9],
            cls_ids=[0],
        )
        detector = make_detector_with_mock(mock, device="cpu")
        result = detector.detect(_blank_frame())
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test 13 – Model loading failure raises ModelLoadError
# ---------------------------------------------------------------------------


class TestModelLoadFailure:
    def test_file_not_found_raises_model_load_error(self):
        with patch("src.detection.detector._resolve_device", return_value="cpu"), \
             patch("ultralytics.YOLO", side_effect=FileNotFoundError("no file")):
            with pytest.raises(ModelLoadError, match="not found"):
                Detector(model_path="/nonexistent/path/model.pt", device="cpu")

    def test_ultralytics_not_installed_raises_model_load_error(self):
        with patch("src.detection.detector._resolve_device", return_value="cpu"), \
             patch.dict("sys.modules", {"ultralytics": None}):
            with pytest.raises((ModelLoadError, ImportError)):
                Detector(model_path="yolo11n.pt", device="cpu")


# ---------------------------------------------------------------------------
# Test 14 – Output is always list, never None
# ---------------------------------------------------------------------------


class TestOutputIsAlwaysList:
    def test_empty_detections_is_list(self):
        mock = _make_mock_model()
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        assert isinstance(result, list)

    def test_non_empty_detections_is_list(self):
        mock = _make_mock_model(
            boxes_xyxy=[[10, 10, 200, 300]],
            confs=[0.9],
            cls_ids=[0],
        )
        detector = make_detector_with_mock(mock)
        result = detector.detect(_blank_frame())
        assert isinstance(result, list)
        assert result is not None


# ---------------------------------------------------------------------------
# Integration test (skipped unless --run-integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    """Requires real YOLO11n model — skipped by default."""

    @pytest.fixture(autouse=True)
    def skip_without_flag(self, request):
        if not request.config.getoption("--run-integration", default=False):
            pytest.skip("Pass --run-integration to run this test.")

    def test_model_loads_from_auto_download(self):
        detector = Detector(model_path="yolo11n.pt", device="cpu")
        assert detector is not None

    def test_detect_blank_frame_returns_list(self):
        detector = Detector(model_path="yolo11n.pt", device="cpu")
        result = detector.detect(_blank_frame())
        assert isinstance(result, list)

    def test_detect_with_synthetic_person_frame(self):
        """Synthetic white rectangle — model may or may not detect anything."""
        detector = Detector(model_path="yolo11n.pt", device="cpu")
        frame = _blank_frame()
        # Draw a white rectangle vaguely person-shaped
        frame[100:400, 280:360] = 255
        result = detector.detect(frame)
        assert isinstance(result, list)
        assert all(isinstance(d, Detection) for d in result)
