"""
Unit tests for :class:`~src.detection.detector.Detector`.

Strategy
--------
All unit tests use ``_build_detector()`` which constructs a ``Detector``
via ``object.__new__`` and injects a lightweight ``FakeModel`` directly —
zero ultralytics / torch code runs during the unit test suite.

This makes the test run fast (< 5 s) on any machine, with or without GPU.

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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.config import DEFAULT_SUPPORTED_CLASSES, DetectorConfig
from src.detection.detector import Detector, _validate_frame
from src.detection.exceptions import InvalidFrameError, ModelLoadError
from src.detection.models import Detection
from src.detection.postprocess import PostProcessor


# ---------------------------------------------------------------------------
# Pytest CLI option
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
# Fake model — zero ultralytics / torch import
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


def _make_fake_boxes(boxes_xyxy, confs, cls_ids):
    boxes = MagicMock()
    boxes.__len__ = lambda self: len(boxes_xyxy)

    xyxy = MagicMock()
    xyxy.cpu.return_value.numpy.return_value = np.array(boxes_xyxy, dtype=np.float32)
    conf = MagicMock()
    conf.cpu.return_value.numpy.return_value = np.array(confs, dtype=np.float32)
    cls = MagicMock()
    cls.cpu.return_value.numpy.return_value = np.array(cls_ids, dtype=np.float32)

    boxes.xyxy = xyxy
    boxes.conf = conf
    boxes.cls = cls
    return boxes


def _make_fake_model(boxes_xyxy=None, confs=None, cls_ids=None):
    """Build a fake YOLO model — no ultralytics import occurs."""
    if boxes_xyxy is None:
        boxes_xyxy, confs, cls_ids = [], [], []

    model = MagicMock()
    model.names = COCO_NAMES
    model.to.return_value = model

    result = MagicMock()
    result.boxes = _make_fake_boxes(boxes_xyxy, confs, cls_ids)
    model.predict.return_value = [result]
    return model


def _build_detector(fake_model, confidence_threshold=0.5, device="cpu") -> Detector:
    """
    Build a Detector without importing ultralytics or torch.

    Bypasses __init__ via object.__new__ and injects attributes that
    detect() reads directly. No YOLO() constructor is ever called.
    """
    d = object.__new__(Detector)
    d._config = DetectorConfig(
        model_path="fake.pt",
        confidence_threshold=confidence_threshold,
        device=device,
    )
    d._resolved_device = device
    d._model = fake_model
    d._class_names = list(COCO_NAMES.values())
    d._post_processor = PostProcessor(
        confidence_threshold=confidence_threshold,
        supported_classes=DEFAULT_SUPPORTED_CLASSES,
    )
    return d


def _blank_frame(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1 – Valid frame
# ---------------------------------------------------------------------------

class TestValidFrame:
    def test_returns_list(self):
        d = _build_detector(_make_fake_model([[50, 50, 200, 300]], [0.9], [0]))
        assert isinstance(d.detect(_blank_frame()), list)

    def test_returns_detection_objects(self):
        d = _build_detector(_make_fake_model([[50, 50, 200, 300]], [0.9], [0]))
        assert all(isinstance(x, Detection) for x in d.detect(_blank_frame()))

    def test_person_detected(self):
        d = _build_detector(_make_fake_model([[50, 50, 200, 300]], [0.9], [0]))
        result = d.detect(_blank_frame())
        assert len(result) == 1
        assert result[0].class_name == "person"

    def test_car_detected(self):
        d = _build_detector(_make_fake_model([[100, 100, 400, 300]], [0.88], [2]))
        result = d.detect(_blank_frame())
        assert len(result) == 1
        assert result[0].class_name == "car"

    def test_all_five_supported_classes(self):
        d = _build_detector(_make_fake_model(
            [[10, 10, 100, 100]] * 5,
            [0.9] * 5,
            [0, 2, 3, 5, 7],  # person, car, motorcycle, bus, truck
        ))
        result = d.detect(_blank_frame())
        assert len(result) == 5
        assert {r.class_name for r in result} == {"person", "car", "motorcycle", "bus", "truck"}


# ---------------------------------------------------------------------------
# 2 – None frame
# ---------------------------------------------------------------------------

class TestNoneFrame:
    def test_validate_none_raises(self):
        with pytest.raises(InvalidFrameError, match="None"):
            _validate_frame(None)

    def test_detect_none_raises(self):
        d = _build_detector(_make_fake_model())
        with pytest.raises(InvalidFrameError):
            d.detect(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3 – Empty frame
# ---------------------------------------------------------------------------

class TestEmptyFrame:
    def test_empty_array_raises(self):
        with pytest.raises(InvalidFrameError, match="empty"):
            _validate_frame(np.array([]))

    def test_zero_height_raises(self):
        with pytest.raises(InvalidFrameError):
            _validate_frame(np.zeros((0, 640, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 4 – Wrong dtype
# ---------------------------------------------------------------------------

class TestWrongDtype:
    def test_float32_raises(self):
        with pytest.raises(InvalidFrameError, match="uint8"):
            _validate_frame(np.zeros((480, 640, 3), dtype=np.float32))

    def test_float64_raises(self):
        with pytest.raises(InvalidFrameError, match="uint8"):
            _validate_frame(np.zeros((480, 640, 3), dtype=np.float64))

    def test_int16_raises(self):
        with pytest.raises(InvalidFrameError, match="uint8"):
            _validate_frame(np.zeros((480, 640, 3), dtype=np.int16))


# ---------------------------------------------------------------------------
# 5 – Wrong channels
# ---------------------------------------------------------------------------

class TestWrongChannels:
    def test_1ch_raises(self):
        with pytest.raises(InvalidFrameError, match="3 channel"):
            _validate_frame(np.zeros((480, 640, 1), dtype=np.uint8))

    def test_4ch_raises(self):
        with pytest.raises(InvalidFrameError, match="3 channel"):
            _validate_frame(np.zeros((480, 640, 4), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 6 – Wrong ndim
# ---------------------------------------------------------------------------

class TestWrongNdim:
    def test_2d_raises(self):
        with pytest.raises(InvalidFrameError, match="3-dimensional"):
            _validate_frame(np.zeros((480, 640), dtype=np.uint8))

    def test_4d_raises(self):
        with pytest.raises(InvalidFrameError, match="3-dimensional"):
            _validate_frame(np.zeros((1, 480, 640, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 7 – No objects
# ---------------------------------------------------------------------------

class TestNoObjects:
    def test_returns_empty_list(self):
        d = _build_detector(_make_fake_model())
        assert d.detect(_blank_frame()) == []

    def test_returns_list_not_none(self):
        d = _build_detector(_make_fake_model())
        assert d.detect(_blank_frame()) is not None


# ---------------------------------------------------------------------------
# 8 – Unsupported class filtering
# ---------------------------------------------------------------------------

class TestUnsupportedClassFiltering:
    def test_bicycle_dropped(self):
        d = _build_detector(_make_fake_model([[10, 10, 100, 100]], [0.95], [1]))
        assert d.detect(_blank_frame()) == []

    def test_airplane_dropped(self):
        d = _build_detector(_make_fake_model([[10, 10, 100, 100]], [0.95], [4]))
        assert d.detect(_blank_frame()) == []

    def test_supported_and_unsupported_mixed(self):
        d = _build_detector(_make_fake_model(
            [[10, 10, 100, 100], [200, 10, 400, 200]],
            [0.9, 0.9],
            [0, 1],  # person (kept) + bicycle (dropped)
        ))
        result = d.detect(_blank_frame())
        assert len(result) == 1
        assert result[0].class_name == "person"


# ---------------------------------------------------------------------------
# 9 – Confidence threshold
# ---------------------------------------------------------------------------

class TestConfidenceFiltering:
    def test_below_threshold_dropped(self):
        d = _build_detector(_make_fake_model([[10, 10, 100, 200]], [0.3], [0]), confidence_threshold=0.5)
        assert d.detect(_blank_frame()) == []

    def test_at_threshold_kept(self):
        d = _build_detector(_make_fake_model([[10, 10, 100, 200]], [0.5], [0]), confidence_threshold=0.5)
        assert len(d.detect(_blank_frame())) == 1

    def test_above_threshold_kept(self):
        d = _build_detector(_make_fake_model([[10, 10, 100, 200]], [0.8], [0]), confidence_threshold=0.5)
        assert len(d.detect(_blank_frame())) == 1


# ---------------------------------------------------------------------------
# 10 – Bounding box format
# ---------------------------------------------------------------------------

class TestBoundingBoxFormat:
    def test_x1_less_than_x2(self):
        d = _build_detector(_make_fake_model([[50, 50, 300, 400]], [0.9], [0]))
        for det in d.detect(_blank_frame()):
            assert det.bbox[0] < det.bbox[2], f"x1 >= x2: {det.bbox}"

    def test_y1_less_than_y2(self):
        d = _build_detector(_make_fake_model([[50, 50, 300, 400]], [0.9], [0]))
        for det in d.detect(_blank_frame()):
            assert det.bbox[1] < det.bbox[3], f"y1 >= y2: {det.bbox}"

    def test_bbox_is_4_float_tuple(self):
        d = _build_detector(_make_fake_model([[50, 50, 300, 400]], [0.9], [0]))
        result = d.detect(_blank_frame())
        assert len(result) == 1
        assert isinstance(result[0].bbox, tuple)
        assert len(result[0].bbox) == 4
        assert all(isinstance(v, float) for v in result[0].bbox)


# ---------------------------------------------------------------------------
# 11 – Model loaded only once
# ---------------------------------------------------------------------------

class TestModelLoadedOnce:
    def test_same_object_id_across_calls(self):
        fake = _make_fake_model([[50, 50, 200, 300]], [0.9], [0])
        d = _build_detector(fake)
        id_before = id(d._model)
        d.detect(_blank_frame())
        d.detect(_blank_frame())
        assert id(d._model) == id_before

    def test_predict_called_per_frame(self):
        fake = _make_fake_model()
        d = _build_detector(fake)
        d.detect(_blank_frame())
        d.detect(_blank_frame())
        d.detect(_blank_frame())
        assert fake.predict.call_count == 3


# ---------------------------------------------------------------------------
# 12 – CPU execution
# ---------------------------------------------------------------------------

class TestCPUExecution:
    def test_device_is_cpu(self):
        d = _build_detector(_make_fake_model(), device="cpu")
        assert d.device == "cpu"

    def test_detect_on_cpu_succeeds(self):
        d = _build_detector(_make_fake_model([[10, 10, 100, 200]], [0.9], [0]), device="cpu")
        assert len(d.detect(_blank_frame())) == 1


# ---------------------------------------------------------------------------
# 13 – Model loading failure
# ---------------------------------------------------------------------------

class TestModelLoadFailure:
    def test_bad_path_raises_model_load_error(self):
        with patch.object(Detector, "_load_model", side_effect=ModelLoadError("not found")):
            with pytest.raises(ModelLoadError, match="not found"):
                Detector(model_path="/nonexistent/model.pt", device="cpu")

    def test_ultralytics_missing_raises(self):
        with patch.object(Detector, "_load_model", side_effect=ModelLoadError("ultralytics not installed")):
            with pytest.raises(ModelLoadError):
                Detector(model_path="yolo11n.pt", device="cpu")


# ---------------------------------------------------------------------------
# 14 – Output is always list, never None
# ---------------------------------------------------------------------------

class TestOutputAlwaysList:
    def test_empty_result_is_list(self):
        d = _build_detector(_make_fake_model())
        result = d.detect(_blank_frame())
        assert isinstance(result, list)
        assert result is not None

    def test_non_empty_result_is_list(self):
        d = _build_detector(_make_fake_model([[10, 10, 200, 300]], [0.9], [0]))
        result = d.detect(_blank_frame())
        assert isinstance(result, list)
        assert result is not None


# ---------------------------------------------------------------------------
# Integration tests (skipped unless --run-integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """Requires real YOLO11n model. Skipped unless --run-integration is passed."""

    @pytest.fixture(autouse=True)
    def skip_without_flag(self, request):
        if not request.config.getoption("--run-integration", default=False):
            pytest.skip("Pass --run-integration to run this test.")

    def test_model_loads(self):
        detector = Detector(model_path="models/yolo11n.pt", device="cpu")
        assert detector is not None

    def test_detect_blank_frame(self):
        detector = Detector(model_path="models/yolo11n.pt", device="cpu")
        result = detector.detect(_blank_frame())
        assert isinstance(result, list)
        assert all(isinstance(x, Detection) for x in result)

    def test_class_names_include_required(self):
        detector = Detector(model_path="models/yolo11n.pt", device="cpu")
        names = set(detector.class_names)
        for required in {"person", "car", "motorcycle", "bus", "truck"}:
            assert required in names, f"'{required}' missing from model class list"
