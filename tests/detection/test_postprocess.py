"""
Unit tests for :class:`~src.detection.postprocess.PostProcessor`.

All tests use synthetic NumPy arrays — no model is loaded.

Tests:
- Returns empty list when no boxes are provided
- Confidence filtering: low-confidence boxes are dropped
- Class filtering: unsupported classes are dropped
- Bbox clamping: coordinates clipped to frame boundaries
- Degenerate bbox after clamping: box is dropped
- Correct conversion to Detection objects
- Bounding box geometry guaranteed: x1 < x2, y1 < y2
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.postprocess import PostProcessor

# COCO-style class list subset for tests
COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]

SUPPORTED = frozenset({"person", "car", "motorcycle", "bus", "truck"})
CONF_THRESHOLD = 0.5


def make_processor(**kwargs) -> PostProcessor:
    defaults = dict(
        confidence_threshold=CONF_THRESHOLD,
        supported_classes=SUPPORTED,
    )
    defaults.update(kwargs)
    return PostProcessor(**defaults)


def run(processor, boxes, confs, cls_ids, h=480, w=640, names=None):
    """Convenience wrapper for processor.process()."""
    return processor.process(
        boxes_xyxy=np.array(boxes, dtype=np.float32),
        confidences=np.array(confs, dtype=np.float32),
        class_ids=np.array(cls_ids, dtype=np.int32),
        class_names=names or COCO_NAMES,
        frame_height=h,
        frame_width=w,
    )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_no_boxes_returns_empty_list(self):
        p = make_processor()
        result = run(p, [], [], [])
        assert result == []

    def test_returns_list_not_none(self):
        p = make_processor()
        result = run(p, [], [], [])
        assert result is not None
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Confidence filtering
# ---------------------------------------------------------------------------


class TestConfidenceFiltering:
    def test_below_threshold_dropped(self):
        p = make_processor(confidence_threshold=0.5)
        # person detected at 0.3 — below threshold
        result = run(p, [[10, 20, 100, 200]], [0.3], [0])
        assert result == []

    def test_at_threshold_kept(self):
        p = make_processor(confidence_threshold=0.5)
        result = run(p, [[10, 20, 100, 200]], [0.5], [0])
        assert len(result) == 1

    def test_above_threshold_kept(self):
        p = make_processor(confidence_threshold=0.5)
        result = run(p, [[10, 20, 100, 200]], [0.9], [0])
        assert len(result) == 1

    def test_mixed_confidences(self):
        p = make_processor(confidence_threshold=0.5)
        boxes = [[10, 20, 100, 200], [50, 60, 150, 300]]
        confs = [0.9, 0.2]   # second below threshold
        cls_ids = [0, 0]     # both person
        result = run(p, boxes, confs, cls_ids)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9, abs=1e-5)


# ---------------------------------------------------------------------------
# Class filtering
# ---------------------------------------------------------------------------


class TestClassFiltering:
    def test_unsupported_class_dropped(self):
        p = make_processor()
        # class_id=1 is "bicycle" — not in SUPPORTED
        result = run(p, [[10, 20, 100, 200]], [0.9], [1])
        assert result == []

    def test_supported_class_kept(self):
        p = make_processor()
        # class_id=2 is "car" — in SUPPORTED
        result = run(p, [[10, 20, 100, 200]], [0.9], [2])
        assert len(result) == 1
        assert result[0].class_name == "car"

    def test_all_five_supported_classes(self):
        # person=0, car=2, motorcycle=3, bus=5, truck=7
        boxes = [
            [10, 10, 100, 100],
            [10, 10, 100, 100],
            [10, 10, 100, 100],
            [10, 10, 100, 100],
            [10, 10, 100, 100],
        ]
        confs = [0.9, 0.9, 0.9, 0.9, 0.9]
        cls_ids = [0, 2, 3, 5, 7]
        p = make_processor()
        result = run(p, boxes, confs, cls_ids)
        assert len(result) == 5
        names = {d.class_name for d in result}
        assert names == {"person", "car", "motorcycle", "bus", "truck"}

    def test_mixed_supported_unsupported(self):
        # person(supported) + bicycle(unsupported) + car(supported)
        boxes = [
            [10, 10, 50, 50],
            [60, 60, 120, 120],
            [130, 130, 200, 200],
        ]
        confs = [0.9, 0.9, 0.9]
        cls_ids = [0, 1, 2]
        p = make_processor()
        result = run(p, boxes, confs, cls_ids)
        assert len(result) == 2
        names = {d.class_name for d in result}
        assert names == {"person", "car"}


# ---------------------------------------------------------------------------
# Bounding box clamping
# ---------------------------------------------------------------------------


class TestBboxClamping:
    def test_negative_coords_clamped_to_zero(self):
        p = make_processor()
        # x1=-10 should be clamped to 0
        result = run(p, [[-10, -5, 100, 200]], [0.9], [0], h=480, w=640)
        assert len(result) == 1
        x1, y1, x2, y2 = result[0].bbox
        assert x1 == 0.0
        assert y1 == 0.0

    def test_coords_beyond_frame_clamped(self):
        p = make_processor()
        # x2=700 and y2=500 should be clamped to w=640, h=480
        result = run(p, [[10, 10, 700, 500]], [0.9], [0], h=480, w=640)
        assert len(result) == 1
        x1, y1, x2, y2 = result[0].bbox
        assert x2 == 640.0
        assert y2 == 480.0

    def test_degenerate_after_clamping_dropped(self):
        p = make_processor()
        # x1=650, x2=700 both beyond w=640 → clamped to x1=x2=640 → degenerate
        result = run(p, [[650, 10, 700, 200]], [0.9], [0], h=480, w=640)
        assert result == []


# ---------------------------------------------------------------------------
# Bbox format guarantee
# ---------------------------------------------------------------------------


class TestBboxFormat:
    def test_x1_less_than_x2(self):
        p = make_processor()
        result = run(p, [[50, 50, 200, 300]], [0.9], [0])
        for d in result:
            x1, y1, x2, y2 = d.bbox
            assert x1 < x2, f"Expected x1 < x2, got {x1=} {x2=}"

    def test_y1_less_than_y2(self):
        p = make_processor()
        result = run(p, [[50, 50, 200, 300]], [0.9], [0])
        for d in result:
            x1, y1, x2, y2 = d.bbox
            assert y1 < y2, f"Expected y1 < y2, got {y1=} {y2=}"

    def test_bbox_is_four_tuple(self):
        p = make_processor()
        result = run(p, [[50, 50, 200, 300]], [0.9], [0])
        assert len(result) == 1
        assert isinstance(result[0].bbox, tuple)
        assert len(result[0].bbox) == 4


# ---------------------------------------------------------------------------
# Detection field accuracy
# ---------------------------------------------------------------------------


class TestDetectionFields:
    def test_class_id_correct(self):
        p = make_processor()
        # class_id=2 → "car"
        result = run(p, [[10, 10, 100, 100]], [0.8], [2])
        assert result[0].class_id == 2

    def test_class_name_correct(self):
        p = make_processor()
        result = run(p, [[10, 10, 100, 100]], [0.8], [2])
        assert result[0].class_name == "car"

    def test_confidence_preserved(self):
        p = make_processor()
        result = run(p, [[10, 10, 100, 100]], [0.76543], [0])
        assert result[0].confidence == pytest.approx(0.76543, abs=1e-5)

    def test_out_of_range_class_id_skipped(self):
        p = make_processor()
        # cls_id=999 is way beyond len(COCO_NAMES)
        result = run(p, [[10, 10, 100, 100]], [0.9], [999])
        assert result == []
