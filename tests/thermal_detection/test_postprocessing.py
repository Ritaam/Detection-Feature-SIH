"""
Unit tests for thermal post-processing.

Tests:
- Confidence filtering (above/below threshold)
- Class filtering (allowed/disallowed, None = all)
- BBox clamping (negative coords, out-of-frame coords)
- Degenerate bbox discarded (zero area after clamp)
- Empty input
- Invalid class ids
"""

from __future__ import annotations

import numpy as np
import pytest

from src.thermal_detection.models import ThermalDetection
from src.thermal_detection.postprocessing import ThermalPostProcessor


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

_CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


def _make_processor(
    confidence_threshold: float = 0.25,
    allowed_classes: frozenset[str] | None = frozenset({"person", "car", "truck"}),
) -> ThermalPostProcessor:
    return ThermalPostProcessor(
        confidence_threshold=confidence_threshold,
        allowed_classes=allowed_classes,
    )


# -----------------------------------------------------------------------
# Confidence filtering
# -----------------------------------------------------------------------


class TestConfidenceFiltering:
    """Test that detections below threshold are removed."""

    def test_high_confidence_retained(self):
        proc = _make_processor(confidence_threshold=0.25)
        result = proc.process(
            boxes_xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidences=np.array([0.91], dtype=np.float32),
            class_ids=np.array([0]),
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.91, abs=1e-4)

    def test_low_confidence_removed(self):
        proc = _make_processor(confidence_threshold=0.25)
        result = proc.process(
            boxes_xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidences=np.array([0.10], dtype=np.float32),
            class_ids=np.array([0]),
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 0

    def test_exact_threshold_retained(self):
        proc = _make_processor(confidence_threshold=0.50)
        result = proc.process(
            boxes_xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidences=np.array([0.50], dtype=np.float32),
            class_ids=np.array([0]),
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 1

    def test_mixed_confidences(self):
        proc = _make_processor(confidence_threshold=0.5)
        result = proc.process(
            boxes_xyxy=np.array(
                [[10, 20, 100, 200], [10, 20, 100, 200], [10, 20, 100, 200]],
                dtype=np.float32,
            ),
            confidences=np.array([0.91, 0.10, 0.75], dtype=np.float32),
            class_ids=np.array([0, 0, 0]),
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 2


# -----------------------------------------------------------------------
# Class filtering
# -----------------------------------------------------------------------


class TestClassFiltering:
    """Test that disallowed classes are removed."""

    def test_person_retained(self):
        proc = _make_processor(allowed_classes=frozenset({"person", "car"}))
        result = proc.process(
            boxes_xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidences=np.array([0.9], dtype=np.float32),
            class_ids=np.array([0]),  # person
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 1
        assert result[0].class_name == "person"

    def test_car_retained(self):
        proc = _make_processor(allowed_classes=frozenset({"person", "car"}))
        result = proc.process(
            boxes_xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidences=np.array([0.9], dtype=np.float32),
            class_ids=np.array([2]),  # car
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 1
        assert result[0].class_name == "car"

    def test_bicycle_removed(self):
        proc = _make_processor(allowed_classes=frozenset({"person", "car"}))
        result = proc.process(
            boxes_xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidences=np.array([0.9], dtype=np.float32),
            class_ids=np.array([1]),  # bicycle
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 0

    def test_none_allowed_classes_accepts_all(self):
        """When allowed_classes is None, all classes should pass."""
        proc = _make_processor(allowed_classes=None)
        result = proc.process(
            boxes_xyxy=np.array(
                [[10, 20, 100, 200], [10, 20, 100, 200]],
                dtype=np.float32,
            ),
            confidences=np.array([0.9, 0.9], dtype=np.float32),
            class_ids=np.array([0, 1]),  # person, bicycle
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 2

    def test_invalid_class_id_skipped(self):
        proc = _make_processor()
        result = proc.process(
            boxes_xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidences=np.array([0.9], dtype=np.float32),
            class_ids=np.array([999]),  # out of range
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 0


# -----------------------------------------------------------------------
# Bounding box validation
# -----------------------------------------------------------------------


class TestBBoxValidation:
    """Test bbox clamping and degenerate box discard."""

    def test_valid_box(self):
        proc = _make_processor()
        result = proc.process(
            boxes_xyxy=np.array([[50, 50, 200, 300]], dtype=np.float32),
            confidences=np.array([0.9], dtype=np.float32),
            class_ids=np.array([0]),
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 1
        assert result[0].bbox == (50.0, 50.0, 200.0, 300.0)

    def test_negative_coords_clamped(self):
        proc = _make_processor()
        result = proc.process(
            boxes_xyxy=np.array([[-10, -20, 100, 200]], dtype=np.float32),
            confidences=np.array([0.9], dtype=np.float32),
            class_ids=np.array([0]),
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 1
        x1, y1, x2, y2 = result[0].bbox
        assert x1 == 0.0
        assert y1 == 0.0

    def test_out_of_frame_clamped(self):
        proc = _make_processor()
        result = proc.process(
            boxes_xyxy=np.array([[500, 400, 700, 600]], dtype=np.float32),
            confidences=np.array([0.9], dtype=np.float32),
            class_ids=np.array([0]),
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 1
        x1, y1, x2, y2 = result[0].bbox
        assert x2 <= 640.0
        assert y2 <= 480.0

    def test_zero_area_after_clamp_discarded(self):
        """Box fully outside frame → collapses to zero-area → discarded."""
        proc = _make_processor()
        result = proc.process(
            boxes_xyxy=np.array([[700, 500, 800, 600]], dtype=np.float32),
            confidences=np.array([0.9], dtype=np.float32),
            class_ids=np.array([0]),
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 0

    def test_degenerate_box_x1_equals_x2(self):
        """Box where x1==x2 after clamping should be discarded."""
        proc = _make_processor()
        result = proc.process(
            boxes_xyxy=np.array([[640, 50, 640, 200]], dtype=np.float32),
            confidences=np.array([0.9], dtype=np.float32),
            class_ids=np.array([0]),
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert len(result) == 0


# -----------------------------------------------------------------------
# Empty input
# -----------------------------------------------------------------------


class TestEmptyInput:
    """Test with empty arrays."""

    def test_empty_arrays(self):
        proc = _make_processor()
        result = proc.process(
            boxes_xyxy=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty((0,), dtype=np.float32),
            class_ids=np.empty((0,), dtype=int),
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert result == []


# -----------------------------------------------------------------------
# Output type
# -----------------------------------------------------------------------


class TestOutputType:
    """Test that output is list[ThermalDetection]."""

    def test_output_type(self):
        proc = _make_processor()
        result = proc.process(
            boxes_xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidences=np.array([0.9], dtype=np.float32),
            class_ids=np.array([0]),
            class_names=_CLASS_NAMES,
            frame_height=480,
            frame_width=640,
        )
        assert isinstance(result, list)
        assert all(isinstance(d, ThermalDetection) for d in result)
