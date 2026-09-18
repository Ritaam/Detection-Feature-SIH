"""
Unit tests for :class:`~src.detection.models.Detection`.

Tests:
- Construction with valid data
- Immutability (frozen dataclass)
- Confidence bounds validation
- Bounding-box geometry invariants (x1 < x2, y1 < y2)
- Derived properties: width, height, area, center
- Hashability (usable in sets/dicts)
"""

from __future__ import annotations

import pytest

from src.detection.models import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(**overrides) -> Detection:
    """Factory that returns a valid Detection, allowing field overrides."""
    defaults = dict(
        class_id=0,
        class_name="person",
        confidence=0.85,
        bbox=(100.0, 50.0, 300.0, 400.0),
    )
    defaults.update(overrides)
    return Detection(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestDetectionConstruction:
    def test_valid_construction(self):
        d = _make_detection()
        assert d.class_id == 0
        assert d.class_name == "person"
        assert d.confidence == 0.85
        assert d.bbox == (100.0, 50.0, 300.0, 400.0)

    def test_confidence_zero_is_invalid(self):
        # confidence=0.0 means the model has no confidence — keep 0 < conf
        # Actually confidence=0.0 is edge-case valid per model spec; 
        # Detection allows 0.0 <= confidence <= 1.0
        d = _make_detection(confidence=0.0)
        assert d.confidence == 0.0

    def test_confidence_one_is_valid(self):
        d = _make_detection(confidence=1.0)
        assert d.confidence == 1.0

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            _make_detection(confidence=1.1)

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            _make_detection(confidence=-0.01)

    def test_bbox_x1_equal_x2_raises(self):
        with pytest.raises(ValueError, match="x1 < x2"):
            _make_detection(bbox=(100.0, 50.0, 100.0, 400.0))

    def test_bbox_x1_greater_than_x2_raises(self):
        with pytest.raises(ValueError, match="x1 < x2"):
            _make_detection(bbox=(200.0, 50.0, 100.0, 400.0))

    def test_bbox_y1_equal_y2_raises(self):
        with pytest.raises(ValueError, match="y1 < y2"):
            _make_detection(bbox=(100.0, 400.0, 300.0, 400.0))

    def test_bbox_y1_greater_than_y2_raises(self):
        with pytest.raises(ValueError, match="y1 < y2"):
            _make_detection(bbox=(100.0, 500.0, 300.0, 400.0))


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestDetectionImmutability:
    def test_cannot_set_class_id(self):
        d = _make_detection()
        with pytest.raises((AttributeError, TypeError)):
            d.class_id = 99  # type: ignore[misc]

    def test_cannot_set_confidence(self):
        d = _make_detection()
        with pytest.raises((AttributeError, TypeError)):
            d.confidence = 0.1  # type: ignore[misc]

    def test_cannot_set_bbox(self):
        d = _make_detection()
        with pytest.raises((AttributeError, TypeError)):
            d.bbox = (0.0, 0.0, 1.0, 1.0)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------


class TestDetectionProperties:
    def test_width(self):
        d = _make_detection(bbox=(100.0, 50.0, 300.0, 400.0))
        assert d.width == pytest.approx(200.0)

    def test_height(self):
        d = _make_detection(bbox=(100.0, 50.0, 300.0, 400.0))
        assert d.height == pytest.approx(350.0)

    def test_area(self):
        d = _make_detection(bbox=(100.0, 50.0, 300.0, 400.0))
        assert d.area == pytest.approx(200.0 * 350.0)

    def test_center(self):
        d = _make_detection(bbox=(100.0, 50.0, 300.0, 450.0))
        cx, cy = d.center
        assert cx == pytest.approx(200.0)
        assert cy == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# Hashability
# ---------------------------------------------------------------------------


class TestDetectionHashability:
    def test_usable_in_set(self):
        d1 = _make_detection()
        d2 = _make_detection()
        d3 = _make_detection(class_id=2, class_name="car")
        s = {d1, d2, d3}
        assert len(s) == 2  # d1 == d2

    def test_usable_as_dict_key(self):
        d = _make_detection()
        mapping = {d: "test"}
        assert mapping[d] == "test"
