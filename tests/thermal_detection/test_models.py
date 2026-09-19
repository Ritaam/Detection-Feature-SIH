"""
Unit tests for ThermalDetection data model.

Tests:
- Construction with valid parameters
- Invariant violations (bad confidence, bad bbox geometry)
- Computed properties (width, height, area, center)
- Optional thermal metadata
- Immutability (frozen dataclass)
"""

from __future__ import annotations

import pytest

from src.thermal_detection.models import ThermalDetection


# -----------------------------------------------------------------------
# Construction — valid
# -----------------------------------------------------------------------


class TestThermalDetectionConstruction:
    """Test creating ThermalDetection with valid arguments."""

    def test_minimal_construction(self):
        d = ThermalDetection(
            bbox=(10.0, 20.0, 100.0, 200.0),
            confidence=0.95,
            class_id=0,
            class_name="person",
        )
        assert d.bbox == (10.0, 20.0, 100.0, 200.0)
        assert d.confidence == 0.95
        assert d.class_id == 0
        assert d.class_name == "person"
        assert d.mean_intensity is None
        assert d.temperature_range is None

    def test_with_mean_intensity(self):
        d = ThermalDetection(
            bbox=(10.0, 20.0, 100.0, 200.0),
            confidence=0.8,
            class_id=2,
            class_name="car",
            mean_intensity=195.5,
        )
        assert d.mean_intensity == 195.5

    def test_with_temperature_range(self):
        d = ThermalDetection(
            bbox=(10.0, 20.0, 100.0, 200.0),
            confidence=0.8,
            class_id=0,
            class_name="person",
            temperature_range=(32.0, 37.5),
        )
        assert d.temperature_range == (32.0, 37.5)

    def test_confidence_zero(self):
        d = ThermalDetection(
            bbox=(0.0, 0.0, 1.0, 1.0),
            confidence=0.0,
            class_id=0,
            class_name="person",
        )
        assert d.confidence == 0.0

    def test_confidence_one(self):
        d = ThermalDetection(
            bbox=(0.0, 0.0, 1.0, 1.0),
            confidence=1.0,
            class_id=0,
            class_name="person",
        )
        assert d.confidence == 1.0


# -----------------------------------------------------------------------
# Construction — invalid (invariant violations)
# -----------------------------------------------------------------------


class TestThermalDetectionInvariants:
    """Test that invalid construction raises ValueError."""

    def test_confidence_too_high(self):
        with pytest.raises(ValueError, match="confidence"):
            ThermalDetection(
                bbox=(0.0, 0.0, 10.0, 10.0),
                confidence=1.5,
                class_id=0,
                class_name="person",
            )

    def test_confidence_negative(self):
        with pytest.raises(ValueError, match="confidence"):
            ThermalDetection(
                bbox=(0.0, 0.0, 10.0, 10.0),
                confidence=-0.1,
                class_id=0,
                class_name="person",
            )

    def test_bbox_x1_equals_x2(self):
        with pytest.raises(ValueError, match="x1 < x2"):
            ThermalDetection(
                bbox=(50.0, 0.0, 50.0, 10.0),
                confidence=0.9,
                class_id=0,
                class_name="person",
            )

    def test_bbox_x1_greater_than_x2(self):
        with pytest.raises(ValueError, match="x1 < x2"):
            ThermalDetection(
                bbox=(100.0, 0.0, 50.0, 10.0),
                confidence=0.9,
                class_id=0,
                class_name="person",
            )

    def test_bbox_y1_equals_y2(self):
        with pytest.raises(ValueError, match="y1 < y2"):
            ThermalDetection(
                bbox=(0.0, 50.0, 10.0, 50.0),
                confidence=0.9,
                class_id=0,
                class_name="person",
            )

    def test_bbox_y1_greater_than_y2(self):
        with pytest.raises(ValueError, match="y1 < y2"):
            ThermalDetection(
                bbox=(0.0, 100.0, 10.0, 50.0),
                confidence=0.9,
                class_id=0,
                class_name="person",
            )

    def test_temperature_range_inverted(self):
        with pytest.raises(ValueError, match="temperature_range"):
            ThermalDetection(
                bbox=(0.0, 0.0, 10.0, 10.0),
                confidence=0.9,
                class_id=0,
                class_name="person",
                temperature_range=(40.0, 35.0),
            )


# -----------------------------------------------------------------------
# Computed properties
# -----------------------------------------------------------------------


class TestThermalDetectionProperties:
    """Test computed properties."""

    def test_width(self):
        d = ThermalDetection(
            bbox=(10.0, 20.0, 110.0, 120.0),
            confidence=0.5,
            class_id=0,
            class_name="person",
        )
        assert d.width == pytest.approx(100.0)

    def test_height(self):
        d = ThermalDetection(
            bbox=(10.0, 20.0, 110.0, 120.0),
            confidence=0.5,
            class_id=0,
            class_name="person",
        )
        assert d.height == pytest.approx(100.0)

    def test_area(self):
        d = ThermalDetection(
            bbox=(0.0, 0.0, 50.0, 100.0),
            confidence=0.5,
            class_id=0,
            class_name="person",
        )
        assert d.area == pytest.approx(5000.0)

    def test_center(self):
        d = ThermalDetection(
            bbox=(10.0, 20.0, 110.0, 120.0),
            confidence=0.5,
            class_id=0,
            class_name="person",
        )
        cx, cy = d.center
        assert cx == pytest.approx(60.0)
        assert cy == pytest.approx(70.0)


# -----------------------------------------------------------------------
# Immutability
# -----------------------------------------------------------------------


class TestThermalDetectionImmutability:
    """Test that ThermalDetection is frozen."""

    def test_cannot_set_attribute(self):
        d = ThermalDetection(
            bbox=(0.0, 0.0, 10.0, 10.0),
            confidence=0.5,
            class_id=0,
            class_name="person",
        )
        with pytest.raises(AttributeError):
            d.confidence = 0.99  # type: ignore[misc]

    def test_equality(self):
        d1 = ThermalDetection(
            bbox=(1.0, 2.0, 3.0, 4.0),
            confidence=0.5,
            class_id=0,
            class_name="person",
        )
        d2 = ThermalDetection(
            bbox=(1.0, 2.0, 3.0, 4.0),
            confidence=0.5,
            class_id=0,
            class_name="person",
        )
        assert d1 == d2

    def test_hashable(self):
        d = ThermalDetection(
            bbox=(1.0, 2.0, 3.0, 4.0),
            confidence=0.5,
            class_id=0,
            class_name="person",
        )
        # Should not raise
        {d}
        {d: "value"}
