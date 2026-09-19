"""
Unit tests for hotspot segmentation.

Tests:
- Dark frame → 0 hotspots
- Frame with bright blob → 1+ hotspots
- Hotspot bbox geometry is valid
- Output is ThermalHotspot (not classified)
- Min area filtering
- Sorted by area (largest first)
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.thermal_detection.hotspot_segmentation import (
    HotspotSegmenter,
    ThermalHotspot,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_frame_with_bright_blob(
    h: int = 200,
    w: int = 300,
    blob_center: tuple[int, int] = (150, 100),
    blob_radius: int = 30,
    blob_intensity: int = 220,
) -> np.ndarray:
    """Create a uint8 grayscale frame with a single bright circular blob."""
    frame = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(
        frame, blob_center, blob_radius, int(blob_intensity), cv2.FILLED
    )
    return frame


# -----------------------------------------------------------------------
# Basic segmentation
# -----------------------------------------------------------------------


class TestHotspotSegmentation:
    """Test HotspotSegmenter.segment()."""

    def test_dark_frame_no_hotspots(self):
        """All-dark frame should produce zero hotspots."""
        frame = np.zeros((200, 300), dtype=np.uint8)
        segmenter = HotspotSegmenter(intensity_threshold=180, min_area=100)
        hotspots = segmenter.segment(frame)
        assert hotspots == []

    def test_single_bright_blob(self):
        """Frame with one bright blob should produce at least one hotspot."""
        frame = _make_frame_with_bright_blob()
        segmenter = HotspotSegmenter(intensity_threshold=180, min_area=100)
        hotspots = segmenter.segment(frame)
        assert len(hotspots) >= 1

    def test_hotspot_is_not_classified(self):
        """ThermalHotspot should NOT have class_name or class_id."""
        frame = _make_frame_with_bright_blob()
        segmenter = HotspotSegmenter(intensity_threshold=180, min_area=100)
        hotspots = segmenter.segment(frame)
        assert len(hotspots) >= 1
        h = hotspots[0]
        assert isinstance(h, ThermalHotspot)
        assert not hasattr(h, "class_name")
        assert not hasattr(h, "class_id")

    def test_hotspot_bbox_geometry(self):
        """Hotspot bbox should have x1 < x2 and y1 < y2."""
        frame = _make_frame_with_bright_blob()
        segmenter = HotspotSegmenter(intensity_threshold=180, min_area=100)
        hotspots = segmenter.segment(frame)
        for h in hotspots:
            x1, y1, x2, y2 = h.bbox
            assert x1 < x2, f"x1={x1} >= x2={x2}"
            assert y1 < y2, f"y1={y1} >= y2={y2}"

    def test_hotspot_area_positive(self):
        frame = _make_frame_with_bright_blob()
        segmenter = HotspotSegmenter(intensity_threshold=180, min_area=100)
        hotspots = segmenter.segment(frame)
        for h in hotspots:
            assert h.area > 0

    def test_mean_intensity_in_range(self):
        frame = _make_frame_with_bright_blob(blob_intensity=230)
        segmenter = HotspotSegmenter(intensity_threshold=180, min_area=100)
        hotspots = segmenter.segment(frame)
        for h in hotspots:
            assert 0.0 <= h.mean_intensity <= 255.0


# -----------------------------------------------------------------------
# Filtering
# -----------------------------------------------------------------------


class TestHotspotFiltering:
    """Test area filtering and threshold sensitivity."""

    def test_min_area_filters_small(self):
        """Blobs smaller than min_area should be filtered out."""
        frame = _make_frame_with_bright_blob(blob_radius=3)
        segmenter = HotspotSegmenter(
            intensity_threshold=180, min_area=50000
        )
        hotspots = segmenter.segment(frame)
        assert len(hotspots) == 0

    def test_high_threshold_filters_dim_blobs(self):
        """Blobs below the intensity threshold should not be detected."""
        frame = _make_frame_with_bright_blob(blob_intensity=150)
        segmenter = HotspotSegmenter(intensity_threshold=200, min_area=50)
        hotspots = segmenter.segment(frame)
        assert len(hotspots) == 0

    def test_multiple_blobs_sorted_by_area(self):
        """Multiple blobs should be returned sorted by area (largest first)."""
        frame = np.zeros((400, 400), dtype=np.uint8)
        # Small blob
        cv2.circle(frame, (100, 100), 15, 220, cv2.FILLED)
        # Large blob
        cv2.circle(frame, (300, 300), 50, 230, cv2.FILLED)
        segmenter = HotspotSegmenter(intensity_threshold=180, min_area=50)
        hotspots = segmenter.segment(frame)
        assert len(hotspots) >= 2
        # Largest first
        assert hotspots[0].area >= hotspots[1].area


# -----------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------


class TestHotspotInputValidation:
    """Test input validation in segment()."""

    def test_none_raises(self):
        segmenter = HotspotSegmenter()
        with pytest.raises(ValueError, match="numpy.ndarray"):
            segmenter.segment(None)  # type: ignore[arg-type]

    def test_3d_raises(self):
        segmenter = HotspotSegmenter()
        with pytest.raises(ValueError, match="2-D"):
            segmenter.segment(
                np.zeros((100, 100, 3), dtype=np.uint8)
            )

    def test_non_uint8_raises(self):
        segmenter = HotspotSegmenter()
        with pytest.raises(ValueError, match="uint8"):
            segmenter.segment(
                np.zeros((100, 100), dtype=np.float32)
            )


# -----------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------


class TestHotspotConstructorValidation:
    """Test constructor parameter validation."""

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="intensity_threshold"):
            HotspotSegmenter(intensity_threshold=-1)

    def test_threshold_above_255_raises(self):
        with pytest.raises(ValueError, match="intensity_threshold"):
            HotspotSegmenter(intensity_threshold=256)

    def test_negative_min_area_raises(self):
        with pytest.raises(ValueError, match="min_area"):
            HotspotSegmenter(min_area=-10)
