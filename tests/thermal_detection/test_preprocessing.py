"""
Unit tests for thermal preprocessing pipeline.

Tests:
- Frame validation (None, empty, wrong ndim, wrong dtype, NaN/Inf)
- Normalization (uint8, uint16, float32, constant, NaN, Inf)
- CLAHE contrast enhancement
- Denoising
- 3-channel conversion
- Full preprocessing pipeline
"""

from __future__ import annotations

import numpy as np
import pytest

from src.thermal_detection.config import ThermalDetectorConfig
from src.thermal_detection.exceptions import ThermalInvalidFrameError
from src.thermal_detection.thermal_preprocessing import (
    ThermalPreprocessor,
    apply_clahe,
    apply_denoising,
    normalize_thermal,
    to_3channel,
    validate_thermal_frame,
)


# -----------------------------------------------------------------------
# Frame validation
# -----------------------------------------------------------------------


class TestValidateThermalFrame:
    """Test validate_thermal_frame()."""

    def test_none_frame(self):
        with pytest.raises(ThermalInvalidFrameError, match="None"):
            validate_thermal_frame(None)

    def test_non_ndarray(self):
        with pytest.raises(ThermalInvalidFrameError, match="numpy.ndarray"):
            validate_thermal_frame([[1, 2], [3, 4]])  # type: ignore[arg-type]

    def test_empty_frame(self):
        with pytest.raises(ThermalInvalidFrameError, match="empty"):
            validate_thermal_frame(np.array([], dtype=np.uint8))

    def test_zero_height(self):
        with pytest.raises(ThermalInvalidFrameError, match="empty"):
            validate_thermal_frame(
                np.empty((0, 100), dtype=np.uint8)
            )

    def test_zero_width(self):
        with pytest.raises(ThermalInvalidFrameError, match="empty"):
            validate_thermal_frame(
                np.empty((100, 0), dtype=np.uint8)
            )

    def test_1d_frame(self):
        with pytest.raises(ThermalInvalidFrameError, match="2-D.*3-D"):
            validate_thermal_frame(np.zeros(100, dtype=np.uint8))

    def test_4d_frame(self):
        with pytest.raises(ThermalInvalidFrameError, match="2-D.*3-D"):
            validate_thermal_frame(
                np.zeros((1, 100, 100, 3), dtype=np.uint8)
            )

    def test_wrong_channels(self):
        with pytest.raises(ThermalInvalidFrameError, match="1 or 3 channels"):
            validate_thermal_frame(
                np.zeros((100, 100, 2), dtype=np.uint8)
            )

    def test_unsupported_dtype(self):
        with pytest.raises(ThermalInvalidFrameError, match="dtype"):
            validate_thermal_frame(
                np.zeros((100, 100), dtype=np.complex64)
            )

    def test_valid_2d_uint8(self):
        validate_thermal_frame(np.zeros((100, 100), dtype=np.uint8))

    def test_valid_2d_uint16(self):
        validate_thermal_frame(np.zeros((100, 100), dtype=np.uint16))

    def test_valid_2d_float32(self):
        validate_thermal_frame(np.zeros((100, 100), dtype=np.float32))

    def test_valid_3d_1channel(self):
        validate_thermal_frame(np.zeros((100, 100, 1), dtype=np.uint8))

    def test_valid_3d_3channel(self):
        validate_thermal_frame(np.zeros((100, 100, 3), dtype=np.uint8))


# -----------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------


class TestNormalizeThermal:
    """Test normalize_thermal()."""

    def test_uint8_passthrough(self):
        """uint8 frames should be returned unchanged."""
        frame = np.array([[0, 128, 255]], dtype=np.uint8)
        result = normalize_thermal(frame)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, frame)

    def test_uint16_scaling(self):
        """uint16 frames should be scaled to [0, 255]."""
        frame = np.array([[0, 32768, 65535]], dtype=np.uint16)
        result = normalize_thermal(frame)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255
        # Middle value should be approximately 128
        assert 125 <= result[0, 1] <= 130

    def test_float32_scaling(self):
        """float32 frames should be min-max scaled to [0, 255]."""
        frame = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = normalize_thermal(frame)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255
        assert 125 <= result[0, 1] <= 130

    def test_float32_arbitrary_range(self):
        """float32 with non-[0,1] range should still normalize correctly."""
        frame = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
        result = normalize_thermal(frame)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_constant_frame(self):
        """Constant-value frame should return mid-gray (128)."""
        frame = np.full((50, 50), 42.0, dtype=np.float32)
        result = normalize_thermal(frame)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, 128)

    def test_nan_handling(self):
        """NaN values should be replaced with 0 and not crash."""
        frame = np.array([[np.nan, 0.5, 1.0]], dtype=np.float32)
        result = normalize_thermal(frame)
        assert result.dtype == np.uint8
        assert not np.any(np.isnan(result.astype(float)))

    def test_inf_handling(self):
        """Inf values should be replaced with 0 and not crash."""
        frame = np.array([[np.inf, 0.5, 1.0]], dtype=np.float32)
        result = normalize_thermal(frame)
        assert result.dtype == np.uint8
        assert not np.any(np.isinf(result.astype(float)))

    def test_all_nan_frame(self):
        """Frame of all NaN → should become constant → mid-gray."""
        frame = np.full((10, 10), np.nan, dtype=np.float32)
        result = normalize_thermal(frame)
        assert result.dtype == np.uint8
        # After NaN → 0, it's a constant frame → mid-gray
        np.testing.assert_array_equal(result, 128)

    def test_does_not_modify_input(self):
        """Normalization must not modify the input array."""
        frame = np.array([[0, 32768, 65535]], dtype=np.uint16)
        original = frame.copy()
        normalize_thermal(frame)
        np.testing.assert_array_equal(frame, original)


# -----------------------------------------------------------------------
# CLAHE
# -----------------------------------------------------------------------


class TestApplyCLAHE:
    """Test apply_clahe()."""

    def test_output_shape(self):
        frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = apply_clahe(frame)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_low_contrast_enhanced(self):
        """A low-contrast frame should have increased contrast after CLAHE."""
        frame = np.random.randint(100, 110, (100, 100), dtype=np.uint8)
        result = apply_clahe(frame, clip_limit=3.0)
        # CLAHE should increase the dynamic range
        assert result.max() - result.min() >= frame.max() - frame.min()


# -----------------------------------------------------------------------
# Denoising
# -----------------------------------------------------------------------


class TestApplyDenoising:
    """Test apply_denoising()."""

    def test_output_shape(self):
        frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = apply_denoising(frame, kernel_size=5)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_reduces_noise(self):
        """Denoising a noisy frame should reduce pixel variance."""
        np.random.seed(42)
        frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = apply_denoising(frame, kernel_size=5)
        assert result.var() < frame.var()


# -----------------------------------------------------------------------
# 3-channel conversion
# -----------------------------------------------------------------------


class TestTo3Channel:
    """Test to_3channel()."""

    def test_output_shape(self):
        frame = np.zeros((100, 100), dtype=np.uint8)
        result = to_3channel(frame)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_channels_identical(self):
        """All three channels should be identical for a grayscale input."""
        frame = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result = to_3channel(frame)
        np.testing.assert_array_equal(result[:, :, 0], frame)
        np.testing.assert_array_equal(result[:, :, 1], frame)
        np.testing.assert_array_equal(result[:, :, 2], frame)


# -----------------------------------------------------------------------
# Full preprocessor pipeline
# -----------------------------------------------------------------------


class TestThermalPreprocessor:
    """Test the full ThermalPreprocessor class."""

    def _make_config(self, **kwargs) -> ThermalDetectorConfig:
        defaults = {"model_path": "dummy.pt"}
        defaults.update(kwargs)
        return ThermalDetectorConfig(**defaults)

    def test_uint8_grayscale(self):
        config = self._make_config(enable_clahe=False, enable_denoising=False)
        preprocessor = ThermalPreprocessor(config)
        frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        preprocessed, normalized = preprocessor.preprocess(frame)
        assert preprocessed.shape == (100, 100, 3)
        assert preprocessed.dtype == np.uint8
        assert normalized.shape == (100, 100)
        assert normalized.dtype == np.uint8

    def test_uint16_grayscale(self):
        config = self._make_config(enable_clahe=False, enable_denoising=False)
        preprocessor = ThermalPreprocessor(config)
        frame = np.random.randint(0, 65536, (80, 120), dtype=np.uint16)
        preprocessed, normalized = preprocessor.preprocess(frame)
        assert preprocessed.shape == (80, 120, 3)
        assert preprocessed.dtype == np.uint8

    def test_float32_grayscale(self):
        config = self._make_config(enable_clahe=False, enable_denoising=False)
        preprocessor = ThermalPreprocessor(config)
        frame = np.random.rand(60, 80).astype(np.float32)
        preprocessed, normalized = preprocessor.preprocess(frame)
        assert preprocessed.shape == (60, 80, 3)
        assert preprocessed.dtype == np.uint8

    def test_with_clahe(self):
        config = self._make_config(enable_clahe=True, enable_denoising=False)
        preprocessor = ThermalPreprocessor(config)
        frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        preprocessed, normalized = preprocessor.preprocess(frame)
        assert preprocessed.shape == (100, 100, 3)

    def test_with_denoising(self):
        config = self._make_config(enable_clahe=False, enable_denoising=True)
        preprocessor = ThermalPreprocessor(config)
        frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        preprocessed, normalized = preprocessor.preprocess(frame)
        assert preprocessed.shape == (100, 100, 3)

    def test_with_clahe_and_denoising(self):
        config = self._make_config(enable_clahe=True, enable_denoising=True)
        preprocessor = ThermalPreprocessor(config)
        frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        preprocessed, normalized = preprocessor.preprocess(frame)
        assert preprocessed.shape == (100, 100, 3)

    def test_single_channel_3d(self):
        """H×W×1 input should be handled."""
        config = self._make_config(enable_clahe=False, enable_denoising=False)
        preprocessor = ThermalPreprocessor(config)
        frame = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
        preprocessed, normalized = preprocessor.preprocess(frame)
        assert preprocessed.shape == (100, 100, 3)

    def test_none_raises(self):
        config = self._make_config()
        preprocessor = ThermalPreprocessor(config)
        with pytest.raises(ThermalInvalidFrameError):
            preprocessor.preprocess(None)  # type: ignore[arg-type]

    def test_empty_raises(self):
        config = self._make_config()
        preprocessor = ThermalPreprocessor(config)
        with pytest.raises(ThermalInvalidFrameError):
            preprocessor.preprocess(np.array([], dtype=np.uint8))
