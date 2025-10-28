"""
Tests for the DTUTMO pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from dtutmo import (
    CAMType,
    CompleteDTUTMO,
    DTUTMOConfig,
    DisplayMapping,
    DisplayStandard,
    tone_map_hdr,
)


def test_basic_processing() -> None:
    img_hdr = np.random.rand(64, 64, 3) * 100.0
    tmo = CompleteDTUTMO()
    result = tmo.process(img_hdr)
    assert result.shape == img_hdr.shape
    assert np.isfinite(result).all()
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_dtucam_processing() -> None:
    img_hdr = np.random.rand(32, 32, 3) * 500.0
    config = DTUTMOConfig(use_cam=CAMType.DTUCAM)
    tmo = CompleteDTUTMO(config)
    result = tmo.process(img_hdr)
    assert result.shape == img_hdr.shape
    assert np.isfinite(result).all()


def test_xlrcam_processing() -> None:
    img_hdr = np.random.rand(64, 64, 3) * 1000.0
    config = DTUTMOConfig(use_cam=CAMType.XLRCAM)
    tmo = CompleteDTUTMO(config)
    result = tmo.process(img_hdr)
    assert result.shape == img_hdr.shape
    assert np.isfinite(result).all()


def test_ciecam16_processing() -> None:
    img_hdr = np.random.rand(64, 64, 3) * 1000.0
    config = DTUTMOConfig(use_cam=CAMType.CIECAM16)
    tmo = CompleteDTUTMO(config)
    result = tmo.process(img_hdr)
    assert result.shape == img_hdr.shape
    assert np.isfinite(result).all()


def test_processing_without_cam() -> None:
    img_hdr = np.random.rand(32, 32, 3) * 200.0
    config = DTUTMOConfig(use_cam=CAMType.NONE)
    tmo = CompleteDTUTMO(config)
    result = tmo.process(img_hdr)
    assert result.shape == img_hdr.shape


def test_intermediate_results() -> None:
    img_hdr = np.random.rand(16, 16, 3) * 100.0
    tmo = CompleteDTUTMO()
    results = tmo.process(img_hdr, return_intermediate=True)
    assert "input" in results
    assert "output" in results
    assert "adaptation" in results
    assert "photoreceptors" in results
    photo = results["photoreceptors"]
    assert "cones" in photo
    assert "cone_channels" in photo
    assert tuple(photo["cone_channels"]) == ("R", "G", "B")
    assert photo["cones"].shape[-1] == 3


def test_hdr10_output() -> None:
    img_hdr = np.random.rand(64, 64, 3) * 5000.0
    config = DTUTMOConfig(target_display=DisplayStandard.REC_2100_PQ)
    tmo = CompleteDTUTMO(config)
    result = tmo.process(img_hdr)
    assert result.shape == img_hdr.shape
    assert result.max() <= 1.0


def test_valid_config() -> None:
    config = DTUTMOConfig(observer_age=30.0, field_diameter=60.0)
    config.validate()


def test_invalid_age() -> None:
    config = DTUTMOConfig(observer_age=150.0)
    with pytest.raises(ValueError):
        config.validate()


def test_invalid_field() -> None:
    config = DTUTMOConfig(field_diameter=200.0)
    with pytest.raises(ValueError):
        config.validate()


def test_invalid_image_shape_raises() -> None:
    tmo = CompleteDTUTMO()
    invalid = np.random.rand(32, 32)
    with pytest.raises(ValueError):
        tmo.process(invalid)


def test_nan_input_raises() -> None:
    tmo = CompleteDTUTMO()
    img_hdr = np.random.rand(8, 8, 3)
    img_hdr[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        tmo.process(img_hdr)


def test_production_hybrid_mapping_without_cam() -> None:
    img_hdr = np.random.rand(16, 16, 3) * 500.0
    config = DTUTMOConfig(
        use_cam=CAMType.NONE, display_mapping=DisplayMapping.PRODUCTION_HYBRID
    )
    tmo = CompleteDTUTMO(config)
    result = tmo.process(img_hdr)
    assert result.shape == img_hdr.shape
    assert np.isfinite(result).all()


def test_tone_map_hdr_helper() -> None:
    img_hdr = np.random.rand(10, 10, 3) * 100.0
    result = tone_map_hdr(img_hdr)
    assert result.shape == img_hdr.shape
    assert np.isfinite(result).all()
