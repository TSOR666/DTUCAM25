"""Tests for glare point spread function handling."""

from __future__ import annotations

import numpy as np

from dtutmo.optics import GlareModel, GlareParameters


def test_effective_psf_matches_image_size() -> None:
    params = GlareParameters(psf_samples=512)
    model = GlareModel(params)
    image_shape = (32, 48)

    psf = model.get_effective_psf(4.0, image_shape)

    assert psf.shape[0] <= 2 * image_shape[0] + 1
    assert psf.shape[1] <= 2 * image_shape[1] + 1
    np.testing.assert_allclose(np.sum(psf), 1.0, rtol=1e-6, atol=1e-6)


def test_apply_spectral_glare_preserves_shape() -> None:
    rng = np.random.default_rng(0)
    img = rng.random((16, 16, 3))
    pupil_map = np.full((16, 16), 4.0)

    params = GlareParameters(psf_samples=128)
    model = GlareModel(params)
    result = model.apply_spectral_glare(img, pupil_map)

    assert result.shape == img.shape
    assert np.isfinite(result).all()
