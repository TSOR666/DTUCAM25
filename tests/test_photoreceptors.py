"""
Tests for photoreceptor models.
"""

from __future__ import annotations

import numpy as np

from dtutmo.photoreceptors import CorrectedPhotoreceptorResponse
from dtutmo.photoreceptors.constants import PHOTORECEPTOR_CHANNELS


def test_pupil_diameter_model() -> None:
    photo = CorrectedPhotoreceptorResponse()
    lum = np.array([[10.0, 100.0], [1000.0, 10000.0]])
    pupil = photo.pupil_diameter_watson(lum, age=30.0, field_deg=60.0)
    assert pupil.shape == lum.shape
    assert (pupil >= 2.0).all() and (pupil <= 8.0).all()


def test_cone_response_shapes() -> None:
    photo = CorrectedPhotoreceptorResponse()
    lms = np.random.rand(32, 32, 3) * 100.0
    adapt = np.random.rand(32, 32) * 50.0
    response = photo.process_cones_original(lms, adapt, pupil_size=4.0)
    assert response.shape == lms.shape
    assert np.isfinite(response).all()


def test_channel_mapping_alignment() -> None:
    photo = CorrectedPhotoreceptorResponse()
    assert photo.channel_order == PHOTORECEPTOR_CHANNELS
    for channel in PHOTORECEPTOR_CHANNELS:
        cone_type = photo.cone_for_channel(channel)
        assert cone_type.endswith("cone")
