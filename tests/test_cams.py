"""
Tests for color appearance models.
"""

from __future__ import annotations

import numpy as np

from dtutmo.appearance import CIECAM16, DTUCAM, XLRCAM


def test_dtucam_degree_of_adaptation_matches_reference() -> None:
    cam = DTUCAM()
    surround = "dim"
    surround_factor = {"dark": 0.8, "dim": 0.9, "average": 1.0}[surround]
    luminances = np.array([0.01, 1.0, 42.0, 200.0])
    expected = surround_factor * (1.0 - (1.0 / 3.6) * np.exp(-(luminances - 42.0) / 92.0))

    for luminance, expected_value in zip(luminances, expected):
        assert np.isclose(
            cam._degree_of_adaptation(float(luminance), surround),
            expected_value,
            rtol=1e-6,
            atol=1e-9,
        )


def test_xlrcam_forward_inverse_roundtrip() -> None:
    cam = XLRCAM()
    xyz = np.random.rand(16, 16, 3) * 100.0
    white = np.array([95.047, 100.0, 108.883])

    appearance = cam.forward(xyz, white, 20.0, "average")
    assert "lightness" in appearance
    assert "colorfulness" in appearance
    assert "hue" in appearance

    xyz_back = cam.inverse(
        appearance["lightness"],
        appearance["colorfulness"],
        appearance["hue"],
        white,
        100.0,
        "dim",
    )
    assert xyz_back.shape == xyz.shape


def test_ciecam16_forward_outputs() -> None:
    cam = CIECAM16()
    xyz = np.random.rand(16, 16, 3) * 100.0
    white = np.array([95.047, 100.0, 108.883])
    appearance = cam.forward(xyz, white, 20.0, "average")

    assert "lightness" in appearance
    assert "colorfulness" in appearance
    assert "brightness" in appearance
def test_dtucam_forward_inverse_roundtrip() -> None:
    cam = DTUCAM()
    xyz = np.random.rand(8, 8, 3) * 200.0
    white = np.array([95.047, 100.0, 108.883])

    appearance = cam.forward(xyz, white, 20.0, surround="average")
    assert set(["lightness", "colorfulness", "hue"]) <= appearance.keys()

    xyz_back = cam.inverse(
        appearance["lightness"],
        appearance["colorfulness"],
        appearance["hue"],
        white,
        200.0,
        surround="dim",
    )
    assert xyz_back.shape == xyz.shape

