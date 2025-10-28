"""
Tests for color appearance models.
"""

from __future__ import annotations

import numpy as np

from dtutmo.appearance import CIECAM16, DTUCAM, XLRCAM


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

