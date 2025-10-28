"""
Factory utilities for selecting a color appearance model.
"""

from __future__ import annotations

from dtutmo.core.config import CAMType, DTUTMOConfig


def create_cam(cam_type: CAMType, *, config: DTUTMOConfig | None = None):
    """Instantiate the requested color appearance model."""

    if cam_type == CAMType.DTUCAM:
        from dtutmo.appearance.dtucam import DTUCAM

        kwargs = {}
        if config is not None:
            kwargs = {
                "observer_age": config.observer_age,
                "field_diameter": config.field_diameter,
                "peak_sensitivity": config.peak_sensitivity,
                "pixels_per_degree": config.pixels_per_degree,
            }
        return DTUCAM(**kwargs)

    if cam_type == CAMType.XLRCAM:
        from dtutmo.appearance.xlr_cam import XLRCAM

        return XLRCAM()

    if cam_type == CAMType.CIECAM16:
        from dtutmo.appearance.ciecam16 import CIECAM16

        return CIECAM16()

    if cam_type == CAMType.NONE:
        return None

    raise ValueError(f"Unknown CAM type: {cam_type}")
