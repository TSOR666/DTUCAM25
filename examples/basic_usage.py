"""
Basic usage examples for DTUTMO.
"""

from __future__ import annotations

import numpy as np

from dtutmo import (
    CAMType,
    CompleteDTUTMO,
    DTUTMOConfig,
    DisplayStandard,
    ViewingCondition,
    tone_map_hdr,
)


def example_simple() -> np.ndarray:
    """Run the pipeline with default configuration."""

    img_hdr = np.random.rand(512, 512, 3) * 1000.0
    tmo = CompleteDTUTMO()
    img_display = tmo.process(img_hdr)
    print(f"Simple example output range: [{img_display.min():0.3f}, {img_display.max():0.3f}]")
    return img_display


def example_with_xlrcam() -> np.ndarray:
    """Use XLR-CAM for extended luminance ranges."""

    img_hdr = np.random.rand(512, 512, 3) * 10000.0
    config = DTUTMOConfig(
        use_cam=CAMType.XLRCAM,
        target_display=DisplayStandard.REC_709,
        viewing_condition=ViewingCondition.DIM,
    )
    tmo = CompleteDTUTMO(config)
    img_display = tmo.process(img_hdr)
    print(f"XLR-CAM example output range: [{img_display.min():0.3f}, {img_display.max():0.3f}]")
    return img_display


def example_convenience_function() -> np.ndarray:
    """Tone-map using the high-level convenience wrapper."""

    img_hdr = np.random.rand(256, 256, 3) * 1500.0
    img_display = tone_map_hdr(
        img_hdr,
        target_display=DisplayStandard.REC_709,
        viewing_condition=ViewingCondition.DIM,
        use_cam=CAMType.DTUCAM,
    )
    print(f"Convenience example output range: [{img_display.min():0.3f}, {img_display.max():0.3f}]")
    return img_display


if __name__ == "__main__":
    print("Running DTUTMO basic examples...")
    example_simple()
    example_with_xlrcam()
    example_convenience_function()
