"""
Advanced DTUTMO usage scenarios.
"""

from __future__ import annotations

import numpy as np

from dtutmo import CAMType, CompleteDTUTMO, DTUTMOConfig, DisplayStandard, ViewingCondition


def example_hdr10_output() -> np.ndarray:
    """Tone-map for a Rec.2100 PQ (HDR10) display."""

    img_hdr = np.random.rand(512, 512, 3) * 5000.0
    config = DTUTMOConfig(
        use_cam=CAMType.DTUCAM,
        target_display=DisplayStandard.REC_2100_PQ,
        viewing_condition=ViewingCondition.DARK,
    )
    tmo = CompleteDTUTMO(config)
    img_display = tmo.process(img_hdr)
    print(f"HDR10 example output range: [{img_display.min():0.3f}, {img_display.max():0.3f}]")
    return img_display


def example_with_intermediate_results() -> dict:
    """Retrieve intermediate maps for inspection."""

    img_hdr = np.random.rand(256, 256, 3) * 500.0
    tmo = CompleteDTUTMO()
    results = tmo.process(img_hdr, return_intermediate=True)
    keys = ", ".join(results.keys())
    print(f"Intermediate results available: {keys}")
    return results


def example_custom_configuration() -> np.ndarray:
    """Demonstrate a fully customized configuration."""

    img_hdr = np.random.rand(512, 512, 3) * 2000.0
    config = DTUTMOConfig(
        viewing_condition=ViewingCondition.DIM,
        observer_age=35.0,
        field_diameter=60.0,
        use_otf=True,
        use_glare=True,
        use_bilateral=True,
        use_local_adapt=True,
        use_cam=CAMType.DTUCAM,
        target_display=DisplayStandard.REC_709,
        glare_model="cie_general",
        peak_sensitivity=6.0,
        pixels_per_degree=45.0,
    )
    tmo = CompleteDTUTMO(config)
    img_display = tmo.process(img_hdr)
    print("Custom configuration example complete.")
    return img_display


def example_comparison() -> dict:
    """Compare outputs across CAM choices."""

    img_hdr = np.random.rand(256, 256, 3) * 1000.0
    results = {}

    for cam in (CAMType.NONE, CAMType.DTUCAM, CAMType.XLRCAM, CAMType.CIECAM16):
        config = DTUTMOConfig(use_cam=cam)
        tmo = CompleteDTUTMO(config)
        results[cam.value] = tmo.process(img_hdr)

    for name, img in results.items():
        print(f"{name:>12}: mean={np.mean(img):0.3f}, std={np.std(img):0.3f}")

    return results


def example_torch_pipeline():
    """Demonstrate the PyTorch accelerated pipeline (requires torch)."""
    try:
        import torch
        from dtutmo.torch import TorchDTUTMO  # type: ignore
    except Exception:  # pragma: no cover - torch optional
        print("PyTorch is not available; skipping GPU example.")
        return None

    img_hdr = torch.rand(1, 3, 256, 256) * 2000.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmo = TorchDTUTMO(device=device)
    img_display = tmo.process(img_hdr.to(device))
    print(f"Torch pipeline output on {device}: min={img_display.min():0.3f}, max={img_display.max():0.3f}")
    return img_display


if __name__ == "__main__":
    print("Running DTUTMO advanced examples...")
    example_hdr10_output()
    example_with_intermediate_results()
    example_custom_configuration()
    example_comparison()
    example_torch_pipeline()
