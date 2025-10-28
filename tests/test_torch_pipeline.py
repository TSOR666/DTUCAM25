"""
Smoke tests for the torch-based pipeline. Skipped automatically when torch is unavailable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dtutmo import CAMType, DTUTMOConfig, DisplayMapping  # noqa: E402
from dtutmo.torch import TorchDTUTMO, tone_map_hdr_torch  # noqa: E402


def test_torch_pipeline_forward():
    img_hdr = torch.rand(1, 3, 32, 32) * 500.0
    tmo = TorchDTUTMO(device=torch.device("cpu"))
    result = tmo.process(img_hdr)
    assert result.shape == img_hdr.shape
    assert torch.isfinite(result).all()


def test_tone_map_hdr_torch_helper():
    img_hdr = torch.rand(32, 32, 3) * 100.0
    result = tone_map_hdr_torch(img_hdr, device=torch.device("cpu"))
    assert result.shape == img_hdr.shape


def test_torch_pipeline_respects_display_mapping():
    img_hdr = torch.rand(1, 3, 16, 16) * 250.0
    config = DTUTMOConfig(
        display_mapping=DisplayMapping.PRODUCTION_HYBRID,
        use_cam=CAMType.NONE,
        use_bilateral=False,
        use_otf=False,
        use_glare=False,
    )
    tmo = TorchDTUTMO(config=config, device=torch.device("cpu"))
    result = tmo.process(img_hdr)
    assert result.shape == img_hdr.shape
    assert torch.isfinite(result).all()
