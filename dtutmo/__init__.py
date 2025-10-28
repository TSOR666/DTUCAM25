"""DTU Tone Mapping Operator (DTUTMO).

Biologically-inspired HDR tone mapping with full color appearance modeling
and display adaptation.

Author : Thierry Silvio Claude Soreze 
"""

from dtutmo.core.config import (
    CAMType,
    DTUTMOConfig,
    DisplayMapping,
    DisplayStandard,
    ViewingCondition,
)
from dtutmo.core.pipeline import CompleteDTUTMO, tone_map_hdr

__all__ = [
    "CompleteDTUTMO",
    "DTUTMOConfig",
    "ViewingCondition",
    "CAMType",
    "DisplayStandard",
    "DisplayMapping",
    "tone_map_hdr",
]

try:  # Optional PyTorch acceleration
    from dtutmo.torch.pipeline import TorchDTUTMO, tone_map_hdr_torch  # type: ignore

    __all__.extend(["TorchDTUTMO", "tone_map_hdr_torch"])
except Exception:  # pragma: no cover - torch not installed
    TorchDTUTMO = None  # type: ignore
    tone_map_hdr_torch = None  # type: ignore

__version__ = "2.0.1"
