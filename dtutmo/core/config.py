"""
Configuration primitives for DTUTMO.

Defines enums for viewing conditions, color appearance models, display
standards, and a dataclass collecting configurable parameters.
"""

from dataclasses import dataclass
from enum import Enum


class ViewingCondition(Enum):
    """Viewing environment surround conditions."""

    DARK = "dark"       # Cinema, <0.2 cd/m^2
    DIM = "dim"         # Home TV, 0.2-20 cd/m^2
    AVERAGE = "average"  # Office, >20 cd/m^2


class CAMType(Enum):
    """Color appearance model selection."""

    NONE = "none"            # No CAM (direct mapping)
    DTUCAM = "dtucam"        # DTU proprietary CAM
    XLRCAM = "xlr_cam"       # Extended Luminance Range CAM (2009)
    CIECAM16 = "ciecam16"    # CIE standard CAM (2022)


class DisplayStandard(Enum):
    """Target display standards."""

    REC_709 = "rec_709"            # SDR, 100 cd/m^2
    REC_2020 = "rec_2020"          # Wide gamut SDR
    DCI_P3 = "dci_p3"              # Digital cinema
    REC_2100_PQ = "rec_2100_pq"    # HDR10, up to 10,000 cd/m^2
    REC_2100_HLG = "rec_2100_hlg"  # Hybrid log-gamma HDR


class DisplayMapping(Enum):
    """Display output mapping strategy."""

    LEGACY = "legacy"  # Existing display adaptation module
    WHITEBOARD = "whiteboard"  # Simplified inverse mapping
    FULL_INVERSE = "full_inverse"  # Complete photoreceptor inverse
    HYBRID = "hybrid"  # Blend of whiteboard + inverse
    PRODUCTION_HYBRID = "production_hybrid"  # Gradient-aware hybrid mapper


@dataclass
class DTUTMOConfig:
    """
    Complete configuration for DTUTMO processing.

    All parameters have sensible defaults for typical HDR tone mapping.
    """

    # Viewing conditions
    viewing_condition: ViewingCondition = ViewingCondition.DIM
    observer_age: float = 24.0  # years
    field_diameter: float = 60.0  # degrees

    # Processing stages
    use_otf: bool = True  # Optical transfer function
    use_glare: bool = True  # CIE disability glare (critical)
    use_bilateral: bool = True  # Base/detail separation
    use_local_adapt: bool = True  # Vangorp et al. model
    use_cam: CAMType = CAMType.DTUCAM  # Color appearance model

    # Display target
    target_display: DisplayStandard = DisplayStandard.REC_709
    display_mapping: DisplayMapping = DisplayMapping.HYBRID

    # Advanced options
    glare_model: str = "cie_general"  # cie_general | stiles_holladay | vos_vandenberg
    csf_model: str = "castle"  # currently only "castle"
    cone_integration: float = 0.1  # seconds
    rod_integration: float = 1.0  # seconds
    peak_sensitivity: float = 6.0  # for local adaptation
    pixels_per_degree: float = 45.0

    def validate(self) -> None:
        """Validate configuration parameters."""

        if not (0 < self.observer_age < 120):
            raise ValueError(f"Observer age {self.observer_age} out of range [0, 120]")

        if not (10 < self.field_diameter < 180):
            raise ValueError(f"Field diameter {self.field_diameter} out of range [10, 180]")

        if not (0 < self.cone_integration < 10):
            raise ValueError(f"Cone integration {self.cone_integration} out of range (0, 10)")

        if not (0 < self.pixels_per_degree < 200):
            raise ValueError(f"Pixels per degree {self.pixels_per_degree} out of range (0, 200)")
