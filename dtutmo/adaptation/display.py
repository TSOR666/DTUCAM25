"""
Display adaptation and encoding utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from dtutmo.core.config import DisplayStandard


@dataclass
class DisplaySpec:
    """Display specification."""

    name: str
    max_luminance: float  # cd/m^2
    white_point: np.ndarray  # XYZ
    primaries: np.ndarray  # RGB primaries in xy (placeholder)
    gamma: float
    bit_depth: int
    eotf_type: str  # gamma | pq | hlg


class DisplayAdaptation:
    """
    Convert scene-referred XYZ to display-referred RGB.
    """

    def __init__(self) -> None:
        self.display_specs = self._init_display_specs()

    def _init_display_specs(self) -> Dict[DisplayStandard, DisplaySpec]:
        """Initialize standard display specifications."""

        d65 = np.array([95.047, 100.0, 108.883])

        return {
            DisplayStandard.REC_709: DisplaySpec(
                name="Rec. 709",
                max_luminance=100.0,
                white_point=d65,
                primaries=np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]),
                gamma=2.4,
                bit_depth=8,
                eotf_type="gamma",
            ),
            DisplayStandard.REC_2020: DisplaySpec(
                name="Rec. 2020",
                max_luminance=100.0,
                white_point=d65,
                primaries=np.array([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]]),
                gamma=2.4,
                bit_depth=10,
                eotf_type="gamma",
            ),
            DisplayStandard.DCI_P3: DisplaySpec(
                name="DCI-P3",
                max_luminance=48.0,
                white_point=d65,
                primaries=np.array([[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]]),
                gamma=2.6,
                bit_depth=12,
                eotf_type="gamma",
            ),
            DisplayStandard.REC_2100_PQ: DisplaySpec(
                name="Rec. 2100 PQ (HDR10)",
                max_luminance=10000.0,
                white_point=d65,
                primaries=np.array([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]]),
                gamma=2.4,
                bit_depth=10,
                eotf_type="pq",
            ),
            DisplayStandard.REC_2100_HLG: DisplaySpec(
                name="Rec. 2100 HLG",
                max_luminance=1000.0,
                white_point=d65,
                primaries=np.array([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]]),
                gamma=1.2,
                bit_depth=10,
                eotf_type="hlg",
            ),
        }

    def get_display_spec(self, standard: DisplayStandard) -> Dict[str, float]:
        """Get a display specification as a plain dictionary."""

        spec = self.display_specs[standard]
        return {
            "name": spec.name,
            "max_luminance": spec.max_luminance,
            "white_point": spec.white_point,
            "gamma": spec.gamma,
            "bit_depth": spec.bit_depth,
        }

    def adapt_to_display(self, img_xyz: np.ndarray, standard: DisplayStandard) -> np.ndarray:
        """Adapt scene-referred XYZ to display-referred RGB."""

        spec = self.display_specs[standard]

        Y = img_xyz[:, :, 1]
        scale = spec.max_luminance / max(float(np.max(Y)), 1e-6)
        img_xyz_scaled = img_xyz * scale

        rgb = self.xyz_to_display_rgb(img_xyz_scaled, standard)

        if spec.eotf_type == "gamma":
            encoded = self._apply_gamma_encoding(rgb, spec.gamma)
        elif spec.eotf_type == "pq":
            encoded = self._apply_pq_encoding(rgb, spec.max_luminance)
        elif spec.eotf_type == "hlg":
            encoded = self._apply_hlg_encoding(rgb)
        else:
            encoded = rgb

        return np.clip(encoded, 0.0, 1.0)

    def xyz_to_display_rgb(self, xyz: np.ndarray, standard: DisplayStandard) -> np.ndarray:
        """
        Convert XYZ to display RGB (linear).

        Currently uses the Rec. 709 matrix as a placeholder for all standards.
        """

        del standard  # placeholder until bespoke matrices are provided

        matrix = np.array(
            [
                [3.2406, -1.5372, -0.4986],
                [-0.9689, 1.8758, 0.0415],
                [0.0557, -0.2040, 1.0570],
            ]
        )

        rgb = np.dot(xyz, matrix.T)
        return np.maximum(rgb, 0.0)

    @staticmethod
    def _apply_gamma_encoding(rgb: np.ndarray, gamma: float) -> np.ndarray:
        return np.power(np.maximum(rgb, 0.0), 1.0 / gamma)

    @staticmethod
    def _apply_pq_encoding(rgb: np.ndarray, max_lum: float) -> np.ndarray:
        """Apply SMPTE ST 2084 (PQ) encoding."""

        del max_lum  # currently unused because the content is already scaled to cd/m^2

        L = rgb / 10000.0

        m1 = 2610.0 / 16384.0
        m2 = (2523.0 / 4096.0) * 128.0
        c1 = 3424.0 / 4096.0
        c2 = (2413.0 / 4096.0) * 32.0
        c3 = (2392.0 / 4096.0) * 32.0

        L_m1 = np.power(np.maximum(L, 0.0), m1)
        encoded = np.power((c1 + c2 * L_m1) / (1.0 + c3 * L_m1), m2)

        return encoded

    @staticmethod
    def _apply_hlg_encoding(rgb: np.ndarray) -> np.ndarray:
        """Apply ITU-R BT.2100 HLG encoding."""

        encoded = np.zeros_like(rgb)

        mask_low = rgb <= 1.0 / 12.0
        encoded[mask_low] = np.sqrt(3.0 * rgb[mask_low])

        mask_high = ~mask_low
        a = 0.17883277
        b = 0.28466892
        c = 0.55991073
        encoded[mask_high] = a * np.log(12.0 * rgb[mask_high] - b) + c

        return encoded
