"""
CIECAM16: CIE 2016 color appearance model (simplified implementation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class CIECAM16ViewingConditions:
    """Viewing conditions used by CIECAM16."""

    L_A: float = 20.0  # adapting luminance (cd/m^2)
    Y_b: float = 20.0  # background luminance (cd/m^2)
    surround: str = "average"

    def __post_init__(self) -> None:
        surround_params = {
            "dark": (0.8, 0.525, 0.8),
            "dim": (0.9, 0.59, 0.9),
            "average": (1.0, 0.69, 1.0),
        }
        self.F, self.c, self.N_c = surround_params[self.surround]

        k = 1.0 / (5.0 * self.L_A + 1.0)
        self.F_L = 0.2 * (k**4) * (5.0 * self.L_A) + 0.1 * ((1.0 - k**4) ** 2) * ((5.0 * self.L_A) ** (1.0 / 3.0))

        self.D = self.F * (1.0 - (1.0 / 3.6) * np.exp((-self.L_A - 42.0) / 92.0))
        self.N_bb = 0.725 * (1.0 / self.Y_b) ** 0.2
        self.N_cb = self.N_bb
        self.z = 1.48 + np.sqrt(self.N_c)


class CIECAM16:
    """
    CIECAM16 forward and approximate inverse transforms.
    """

    def __init__(self) -> None:
        self.M16 = np.array(
            [
                [0.401288, 0.650173, -0.051461],
                [-0.250268, 1.204414, 0.045854],
                [-0.002079, 0.048952, 0.953127],
            ]
        )
        self.M16_inv = np.linalg.inv(self.M16)

    def forward(
        self,
        img_xyz: np.ndarray,
        white_xyz: np.ndarray,
        background_lum: float,
        surround: str = "average",
    ) -> Dict[str, np.ndarray]:
        """
        Forward CIECAM16: XYZ to perceptual correlates.
        """

        vc = CIECAM16ViewingConditions(
            L_A=float(np.mean(img_xyz[:, :, 1])),
            Y_b=background_lum,
            surround=surround,
        )

        rgb = self._xyz_to_cat16(img_xyz)
        rgb_w = self._xyz_to_cat16(white_xyz.reshape(1, 1, 3))

        D = vc.D
        rgb_c = (D * (white_xyz.reshape(1, 1, 3) / np.maximum(rgb_w, 1e-6)) + (1.0 - D)) * rgb
        rgb_aw = (D * (white_xyz.reshape(1, 1, 3) / np.maximum(rgb_w, 1e-6)) + (1.0 - D)) * rgb_w

        rgb_a = self._post_adaptation(rgb_c, vc.F_L)
        rgb_aw = self._post_adaptation(rgb_aw, vc.F_L)

        a = rgb_a[:, :, 0] - 12.0 * rgb_a[:, :, 1] / 11.0 + rgb_a[:, :, 2] / 11.0
        b = (rgb_a[:, :, 0] + rgb_a[:, :, 1] - 2.0 * rgb_a[:, :, 2]) / 9.0

        # Ensure float64 for NumPy 2.x compatibility with arctan2
        h = np.arctan2(b.astype(np.float64), a.astype(np.float64))
        h = np.where(h < 0.0, h + 2.0 * np.pi, h)

        e_t = 0.25 * (np.cos(h + 2.0) + 3.8)

        A = (2.0 * rgb_a[:, :, 0] + rgb_a[:, :, 1] + rgb_a[:, :, 2] / 20.0 - 0.305) * vc.N_bb
        A_w = (2.0 * rgb_aw[:, :, 0] + rgb_aw[:, :, 1] + rgb_aw[:, :, 2] / 20.0 - 0.305) * vc.N_bb
        A = np.clip(A, 0.0, None)
        A_w = np.clip(A_w, 1e-6, None)

        with np.errstate(invalid="ignore"):
            ratio = np.clip(A / A_w, 0.0, None)
            J = 100.0 * np.power(ratio, vc.c * vc.z)
        Q = (4.0 / vc.c) * np.sqrt(J / 100.0) * (A_w + 4.0) * (vc.F_L ** 0.25)

        denominator = np.maximum(rgb_a[:, :, 0] + rgb_a[:, :, 1] + 21.0 * rgb_a[:, :, 2] / 20.0, 1e-6)
        t = (5e4 / 13.0) * vc.N_c * vc.N_cb * e_t * np.sqrt(a**2 + b**2) / denominator
        C = (t ** 0.9) * np.sqrt(J / 100.0) * ((1.64 - 0.29 ** vc.N_cb) ** 0.73)
        M = C * (vc.F_L ** 0.25)
        s = 50.0 * np.sqrt((vc.c * M) / (Q + 1e-4))

        return {
            "lightness": J,
            "brightness": Q,
            "colorfulness": M,
            "chroma": C,
            "hue": h * 180.0 / np.pi,
            "saturation": s,
        }

    def inverse(
        self,
        lightness: np.ndarray,
        colorfulness: np.ndarray,
        hue: np.ndarray,
        display_white: np.ndarray,
        display_max_lum: float,
        surround: str = "dim",
    ) -> np.ndarray:
        """
        Approximate inverse CIECAM16 (sufficient for tone mapping previews).
        """

        vc = CIECAM16ViewingConditions(
            L_A=display_max_lum / 5.0,
            Y_b=display_max_lum * 0.2,
            surround=surround,
        )

        C = colorfulness / (vc.F_L ** 0.25)
        h_rad = hue * np.pi / 180.0
        Y = display_max_lum * np.power(lightness / 100.0, 1.0 / (vc.c * vc.z))

        X = Y * (1.0 + C * np.cos(h_rad) / 100.0)
        Z = Y * (1.0 + C * np.sin(h_rad) / 100.0)

        xyz = np.stack([X, Y, Z], axis=2)
        xyz = np.clip(xyz, 0.0, display_max_lum * 2.0)
        return xyz

    def _xyz_to_cat16(self, xyz: np.ndarray) -> np.ndarray:
        return np.dot(xyz, self.M16.T)

    def _post_adaptation(self, rgb: np.ndarray, F_L: float) -> np.ndarray:
        F_L_rgb = (F_L * np.abs(rgb) / 100.0) ** 0.42
        # Ensure float64 for NumPy 2.x compatibility - np.sign can return int dtype
        rgb_a = 400.0 * np.sign(rgb).astype(np.float64) * F_L_rgb / (27.13 + F_L_rgb) + 0.1
        return rgb_a
