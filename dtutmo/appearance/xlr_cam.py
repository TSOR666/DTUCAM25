"""
XLR-CAM: Extended Luminance Range Color Appearance Model (SIGGRAPH 2009).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class XLRCAMParameters:
    """Parameter bundle for XLR-CAM."""

    n: float = 0.73
    sigma_base: float = 0.3
    sigma_adapt_factor: float = 0.5
    background_weight: float = 0.2


class XLRCAM:
    """
    Extended Luminance Range Color Appearance Model.
    """

    def __init__(self, params: Optional[XLRCAMParameters] = None) -> None:
        self.params = params or XLRCAMParameters()
        self.hpe_matrix = np.array(
            [
                [0.38971, 0.68898, -0.07868],
                [-0.22981, 1.18340, 0.04641],
                [0.00000, 0.00000, 1.00000],
            ]
        )
        self.hpe_inv = np.linalg.inv(self.hpe_matrix)

    def forward(
        self,
        img_xyz: np.ndarray,
        white_xyz: np.ndarray,
        background_lum: float,
        surround: str = "average",
    ) -> Dict[str, np.ndarray]:
        """Forward transform: XYZ to perceptual attributes."""

        lms = self._xyz_to_lms(img_xyz)
        white_lms = self._xyz_to_lms(white_xyz.reshape(1, 1, 3))

        lms_adapted = np.clip(lms / white_lms, 0.0, None)

        avg_lum = float(np.mean(img_xyz[:, :, 1]))
        sigma = self._compute_sigma(avg_lum, background_lum)

        lms_response = self._cone_response(lms_adapted, sigma)

        achrom, red_green, blue_yellow = self._opponent_channels(lms_response)

        lightness = self._compute_lightness(achrom, background_lum, surround)
        colorfulness = self._compute_colorfulness(red_green, blue_yellow, achrom)
        hue = np.arctan2(blue_yellow, red_green)
        chroma = np.sqrt(red_green**2 + blue_yellow**2)

        return {
            "lightness": lightness,
            "colorfulness": colorfulness,
            "hue": hue,
            "chroma": chroma,
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
        """Inverse transform: perceptual attributes to display XYZ."""

        achrom = self._inverse_lightness(lightness, display_max_lum, surround)
        red_green = colorfulness / (1.0 + 0.3 * achrom) * np.cos(hue)
        blue_yellow = colorfulness / (1.0 + 0.3 * achrom) * np.sin(hue)

        lms_response = self._inverse_opponent(achrom, red_green, blue_yellow)

        sigma_display = self._compute_sigma(display_max_lum / 2.0, display_max_lum * 0.2)
        lms_adapted = self._inverse_cone_response(lms_response, sigma_display)

        display_white_lms = self._xyz_to_lms(display_white.reshape(1, 1, 3))
        lms_display = lms_adapted * display_white_lms

        xyz_display = self._lms_to_xyz(lms_display)
        return xyz_display

    def _xyz_to_lms(self, xyz: np.ndarray) -> np.ndarray:
        return np.dot(xyz, self.hpe_matrix.T)

    def _lms_to_xyz(self, lms: np.ndarray) -> np.ndarray:
        return np.dot(lms, self.hpe_inv.T)

    def _compute_sigma(self, adaptation_lum: float, background_lum: float) -> float:
        log_adapt = np.log10(max(adaptation_lum, 0.01))
        bg_factor = 1.0 + self.params.background_weight * np.log10(max(background_lum, 0.01))

        sigma = self.params.sigma_base * (1.0 + self.params.sigma_adapt_factor * log_adapt) * bg_factor
        return sigma

    def _cone_response(self, lms: np.ndarray, sigma: float) -> np.ndarray:
        n = self.params.n
        lms = np.clip(lms, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = np.power(lms, n)
            denominator = numerator + sigma**n
            response = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        return response

    def _inverse_cone_response(self, response: np.ndarray, sigma: float) -> np.ndarray:
        n = self.params.n
        R = np.clip(response, 0.0, 0.99)
        intensity = sigma * np.power(R / (1.0 - R), 1.0 / n)
        return intensity

    def _opponent_channels(self, lms: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        achrom = (lms[:, :, 0] + lms[:, :, 1] + lms[:, :, 2]) / 3.0
        red_green = lms[:, :, 0] - lms[:, :, 1]
        blue_yellow = lms[:, :, 2] - (lms[:, :, 0] + lms[:, :, 1]) / 2.0
        return achrom, red_green, blue_yellow

    def _inverse_opponent(
        self,
        achrom: np.ndarray,
        red_green: np.ndarray,
        blue_yellow: np.ndarray,
    ) -> np.ndarray:
        lms = np.zeros(achrom.shape + (3,))
        lms[:, :, 0] = achrom + red_green / 2.0 + blue_yellow / 3.0
        lms[:, :, 1] = achrom - red_green / 2.0 + blue_yellow / 3.0
        lms[:, :, 2] = achrom + 2.0 * blue_yellow / 3.0
        return lms

    def _compute_lightness(self, achrom: np.ndarray, background: float, surround: str) -> np.ndarray:
        surround_factors = {"dark": 0.8, "dim": 0.9, "average": 1.0}
        factor = surround_factors.get(surround, 0.9)
        bg_factor = 1.0 + 0.2 * np.log10(max(background, 0.01))
        return 100.0 * np.power(achrom * bg_factor * factor, 0.67)

    def _inverse_lightness(self, lightness: np.ndarray, max_lum: float, surround: str) -> np.ndarray:
        surround_factors = {"dark": 0.8, "dim": 0.9, "average": 1.0}
        factor = surround_factors.get(surround, 0.9)
        bg_factor = 1.0 + 0.2 * np.log10(max(max_lum * 0.2, 0.01))
        return np.power(lightness / 100.0, 1.0 / 0.67) / (bg_factor * factor)

    def _compute_colorfulness(
        self,
        red_green: np.ndarray,
        blue_yellow: np.ndarray,
        achrom: np.ndarray,
    ) -> np.ndarray:
        chroma = np.sqrt(red_green**2 + blue_yellow**2)
        return chroma * (1.0 + 0.3 * achrom)
