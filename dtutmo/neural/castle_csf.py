"""
CastleCSF: Comprehensive contrast sensitivity function (Ashraf et al. 2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CastleCSFParams:
    """Parameter bundle for CastleCSF."""

    achrom_peak_freq: float = 3.0  # cycles per degree
    achrom_peak_sens: float = 100.0
    achrom_bandwidth: float = 2.0

    rg_peak_freq: float = 1.0
    rg_peak_sens: float = 30.0
    by_peak_freq: float = 0.8
    by_peak_sens: float = 25.0


class CastleCSF:
    """
    CastleCSF neural filtering model.
    """

    def __init__(self, ppd: float = 45.0) -> None:
        self.ppd = ppd
        self.params = CastleCSFParams()

    def apply_csf(self, img_xyz: np.ndarray, adaptation_lum: np.ndarray) -> np.ndarray:
        """
        Apply CSF filtering to an XYZ image.
        """

        opponent = self._xyz_to_opponent(img_xyz)
        opponent_fft = np.fft.fft2(opponent, axes=(0, 1))
        freq_map = self._create_frequency_map(opponent.shape[:2])
        avg_adapt = float(np.mean(adaptation_lum))

        achrom_csf = self._achromatic_csf(freq_map, avg_adapt)
        rg_csf = self._chromatic_csf(freq_map, avg_adapt, "rg")
        by_csf = self._chromatic_csf(freq_map, avg_adapt, "by")

        filtered_fft = opponent_fft.copy()
        filtered_fft[:, :, 0] *= achrom_csf
        filtered_fft[:, :, 1] *= rg_csf
        filtered_fft[:, :, 2] *= by_csf

        filtered = np.real(np.fft.ifft2(filtered_fft, axes=(0, 1)))
        xyz_filtered = self._opponent_to_xyz(filtered)

        return xyz_filtered

    def _create_frequency_map(self, shape: Tuple[int, int]) -> np.ndarray:
        height, width = shape
        freq_x = np.fft.fftfreq(width, d=1.0 / self.ppd)
        freq_y = np.fft.fftfreq(height, d=1.0 / self.ppd)
        fx, fy = np.meshgrid(freq_x, freq_y)
        return np.sqrt(fx**2 + fy**2)

    def _achromatic_csf(self, freq: np.ndarray, luminance: float) -> np.ndarray:
        log_freq = np.log10(np.maximum(freq, 0.1))
        log_peak = np.log10(self.params.achrom_peak_freq)

        csf = self.params.achrom_peak_sens * np.exp(
            -((log_freq - log_peak) / self.params.achrom_bandwidth) ** 2
        )

        csf *= freq / (freq + 0.5)
        lum_factor = np.sqrt(luminance / 100.0) if luminance < 200.0 else 1.0
        csf *= lum_factor
        csf = csf / np.max(csf)

        return csf

    def _chromatic_csf(self, freq: np.ndarray, luminance: float, channel: str) -> np.ndarray:
        if channel == "rg":
            peak_freq = self.params.rg_peak_freq
            peak_sens = self.params.rg_peak_sens
        else:
            peak_freq = self.params.by_peak_freq
            peak_sens = self.params.by_peak_sens

        csf = peak_sens * np.exp(-freq / peak_freq)
        lum_factor = min(luminance / 200.0, 1.0)
        csf *= lum_factor
        csf = csf / np.max(csf)

        return csf

    @staticmethod
    def _xyz_to_opponent(xyz: np.ndarray) -> np.ndarray:
        opponent = np.zeros_like(xyz)
        opponent[:, :, 0] = xyz[:, :, 1]
        opponent[:, :, 1] = xyz[:, :, 0] - xyz[:, :, 1]
        opponent[:, :, 2] = xyz[:, :, 1] - xyz[:, :, 2]
        return opponent

    @staticmethod
    def _opponent_to_xyz(opponent: np.ndarray) -> np.ndarray:
        xyz = np.zeros_like(opponent)
        xyz[:, :, 1] = opponent[:, :, 0]
        xyz[:, :, 0] = opponent[:, :, 0] + opponent[:, :, 1]
        xyz[:, :, 2] = opponent[:, :, 0] - opponent[:, :, 2]
        return xyz
