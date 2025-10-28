"""
Torch CastleCSF implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class CastleCSFParams:
    achrom_peak_freq: float = 3.0
    achrom_peak_sens: float = 100.0
    achrom_bandwidth: float = 2.0
    rg_peak_freq: float = 1.0
    rg_peak_sens: float = 30.0
    by_peak_freq: float = 0.8
    by_peak_sens: float = 25.0


class CastleCSF:
    def __init__(self, ppd: float = 45.0, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> None:
        self.ppd = ppd
        self.params = CastleCSFParams()
        self.device = device
        self.dtype = dtype

    def apply_csf(self, img_xyz: torch.Tensor, adaptation_lum: torch.Tensor) -> torch.Tensor:
        opponent = self._xyz_to_opponent(img_xyz)
        opponent_fft = torch.fft.fft2(opponent, dim=(-2, -1))
        freq_map = self._create_frequency_map(opponent.shape[-2:])
        avg_adapt = float(adaptation_lum.mean().item())

        achrom_csf = self._achromatic_csf(freq_map, avg_adapt)
        rg_csf = self._chromatic_csf(freq_map, avg_adapt, "rg")
        by_csf = self._chromatic_csf(freq_map, avg_adapt, "by")

        opponent_fft[:, 0, :, :] *= achrom_csf
        opponent_fft[:, 1, :, :] *= rg_csf
        opponent_fft[:, 2, :, :] *= by_csf

        filtered = torch.fft.ifft2(opponent_fft, dim=(-2, -1)).real
        xyz_filtered = self._opponent_to_xyz(filtered)
        return xyz_filtered

    def _create_frequency_map(self, shape: Tuple[int, int]) -> torch.Tensor:
        height, width = shape
        ppd_value = self.ppd
        fx = torch.fft.fftfreq(width, d=1.0 / ppd_value, device=self.device, dtype=self.dtype)
        fy = torch.fft.fftfreq(height, d=1.0 / ppd_value, device=self.device, dtype=self.dtype)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        return torch.sqrt(fx_grid**2 + fy_grid**2)

    def _achromatic_csf(self, freq: torch.Tensor, luminance: float) -> torch.Tensor:
        eps = torch.tensor(0.1, device=self.device, dtype=self.dtype)
        log_freq = torch.log10(torch.clamp(freq, min=eps))
        log_peak = torch.log10(torch.tensor(self.params.achrom_peak_freq, device=self.device, dtype=self.dtype))
        csf = self.params.achrom_peak_sens * torch.exp(-((log_freq - log_peak) / self.params.achrom_bandwidth) ** 2)
        csf = csf * freq / (freq + 0.5)
        lum_factor = torch.sqrt(torch.tensor(luminance / 100.0, device=self.device, dtype=self.dtype)) if luminance < 200.0 else torch.tensor(1.0, device=self.device, dtype=self.dtype)
        csf = csf * lum_factor
        csf = csf / torch.max(csf)
        return csf

    def _chromatic_csf(self, freq: torch.Tensor, luminance: float, channel: str) -> torch.Tensor:
        if channel == "rg":
            peak_freq = self.params.rg_peak_freq
            peak_sens = self.params.rg_peak_sens
        else:
            peak_freq = self.params.by_peak_freq
            peak_sens = self.params.by_peak_sens

        csf = peak_sens * torch.exp(-freq / peak_freq)
        lum_factor = min(luminance / 200.0, 1.0)
        csf = csf * lum_factor
        csf = csf / torch.max(csf)
        return csf

    @staticmethod
    def _xyz_to_opponent(xyz: torch.Tensor) -> torch.Tensor:
        opponent = torch.zeros_like(xyz)
        opponent[:, 0, :, :] = xyz[:, 1, :, :]
        opponent[:, 1, :, :] = xyz[:, 0, :, :] - xyz[:, 1, :, :]
        opponent[:, 2, :, :] = xyz[:, 1, :, :] - xyz[:, 2, :, :]
        return opponent

    @staticmethod
    def _opponent_to_xyz(opponent: torch.Tensor) -> torch.Tensor:
        xyz = torch.zeros_like(opponent)
        xyz[:, 1, :, :] = opponent[:, 0, :, :]
        xyz[:, 0, :, :] = opponent[:, 0, :, :] + opponent[:, 1, :, :]
        xyz[:, 2, :, :] = opponent[:, 0, :, :] - opponent[:, 2, :, :]
        return xyz
