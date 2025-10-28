"""
Torch implementations of adaptation stages.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(int(3.0 * sigma), 1)
    positions = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(positions**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply a separable Gaussian blur to a tensor (N, 1, H, W).
    """

    if sigma <= 0:
        return img

    kernel_1d = _gaussian_kernel1d(sigma, img.device, img.dtype)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1])
    pad = kernel_2d.shape[-1] // 2
    padded = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    blurred = F.conv2d(padded, kernel_2d)
    return blurred


class TorchLocalAdaptation:
    """
    Torch analogue of LocalAdaptation.
    """

    def __init__(self, peak_sensitivity: float = 6.0, ppd: float = 45.0) -> None:
        self.peak_sensitivity = peak_sensitivity
        self.ppd = ppd

    def compute(self, luminance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute contrast threshold and adaptation luminance.

        Parameters
        ----------
        luminance : torch.Tensor
            Tensor of shape (N, 1, H, W)
        """

        device = luminance.device
        dtype = luminance.dtype

        scales = [1, 2, 4, 8]
        adapted = torch.zeros_like(luminance)
        total_weight = 0.0

        for scale in scales:
            sigma_pix = scale * self.ppd
            if sigma_pix < 1.0:
                weight = 1.0
                smoothed = luminance
            else:
                sigma_norm = min(sigma_pix / 6.0, 10.0)  # cap radius for efficiency
                smoothed = gaussian_blur(luminance, sigma_norm)
                weight = float(scale)
            adapted = adapted + smoothed * weight
            total_weight += weight

        adapted = adapted / total_weight
        threshold = self._contrast_threshold(adapted, device, dtype)
        return threshold, adapted

    def _contrast_threshold(self, adaptation_lum: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        threshold = torch.zeros_like(adaptation_lum)
        photopic_mask = adaptation_lum > 10.0
        scotopic_mask = adaptation_lum < 0.01
        mesopic_mask = ~(photopic_mask | scotopic_mask)

        threshold = threshold + torch.where(photopic_mask, torch.tensor(0.01, device=device, dtype=dtype), torch.zeros_like(adaptation_lum))
        threshold = threshold + torch.where(scotopic_mask, torch.tensor(0.1, device=device, dtype=dtype), torch.zeros_like(adaptation_lum))

        if mesopic_mask.any():
            log_lum = torch.log10(torch.clamp(adaptation_lum[mesopic_mask], min=1e-4))
            low = torch.log10(torch.tensor(0.01, device=device, dtype=dtype))
            high = torch.log10(torch.tensor(10.0, device=device, dtype=dtype))
            interp = (log_lum - low) / (high - low)
            log_thresh_low = torch.log10(torch.tensor(0.1, device=device, dtype=dtype))
            log_thresh_high = torch.log10(torch.tensor(0.01, device=device, dtype=dtype))
            log_thresh = log_thresh_low + interp * (log_thresh_high - log_thresh_low)
            threshold[mesopic_mask] = torch.pow(10.0, log_thresh)

        return threshold
