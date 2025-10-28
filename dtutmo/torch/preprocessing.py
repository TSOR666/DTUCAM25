"""
Preprocessing helpers for the torch pipeline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class TorchBilateralFilter:
    """
    Bilateral filter using torch unfold for GPU execution.
    """

    def __init__(self, sigma_spatial: float = 10.0, sigma_range: float = 0.1) -> None:
        self.sigma_spatial = max(sigma_spatial, 1.0)
        self.sigma_range = max(sigma_range, 1e-3)
        self.kernel_radius = int(max(round(self.sigma_spatial), 1))
        self.kernel_size = 2 * self.kernel_radius + 1
        self._spatial_weights = None

    def _prepare_spatial_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._spatial_weights is not None and self._spatial_weights.device == device and self._spatial_weights.dtype == dtype:
            return self._spatial_weights

        coords = torch.arange(-self.kernel_radius, self.kernel_radius + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        spatial = torch.exp(-(xx**2 + yy**2) / (2.0 * self.sigma_spatial**2))
        spatial = spatial / spatial.sum()
        self._spatial_weights = spatial.reshape(1, -1, 1)
        return self._spatial_weights

    def filter(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply the bilateral filter to a grayscale tensor in NCHW format.
        """

        if img.shape[1] != 1:
            raise ValueError("TorchBilateralFilter expects a single-channel tensor.")

        spatial = self._prepare_spatial_weights(img.device, img.dtype)
        patches = F.unfold(img, kernel_size=self.kernel_size, padding=self.kernel_radius)
        center = img.reshape(img.shape[0], 1, -1)
        range_weights = torch.exp(-((patches - center) ** 2) / (2.0 * self.sigma_range**2))
        weights = spatial * range_weights
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-6)
        filtered = (weights * patches).sum(dim=1).reshape(img.shape[0], 1, img.shape[2], img.shape[3])
        return filtered
