"""
Edge-preserving bilateral filtering for base/detail separation.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


class BilateralFilter:
    """
    Simplified bilateral filter for grayscale images.
    """

    def __init__(self, sigma_spatial: float = 10.0, sigma_range: float = 0.1) -> None:
        self.sigma_spatial = sigma_spatial
        self.sigma_range = sigma_range

    def filter(self, img: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering.
        """

        spatial = gaussian_filter(img, sigma=self.sigma_spatial)
        diff = np.abs(img - spatial)
        weights = np.exp(-(diff / self.sigma_range) ** 2)
        filtered = spatial * weights + img * (1.0 - weights)

        return filtered
