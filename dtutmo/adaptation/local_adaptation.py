"""
Local adaptation model based on Vangorp et al. (2015).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


class LocalAdaptation:
    """
    Compute spatially varying adaptation state based on local luminance.
    """

    def __init__(self, peak_sensitivity: float = 6.0, ppd: float = 45.0) -> None:
        self.peak_sensitivity = peak_sensitivity
        self.ppd = ppd

    def compute(self, luminance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute contrast threshold and adaptation luminance maps.
        """

        scales = [1, 2, 4, 8]  # degrees
        adapted_lum = np.zeros_like(luminance, dtype=float)
        total_weight = 0.0

        for scale in scales:
            sigma_pix = scale * self.ppd
            smoothed = gaussian_filter(luminance, sigma=sigma_pix)
            weight = float(scale)
            adapted_lum += smoothed * weight
            total_weight += weight

        adapted_lum /= total_weight
        contrast_threshold = self._contrast_threshold(adapted_lum)

        return contrast_threshold, adapted_lum

    def _contrast_threshold(self, adaptation_lum: np.ndarray) -> np.ndarray:
        """
        Simplified TVI (threshold vs intensity) function.
        """

        threshold = np.zeros_like(adaptation_lum, dtype=float)

        photopic_mask = adaptation_lum > 10.0
        scotopic_mask = adaptation_lum < 0.01
        mesopic_mask = ~(photopic_mask | scotopic_mask)

        threshold[photopic_mask] = 0.01
        threshold[scotopic_mask] = 0.1

        if np.any(mesopic_mask):
            log_lum = np.log10(adaptation_lum[mesopic_mask])
            log_thresh = np.interp(
                log_lum,
                [np.log10(0.01), np.log10(10.0)],
                [np.log10(0.1), np.log10(0.01)],
            )
            threshold[mesopic_mask] = 10.0 ** log_thresh

        return threshold
