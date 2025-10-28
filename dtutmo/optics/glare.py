"""
CIE disability glare model based on CIE 180:2010.

Provides a wavelength-dependent glare PSF and utilities to apply it to
HDR imagery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal

from dtutmo.photoreceptors.constants import PHOTORECEPTOR_CHANNELS


@dataclass
class GlareParameters:
    """Configuration for the glare model."""

    age: float = 24.0
    eye_pigmentation: str = "blue"  # blue | brown | dark_brown
    model: str = "cie_general"
    include_wavelength_dependence: bool = True
    include_corneal_reflections: bool = True
    min_angle: float = 0.1  # degrees
    max_angle: float = 100.0  # degrees
    psf_samples: int = 512


class GlareModel:
    """
    Disability glare model following CIE 180:2010.
    """

    def __init__(self, params: GlareParameters):
        self.params = params
        self.age_factor = 1.0 + (params.age / 70.0) ** 4
        self._psf_cache: Dict[Tuple[float, float, Tuple[int, int]], np.ndarray] = {}

    def compute_glare_psf(
        self,
        pupil_diameter: float,
        wavelength: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute the glare point spread function.
        """

        angles = self._create_angular_grid()
        psf = self._cie_general_psf(angles, pupil_diameter, wavelength)

        if self.params.include_corneal_reflections:
            psf += self._corneal_reflections(angles, pupil_diameter)

        psf = psf / np.sum(psf)
        return psf

    def apply_spectral_glare(self, img_rgb: np.ndarray, pupil_map: np.ndarray) -> np.ndarray:
        """
        Apply wavelength-dependent glare to an RGB image.
        """

        wavelengths = {"R": 620.0, "G": 555.0, "B": 470.0}  # nanometers
        result = np.zeros_like(img_rgb)
        avg_pupil = float(np.mean(pupil_map))

        image_shape = img_rgb.shape[:2]

        for channel_index, channel in enumerate(PHOTORECEPTOR_CHANNELS):
            wavelength = wavelengths[channel]
            psf = self._get_effective_psf(avg_pupil, wavelength, image_shape)
            veiling = signal.fftconvolve(
                img_rgb[:, :, channel_index],
                psf,
                mode="same",
            ).real
            result[:, :, channel_index] = img_rgb[:, :, channel_index] + veiling

        return result

    def get_effective_psf(
        self,
        pupil_diameter: float,
        image_shape: Tuple[int, int],
        wavelength: Optional[float] = None,
    ) -> np.ndarray:
        """Expose the trimmed PSF used for convolution."""

        return self._get_effective_psf(pupil_diameter, wavelength, image_shape).copy()

    def _get_effective_psf(
        self,
        pupil_diameter: float,
        wavelength: Optional[float],
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        target_shape = self._target_psf_shape(image_shape)
        cache_key = (
            round(pupil_diameter, 2),
            round(0.0 if wavelength is None else wavelength, 1),
            target_shape,
        )

        psf = self._psf_cache.get(cache_key)
        if psf is None:
            base_psf = self.compute_glare_psf(pupil_diameter, wavelength)
            psf = self._trim_and_normalize_psf(base_psf, target_shape)
            self._psf_cache[cache_key] = psf

        return psf

    def _target_psf_shape(self, image_shape: Tuple[int, int]) -> Tuple[int, int]:
        height, width = image_shape
        target_h = min(self.params.psf_samples, 2 * height + 1)
        target_w = min(self.params.psf_samples, 2 * width + 1)

        if target_h % 2 == 0:
            target_h -= 1
        if target_w % 2 == 0:
            target_w -= 1

        target_h = max(target_h, 3)
        target_w = max(target_w, 3)

        return target_h, target_w

    def _trim_and_normalize_psf(
        self, psf: np.ndarray, target_shape: Tuple[int, int]
    ) -> np.ndarray:
        target_h, target_w = target_shape

        center_y = psf.shape[0] // 2
        center_x = psf.shape[1] // 2
        half_h = target_h // 2
        half_w = target_w // 2

        start_y = max(center_y - half_h, 0)
        end_y = min(start_y + target_h, psf.shape[0])
        start_y = end_y - target_h

        start_x = max(center_x - half_w, 0)
        end_x = min(start_x + target_w, psf.shape[1])
        start_x = end_x - target_w

        trimmed = psf[start_y:end_y, start_x:end_x].copy()
        trimmed = np.clip(trimmed, 0.0, None)

        total = float(np.sum(trimmed))
        if not np.isfinite(total) or total <= 0.0:
            trimmed = np.full((target_h, target_w), 1.0 / (target_h * target_w))
        else:
            trimmed /= total

        return trimmed

    def _create_angular_grid(self) -> np.ndarray:
        """Create a 2D angular grid for PSF evaluation."""

        n = self.params.psf_samples
        max_angle = self.params.max_angle

        axis = np.linspace(-max_angle, max_angle, n)
        xx, yy = np.meshgrid(axis, axis)
        angles = np.sqrt(xx**2 + yy**2)
        angles = np.maximum(angles, self.params.min_angle)

        return angles

    def _cie_general_psf(
        self,
        angles: np.ndarray,
        pupil_diameter: float,
        wavelength: Optional[float],
    ) -> np.ndarray:
        """CIE general disability glare equation."""

        psf = np.zeros_like(angles)

        mask_small = (angles >= 0.1) & (angles < 1.0)
        psf[mask_small] = 10.0 * self.age_factor / (angles[mask_small] ** 3)

        mask_medium = (angles >= 1.0) & (angles < 30.0)
        psf[mask_medium] = 10.0 * self.age_factor / (angles[mask_medium] ** 2)

        mask_large = (angles >= 30.0) & (angles <= 100.0)
        psf[mask_large] = 5.0 * self.age_factor / (angles[mask_large] ** 1.5)

        if wavelength and self.params.include_wavelength_dependence:
            lambda_factor = (550.0 / wavelength) ** 4
            psf *= lambda_factor

        pupil_factor = (pupil_diameter / 5.0) ** 0.5
        psf *= pupil_factor

        return psf

    def _corneal_reflections(self, angles: np.ndarray, pupil_diameter: float) -> np.ndarray:
        """Add Purkinje images (corneal reflections)."""

        reflection = np.zeros_like(angles)

        mask1 = np.abs(angles - 3.0) < 0.5
        reflection[mask1] = 0.001 * self.age_factor

        mask2 = np.abs(angles - 3.5) < 0.7
        reflection[mask2] = 0.0001 * self.age_factor

        reflection *= np.sqrt(pupil_diameter / 5.0)

        return reflection
