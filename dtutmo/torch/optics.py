"""
Optics stage implemented with torch tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_otf(
    luminance: float,
    img_shape: Tuple[int, int],
    diagonal_inches: float,
    viewing_distance: float,
    field_deg: float,
    age: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Torch-based OTF computation producing tensors on the requested device.

    Parameters mirror the NumPy implementation; luminance is reserved for
    future refinement but kept for API compatibility.
    """

    del luminance  # currently unused

    height, width = img_shape

    diag_m = diagonal_inches * 0.0254
    diag_pix = torch.tensor((height**2 + width**2) ** 0.5, dtype=dtype, device=device)
    half_fov = torch.tensor(field_deg / 2.0, dtype=dtype, device=device)
    radians = half_fov * torch.pi / 180.0
    ppd = diag_pix * viewing_distance / (2.0 * diag_m * torch.tan(radians))
    ppd_value = float(ppd.item())

    # Frequency grid
    fx = torch.fft.fftfreq(width, d=1.0 / ppd_value, device=device, dtype=dtype)
    fy = torch.fft.fftfreq(height, d=1.0 / ppd_value, device=device, dtype=dtype)
    freq_y, freq_x = torch.meshgrid(fy, fx, indexing="ij")
    freq_map = torch.sqrt(freq_x**2 + freq_y**2)

    age_factor = 1.0 + (age - 20.0) / 100.0
    cutoff_freq = 60.0 / age_factor
    otf = torch.exp(-((freq_map / cutoff_freq) ** 2))
    return otf, freq_map


def apply_otf(img: torch.Tensor, otf: torch.Tensor) -> torch.Tensor:
    """
    Apply an OTF to an RGB tensor in NCHW format.
    """

    # Expand OTF to match spatial dimensions.
    otf_shifted = torch.fft.ifftshift(otf)

    channels = []
    for c in range(img.shape[1]):
        img_fft = torch.fft.fft2(img[:, c, :, :])
        filtered_fft = img_fft * otf_shifted
        filtered = torch.fft.ifft2(filtered_fft).real
        channels.append(filtered.unsqueeze(1))

    return torch.cat(channels, dim=1)


@dataclass
class GlareParameters:
    age: float = 24.0
    model: str = "cie_general"
    include_wavelength_dependence: bool = True
    include_corneal_reflections: bool = True
    min_angle: float = 0.1
    max_angle: float = 100.0
    psf_samples: int = 256


class GlareModel:
    """Torch version of the disability glare model."""

    def __init__(self, params: GlareParameters, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.params = params
        self.device = device
        self.dtype = dtype
        self.age_factor = 1.0 + (params.age / 70.0) ** 4

    def compute_glare_psf(
        self,
        pupil_diameter: float,
        wavelength: Optional[float] = None,
    ) -> torch.Tensor:
        angles = self._create_angular_grid()
        psf = self._cie_general_psf(angles, pupil_diameter, wavelength)
        if self.params.include_corneal_reflections:
            psf = psf + self._corneal_reflections(angles, pupil_diameter)
        psf = psf / psf.sum()
        return psf

    def apply_spectral_glare(self, img_rgb: torch.Tensor, pupil_map: torch.Tensor) -> torch.Tensor:
        """
        Apply glare to an image in NCHW format.
        """

        wavelengths = [620.0, 555.0, 470.0]
        result = torch.zeros_like(img_rgb)
        avg_pupil = float(pupil_map.mean().item())

        for channel, wavelength in enumerate(wavelengths):
            psf = self.compute_glare_psf(avg_pupil, wavelength)
            kernel = psf.to(device=img_rgb.device, dtype=img_rgb.dtype)
            kernel = kernel / kernel.sum()

            # Prepare convolution kernel: (out_channels, in_channels/groups, kH, kW)
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            pad_y = kernel.shape[-2] // 2
            pad_x = kernel.shape[-1] // 2
            padded = F.conv2d(
                img_rgb[:, channel : channel + 1, :, :],
                kernel,
                padding=(pad_y, pad_x),
            )
            result[:, channel : channel + 1, :, :] = img_rgb[:, channel : channel + 1, :, :] + padded

        return result

    def _create_angular_grid(self) -> torch.Tensor:
        params = self.params
        axis = torch.linspace(
            -params.max_angle,
            params.max_angle,
            params.psf_samples,
            device=self.device,
            dtype=self.dtype,
        )
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        angles = torch.sqrt(xx**2 + yy**2)
        return torch.clamp(angles, min=params.min_angle)

    def _cie_general_psf(
        self,
        angles: torch.Tensor,
        pupil_diameter: float,
        wavelength: Optional[float],
    ) -> torch.Tensor:
        psf = torch.zeros_like(angles)

        mask_small = (angles >= 0.1) & (angles < 1.0)
        mask_medium = (angles >= 1.0) & (angles < 30.0)
        mask_large = (angles >= 30.0) & (angles <= 100.0)

        psf = psf + torch.where(mask_small, 10.0 * self.age_factor / (angles**3), torch.zeros_like(angles))
        psf = psf + torch.where(mask_medium, 10.0 * self.age_factor / (angles**2), torch.zeros_like(angles))
        psf = psf + torch.where(mask_large, 5.0 * self.age_factor / (angles**1.5), torch.zeros_like(angles))

        if wavelength and self.params.include_wavelength_dependence:
            lambda_factor = (550.0 / wavelength) ** 4
            psf = psf * lambda_factor

        psf = psf * (pupil_diameter / 5.0) ** 0.5
        return psf

    def _corneal_reflections(
        self,
        angles: torch.Tensor,
        pupil_diameter: float,
    ) -> torch.Tensor:
        reflection = torch.zeros_like(angles)

        mask1 = torch.abs(angles - 3.0) < 0.5
        reflection = reflection + torch.where(mask1, 0.001 * self.age_factor, torch.zeros_like(angles))

        mask2 = torch.abs(angles - 3.5) < 0.7
        reflection = reflection + torch.where(mask2, 0.0001 * self.age_factor, torch.zeros_like(angles))

        reflection = reflection * torch.sqrt(torch.tensor(pupil_diameter / 5.0, device=angles.device, dtype=angles.dtype))
        return reflection
