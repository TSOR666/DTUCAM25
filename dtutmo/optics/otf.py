"""
Optical transfer function utilities for simulating ocular blur.
"""

from __future__ import annotations

import numpy as np


def compute_otf(
    luminance: float,
    img_shape: tuple[int, int],
    diagonal_inches: float,
    viewing_distance: float,
    field_deg: float,
    age: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute an optical transfer function estimate.

    Parameters
    ----------
    luminance : float
        Average scene luminance in cd/m^2 (currently unused but kept for future work).
    img_shape : tuple
        Image shape (height, width).
    diagonal_inches : float
        Display diagonal in inches.
    viewing_distance : float
        Viewing distance in meters.
    field_deg : float
        Field of view in degrees.
    age : float
        Observer age in years.
    """

    del luminance  # reserved for future refinement

    height, width = img_shape

    # Compute pixels per degree
    diag_m = diagonal_inches * 0.0254
    diag_pix = np.sqrt(height**2 + width**2)
    ppd = diag_pix * viewing_distance / (2 * diag_m * np.tan(np.radians(field_deg / 2)))

    # Frequency grid
    fx = np.fft.fftfreq(width, d=1.0 / ppd)
    fy = np.fft.fftfreq(height, d=1.0 / ppd)
    freq_x, freq_y = np.meshgrid(fx, fy)
    freq_map = np.sqrt(freq_x**2 + freq_y**2)

    # Age-dependent OTF (simplified Gaussian falloff)
    age_factor = 1.0 + (age - 20.0) / 100.0
    cutoff_freq = 60.0 / age_factor  # cycles/degree
    otf = np.exp(-(freq_map / cutoff_freq) ** 2)

    return otf, freq_map


def apply_otf(img: np.ndarray, otf: np.ndarray) -> np.ndarray:
    """
    Apply the OTF to an RGB image via FFT convolution.
    """

    result = np.zeros_like(img)

    for channel in range(img.shape[2]):
        img_fft = np.fft.fft2(img[:, :, channel])
        filtered_fft = img_fft * np.fft.ifftshift(otf)
        result[:, :, channel] = np.real(np.fft.ifft2(filtered_fft))

    return result
