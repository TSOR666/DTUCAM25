"""
Mesopic (rod and cone) combination utilities.
"""

from __future__ import annotations

import numpy as np

from dtutmo.photoreceptors.constants import PHOTORECEPTOR_CHANNELS


def mesopic_global(
    cones: np.ndarray,
    rods: np.ndarray,
    photopic_lum: float,
    scotopic_lum: float,
) -> np.ndarray:
    """
    Global mesopic combination with a single rod weight.
    """

    if photopic_lum < 0.01:
        rod_weight = 1.0
    elif photopic_lum > 10.0:
        rod_weight = 0.0
    else:
        log_lum = np.log10(photopic_lum)
        rod_weight = np.interp(log_lum, [np.log10(0.01), np.log10(10.0)], [1.0, 0.0])

    mesopic = np.zeros_like(cones)

    for channel_index, _ in enumerate(PHOTORECEPTOR_CHANNELS):
        mesopic[:, :, channel_index] = (1.0 - rod_weight) * cones[:, :, channel_index] + rod_weight * rods

    return mesopic


def mesopic_local(
    cones: np.ndarray,
    rods: np.ndarray,
    photopic_lum_map: np.ndarray,
    scotopic_lum_map: np.ndarray,
) -> np.ndarray:
    """
    Local mesopic combination with spatially varying rod weights.
    """

    del scotopic_lum_map  # retained for future use

    rod_weight = np.zeros_like(photopic_lum_map, dtype=float)

    mask_scotopic = photopic_lum_map < 0.01
    rod_weight[mask_scotopic] = 1.0

    mask_photopic = photopic_lum_map > 10.0
    rod_weight[mask_photopic] = 0.0

    mask_mesopic = ~(mask_scotopic | mask_photopic)
    if np.any(mask_mesopic):
        log_lum = np.log10(photopic_lum_map[mask_mesopic])
        rod_weight[mask_mesopic] = np.interp(
            log_lum,
            [np.log10(0.01), np.log10(10.0)],
            [1.0, 0.0],
        )

    mesopic = np.zeros_like(cones)

    for channel_index, _ in enumerate(PHOTORECEPTOR_CHANNELS):
        mesopic[:, :, channel_index] = (
            (1.0 - rod_weight) * cones[:, :, channel_index]
            + rod_weight * rods
        )

    return mesopic
