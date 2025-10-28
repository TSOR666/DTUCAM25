"""Complete photoreceptor response model with Hood & Finkelstein adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from dtutmo.photoreceptors.constants import CHANNEL_TO_CONE, PHOTORECEPTOR_CHANNELS


@dataclass
class PhotoreceptorParams:
    """Hood & Finkelstein (1986) adaptation parameters."""

    # Cone parameters
    k1_L: float = 35.0
    k1_M: float = 35.0
    k1_S: float = 25.0
    O1_cone: float = 100.0
    m_cone: float = 0.74

    # Rod parameters
    k1_rod: float = 5.0
    O1_rod: float = 1.0
    m_rod: float = 0.73

    # Bleaching constants
    bleach_const_cone: float = np.log(4.5)
    bleach_const_rod: float = np.log(4.4)

    # Neural noise parameters
    sigma_neural_cone: float = 1.0
    sigma_neural_rod: float = 1.0

    # Adaptation dependent offset s(I_a)
    s_base_cone: float = 1.0
    s_adapt_factor_cone: float = 0.1
    s_base_rod: float = 1.0
    s_adapt_factor_rod: float = 0.15


class CorrectedPhotoreceptorResponse:
    """Full photoreceptor model including Hood & Finkelstein adaptation."""

    def __init__(
        self,
        params: Optional[PhotoreceptorParams] = None,
        channel_order: Sequence[str] = PHOTORECEPTOR_CHANNELS,
    ) -> None:
        self.params = params or PhotoreceptorParams()
        self.epsilon = 0.01
        self.channel_order: Tuple[str, ...] = tuple(channel_order)
        self._validate_channel_order()
        self.cone_types: Tuple[str, ...] = tuple(
            CHANNEL_TO_CONE[channel] for channel in self.channel_order
        )

    def pupil_diameter_watson(
        self,
        adapt_lum: np.ndarray,
        age: float = 24.0,
        field_deg: float = 60.0,
    ) -> np.ndarray:
        """Watson's pupil diameter model."""

        field_area = ((field_deg / 2.0) ** 2) * np.pi
        pupil_diameter = np.full_like(adapt_lum, 4.0, dtype=float)

        for _ in range(5):
            pupil_area = np.pi * (pupil_diameter / 2.0) ** 2
            trolands = adapt_lum * pupil_area
            k = np.power((trolands * field_area) / 846.0, 0.41)
            D_sd = 7.75 - (5.75 * k) / (k + 2.0)
            pupil_diameter = D_sd + (age - 28.58) * (0.02132 - 0.009562 * D_sd)
            pupil_diameter = np.clip(pupil_diameter, 2.0, 8.0)

        return pupil_diameter

    def compute_hood_sigma(self, I_adapt_td: np.ndarray, phototype: str = "L_cone") -> np.ndarray:
        """Hood & Finkelstein semi-saturation parameter."""

        if phototype == "L_cone":
            k1 = self.params.k1_L
            O1 = self.params.O1_cone
            m = self.params.m_cone
        elif phototype == "M_cone":
            k1 = self.params.k1_M
            O1 = self.params.O1_cone
            m = self.params.m_cone
        elif phototype == "S_cone":
            k1 = self.params.k1_S
            O1 = self.params.O1_cone
            m = self.params.m_cone
        else:
            k1 = self.params.k1_rod
            O1 = self.params.O1_rod
            m = self.params.m_rod

        ratio = (O1 + I_adapt_td) / O1
        return k1 * np.power(ratio, m)

    def get_adaptation_characteristics(self, phototype: str = "L_cone") -> Dict[str, float]:
        """Return key Hood adaptation parameters for a photoreceptor type."""

        if phototype == "L_cone":
            return {
                "k1": self.params.k1_L,
                "O1": self.params.O1_cone,
                "m": self.params.m_cone,
                "type": "L-cone",
            }
        if phototype == "M_cone":
            return {
                "k1": self.params.k1_M,
                "O1": self.params.O1_cone,
                "m": self.params.m_cone,
                "type": "M-cone",
            }
        if phototype == "S_cone":
            return {
                "k1": self.params.k1_S,
                "O1": self.params.O1_cone,
                "m": self.params.m_cone,
                "type": "S-cone",
            }
        return {
            "k1": self.params.k1_rod,
            "O1": self.params.O1_rod,
            "m": self.params.m_rod,
            "type": "Rod",
        }

    def compute_bleaching_factor(
        self, I_adapt_td: np.ndarray, phototype: str = "L_cone"
    ) -> np.ndarray:
        """Photopigment bleaching factor."""

        bleach_const = (
            self.params.bleach_const_cone if "cone" in phototype.lower() else self.params.bleach_const_rod
        )
        p = bleach_const / (bleach_const + np.log(I_adapt_td + self.epsilon))
        return np.clip(p, 0.1, 1.0)

    def compute_s_adaptation(self, I_adapt_td: np.ndarray, phototype: str = "L_cone") -> np.ndarray:
        """Adaptation-dependent offset term."""

        if "cone" in phototype.lower():
            s_base = self.params.s_base_cone
            s_factor = self.params.s_adapt_factor_cone
        else:
            s_base = self.params.s_base_rod
            s_factor = self.params.s_adapt_factor_rod

        return s_base + s_factor * np.log10(I_adapt_td + self.epsilon)

    def compute_rmax_adaptation(
        self, I_adapt_td: np.ndarray, p: np.ndarray, phototype: str = "L_cone"
    ) -> np.ndarray:
        """Adaptation-dependent maximum response."""

        if "cone" in phototype.lower():
            k2 = 700.0
            O2 = 100.0
        else:
            k2 = 500.0
            O2 = 10.0

        effective_adapt = O2 + (p * I_adapt_td)
        ratio = effective_adapt / O2
        return k2 * np.power(ratio, -0.5)

    def photo_response_complete(
        self,
        I_signal: np.ndarray,
        I_adapt: np.ndarray,
        pupil_size: float,
        n: float = 1.0,
        phototype: str = "L_cone",
    ) -> np.ndarray:
        """Compute the complete photoreceptor response."""

        d = pupil_size
        area = (np.pi * (d**2)) / 4.0
        factor = 1.0 - (d / 9.7) ** 2 + (d / 12.4) ** 4

        I_signal_td = np.clip(I_signal * factor * area, 0.0, None)
        log_retinal_ill = np.log(I_signal_td + self.epsilon)
        I_adapt_td = np.clip(I_adapt * factor * area, 0.0, None)

        sigma_hood = self.compute_hood_sigma(I_adapt_td, phototype)
        p = self.compute_bleaching_factor(I_adapt_td, phototype)
        s_adapted = self.compute_s_adaptation(I_adapt_td, phototype)

        sigma_neural = (
            self.params.sigma_neural_cone if "cone" in phototype.lower() else self.params.sigma_neural_rod
        )
        sigma_effective = sigma_hood + (sigma_neural / p)
        sigma_effective = np.maximum(sigma_effective, self.epsilon)

        R_max = self.compute_rmax_adaptation(I_adapt_td, p, phototype)

        signal_term = p * (log_retinal_ill - s_adapted)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            modulated_magnitude = np.power(np.abs(signal_term), n)
        modulated_signal = np.sign(signal_term) * modulated_magnitude

        denom = np.power(sigma_effective, n) + np.abs(modulated_signal)
        response = np.sign(modulated_signal) * (np.abs(modulated_signal) / denom)
        response = response * R_max
        response = np.clip(response, -60.0, 60.0)

        return np.exp(response - self.epsilon) / np.pi

    def process_cones(
        self,
        img_lms: np.ndarray,
        adapt_luminance: np.ndarray,
        pupil_size: float,
        n: float = 0.74,
    ) -> np.ndarray:
        """Process L, M, S cone responses."""

        cone_responses = np.zeros_like(img_lms)
        for idx, cone_type in enumerate(self.cone_types):
            cone_responses[:, :, idx] = self.photo_response_complete(
                img_lms[:, :, idx], adapt_luminance, pupil_size, n, cone_type
            )

        return cone_responses

    def process_rods(
        self,
        rod_signal: np.ndarray,
        adapt_luminance: np.ndarray,
        pupil_size: float,
        n: float = 0.73,
    ) -> np.ndarray:
        """Process rod responses."""

        return self.photo_response_complete(rod_signal, adapt_luminance, pupil_size, n, "rod")

    # ------------------------------------------------------------------
    # Backwards compatibility helpers
    # ------------------------------------------------------------------
    def process_cones_original(
        self,
        img_lms: np.ndarray,
        adapt_luminance: np.ndarray,
        pupil_size: float,
        n: float = 0.74,
    ) -> np.ndarray:
        """Alias for the updated cone response computation."""

        return self.process_cones(img_lms, adapt_luminance, pupil_size, n)

    def process_rods_original(
        self,
        rod_signal: np.ndarray,
        adapt_luminance: np.ndarray,
        pupil_size: float,
        n: float = 0.73,
    ) -> np.ndarray:
        """Alias for the updated rod response computation."""

        return self.process_rods(rod_signal, adapt_luminance, pupil_size, n)

    # ------------------------------------------------------------------
    # Channel mapping helpers
    # ------------------------------------------------------------------
    def _validate_channel_order(self) -> None:
        if set(self.channel_order) != set(PHOTORECEPTOR_CHANNELS):
            raise ValueError(
                "Channel order must include R, G, and B exactly once to align "
                "with the glare and mesopic modules."
            )

    def cone_for_channel(self, channel: str) -> str:
        """Return the photoreceptor type associated with a rendering channel."""

        if channel not in CHANNEL_TO_CONE:
            raise KeyError(f"Unknown channel '{channel}'.")
        return CHANNEL_TO_CONE[channel]


CompletePhotoreceptorWithHood = CorrectedPhotoreceptorResponse
