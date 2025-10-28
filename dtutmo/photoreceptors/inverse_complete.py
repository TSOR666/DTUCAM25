"""Analytical inverse of the dual-adaptation photoreceptor model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .response import CorrectedPhotoreceptorResponse


@dataclass
class InverseParameters:
    """Container for per-photoreceptor parameters."""

    k1_L: float = 35.0
    k1_M: float = 35.0
    k1_S: float = 25.0
    O1_cone: float = 100.0
    m_cone: float = 0.74
    k2_cone: float = 700.0
    O2_cone: float = 100.0
    sigma_neural_cone: float = 1.0
    bleach_const_cone: float = np.log(4.5)
    s_base_cone: float = 1.0
    s_factor_cone: float = 0.1
    k1_rod: float = 5.0
    O1_rod: float = 1.0
    m_rod: float = 0.73
    k2_rod: float = 500.0
    O2_rod: float = 10.0
    sigma_neural_rod: float = 1.0
    bleach_const_rod: float = np.log(4.4)
    s_base_rod: float = 1.0
    s_factor_rod: float = 0.15


class InversePhotoreceptorComplete:
    """Explicit inverse of the complete photoreceptor response."""

    def __init__(self, params: Optional[InverseParameters] = None) -> None:
        self.params = params or InverseParameters()
        self.epsilon = 0.01

    def inverse_response(
        self,
        responses: np.ndarray,
        adaptation: np.ndarray,
        pupil_size: float,
        n: float = 0.74,
        phototype: str = "L_cone",
    ) -> np.ndarray:
        """Invert the photoreceptor response for a single photoreceptor type."""

        p = self._bleaching_factor(adaptation, phototype, pupil_size)
        sigma = self._sigma_total(adaptation, phototype, pupil_size, p)
        R_max = self._rmax(adaptation, phototype, pupil_size, p)

        r = np.clip(responses / (R_max + 1e-6), 0.0, 0.99)
        numerator = r * np.power(sigma, n)
        denominator = np.maximum(1.0 - r, 1e-6)
        x = np.power(numerator / denominator, 1.0 / n)

        s = self._s_offset(adaptation, phototype, pupil_size)
        E = (x / p) + s

        # Guard against numerical overflow.  The exponential can easily exceed
        # the floating point range for large excitations which, in turn,
        # propagates ``inf`` values through the pipeline.  Clipping the
        # exponent keeps the inverse stable without materially affecting the
        # perceptual output because values above this threshold already map to
        # the brightest display intensities.
        max_exponent = np.log(np.finfo(np.float64).max) - 1.0
        safe_E = np.clip(E, None, max_exponent)
        I_signal_td = np.exp(safe_E) - self.epsilon

        factor, area = self._pupil_factors(pupil_size)
        return np.maximum(I_signal_td / (factor * area), 0.0)

    def inverse_cones(
        self,
        cone_responses: np.ndarray,
        adaptation: np.ndarray,
        pupil_size: float,
        n: float = 0.74,
    ) -> np.ndarray:
        """Invert responses for the three cone classes."""

        cone_types = ("L_cone", "M_cone", "S_cone")
        lms = np.zeros_like(cone_responses)
        for idx, cone_type in enumerate(cone_types):
            lms[:, :, idx] = self.inverse_response(
                cone_responses[:, :, idx], adaptation, pupil_size, n, cone_type
            )
        return lms

    def inverse_rods(
        self,
        rod_response: np.ndarray,
        adaptation: np.ndarray,
        pupil_size: float,
        n: float = 0.73,
    ) -> np.ndarray:
        """Invert the rod response."""

        return self.inverse_response(rod_response, adaptation, pupil_size, n, "rod")

    def verify_inverse(
        self,
        test_signal: np.ndarray,
        adaptation: np.ndarray,
        pupil_size: float,
        n: float = 0.74,
        phototype: str = "L_cone",
    ) -> Tuple[float, np.ndarray]:
        """Verify the inverse by round-tripping through the forward model."""

        forward = CorrectedPhotoreceptorResponse()
        response = forward.photo_response_complete(
            test_signal, adaptation, pupil_size, n, phototype
        )
        recovered = self.inverse_response(response, adaptation, pupil_size, n, phototype)
        relative_error = np.abs(test_signal - recovered) / (np.abs(test_signal) + 1e-6)
        return float(np.max(relative_error)), recovered

    def _bleaching_factor(
        self, adaptation: np.ndarray, phototype: str, pupil_size: float
    ) -> np.ndarray:
        bleach_const = (
            self.params.bleach_const_cone
            if "cone" in phototype.lower()
            else self.params.bleach_const_rod
        )
        factor, area = self._pupil_factors(pupil_size)
        adaptation_td = adaptation * factor * area
        p = bleach_const / (bleach_const + np.log(adaptation_td + self.epsilon))
        return np.clip(p, 0.1, 1.0)

    def _s_offset(
        self, adaptation: np.ndarray, phototype: str, pupil_size: float
    ) -> np.ndarray:
        if "cone" in phototype.lower():
            s_base = self.params.s_base_cone
            s_factor = self.params.s_factor_cone
        else:
            s_base = self.params.s_base_rod
            s_factor = self.params.s_factor_rod
        factor, area = self._pupil_factors(pupil_size)
        adaptation_td = adaptation * factor * area
        return s_base + s_factor * np.log10(adaptation_td + self.epsilon)

    def _sigma_total(
        self,
        adaptation: np.ndarray,
        phototype: str,
        pupil_size: float,
        p: np.ndarray,
    ) -> np.ndarray:
        if phototype == "L_cone":
            k1, O1, m = self.params.k1_L, self.params.O1_cone, self.params.m_cone
            sigma_neural = self.params.sigma_neural_cone
        elif phototype == "M_cone":
            k1, O1, m = self.params.k1_M, self.params.O1_cone, self.params.m_cone
            sigma_neural = self.params.sigma_neural_cone
        elif phototype == "S_cone":
            k1, O1, m = self.params.k1_S, self.params.O1_cone, self.params.m_cone
            sigma_neural = self.params.sigma_neural_cone
        else:
            k1, O1, m = self.params.k1_rod, self.params.O1_rod, self.params.m_rod
            sigma_neural = self.params.sigma_neural_rod

        factor, area = self._pupil_factors(pupil_size)
        adaptation_td = adaptation * factor * area
        sigma_hood = k1 * np.power((O1 + adaptation_td) / O1, m)
        return sigma_hood + sigma_neural / p

    def _rmax(
        self,
        adaptation: np.ndarray,
        phototype: str,
        pupil_size: float,
        p: np.ndarray,
    ) -> np.ndarray:
        if "cone" in phototype.lower():
            k2 = self.params.k2_cone
            O2 = self.params.O2_cone
        else:
            k2 = self.params.k2_rod
            O2 = self.params.O2_rod
        factor, area = self._pupil_factors(pupil_size)
        adaptation_td = adaptation * factor * area
        return k2 * np.power((O2 + p * adaptation_td) / O2, -0.5)

    @staticmethod
    def _pupil_factors(pupil_size: float) -> Tuple[float, float]:
        area = (np.pi * (pupil_size**2)) / 4.0
        factor = 1.0 - (pupil_size / 9.7) ** 2 + (pupil_size / 12.4) ** 4
        return factor, area
