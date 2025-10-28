"""
Inverse photoreceptor response model for display adaptation.
"""

from __future__ import annotations

import numpy as np


class InversePhotoreceptorResponse:
    """
    Inverse photoreceptor model translating target responses to display luminance.
    """

    def __init__(self) -> None:
        self.epsilon = 0.01

    def inverse_naka_rushton(
        self,
        response: np.ndarray,
        sigma: float,
        R_max: float = 1.0,
        n: float = 1.0,
    ) -> np.ndarray:
        """
        Invert the Naka-Rushton equation: R = R_max * (I^n / (I^n + sigma^n)).
        """

        R = np.clip(response, 0.0, R_max * 0.99)
        ratio = R / R_max
        denominator = np.maximum(1.0 - ratio, 1e-6)
        intensity = sigma * np.power(ratio / denominator, 1.0 / n)
        return intensity

    def display_luminance_from_response(
        self,
        responses: np.ndarray,
        viewer_adaptation: float,
        display_gamma: float = 2.4,
    ) -> np.ndarray:
        """
        Compute display-referred RGB values from desired photoreceptor responses.
        """

        display_linear = np.zeros_like(responses)
        sigma = 0.3 * (1.0 + 0.5 * np.log10(np.maximum(viewer_adaptation, 0.01)))

        for channel in range(responses.shape[2]):
            display_linear[:, :, channel] = self.inverse_naka_rushton(
                responses[:, :, channel],
                sigma,
                R_max=1.0,
                n=1.0,
            )

        display_encoded = np.power(np.maximum(display_linear, 0.0), 1.0 / display_gamma)
        return np.clip(display_encoded, 0.0, 1.0)

    def viewing_condition_correction(
        self,
        display_luminance: np.ndarray,
        surround: str = "dim",
    ) -> np.ndarray:
        """
        Apply a simple surround compensation (Hunt effect).
        """

        surround_factors = {"dark": 0.8, "dim": 0.9, "average": 1.0}
        factor = surround_factors.get(surround, 0.9)
        return display_luminance * factor
