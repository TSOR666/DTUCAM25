"""Display output mapping utilities.

This module integrates three complementary strategies for mapping
photoreceptor responses to display luminances:

1. Whiteboard equation – fast approximation derived from the inverse
   Naka–Rushton relation.
2. Complete photoreceptor inverse – exact analytical inversion of the
   dual-adaptation photoreceptor model.
3. Hybrid approach – blends the two for quality and performance.

The mapper is designed as the final stage of the DTUTMO pipeline when a
full color appearance model is not employed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..adaptation.display import DisplayAdaptation
from ..core.config import DisplayStandard
from ..photoreceptors.inverse_complete import InversePhotoreceptorComplete


@dataclass
class DisplayOutputConfig:
    """Configuration for :class:`DisplayOutputMapper`."""

    use_whiteboard_formula: bool = True
    alpha: float = 1.0
    target_luminance: float = 100.0
    target_standard: DisplayStandard = DisplayStandard.REC_709
    use_full_inverse: bool = True
    viewer_adaptation: float = 20.0


class DisplayOutputMapper:
    """Map photoreceptor responses to display-referred RGB."""

    def __init__(self, config: Optional[DisplayOutputConfig] = None) -> None:
        self.config = config or DisplayOutputConfig()
        self.inverse_photoreceptor = InversePhotoreceptorComplete()
        self.display_adapter = DisplayAdaptation()
        self.display_spec = self.display_adapter.get_display_spec(
            self.config.target_standard
        )

    def map_to_display(
        self,
        photoreceptor_responses: np.ndarray,
        scene_adaptation: np.ndarray,
        method: str = "hybrid",
    ) -> np.ndarray:
        """Convert photoreceptor responses to display RGB."""

        method = method.lower()
        if method == "whiteboard":
            luminance = self._whiteboard_mapping(photoreceptor_responses)
        elif method == "full_inverse":
            luminance = self._full_inverse_mapping(
                photoreceptor_responses, scene_adaptation
            )
        else:
            luminance = self._hybrid_mapping(
                photoreceptor_responses, scene_adaptation
            )

        display_xyz = self._luminance_to_xyz(luminance)
        display_rgb = self.display_adapter.adapt_to_display(
            display_xyz, self.config.target_standard
        )
        return display_rgb

    def _whiteboard_mapping(self, responses: np.ndarray) -> np.ndarray:
        """Simplified inverse using the whiteboard tone curve."""

        L_mean_d = self.config.target_luminance
        n = 0.74
        alpha = np.clip(self.config.alpha, 0.0, 1.0)

        max_val = float(np.max(responses))
        if max_val <= 0.0:
            return np.zeros_like(responses)

        R_normalized = np.clip(responses / (max_val + 1e-6), 0.0, 0.99)
        numerator = R_normalized * L_mean_d
        denominator = np.power(1.0 - R_normalized, n)
        denominator = np.maximum(denominator, 1e-6)
        luminance = numerator / denominator

        if alpha < 1.0:
            luminance_linear = R_normalized * L_mean_d
            luminance = alpha * luminance + (1.0 - alpha) * luminance_linear

        return np.clip(luminance, 0.0, self.display_spec["max_luminance"])

    def _full_inverse_mapping(
        self, responses: np.ndarray, scene_adaptation: np.ndarray
    ) -> np.ndarray:
        """Full photoreceptor inverse for each cone type."""

        display_adaptation = np.full_like(
            scene_adaptation, self.config.viewer_adaptation
        )
        pupil_size = 4.5
        lms_signal = self.inverse_photoreceptor.inverse_cones(
            responses,
            display_adaptation,
            pupil_size,
            n=0.74,
        )

        # Replace NaNs/infs introduced by extreme values with safe defaults.
        lms_signal = np.nan_to_num(
            lms_signal,
            nan=0.0,
            posinf=self.display_spec["max_luminance"],
            neginf=0.0,
        )

        max_val = float(np.max(lms_signal))
        if max_val > 0 and np.isfinite(max_val):
            scale = self.display_spec["max_luminance"] / max_val
        else:
            scale = 1.0
        return np.clip(lms_signal * scale, 0.0, self.display_spec["max_luminance"])

    def _hybrid_mapping(
        self, responses: np.ndarray, scene_adaptation: np.ndarray
    ) -> np.ndarray:
        """Blend whiteboard and full inverse mappings."""

        coarse = self._whiteboard_mapping(responses)
        luminance = np.mean(responses, axis=2)
        grad_y, grad_x = np.gradient(luminance)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        threshold = np.percentile(gradient_mag, 75)
        if threshold <= 0:
            return coarse

        refine_mask = gradient_mag > threshold
        refined = self._full_inverse_mapping(responses, scene_adaptation)
        weight = refine_mask[:, :, np.newaxis].astype(responses.dtype)
        blend = weight * refined + (1.0 - weight) * coarse
        return np.clip(blend, 0.0, self.display_spec["max_luminance"])

    @staticmethod
    def _luminance_to_xyz(luminance: np.ndarray) -> np.ndarray:
        """Approximate XYZ conversion from display luminance."""

        if luminance.ndim == 2:
            Y = luminance
            X = Y * (95.047 / 100.0)
            Z = Y * (108.883 / 100.0)
            return np.stack([X, Y, Z], axis=2)

        Y = luminance[:, :, 1]
        X = luminance[:, :, 0] * (95.047 / 100.0)
        Z = luminance[:, :, 2] * (108.883 / 100.0)
        return np.stack([X, Y, Z], axis=2)
