"""DTUCAM: DTU Color Appearance Model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from dtutmo.adaptation.local_adaptation import LocalAdaptation
from dtutmo.photoreceptors.response import CorrectedPhotoreceptorResponse


@dataclass
class DTUCAMParameters:
    """Parameter bundle for :class:`DTUCAM`."""

    w_rg: float = 1.0
    w_by: float = 0.7
    J_scale: float = 100.0
    J_offset: float = 0.1
    M_scale: float = 50.0
    cone_exponent: float = 0.74
    white_min: float = 1e-4
    hue_quadrature: Dict[str, float] = field(
        default_factory=lambda: {
            "red": 0.0,
            "yellow": 90.0,
            "green": 180.0,
            "blue": 270.0,
        }
    )


class DTUCAM:
    """Physiologically grounded color appearance model."""

    def __init__(
        self,
        params: Optional[DTUCAMParameters] = None,
        *,
        observer_age: float = 24.0,
        field_diameter: float = 60.0,
        peak_sensitivity: float = 6.0,
        pixels_per_degree: float = 45.0,
    ) -> None:
        self.params = params or DTUCAMParameters()
        self.photoreceptor = CorrectedPhotoreceptorResponse()
        self.local_adapt = LocalAdaptation(
            peak_sensitivity=peak_sensitivity, ppd=pixels_per_degree
        )
        self.observer_age = observer_age
        self.field_diameter = field_diameter

        self._init_matrices()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        img_xyz: np.ndarray,
        white_xyz: np.ndarray,
        background_lum: float,
        *,
        surround: str = "average",
        return_intermediate: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Map physical XYZ values to perceptual correlates."""

        img_xyz = np.asarray(img_xyz, dtype=float)
        white_xyz = np.asarray(white_xyz, dtype=float)

        results: Dict[str, np.ndarray] = {} if return_intermediate else {}

        img_lms = self._xyz_to_lms(img_xyz)
        white_lms = self._xyz_to_lms(white_xyz.reshape(1, 1, 3))
        white_lms = np.clip(white_lms, self.params.white_min, None)

        luminance = np.clip(img_xyz[..., 1], self.params.white_min, None)
        _, adapt_lum = self.local_adapt.compute(luminance)
        mean_adapt = float(np.mean(adapt_lum)) if adapt_lum.size else float(np.mean(luminance))
        if not np.isfinite(mean_adapt):
            mean_adapt = max(background_lum, self.params.white_min)

        degree_adapt = self._degree_of_adaptation(mean_adapt, surround)
        img_lms_adapted = degree_adapt * (img_lms / white_lms) + (1.0 - degree_adapt) * img_lms

        pupil = self.photoreceptor.pupil_diameter_watson(
            adapt_lum,
            age=self.observer_age,
            field_deg=self.field_diameter,
        )
        avg_pupil = float(np.mean(pupil)) if pupil.size else 4.0

        cone_responses = self.photoreceptor.process_cones(
            img_lms_adapted,
            adapt_lum,
            avg_pupil,
            n=self.params.cone_exponent,
        )

        opponent = self._lms_to_opponent(cone_responses)
        achromatic = opponent[..., 0]
        red_green = opponent[..., 1]
        blue_yellow = opponent[..., 2]

        lightness = self._lightness(achromatic, surround)
        hue = self._hue(red_green, blue_yellow)
        chroma = self._chroma(red_green, blue_yellow)
        colorfulness = self._colorfulness(chroma, lightness, surround)
        brightness = self._brightness(lightness, achromatic, surround)
        saturation = self._saturation(colorfulness, brightness)

        if return_intermediate:
            results.update(
                {
                    "lms": img_lms,
                    "lms_adapted": img_lms_adapted,
                    "adaptation": adapt_lum,
                    "cone_responses": cone_responses,
                    "opponent": opponent,
                }
            )

        appearance: Dict[str, np.ndarray] = {
            "lightness": lightness,
            "colorfulness": colorfulness,
            "hue": hue,
            "chroma": chroma,
            "brightness": brightness,
            "saturation": saturation,
        }

        if return_intermediate:
            appearance["intermediate"] = results

        return appearance

    def inverse(
        self,
        lightness: np.ndarray,
        colorfulness: np.ndarray,
        hue: np.ndarray,
        display_white_xyz: np.ndarray,
        display_max_lum: float,
        *,
        surround: str = "dim",
    ) -> np.ndarray:
        """Inverse DTUCAM mapping back to XYZ."""

        achromatic = self._inverse_lightness(lightness, surround)
        chroma = self._inverse_colorfulness(colorfulness, lightness, surround)

        hue_rad = np.deg2rad(hue)
        red_green = chroma * np.cos(hue_rad)
        blue_yellow = chroma * np.sin(hue_rad)

        opponent = np.stack((achromatic, red_green, blue_yellow), axis=-1)
        cone_responses = self._opponent_to_lms(opponent)

        lms_est = self._inverse_photoreceptor(cone_responses, display_max_lum, surround)

        white_lms = self._xyz_to_lms(display_white_xyz.reshape(1, 1, 3))
        degree_adapt = self._degree_of_adaptation(display_max_lum / 5.0, surround)
        adapt_gain = degree_adapt / np.clip(white_lms, self.params.white_min, None) + (1.0 - degree_adapt)
        lms_unadapted = lms_est / np.clip(adapt_gain, self.params.white_min, None)
        display_xyz = self._lms_to_xyz(lms_unadapted)
        return np.clip(display_xyz, 0.0, display_max_lum * 2.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_matrices(self) -> None:
        self.xyz_to_lms_matrix = np.array(
            [
                [0.38971, 0.68898, -0.07868],
                [-0.22981, 1.18340, 0.04641],
                [0.0, 0.0, 1.0],
            ]
        )
        self.lms_to_xyz_matrix = np.linalg.inv(self.xyz_to_lms_matrix)
        self.lms_to_opponent_matrix = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, 0.0],
                [0.5, 0.5, -1.0],
            ]
        ) / 3.0
        self.opponent_to_lms_matrix = np.linalg.pinv(self.lms_to_opponent_matrix)

    def _xyz_to_lms(self, xyz: np.ndarray) -> np.ndarray:
        return np.tensordot(xyz, self.xyz_to_lms_matrix.T, axes=([-1], [0]))

    def _lms_to_xyz(self, lms: np.ndarray) -> np.ndarray:
        return np.tensordot(lms, self.lms_to_xyz_matrix.T, axes=([-1], [0]))

    def _lms_to_opponent(self, lms: np.ndarray) -> np.ndarray:
        return np.tensordot(lms, self.lms_to_opponent_matrix.T, axes=([-1], [0]))

    def _opponent_to_lms(self, opponent: np.ndarray) -> np.ndarray:
        return np.tensordot(opponent, self.opponent_to_lms_matrix.T, axes=([-1], [0]))

    def _degree_of_adaptation(self, luminance: float, surround: str) -> float:
        surround_factor = {"dark": 0.8, "dim": 0.9, "average": 1.0}.get(surround, 0.9)
        exponent = -(luminance + 42.0) / 92.0
        degree = surround_factor * (1.0 - (1.0 / 3.6) * np.exp(exponent))
        return float(np.clip(degree, 0.0, 1.0))

    def _lightness(self, achromatic: np.ndarray, surround: str) -> np.ndarray:
        surround_gain = {"dark": 0.8, "dim": 0.9, "average": 1.0}.get(surround, 0.9)
        achromatic = np.maximum(achromatic, 0.0)
        base = np.log10(achromatic + self.params.J_offset) - np.log10(self.params.J_offset)
        return self.params.J_scale * surround_gain * np.clip(base, 0.0, None)

    def _inverse_lightness(self, lightness: np.ndarray, surround: str) -> np.ndarray:
        surround_gain = {"dark": 0.8, "dim": 0.9, "average": 1.0}.get(surround, 0.9)
        exponent = lightness / (self.params.J_scale * max(surround_gain, 1e-6))
        return self.params.J_offset * (np.power(10.0, exponent) - 1.0)

    def _hue(self, red_green: np.ndarray, blue_yellow: np.ndarray) -> np.ndarray:
        hue = np.degrees(np.arctan2(blue_yellow, red_green))
        return np.where(hue < 0.0, hue + 360.0, hue)

    def _chroma(self, red_green: np.ndarray, blue_yellow: np.ndarray) -> np.ndarray:
        return np.sqrt(red_green**2 + blue_yellow**2)

    def _colorfulness(self, chroma: np.ndarray, lightness: np.ndarray, surround: str) -> np.ndarray:
        scale = self._colorfulness_scale(surround)
        return scale * chroma * np.sqrt(np.clip(lightness, 0.0, None) / 100.0 + 1e-6)

    def _inverse_colorfulness(
        self,
        colorfulness: np.ndarray,
        lightness: np.ndarray,
        surround: str,
    ) -> np.ndarray:
        scale = self._colorfulness_scale(surround)
        denom = scale * np.sqrt(np.clip(lightness, 0.0, None) / 100.0 + 1e-6)
        return colorfulness / np.clip(denom, 1e-6, None)

    def _colorfulness_scale(self, surround: str) -> float:
        return {"dark": 0.8, "dim": 0.9, "average": 1.0}.get(surround, 0.9)

    def _brightness(self, lightness: np.ndarray, achromatic: np.ndarray, surround: str) -> np.ndarray:
        surround_gain = {"dark": 0.8, "dim": 0.9, "average": 1.0}.get(surround, 0.9)
        return (4.0 / max(surround_gain, 1e-6)) * np.sqrt(np.clip(lightness, 0.0, None) / 100.0) * (achromatic + 4.0)

    def _saturation(self, colorfulness: np.ndarray, brightness: np.ndarray) -> np.ndarray:
        return 100.0 * np.sqrt(colorfulness / (np.clip(brightness, 1e-6, None)))

    def _inverse_photoreceptor(
        self,
        cone_responses: np.ndarray,
        display_max_lum: float,
        surround: str,
    ) -> np.ndarray:
        response = np.clip(cone_responses, 0.0, None)
        norm = np.max(response, axis=(-1, -2), keepdims=True)
        norm = np.clip(norm, 1e-6, None)
        scaled = response / norm
        return scaled * (display_max_lum / 100.0)

