"""Utilities to extract photoreceptor signals from RGB images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..photoreceptors.response import CorrectedPhotoreceptorResponse


@dataclass
class SpectralSensitivity:
    """Basic spectral sensitivity information."""

    photopic_peak: float = 555.0
    scotopic_peak: float = 507.0
    purkinje_shift: float = 48.0


class PhotoreceptorSignalExtractor:
    """Extract L, M, S cone and rod signals from linear RGB images."""

    def __init__(self, rgb_space: str = "sRGB") -> None:
        self.rgb_space = rgb_space
        self._init_transforms()

    def _init_transforms(self) -> None:
        if self.rgb_space.lower() == "adobergb":
            self.rgb_to_xyz = np.array(
                [
                    [0.5767309, 0.1855540, 0.1881852],
                    [0.2973769, 0.6273491, 0.0752741],
                    [0.0270343, 0.0706872, 0.9911085],
                ]
            )
        else:
            self.rgb_to_xyz = np.array(
                [
                    [0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041],
                ]
            )

        self.xyz_to_lms = np.array(
            [
                [0.38971, 0.68898, -0.07868],
                [-0.22981, 1.18340, 0.04641],
                [0.0, 0.0, 1.0],
            ]
        )
        self.rgb_to_lms = self.xyz_to_lms @ self.rgb_to_xyz
        self.rgb_to_scotopic = np.array([0.05, 0.85, 0.10])
        self.rgb_to_scotopic /= np.sum(self.rgb_to_scotopic)

    def extract_all_signals(
        self, img_rgb: np.ndarray, is_linear: bool = False
    ) -> Dict[str, np.ndarray]:
        if not is_linear:
            img_linear = self._linearize_srgb(img_rgb)
        else:
            img_linear = img_rgb

        lms = self._extract_cone_signals(img_linear)
        rods = self._extract_rod_signal(img_linear)
        photopic = self._photopic_luminance(img_linear)
        scotopic = self._scotopic_luminance(img_linear)

        return {
            "L": lms[:, :, 0],
            "M": lms[:, :, 1],
            "S": lms[:, :, 2],
            "rods": rods,
            "photopic_luminance": photopic,
            "scotopic_luminance": scotopic,
            "lms_combined": lms,
        }

    def _linearize_srgb(self, img: np.ndarray) -> np.ndarray:
        if img.max() > 1.0:
            img = img / 255.0
        linear = np.where(
            img <= 0.04045,
            img / 12.92,
            np.power((img + 0.055) / 1.055, 2.4),
        )
        return linear

    def _extract_cone_signals(self, img_linear: np.ndarray) -> np.ndarray:
        h, w, _ = img_linear.shape
        img_flat = img_linear.reshape(-1, 3)
        lms_flat = img_flat @ self.rgb_to_lms.T
        lms = lms_flat.reshape(h, w, 3)
        return np.maximum(lms, 0.0)

    def _extract_rod_signal(self, img_linear: np.ndarray) -> np.ndarray:
        return np.sum(img_linear * self.rgb_to_scotopic.reshape(1, 1, 3), axis=2)

    def _photopic_luminance(self, img_linear: np.ndarray) -> np.ndarray:
        return (
            0.2126729 * img_linear[:, :, 0]
            + 0.7151522 * img_linear[:, :, 1]
            + 0.0721750 * img_linear[:, :, 2]
        )

    def _scotopic_luminance(self, img_linear: np.ndarray) -> np.ndarray:
        return np.sum(img_linear * self.rgb_to_scotopic.reshape(1, 1, 3), axis=2)


class DTUTMOPhotoreceptorPipeline:
    """Helper that combines extraction and photoreceptor responses."""

    def __init__(self, rgb_space: str = "sRGB") -> None:
        self.extractor = PhotoreceptorSignalExtractor(rgb_space=rgb_space)
        self.photoreceptor_model = CorrectedPhotoreceptorResponse()

    def process(
        self,
        img_rgb: np.ndarray,
        adapt_luminance: np.ndarray,
        pupil_size: float,
        rod_adaptation: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        signals = self.extractor.extract_all_signals(img_rgb, is_linear=True)
        cone_responses = self.photoreceptor_model.process_cones(
            signals["lms_combined"], adapt_luminance, pupil_size, n=0.74
        )
        rod_adapt = rod_adaptation if rod_adaptation is not None else signals["scotopic_luminance"]
        rod_response = self.photoreceptor_model.process_rods(
            signals["rods"], rod_adapt, pupil_size, n=0.73
        )
        return {
            "L_cone_response": cone_responses[:, :, 0],
            "M_cone_response": cone_responses[:, :, 1],
            "S_cone_response": cone_responses[:, :, 2],
            "rod_response": rod_response,
            "lms_responses": cone_responses,
            "signals_extracted": signals,
        }
