"""Hybrid display mapping that blends analytical inverse and whiteboard models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

from ..adaptation.display import DisplayAdaptation
from ..core.config import DisplayStandard
from ..photoreceptors.inverse_complete import InversePhotoreceptorComplete
from ..utils.color import ColorTransform


@dataclass
class HybridDisplayConfig:
    """Configuration for :class:`HybridDisplayMapper`."""

    target_luminance: float = 100.0
    target_mean_luminance: float = 50.0
    target_standard: DisplayStandard = DisplayStandard.REC_709
    hill_coefficient: float = 0.74
    compression_alpha: float = 1.0
    gradient_threshold: float = 0.15
    gradient_smoothing: float = 2.0
    blend_transition: float = 0.3
    viewer_adaptation: float = 20.0
    viewer_pupil: float = 4.5
    viewer_age: float = 24.0
    clip_response_max: float = 0.99
    add_black_level: bool = True
    black_level: float = 0.2
    use_fast_gradient: bool = True
    downsample_gradient: int = 1


class HybridDisplayMapper:
    """Production-ready hybrid display mapper."""

    def __init__(self, config: Optional[HybridDisplayConfig] = None) -> None:
        self.config = config or HybridDisplayConfig()
        self.inverse_model = InversePhotoreceptorComplete()
        self.display_adapter = DisplayAdaptation()
        self.color_transform = ColorTransform()
        self.display_spec = self.display_adapter.get_display_spec(
            self.config.target_standard
        )
        self._precompute_constants()

    def _precompute_constants(self) -> None:
        self.L_mean_d = self.config.target_mean_luminance
        self.n = self.config.hill_coefficient
        self.alpha = self.config.compression_alpha
        self.knee_start = 0.90
        self.knee_range = max(self.config.clip_response_max - self.knee_start, 1e-4)

    def process(
        self,
        photoreceptor_responses: np.ndarray,
        scene_adaptation: np.ndarray,
        return_debug: bool = False,
    ) -> np.ndarray | Dict[str, np.ndarray]:
        """Convert photoreceptor responses to display RGB."""

        debug_info: Dict[str, np.ndarray]
        if return_debug:
            debug_info = {}
        else:
            debug_info = None  # type: ignore

        R_prime = self._normalize_responses(photoreceptor_responses)
        if debug_info is not None:
            debug_info["R_prime"] = R_prime

        gradient_map = self._compute_gradient_map(R_prime)
        if debug_info is not None:
            debug_info["gradient_map"] = gradient_map

        blend_weights = self._compute_blend_weights(gradient_map)
        if debug_info is not None:
            debug_info["blend_weights"] = blend_weights
            debug_info["fast_region_percent"] = np.array(
                [100.0 * np.sum(blend_weights < 0.5) / blend_weights.size]
            )

        L_d_whiteboard = self._whiteboard_mapping(R_prime)
        if debug_info is not None:
            debug_info["L_d_whiteboard"] = L_d_whiteboard

        needs_full_inverse = blend_weights > 0.01
        if np.any(needs_full_inverse):
            L_d_inverse = self._full_inverse_mapping(
                photoreceptor_responses, scene_adaptation, needs_full_inverse
            )
        else:
            L_d_inverse = L_d_whiteboard
        if debug_info is not None:
            debug_info["L_d_inverse"] = L_d_inverse

        L_d_blended = self._adaptive_blend(L_d_whiteboard, L_d_inverse, blend_weights)
        if debug_info is not None:
            debug_info["L_d_blended"] = L_d_blended

        L_d_final = self._handle_edge_cases(L_d_blended)
        if debug_info is not None:
            debug_info["L_d_final"] = L_d_final

        display_xyz = self._luminance_to_xyz(L_d_final)
        display_rgb = self.display_adapter.adapt_to_display(
            display_xyz, self.config.target_standard
        )

        if debug_info is not None:
            debug_info["display_xyz"] = display_xyz
            debug_info["display_rgb"] = display_rgb
            return debug_info

        return display_rgb

    def _normalize_responses(self, responses: np.ndarray) -> np.ndarray:
        normalized = np.zeros_like(responses)
        for idx in range(responses.shape[2]):
            channel = responses[:, :, idx]
            max_val = float(np.percentile(channel, 99.5))
            if max_val > 0:
                normalized[:, :, idx] = channel / max_val
            else:
                normalized[:, :, idx] = channel
        return np.clip(normalized, 0.0, self.config.clip_response_max)

    def _compute_gradient_map(self, responses: np.ndarray) -> np.ndarray:
        luminance = self.color_transform.rgb_to_luminance(responses)
        if self.config.downsample_gradient > 1:
            step = self.config.downsample_gradient
            luminance_ds = luminance[::step, ::step]
        else:
            luminance_ds = luminance

        if self.config.use_fast_gradient:
            grad_x = sobel(luminance_ds, axis=1)
            grad_y = sobel(luminance_ds, axis=0)
        else:
            grad_y, grad_x = np.gradient(luminance_ds)

        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        if self.config.downsample_gradient > 1:
            from scipy.ndimage import zoom

            zoom_factor = self.config.downsample_gradient
            gradient_mag = zoom(gradient_mag, zoom_factor, order=1)
            gradient_mag = gradient_mag[
                : responses.shape[0], : responses.shape[1]
            ]

        if self.config.gradient_smoothing > 0:
            gradient_mag = gaussian_filter(
                gradient_mag, sigma=self.config.gradient_smoothing
            )

        max_val = float(np.max(gradient_mag))
        if max_val > 0:
            gradient_mag = gradient_mag / max_val
        return gradient_mag

    def _compute_blend_weights(self, gradient_map: np.ndarray) -> np.ndarray:
        threshold = self.config.gradient_threshold
        transition = self.config.blend_transition
        low = threshold * (1.0 - transition)
        high = threshold * (1.0 + transition)
        if high <= low:
            return (gradient_map >= threshold).astype(float)

        x = np.clip(gradient_map, low, high)
        t = (x - low) / (high - low)
        weights = t * t * (3.0 - 2.0 * t)
        return np.clip(weights, 0.0, 1.0)

    def _whiteboard_mapping(self, responses: np.ndarray) -> np.ndarray:
        responses_knee = self._apply_soft_knee(responses)
        numerator = responses_knee * self.L_mean_d
        denominator = np.power(1.0 - responses_knee, self.n)
        denominator = np.maximum(denominator, 1e-6)
        luminance = numerator / denominator
        if self.alpha < 1.0:
            luminance_linear = responses_knee * self.L_mean_d
            luminance = self.alpha * luminance + (1.0 - self.alpha) * luminance_linear
        return np.clip(luminance, 0.0, self.config.target_luminance)

    def _apply_soft_knee(self, responses: np.ndarray) -> np.ndarray:
        adjusted = responses.copy()
        mask = responses > self.knee_start
        if np.any(mask):
            r = responses[mask]
            r_norm = (r - self.knee_start) / self.knee_range
            compression = 1.0 - np.exp(-5.0 * r_norm)
            adjusted[mask] = self.knee_start + compression * self.knee_range
        return adjusted

    def _full_inverse_mapping(
        self,
        responses: np.ndarray,
        scene_adaptation: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if mask is None:
            display_adapt = np.full(
                scene_adaptation.shape,
                self.config.viewer_adaptation,
                dtype=scene_adaptation.dtype,
            )
            luminance = self.inverse_model.inverse_cones(
                responses,
                display_adapt,
                self.config.viewer_pupil,
                n=self.n,
            )
        else:
            luminance = np.zeros_like(responses)
            if mask.ndim == 2:
                mask_expanded = mask[:, :, np.newaxis]
            else:
                mask_expanded = mask

            if np.any(mask_expanded):
                display_adapt = np.full(
                    scene_adaptation.shape,
                    self.config.viewer_adaptation,
                    dtype=scene_adaptation.dtype,
                )
                inv = self.inverse_model.inverse_cones(
                    responses,
                    display_adapt,
                    self.config.viewer_pupil,
                    n=self.n,
                )
                mask_broadcast = np.broadcast_to(mask_expanded, inv.shape)
                luminance = np.where(mask_broadcast, inv, luminance)
        max_val = float(np.percentile(luminance, 99.5))
        if max_val > 0:
            luminance = luminance * (self.config.target_luminance / max_val)
        return np.clip(luminance, 0.0, self.config.target_luminance)

    @staticmethod
    def _adaptive_blend(
        fast_luminance: np.ndarray,
        accurate_luminance: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        weights_3d = weights[:, :, np.newaxis]
        return (1.0 - weights_3d) * fast_luminance + weights_3d * accurate_luminance

    def _handle_edge_cases(self, luminance: np.ndarray) -> np.ndarray:
        luminance = np.nan_to_num(
            luminance,
            nan=0.0,
            posinf=self.config.target_luminance,
            neginf=0.0,
        )
        if self.config.add_black_level:
            luminance = luminance + self.config.black_level
        return np.clip(luminance, 0.0, self.config.target_luminance)

    def _luminance_to_xyz(self, luminance: np.ndarray) -> np.ndarray:
        if luminance.ndim == 2:
            Y = luminance
            X = Y * (95.047 / 100.0)
            Z = Y * (108.883 / 100.0)
            return np.stack([X, Y, Z], axis=2)

        Y = luminance[:, :, 1]
        X = luminance[:, :, 0] * (95.047 / 100.0)
        Z = luminance[:, :, 2] * (108.883 / 100.0)
        return np.stack([X, Y, Z], axis=2)
