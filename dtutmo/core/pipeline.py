"""
Main DTUTMO processing pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Union

import numpy as np

from dtutmo.adaptation.display import DisplayAdaptation
from dtutmo.adaptation.local_adaptation import LocalAdaptation
from dtutmo.adaptation.mesopic import mesopic_global, mesopic_local
from dtutmo.appearance.cam_factory import create_cam
from dtutmo.core.config import (
    CAMType,
    DTUTMOConfig,
    DisplayMapping,
    DisplayStandard,
    ViewingCondition,
)
from dtutmo.display import (
    DisplayOutputConfig,
    DisplayOutputMapper,
    HybridDisplayConfig,
    HybridDisplayMapper,
)
from dtutmo.neural.castle_csf import CastleCSF
from dtutmo.optics.glare import GlareModel, GlareParameters
from dtutmo.optics.otf import apply_otf, compute_otf
from dtutmo.photoreceptors.response import CorrectedPhotoreceptorResponse
from dtutmo.preprocessing.bilateral import BilateralFilter
from dtutmo.utils.color import ColorTransform
from dtutmo.utils.photoreceptor_extraction import PhotoreceptorSignalExtractor

logger = logging.getLogger(__name__)


class CompleteDTUTMO:
    """
    Complete biologically inspired HDR tone mapping pipeline.

    Pipeline stages:
        1. Optical transfer function
        2. CIE disability glare
        3. Color conversion (sRGB -> XYZ -> LMS)
        4. Local adaptation
        5. Bilateral separation (optional)
        6. Chromatic adaptation
        7. Photoreceptor responses (corrected model)
        8. Mesopic combination (rods + cones)
        9. Neural CSF filtering (CastleCSF)
        10. Color appearance model (XLR-CAM or CIECAM16)
        11. Display adaptation and encoding
    """

    def __init__(self, config: Optional[DTUTMOConfig] = None) -> None:
        self.config = config or DTUTMOConfig()
        self.config.validate()

        logger.info("Initializing DTUTMO v2.0")
        logger.info("  CAM: %s", self.config.use_cam.value)
        logger.info("  Display: %s", self.config.target_display.value)
        logger.info("  Viewing: %s", self.config.viewing_condition.value)

        self._init_components()

    def _init_components(self) -> None:
        self.color_transform = ColorTransform()

        if self.config.use_glare:
            glare_params = GlareParameters(
                age=self.config.observer_age,
                model=self.config.glare_model,
                include_wavelength_dependence=True,
                include_corneal_reflections=True,
            )
            self.glare_model = GlareModel(glare_params)
        else:
            self.glare_model = None
            logger.warning("Disability glare disabled: perceptual accuracy may suffer.")

        self.bilateral = BilateralFilter() if self.config.use_bilateral else None
        self.local_adapt = (
            LocalAdaptation(
                peak_sensitivity=self.config.peak_sensitivity,
                ppd=self.config.pixels_per_degree,
            )
            if self.config.use_local_adapt
            else None
        )

        self.photoreceptor = CorrectedPhotoreceptorResponse()
        self.photoreceptor_extractor = PhotoreceptorSignalExtractor()
        self.csf = CastleCSF(ppd=self.config.pixels_per_degree)

        self.cam = (
            create_cam(self.config.use_cam, config=self.config)
            if self.config.use_cam != CAMType.NONE
            else None
        )
        if self.cam is None and self.config.use_cam != CAMType.NONE:
            logger.warning("Requested CAM %s could not be created.", self.config.use_cam)

        self.display = DisplayAdaptation()
        self.adaptation_factor = self._get_adaptation_factor()

        self.display_output_mapper: Optional[DisplayOutputMapper]
        self.hybrid_display_mapper: Optional[HybridDisplayMapper]
        self.display_output_mapper = None
        self.hybrid_display_mapper = None

        display_spec = self.display.get_display_spec(self.config.target_display)

        if self.config.display_mapping in (
            DisplayMapping.WHITEBOARD,
            DisplayMapping.FULL_INVERSE,
            DisplayMapping.HYBRID,
        ):
            mapper_config = DisplayOutputConfig(
                target_luminance=display_spec["max_luminance"],
                target_standard=self.config.target_display,
                viewer_adaptation=display_spec["max_luminance"] * self.adaptation_factor,
            )
            self.display_output_mapper = DisplayOutputMapper(mapper_config)
        elif self.config.display_mapping == DisplayMapping.PRODUCTION_HYBRID:
            hybrid_config = HybridDisplayConfig(
                target_luminance=display_spec["max_luminance"],
                target_mean_luminance=display_spec["max_luminance"] * 0.5,
                target_standard=self.config.target_display,
                viewer_adaptation=display_spec["max_luminance"] * self.adaptation_factor,
                viewer_pupil=4.5,
            )
            self.hybrid_display_mapper = HybridDisplayMapper(hybrid_config)

    def _get_adaptation_factor(self) -> float:
        factors = {
            ViewingCondition.DARK: 0.8,
            ViewingCondition.DIM: 0.9,
            ViewingCondition.AVERAGE: 1.0,
        }
        return factors[self.config.viewing_condition]

    def process(
        self,
        img_hdr: np.ndarray,
        display_params: Optional[Dict[str, Union[int, float]]] = None,
        return_intermediate: bool = False,
    ) -> Union[np.ndarray, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
        """
        Run the full tone-mapping pipeline on an HDR image.
        """

        # Input validation
        if img_hdr.ndim != 3 or img_hdr.shape[2] != 3:
            raise ValueError(f"Expected H×W×3 image, got shape {img_hdr.shape}")
        if not np.isfinite(img_hdr).all():
            raise ValueError("Input contains NaN or Inf values")
        if np.any(img_hdr < 0):
            logger.warning("Input contains negative values, clipping to 0")
            img_hdr = np.clip(img_hdr, 0, None)

        logger.info("Processing HDR image: shape=%s", img_hdr.shape)
        logger.info(
            "Input luminance range: [%0.2e, %0.2e] cd/m^2",
            float(np.min(img_hdr)),
            float(np.max(img_hdr)),
        )

        results: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]] = (
            {"input": img_hdr} if return_intermediate else None
        )

        img_otf = self._stage_otf(img_hdr, display_params, results) if self.config.use_otf else img_hdr
        img_glared = self._stage_glare(img_otf, results) if self.config.use_glare else img_otf
        img_xyz = self._stage_color_convert(img_glared, results)
        adaptation_maps = self._stage_adaptation(img_xyz, results)

        if self.config.use_bilateral and self.bilateral is not None:
            img_xyz = self._stage_bilateral(img_xyz, adaptation_maps, results)

        img_lms = self._stage_chromatic_adapt(img_xyz, adaptation_maps, results)
        photo_outputs = self._stage_photoreceptor(img_lms, adaptation_maps, results)
        mesopic = self._stage_mesopic(photo_outputs, adaptation_maps, results)
        csf_filtered = self._stage_csf(mesopic, adaptation_maps, results)

        if self.cam is not None:
            appearance = self._stage_cam_forward(csf_filtered, adaptation_maps, results)
            display_output = self._stage_cam_inverse(appearance, results)
        else:
            display_output = self._stage_display_direct(
                csf_filtered, adaptation_maps, photo_outputs, results
            )

        display_output = np.clip(display_output, 0.0, 1.0)

        logger.info(
            "Processing complete. Output range: [%0.3f, %0.3f]",
            float(np.min(display_output)),
            float(np.max(display_output)),
        )

        if return_intermediate and results is not None:
            results["output"] = display_output
            return results

        return display_output

    # ------------------------------------------------------------------
    # Individual pipeline stages
    # ------------------------------------------------------------------

    def _stage_otf(
        self,
        img: np.ndarray,
        display_params: Optional[Dict[str, Union[int, float]]],
        results: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    ) -> np.ndarray:
        logger.debug("Stage 1: optical transfer function")

        if display_params is None:
            display_params = {
                "diagonal_inches": 24.0,
                "resolution": img.shape[:2][::-1],
                "viewing_distance": 0.5,
            }

        luminance = np.clip(self.color_transform.rgb_to_luminance(img), 0.0, None)
        avg_lum = float(np.mean(luminance))

        otf, freq_map = compute_otf(
            avg_lum,
            img.shape[:2],
            float(display_params["diagonal_inches"]),
            float(display_params["viewing_distance"]),
            self.config.field_diameter,
            self.config.observer_age,
        )

        img_filtered = apply_otf(img, otf)
        img_filtered = np.clip(img_filtered, 0.0, None)

        if results is not None:
            results["otf"] = otf
            results["freq_map"] = freq_map

        return img_filtered

    def _stage_glare(
        self,
        img: np.ndarray,
        results: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    ) -> np.ndarray:
        logger.debug("Stage 2: CIE disability glare")

        luminance = self.color_transform.rgb_to_luminance(img)
        pupil_map = self.photoreceptor.pupil_diameter_watson(
            luminance,
            self.config.observer_age,
            self.config.field_diameter,
        )

        if self.glare_model is None:
            return img

        img_glared = self.glare_model.apply_spectral_glare(img, pupil_map)
        img_glared = np.clip(img_glared, 0.0, None)

        if results is not None:
            results["pupil_map"] = pupil_map
            results["glare_psf"] = self.glare_model.get_effective_psf(
                float(np.mean(pupil_map)),
                img.shape[:2],
            )

        return img_glared

    def _stage_color_convert(
        self,
        img: np.ndarray,
        results: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    ) -> np.ndarray:
        logger.debug("Stage 3: color conversion sRGB -> XYZ")

        img_xyz = self.color_transform.srgb_to_xyz(img)

        if results is not None:
            results["xyz"] = img_xyz

        return img_xyz

    def _stage_adaptation(
        self,
        img_xyz: np.ndarray,
        results: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    ) -> Dict[str, np.ndarray]:
        logger.debug("Stage 4: local adaptation")

        luminance = img_xyz[:, :, 1]
        adaptation_maps: Dict[str, np.ndarray] = {}

        if self.local_adapt is not None:
            contrast_threshold, adapt_lum = self.local_adapt.compute(luminance)
            adapt_lum = np.clip(adapt_lum, 0.0, None)
            adaptation_maps["luminance_cdm2"] = adapt_lum
            adaptation_maps["contrast_threshold"] = contrast_threshold
        else:
            from scipy.ndimage import median_filter

            adapt_lum = median_filter(luminance, size=21)
            adapt_lum = np.clip(adapt_lum, 0.0, None)
            adaptation_maps["luminance_cdm2"] = adapt_lum

        pupil_map = self.photoreceptor.pupil_diameter_watson(
            adaptation_maps["luminance_cdm2"],
            self.config.observer_age,
            self.config.field_diameter,
        )
        adaptation_maps["pupil_diameter"] = pupil_map

        deg_adapt = self.adaptation_factor * (1.0 - (1.0 / 3.6) * np.exp(-(adapt_lum - 42.0) / 92.0))
        adaptation_maps["degree_adaptation"] = deg_adapt
        adaptation_maps["white_xyz"] = np.array([95.047, 100.0, 108.883])

        if results is not None:
            results["adaptation"] = adaptation_maps

        return adaptation_maps

    def _stage_bilateral(
        self,
        img_xyz: np.ndarray,
        adaptation_maps: Dict[str, np.ndarray],
        results: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    ) -> np.ndarray:
        logger.debug("Stage 5: bilateral separation")

        if self.bilateral is None:
            return img_xyz

        luminance = img_xyz[:, :, 1]
        sigma_spatial = max(luminance.shape) * 0.02
        # Improved sigma_range calculation with bounds checking
        ratio = np.maximum(np.max(luminance) / np.maximum(np.min(luminance), 1e-6), 1.0)
        sigma_range = max(0.4 * np.log10(ratio), 1e-3)

        self.bilateral.sigma_spatial = sigma_spatial
        self.bilateral.sigma_range = sigma_range

        epsilon = 1e-6
        log_lum = np.log10(luminance + epsilon)
        base_log = self.bilateral.filter(log_lum)
        base_lum = np.power(10.0, base_log) - epsilon
        detail_ratio = (luminance + epsilon) / (base_lum + epsilon)

        img_base = img_xyz.copy()
        img_base[:, :, 1] = base_lum

        # Store detail layer in adaptation_maps instead of instance variable
        adaptation_maps["detail_layer"] = detail_ratio

        if results is not None:
            results["base_layer"] = base_lum
            results["detail_layer"] = detail_ratio

        return img_base

    def _stage_chromatic_adapt(
        self,
        img_xyz: np.ndarray,
        adaptation_maps: Dict[str, np.ndarray],
        results: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    ) -> np.ndarray:
        logger.debug("Stage 6: chromatic adaptation")

        img_lms = self.color_transform.xyz_to_lms(img_xyz)
        white_xyz = adaptation_maps["white_xyz"]
        white_lms = self.color_transform.xyz_to_lms(white_xyz.reshape(1, 1, 3))
        deg_adapt = adaptation_maps["degree_adaptation"]
        img_adapted = self.color_transform.chromatic_adapt(img_lms, deg_adapt, white_lms)

        if results is not None:
            results["lms_adapted"] = img_adapted

        return img_adapted

    def _stage_photoreceptor(
        self,
        img_lms: np.ndarray,
        adaptation_maps: Dict[str, np.ndarray],
        results: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    ) -> Dict[str, np.ndarray]:
        logger.debug("Stage 7: photoreceptor responses")

        outputs: Dict[str, np.ndarray] = {}
        avg_pupil = float(np.mean(adaptation_maps["pupil_diameter"]))

        img_xyz = self.color_transform.lms_to_xyz(img_lms)
        img_rgb = self.color_transform.xyz_to_srgb(img_xyz)
        signals = self.photoreceptor_extractor.extract_all_signals(
            img_rgb, is_linear=True
        )

        cone_responses = self.photoreceptor.process_cones(
            signals["lms_combined"],
            adaptation_maps["luminance_cdm2"],
            avg_pupil,
            n=0.74,
        )
        outputs["cones"] = cone_responses
        outputs["cone_channels"] = self.photoreceptor.channel_order

        rod_response = self.photoreceptor.process_rods(
            signals["rods"],
            signals["scotopic_luminance"],
            avg_pupil,
            n=0.73,
        )
        outputs["rods"] = rod_response
        outputs["signals"] = signals

        if results is not None:
            results["photoreceptors"] = outputs

        return outputs

    def _stage_mesopic(
        self,
        photo_outputs: Dict[str, np.ndarray],
        adaptation_maps: Dict[str, np.ndarray],
        results: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    ) -> np.ndarray:
        logger.debug("Stage 8: mesopic combination")

        cones = photo_outputs["cones"]
        rods = photo_outputs["rods"]

        photopic_map = photo_outputs.get("signals", {}).get(
            "photopic_luminance", adaptation_maps["luminance_cdm2"]
        )
        scotopic_map = photo_outputs.get("signals", {}).get(
            "scotopic_luminance", rods * 0.2
        )

        photo_lum = float(np.mean(photopic_map))
        scoto_lum = float(np.mean(scotopic_map))

        if self.local_adapt is not None:
            mesopic = mesopic_local(
                cones,
                rods,
                photopic_map,
                scotopic_map,
            )
        else:
            mesopic = mesopic_global(cones, rods, photo_lum, scoto_lum)

        if results is not None:
            results["mesopic"] = mesopic

        return mesopic

    def _stage_csf(
        self,
        mesopic: np.ndarray,
        adaptation_maps: Dict[str, np.ndarray],
        results: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    ) -> np.ndarray:
        logger.debug("Stage 9: neural CSF")

        mesopic_xyz = self.color_transform.lms_to_xyz(mesopic)
        csf_xyz = self.csf.apply_csf(mesopic_xyz, adaptation_maps["luminance_cdm2"])

        # Apply detail layer from adaptation_maps if available (from bilateral filter)
        if "detail_layer" in adaptation_maps:
            detail = np.power(np.clip(adaptation_maps["detail_layer"], 0.01, 100.0), 0.8)
            csf_xyz[:, :, 1] *= detail

        csf_lms = self.color_transform.xyz_to_lms(csf_xyz)

        if results is not None:
            results["csf_filtered"] = csf_xyz
            results["csf_filtered_lms"] = csf_lms

        return csf_lms

    def _stage_cam_forward(
        self,
        csf_lms: np.ndarray,
        adaptation_maps: Dict[str, np.ndarray],
        results: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    ) -> Dict[str, np.ndarray]:
        logger.debug("Stage 10a: CAM forward (%s)", self.config.use_cam.value)

        if self.cam is None:
            raise RuntimeError("CAM forward stage requested but no CAM is configured.")

        img_xyz_ready = self.color_transform.lms_to_xyz(csf_lms)

        appearance = self.cam.forward(
            img_xyz_ready,
            adaptation_maps["white_xyz"],
            float(np.mean(adaptation_maps["luminance_cdm2"])) * 0.2,
            surround=self.config.viewing_condition.value,
        )

        if results is not None:
            results["appearance"] = appearance

        return appearance

    def _stage_cam_inverse(
        self,
        appearance: Dict[str, np.ndarray],
        results: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    ) -> np.ndarray:
        logger.debug("Stage 10b: CAM inverse (%s)", self.config.use_cam.value)

        display_spec = self.display.get_display_spec(self.config.target_display)

        display_xyz = self.cam.inverse(
            appearance["lightness"],
            appearance["colorfulness"],
            appearance["hue"],
            display_spec["white_point"],
            display_spec["max_luminance"],
            surround=self.config.viewing_condition.value,
        )

        display_rgb = self.display.adapt_to_display(display_xyz, self.config.target_display)

        if results is not None:
            results["display_xyz"] = display_xyz

        return display_rgb

    def _stage_display_direct(
        self,
        csf_lms: np.ndarray,
        adaptation_maps: Dict[str, np.ndarray],
        photo_outputs: Dict[str, np.ndarray],
        results: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    ) -> np.ndarray:
        logger.debug("Stage 10: direct display adaptation (no CAM)")

        del photo_outputs  # retained for potential future refinements

        if (
            self.config.display_mapping == DisplayMapping.LEGACY
            or (
                self.display_output_mapper is None
                and self.hybrid_display_mapper is None
            )
        ):
            display_xyz = self.color_transform.lms_to_xyz(csf_lms)
            display_rgb = self.display.adapt_to_display(display_xyz, self.config.target_display)
            if results is not None:
                results["display_xyz"] = display_xyz
            return display_rgb

        if self.config.display_mapping == DisplayMapping.PRODUCTION_HYBRID:
            display_rgb = self.hybrid_display_mapper.process(
                csf_lms,
                adaptation_maps["luminance_cdm2"],
            )
            if results is not None:
                results["display_xyz"] = self.color_transform.lms_to_xyz(csf_lms)
            return display_rgb

        method_lookup = {
            DisplayMapping.WHITEBOARD: "whiteboard",
            DisplayMapping.FULL_INVERSE: "full_inverse",
            DisplayMapping.HYBRID: "hybrid",
        }
        method = method_lookup.get(self.config.display_mapping, "hybrid")
        display_rgb = self.display_output_mapper.map_to_display(
            csf_lms,
            adaptation_maps["luminance_cdm2"],
            method=method,
        )
        if results is not None:
            results["display_xyz"] = self.color_transform.lms_to_xyz(csf_lms)
        return display_rgb


def tone_map_hdr(
    img_hdr: np.ndarray,
    target_display: DisplayStandard = DisplayStandard.REC_709,
    viewing_condition: ViewingCondition = ViewingCondition.DIM,
    use_cam: CAMType = CAMType.DTUCAM,
) -> np.ndarray:
    """
    Convenience wrapper for quick tone mapping.
    """

    config = DTUTMOConfig(
        target_display=target_display,
        viewing_condition=viewing_condition,
        use_cam=use_cam,
    )

    tmo = CompleteDTUTMO(config)
    return tmo.process(img_hdr)
