"""
Torch-powered DTUTMO pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Union

import torch

from dtutmo.core.config import CAMType, DTUTMOConfig, DisplayMapping, DisplayStandard, ViewingCondition
from dtutmo.display import (
    DisplayAdaptation as CpuDisplayAdaptation,
    DisplayOutputConfig,
    DisplayOutputMapper,
    HybridDisplayConfig,
    HybridDisplayMapper,
)
from dtutmo.torch.adaptation import TorchLocalAdaptation
from dtutmo.torch.appearance import CIECAM16 as TorchCIECAM16
from dtutmo.torch.appearance import TorchDTUCAM
from dtutmo.torch.appearance import XLRCAM as TorchXLRCAM
from dtutmo.torch.common import TensorFormat, ensure_tensor, from_nchw, to_nchw
from dtutmo.torch.color import TorchColorTransform
from dtutmo.torch.display import TorchDisplayAdaptation
from dtutmo.torch.neural import CastleCSF
from dtutmo.torch.optics import GlareModel, GlareParameters, apply_otf, compute_otf
from dtutmo.torch.photoreceptors import TorchPhotoreceptorResponse
from dtutmo.torch.preprocessing import TorchBilateralFilter

logger = logging.getLogger(__name__)


class TorchDTUTMO:
    """
    GPU-accelerated DTUTMO implementation using PyTorch tensors.
    """

    def __init__(
        self,
        config: Optional[DTUTMOConfig] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or DTUTMOConfig()
        self.config.validate()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.dtype = dtype

        logger.info("Initializing TorchDTUTMO (%s)", self.device)
        logger.info("  CAM: %s", self.config.use_cam.value)
        logger.info("  Display: %s", self.config.target_display.value)

        self.color = TorchColorTransform(device=device, dtype=dtype)
        self.display = TorchDisplayAdaptation(device=device, dtype=dtype)

        self.photoreceptor = TorchPhotoreceptorResponse(device=device)
        self.local_adapt = (
            TorchLocalAdaptation(
                peak_sensitivity=self.config.peak_sensitivity,
                ppd=self.config.pixels_per_degree,
            )
            if self.config.use_local_adapt
            else None
        )

        self.csf = CastleCSF(ppd=self.config.pixels_per_degree, device=device, dtype=dtype)

        if self.config.use_glare:
            glare_params = GlareParameters(
                age=self.config.observer_age,
                model=self.config.glare_model,
                include_wavelength_dependence=True,
                include_corneal_reflections=True,
            )
            self.glare_model = GlareModel(glare_params, device=device, dtype=dtype)
        else:
            self.glare_model = None

        if self.config.use_cam == CAMType.DTUCAM:
            self.cam = TorchDTUCAM(
                device=device,
                dtype=dtype,
                observer_age=self.config.observer_age,
                field_diameter=self.config.field_diameter,
                peak_sensitivity=self.config.peak_sensitivity,
                pixels_per_degree=self.config.pixels_per_degree,
            )
        elif self.config.use_cam == CAMType.XLRCAM:
            self.cam = TorchXLRCAM(device=device, dtype=dtype)
        elif self.config.use_cam == CAMType.CIECAM16:
            self.cam = TorchCIECAM16(device=device, dtype=dtype)
        else:
            self.cam = None

        self.adaptation_factor = self._get_adaptation_factor()

        self.cpu_display = CpuDisplayAdaptation()
        display_spec = self.cpu_display.get_display_spec(self.config.target_display)
        self.display_output_mapper: Optional[DisplayOutputMapper]
        self.hybrid_display_mapper: Optional[HybridDisplayMapper]
        self.display_output_mapper = None
        self.hybrid_display_mapper = None

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
        img_hdr,
        display_params: Optional[Dict[str, Union[int, float]]] = None,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        """
        Run the torch pipeline. Accepts tensors or array-like data.
        """

        img_tensor = ensure_tensor(img_hdr, device=self.device, dtype=self.dtype)
        img_nchw, fmt = to_nchw(img_tensor)

        results: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]] = (
            {"input": img_nchw.clone()} if return_intermediate else None
        )

        img_otf = self._stage_otf(img_nchw, display_params, results) if self.config.use_otf else img_nchw
        img_glared = self._stage_glare(img_otf, results) if self.config.use_glare else img_otf
        img_xyz = self._stage_color_convert(img_glared, results)
        adaptation_maps = self._stage_adaptation(img_xyz, results)

        if self.config.use_bilateral:
            img_xyz = self._stage_bilateral(img_xyz, adaptation_maps, results)

        img_chromatic = self._stage_chromatic_adapt(img_xyz, adaptation_maps, results)

        photo_outputs = self._stage_photoreceptor(img_chromatic, adaptation_maps, results)
        mesopic = self._stage_mesopic(photo_outputs, adaptation_maps, results)
        csf_filtered = self._stage_csf(mesopic, adaptation_maps, results)

        if self.cam is not None:
            appearance = self._stage_cam_forward(csf_filtered, adaptation_maps, results)
            display_rgb = self._stage_cam_inverse(appearance, results)
        else:
            display_rgb = self._stage_display_direct(csf_filtered, adaptation_maps, results)

        display_rgb = torch.clamp(display_rgb, 0.0, 1.0)

        if return_intermediate and results is not None:
            results["output"] = display_rgb
            return {key: (value.detach() if isinstance(value, torch.Tensor) else value) for key, value in results.items()}

        output = from_nchw(display_rgb, fmt)
        return output

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage_otf(
        self,
        img: torch.Tensor,
        display_params: Optional[Dict[str, Union[int, float]]],
        results: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    ) -> torch.Tensor:
        logger.debug("Torch Stage 1: OTF")

        if display_params is None:
            display_params = {
                "diagonal_inches": 24.0,
                "resolution": (img.shape[-2], img.shape[-1]),
                "viewing_distance": 0.5,
            }

        luminance = self.color.rgb_to_luminance(img)
        avg_lum = float(luminance.mean().item())

        otf, freq_map = compute_otf(
            avg_lum,
            (img.shape[-2], img.shape[-1]),
            float(display_params["diagonal_inches"]),
            float(display_params["viewing_distance"]),
            self.config.field_diameter,
            self.config.observer_age,
            device=self.device,
            dtype=self.dtype,
        )

        img_filtered = apply_otf(img, otf)

        if results is not None:
            results["otf"] = otf
            results["freq_map"] = freq_map

        return img_filtered

    def _stage_glare(
        self,
        img: torch.Tensor,
        results: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    ) -> torch.Tensor:
        logger.debug("Torch Stage 2: Glare")

        luminance = self.color.rgb_to_luminance(img)
        pupil_map = self.photoreceptor.pupil_diameter_watson(
            luminance,
            age=self.config.observer_age,
            field_deg=self.config.field_diameter,
        )

        if self.glare_model is None:
            return img

        img_glared = self.glare_model.apply_spectral_glare(img, pupil_map)

        if results is not None:
            results["pupil_map"] = pupil_map

        return img_glared

    def _stage_color_convert(
        self,
        img: torch.Tensor,
        results: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    ) -> torch.Tensor:
        logger.debug("Torch Stage 3: Color conversion")
        img_xyz = self.color.srgb_to_xyz(img)

        if results is not None:
            results["xyz"] = img_xyz

        return img_xyz

    def _stage_adaptation(
        self,
        img_xyz: torch.Tensor,
        results: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    ) -> Dict[str, torch.Tensor]:
        logger.debug("Torch Stage 4: Adaptation")

        luminance = img_xyz[:, 1:2, :, :]
        maps: Dict[str, torch.Tensor] = {}

        if self.local_adapt is not None:
            contrast, adapt_lum = self.local_adapt.compute(luminance)
            maps["contrast_threshold"] = contrast
            maps["luminance_cdm2"] = adapt_lum
        else:
            maps["luminance_cdm2"] = luminance

        pupil = self.photoreceptor.pupil_diameter_watson(
            maps["luminance_cdm2"],
            age=self.config.observer_age,
            field_deg=self.config.field_diameter,
        )
        maps["pupil_diameter"] = pupil

        deg_adapt = self.adaptation_factor * (
            1.0 - (1.0 / 3.6) * torch.exp(-(maps["luminance_cdm2"] - 42.0) / 92.0)
        )
        maps["degree_adaptation"] = deg_adapt
        white_xyz = torch.tensor([95.047, 100.0, 108.883], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        maps["white_xyz"] = white_xyz

        if results is not None:
            results["adaptation"] = maps

        return maps

    def _stage_chromatic_adapt(
        self,
        img_xyz: torch.Tensor,
        adaptation_maps: Dict[str, torch.Tensor],
        results: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    ) -> torch.Tensor:
        logger.debug("Torch Stage 6: Chromatic adaptation")

        img_lms = self.color.xyz_to_lms(img_xyz)
        deg_adapt = adaptation_maps["degree_adaptation"]
        white_lms = self.color.xyz_to_lms(adaptation_maps["white_xyz"])
        img_adapted = self.color.chromatic_adapt(img_lms, deg_adapt, white_lms)

        if results is not None:
            results["lms_adapted"] = img_adapted

        return img_adapted

    def _stage_bilateral(
        self,
        img_xyz: torch.Tensor,
        adaptation_maps: Dict[str, torch.Tensor],
        results: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    ) -> torch.Tensor:
        logger.debug("Torch Stage 5: Bilateral separation")

        luminance = img_xyz[:, 1:2, :, :]
        sigma_spatial = max(img_xyz.shape[-1], img_xyz.shape[-2]) * 0.02
        # Improved sigma_range calculation with consistent bounds checking
        ratio = torch.clamp(luminance.max() / torch.clamp(luminance.min(), min=1e-6), min=1.0)
        sigma_range = max(0.4 * torch.log10(ratio).item(), 1e-3)

        bilateral = TorchBilateralFilter(sigma_spatial=sigma_spatial, sigma_range=sigma_range)
        epsilon = torch.tensor(1e-6, device=self.device, dtype=self.dtype)
        log_lum = torch.log10(torch.clamp(luminance, min=epsilon))
        base_log = bilateral.filter(log_lum)
        ten = torch.tensor(10.0, device=self.device, dtype=self.dtype)
        base_lum = torch.exp(base_log * torch.log(ten)) - epsilon
        detail = (luminance + epsilon) / (base_lum + epsilon)

        img_base = img_xyz.clone()
        img_base[:, 1:2, :, :] = base_lum
        # Store detail layer in adaptation_maps only, not as instance variable
        adaptation_maps["detail_layer"] = detail

        if results is not None:
            results["base_layer"] = base_lum
            results["detail_layer"] = detail

        return img_base

    def _stage_photoreceptor(
        self,
        img_lms: torch.Tensor,
        adaptation_maps: Dict[str, torch.Tensor],
        results: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    ) -> Dict[str, torch.Tensor]:
        logger.debug("Torch Stage 7: Photoreceptors")

        outputs: Dict[str, torch.Tensor] = {}
        pupil = adaptation_maps["pupil_diameter"]

        cones = self.photoreceptor.process_cones_original(
            img_lms,
            adaptation_maps["luminance_cdm2"],
            pupil,
        )
        outputs["cones"] = cones
        outputs["cone_channels"] = self.photoreceptor.channel_order

        rod_signal = img_lms.mean(dim=1, keepdim=True)
        rods = self.photoreceptor.process_rods_original(
            rod_signal,
            adaptation_maps["luminance_cdm2"],
            pupil,
        )
        outputs["rods"] = rods

        if results is not None:
            results["photoreceptors"] = outputs

        return outputs

    def _stage_mesopic(
        self,
        photo_outputs: Dict[str, torch.Tensor],
        adaptation_maps: Dict[str, torch.Tensor],
        results: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    ) -> torch.Tensor:
        logger.debug("Torch Stage 8: Mesopic combination")

        cones = photo_outputs["cones"]
        rods = photo_outputs["rods"]

        if self.local_adapt is not None:
            rod_weight = self._local_rod_weight(adaptation_maps["luminance_cdm2"])
        else:
            avg_lum = float(adaptation_maps["luminance_cdm2"].mean().item())
            rod_weight = self._scalar_rod_weight(avg_lum, adaptation_maps["luminance_cdm2"].shape)

        rod_weight_expanded = rod_weight.expand_as(cones)
        rods_expanded = rods.expand_as(cones)
        mesopic = (1.0 - rod_weight_expanded) * cones + rod_weight_expanded * rods_expanded

        if results is not None:
            results["mesopic"] = mesopic

        return mesopic

    def _scalar_rod_weight(self, luminance: float, shape) -> torch.Tensor:
        if luminance < 0.01:
            w = 1.0
        elif luminance > 10.0:
            w = 0.0
        else:
            to_tensor = lambda x: torch.tensor(x, device=self.device, dtype=self.dtype)
            low = torch.log10(to_tensor(0.01))
            high = torch.log10(to_tensor(10.0))
            val = torch.log10(to_tensor(luminance))
            w = (val - low) / (high - low)
            w = float(torch.clamp(w, 0.0, 1.0).item())
        return torch.full(shape, w, device=self.device, dtype=self.dtype)

    def _local_rod_weight(self, luminance_map: torch.Tensor) -> torch.Tensor:
        rod_weight = torch.zeros_like(luminance_map)
        rod_weight = rod_weight + torch.where(luminance_map < 0.01, torch.tensor(1.0, device=self.device, dtype=self.dtype), torch.zeros_like(luminance_map))
        rod_weight = rod_weight + torch.where(luminance_map > 10.0, torch.tensor(0.0, device=self.device, dtype=self.dtype), torch.zeros_like(luminance_map))
        mask_mesopic = (luminance_map >= 0.01) & (luminance_map <= 10.0)
        if mask_mesopic.any():
            log_lum = torch.log10(torch.clamp(luminance_map[mask_mesopic], min=1e-4))
            low = torch.log10(torch.tensor(0.01, device=self.device, dtype=self.dtype))
            high = torch.log10(torch.tensor(10.0, device=self.device, dtype=self.dtype))
            weight = (log_lum - low) / (high - low)
            rod_weight[mask_mesopic] = weight
        return rod_weight

    def _stage_csf(
        self,
        mesopic: torch.Tensor,
        adaptation_maps: Dict[str, torch.Tensor],
        results: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    ) -> torch.Tensor:
        logger.debug("Torch Stage 9: CSF")

        xyz = self.color.lms_to_xyz(mesopic)
        csf_xyz = self.csf.apply_csf(xyz, adaptation_maps["luminance_cdm2"])

        # Apply detail layer from adaptation_maps if available (from bilateral filter)
        if "detail_layer" in adaptation_maps:
            detail = torch.clamp(adaptation_maps["detail_layer"], min=0.01, max=100.0) ** 0.8
            csf_xyz[:, 1:2, :, :] = csf_xyz[:, 1:2, :, :] * detail

        csf_lms = self.color.xyz_to_lms(csf_xyz)

        if results is not None:
            results["csf_filtered"] = csf_xyz
            results["csf_filtered_lms"] = csf_lms

        return csf_lms

    def _stage_cam_forward(
        self,
        csf_lms: torch.Tensor,
        adaptation_maps: Dict[str, torch.Tensor],
        results: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    ):
        logger.debug("Torch Stage 10a: CAM forward (%s)", self.config.use_cam.value)
        if self.cam is None:
            raise RuntimeError("CAM requested but not initialised")

        img_xyz = self.color.lms_to_xyz(csf_lms)
        appearance = self.cam.forward(
            img_xyz,
            adaptation_maps["white_xyz"],
            float(adaptation_maps["luminance_cdm2"].mean().item()) * 0.2,
            surround=self.config.viewing_condition.value,
        )

        if results is not None:
            results["appearance"] = appearance

        return appearance

    def _stage_cam_inverse(
        self,
        appearance,
        results: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    ) -> torch.Tensor:
        logger.debug("Torch Stage 10b: CAM inverse (%s)", self.config.use_cam.value)

        spec = self.display.get_spec(self.config.target_display)
        white_point = spec["white_point"].view(1, 3, 1, 1)

        display_xyz = self.cam.inverse(
            appearance["lightness"],
            appearance["colorfulness"],
            appearance["hue"],
            white_point,
            spec["max_luminance"],
            surround=self.config.viewing_condition.value,
        )

        display_rgb = self.display.adapt_to_display(display_xyz, self.config.target_display)

        if results is not None:
            results["display_xyz"] = display_xyz

        return display_rgb

    def _stage_display_direct(
        self,
        csf_lms: torch.Tensor,
        adaptation_maps: Dict[str, torch.Tensor],
        results: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    ) -> torch.Tensor:
        logger.debug("Torch Stage 10: Direct display adaptation")

        csf_xyz = self.color.lms_to_xyz(csf_lms)

        if (
            self.config.display_mapping == DisplayMapping.LEGACY
            or (
                self.display_output_mapper is None
                and self.hybrid_display_mapper is None
            )
        ):
            rgb = self.display.adapt_to_display(csf_xyz, self.config.target_display)
            if results is not None:
                results["display_xyz"] = csf_xyz
            return rgb

        if csf_lms.shape[0] != 1:
            raise ValueError("Batch size > 1 is not supported for advanced display mapping.")

        csf_lms_np = csf_lms[0].permute(1, 2, 0).detach().cpu().numpy()
        adaptation_np = adaptation_maps["luminance_cdm2"][0, 0].detach().cpu().numpy()

        if self.config.display_mapping == DisplayMapping.PRODUCTION_HYBRID:
            mapped_rgb_np = self.hybrid_display_mapper.process(
                csf_lms_np,
                adaptation_np,
            )
        else:
            method_lookup = {
                DisplayMapping.WHITEBOARD: "whiteboard",
                DisplayMapping.FULL_INVERSE: "full_inverse",
                DisplayMapping.HYBRID: "hybrid",
            }
            method = method_lookup.get(self.config.display_mapping, "hybrid")
            mapped_rgb_np = self.display_output_mapper.map_to_display(
                csf_lms_np,
                adaptation_np,
                method=method,
            )

        mapped_rgb = torch.from_numpy(mapped_rgb_np).to(device=self.device, dtype=self.dtype).permute(2, 0, 1).unsqueeze(0)

        if results is not None:
            results["display_xyz"] = csf_xyz

        return mapped_rgb


def tone_map_hdr_torch(
    img_hdr,
    target_display: DisplayStandard = DisplayStandard.REC_709,
    viewing_condition: ViewingCondition = ViewingCondition.DIM,
    use_cam: CAMType = CAMType.DTUCAM,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convenience wrapper mirroring the CPU helper.
    """

    config = DTUTMOConfig(
        target_display=target_display,
        viewing_condition=viewing_condition,
        use_cam=use_cam,
    )
    tmo = TorchDTUTMO(config=config, device=device, dtype=dtype)
    return tmo.process(img_hdr)
