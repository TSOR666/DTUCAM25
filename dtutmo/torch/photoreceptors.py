"""Torch implementation of the Hood & Finkelstein photoreceptor model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch

from dtutmo.photoreceptors.constants import CHANNEL_TO_CONE, PHOTORECEPTOR_CHANNELS


@dataclass
class PhotoreceptorParams:
    k1_L: float = 35.0
    k1_M: float = 35.0
    k1_S: float = 25.0
    O1_cone: float = 100.0
    m_cone: float = 0.74

    k1_rod: float = 5.0
    O1_rod: float = 1.0
    m_rod: float = 0.73

    bleach_const_cone: float = float(torch.log(torch.tensor(4.5)))
    bleach_const_rod: float = float(torch.log(torch.tensor(4.4)))

    sigma_neural_cone: float = 1.0
    sigma_neural_rod: float = 1.0

    s_base_cone: float = 1.0
    s_adapt_factor_cone: float = 0.1
    s_base_rod: float = 1.0
    s_adapt_factor_rod: float = 0.15


class TorchPhotoreceptorResponse:
    """Torch translation of the complete Hood photoreceptor response."""

    def __init__(
        self,
        device: torch.device,
        params: PhotoreceptorParams | None = None,
        channel_order: Sequence[str] = PHOTORECEPTOR_CHANNELS,
    ) -> None:
        self.device = device
        self.params = params or PhotoreceptorParams()
        self.epsilon = 0.01
        self.channel_order: Tuple[str, ...] = tuple(channel_order)
        self._validate_channel_order()
        self.cone_types: Tuple[str, ...] = tuple(
            CHANNEL_TO_CONE[channel] for channel in self.channel_order
        )

    def pupil_diameter_watson(
        self,
        adapt_lum: torch.Tensor,
        age: float = 24.0,
        field_deg: float = 60.0,
    ) -> torch.Tensor:
        field_area = ((field_deg / 2.0) ** 2) * torch.pi
        pupil_diameter = torch.full_like(adapt_lum, 4.0, dtype=adapt_lum.dtype, device=adapt_lum.device)

        for _ in range(5):
            pupil_area = torch.pi * (pupil_diameter / 2.0) ** 2
            trolands = adapt_lum * pupil_area
            k = ((trolands * field_area) / 846.0).clamp(min=1e-6) ** 0.41
            D_sd = 7.75 - (5.75 * k) / (k + 2.0)
            pupil_diameter = D_sd + (age - 28.58) * (0.02132 - 0.009562 * D_sd)
            pupil_diameter = pupil_diameter.clamp(min=2.0, max=8.0)

        return pupil_diameter

    def compute_hood_sigma(self, I_adapt_td: torch.Tensor, phototype: str = "L_cone") -> torch.Tensor:
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
        return k1 * torch.pow(ratio, m)

    def compute_bleaching_factor(self, I_adapt_td: torch.Tensor, phototype: str = "L_cone") -> torch.Tensor:
        bleach_const = (
            self.params.bleach_const_cone if "cone" in phototype.lower() else self.params.bleach_const_rod
        )
        bleach_const_tensor = torch.as_tensor(bleach_const, dtype=I_adapt_td.dtype, device=I_adapt_td.device)
        p = bleach_const_tensor / (bleach_const_tensor + torch.log(I_adapt_td + self.epsilon))
        return p.clamp(min=0.1, max=1.0)

    def compute_s_adaptation(self, I_adapt_td: torch.Tensor, phototype: str = "L_cone") -> torch.Tensor:
        if "cone" in phototype.lower():
            s_base = self.params.s_base_cone
            s_factor = self.params.s_adapt_factor_cone
        else:
            s_base = self.params.s_base_rod
            s_factor = self.params.s_adapt_factor_rod

        s_base_tensor = torch.as_tensor(s_base, dtype=I_adapt_td.dtype, device=I_adapt_td.device)
        s_factor_tensor = torch.as_tensor(s_factor, dtype=I_adapt_td.dtype, device=I_adapt_td.device)
        return s_base_tensor + s_factor_tensor * torch.log10(I_adapt_td + self.epsilon)

    def compute_rmax_adaptation(
        self, I_adapt_td: torch.Tensor, p: torch.Tensor, phototype: str = "L_cone"
    ) -> torch.Tensor:
        if "cone" in phototype.lower():
            k2 = 700.0
            O2 = 100.0
        else:
            k2 = 500.0
            O2 = 10.0

        k2_tensor = torch.as_tensor(k2, dtype=I_adapt_td.dtype, device=I_adapt_td.device)
        O2_tensor = torch.as_tensor(O2, dtype=I_adapt_td.dtype, device=I_adapt_td.device)
        effective_adapt = O2_tensor + (p * I_adapt_td)
        ratio = effective_adapt / O2_tensor
        return k2_tensor * torch.pow(ratio, -0.5)

    def photo_response_complete(
        self,
        I_signal: torch.Tensor,
        I_adapt: torch.Tensor,
        pupil_size: torch.Tensor,
        n: float = 1.0,
        phototype: str = "L_cone",
    ) -> torch.Tensor:
        d = pupil_size
        area = (torch.pi * (d**2)) / 4.0
        factor = 1.0 - (d / 9.7) ** 2 + (d / 12.4) ** 4

        I_signal_td = I_signal * factor * area
        log_retinal = torch.log(I_signal_td + self.epsilon)
        I_adapt_td = I_adapt * factor * area

        sigma_hood = self.compute_hood_sigma(I_adapt_td, phototype)
        p = self.compute_bleaching_factor(I_adapt_td, phototype)
        s_adapted = self.compute_s_adaptation(I_adapt_td, phototype)

        sigma_neural = (
            self.params.sigma_neural_cone if "cone" in phototype.lower() else self.params.sigma_neural_rod
        )
        sigma_neural_tensor = torch.as_tensor(sigma_neural, dtype=I_adapt_td.dtype, device=I_adapt_td.device)
        sigma_effective = sigma_hood + (sigma_neural_tensor / p)

        R_max = self.compute_rmax_adaptation(I_adapt_td, p, phototype)

        modulated = torch.pow(p * (log_retinal - s_adapted), n)
        response = modulated / (modulated + torch.pow(sigma_effective, n))
        response = response * R_max

        return torch.exp(response - self.epsilon) / torch.pi

    def process_cones(
        self,
        img_lms: torch.Tensor,
        adapt_luminance: torch.Tensor,
        pupil_size: torch.Tensor,
        n: float = 0.74,
    ) -> torch.Tensor:
        responses = []

        for channel, cone_type in enumerate(self.cone_types):
            responses.append(
                self.photo_response_complete(
                    img_lms[:, channel : channel + 1, :, :], adapt_luminance, pupil_size, n, cone_type
                )
            )

        return torch.cat(responses, dim=1)

    def process_rods(
        self,
        rod_signal: torch.Tensor,
        adapt_luminance: torch.Tensor,
        pupil_size: torch.Tensor,
        n: float = 0.73,
    ) -> torch.Tensor:
        return self.photo_response_complete(rod_signal, adapt_luminance, pupil_size, n, "rod")

    # ------------------------------------------------------------------
    # Backwards compatibility helpers
    # ------------------------------------------------------------------
    def process_cones_original(
        self,
        img_lms: torch.Tensor,
        adapt_luminance: torch.Tensor,
        pupil_size: torch.Tensor,
        n: float = 0.74,
    ) -> torch.Tensor:
        return self.process_cones(img_lms, adapt_luminance, pupil_size, n)

    def process_rods_original(
        self,
        rod_signal: torch.Tensor,
        adapt_luminance: torch.Tensor,
        pupil_size: torch.Tensor,
        n: float = 0.73,
    ) -> torch.Tensor:
        return self.process_rods(rod_signal, adapt_luminance, pupil_size, n)

    def get_adaptation_characteristics(self, phototype: str = "L_cone") -> Dict[str, float]:
        if phototype == "L_cone":
            return {"k1": self.params.k1_L, "O1": self.params.O1_cone, "m": self.params.m_cone, "type": "L-cone"}
        if phototype == "M_cone":
            return {"k1": self.params.k1_M, "O1": self.params.O1_cone, "m": self.params.m_cone, "type": "M-cone"}
        if phototype == "S_cone":
            return {"k1": self.params.k1_S, "O1": self.params.O1_cone, "m": self.params.m_cone, "type": "S-cone"}
        return {"k1": self.params.k1_rod, "O1": self.params.O1_rod, "m": self.params.m_rod, "type": "Rod"}

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
        if channel not in CHANNEL_TO_CONE:
            raise KeyError(f"Unknown channel '{channel}'.")
        return CHANNEL_TO_CONE[channel]
