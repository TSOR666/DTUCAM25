"""
Torch color appearance models mirroring the CPU implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from dtutmo.appearance.dtucam import DTUCAM, DTUCAMParameters


class TorchDTUCAM:
    """Torch-friendly wrapper around the numpy DTUCAM implementation."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        params: DTUCAMParameters | None = None,
        *,
        observer_age: float = 24.0,
        field_diameter: float = 60.0,
        peak_sensitivity: float = 6.0,
        pixels_per_degree: float = 45.0,
    ) -> None:
        self.device = device
        self.dtype = dtype
        kwargs = {
            "observer_age": observer_age,
            "field_diameter": field_diameter,
            "peak_sensitivity": peak_sensitivity,
            "pixels_per_degree": pixels_per_degree,
        }
        self.cpu_cam = DTUCAM(params=params, **kwargs) if params else DTUCAM(**kwargs)

    def forward(
        self,
        img_xyz: torch.Tensor,
        white_xyz: torch.Tensor,
        background_lum: float,
        *,
        surround: str = "average",
    ) -> Dict[str, torch.Tensor]:
        img_np = img_xyz.detach().cpu().permute(0, 2, 3, 1).numpy()
        white_np = white_xyz.detach().cpu().view(-1).numpy()

        appearance_np: List[Dict[str, torch.Tensor]] = []
        for sample in img_np:
            appearance_np.append(
                {
                    key: torch.from_numpy(value).to(self.device, self.dtype)
                    for key, value in self.cpu_cam.forward(
                        sample, white_np, background_lum, surround=surround
                    ).items()
                    if key != "intermediate"
                }
            )

        def _stack(key: str) -> torch.Tensor:
            tensors = [entry[key] for entry in appearance_np]
            stacked = torch.stack(tensors, dim=0)
            return stacked.unsqueeze(1)

        return {
            "lightness": _stack("lightness"),
            "colorfulness": _stack("colorfulness"),
            "hue": _stack("hue"),
            "chroma": _stack("chroma"),
            "brightness": _stack("brightness"),
            "saturation": _stack("saturation"),
        }

    def inverse(
        self,
        lightness: torch.Tensor,
        colorfulness: torch.Tensor,
        hue: torch.Tensor,
        display_white: torch.Tensor,
        display_max_lum: float,
        *,
        surround: str = "dim",
    ) -> torch.Tensor:
        light_np = lightness.detach().cpu().squeeze(1).numpy()
        color_np = colorfulness.detach().cpu().squeeze(1).numpy()
        hue_np = hue.detach().cpu().squeeze(1).numpy()
        white_np = display_white.detach().cpu().view(-1).numpy()

        xyz_list = []
        for J, M, h in zip(light_np, color_np, hue_np):
            xyz = self.cpu_cam.inverse(
                J,
                M,
                h,
                white_np,
                display_max_lum,
                surround=surround,
            )
            xyz_list.append(torch.from_numpy(xyz).to(self.device, self.dtype))

        stacked = torch.stack(xyz_list, dim=0)
        return stacked.permute(0, 3, 1, 2)


@dataclass
class XLRCAMParameters:
    n: float = 0.73
    sigma_base: float = 0.3
    sigma_adapt_factor: float = 0.5
    background_weight: float = 0.2


class XLRCAM:
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32, params: XLRCAMParameters | None = None) -> None:
        self.device = device
        self.dtype = dtype
        self.params = params or XLRCAMParameters()

        self.hpe = torch.tensor(
            [
                [0.38971, 0.68898, -0.07868],
                [-0.22981, 1.18340, 0.04641],
                [0.00000, 0.00000, 1.00000],
            ],
            device=device,
            dtype=dtype,
        )
        self.hpe_inv = torch.linalg.inv(self.hpe)

    def _matmul(self, mat: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        swapped = img.permute(0, 2, 3, 1)
        result = torch.tensordot(swapped, mat.T, dims=([3], [0]))
        return result.permute(0, 3, 1, 2)

    def forward(
        self,
        img_xyz: torch.Tensor,
        white_xyz: torch.Tensor,
        background_lum: float,
        surround: str = "average",
    ) -> Dict[str, torch.Tensor]:
        lms = self._matmul(self.hpe, img_xyz)
        white_lms = self._matmul(self.hpe, white_xyz)
        lms_adapted = lms / white_lms

        avg_lum = float(img_xyz[:, 1, :, :].mean().item())
        sigma = self._compute_sigma(avg_lum, background_lum)

        n = self.params.n
        lms_response = torch.pow(lms_adapted, n) / (torch.pow(lms_adapted, n) + sigma**n)

        achrom = lms_response.mean(dim=1, keepdim=True)
        red_green = lms_response[:, 0:1, :, :] - lms_response[:, 1:2, :, :]
        blue_yellow = lms_response[:, 2:3, :, :] - 0.5 * (lms_response[:, 0:1, :, :] + lms_response[:, 1:2, :, :])

        lightness = self._compute_lightness(achrom, background_lum, surround)
        colorfulness = self._compute_colorfulness(red_green, blue_yellow, achrom)
        hue = torch.atan2(blue_yellow, red_green)
        chroma = torch.sqrt(red_green**2 + blue_yellow**2)

        return {
            "lightness": lightness,
            "colorfulness": colorfulness,
            "hue": hue,
            "chroma": chroma,
        }

    def inverse(
        self,
        lightness: torch.Tensor,
        colorfulness: torch.Tensor,
        hue: torch.Tensor,
        display_white: torch.Tensor,
        display_max_lum: float,
        surround: str = "dim",
    ) -> torch.Tensor:
        achrom = self._inverse_lightness(lightness, display_max_lum, surround)
        red_green = colorfulness / (1.0 + 0.3 * achrom) * torch.cos(hue)
        blue_yellow = colorfulness / (1.0 + 0.3 * achrom) * torch.sin(hue)

        lms_response = torch.zeros_like(lightness).repeat(1, 3, 1, 1)
        lms_response[:, 0:1, :, :] = achrom + red_green / 2.0 + blue_yellow / 3.0
        lms_response[:, 1:2, :, :] = achrom - red_green / 2.0 + blue_yellow / 3.0
        lms_response[:, 2:3, :, :] = achrom + 2.0 * blue_yellow / 3.0

        sigma_display = self._compute_sigma(display_max_lum / 2.0, display_max_lum * 0.2)
        n = self.params.n
        intensity = sigma_display * torch.pow(torch.clamp(lms_response, max=0.99) / torch.clamp(1.0 - lms_response, min=1e-6), 1.0 / n)

        display_white_lms = self._matmul(self.hpe, display_white)
        lms_display = intensity * display_white_lms
        xyz_display = self._matmul(self.hpe_inv, lms_display)
        return xyz_display

    def _compute_sigma(self, adaptation_lum: float, background_lum: float) -> float:
        log_adapt = torch.log10(torch.tensor(max(adaptation_lum, 0.01), device=self.device, dtype=self.dtype))
        bg_factor = 1.0 + self.params.background_weight * torch.log10(torch.tensor(max(background_lum, 0.01), device=self.device, dtype=self.dtype))
        sigma = self.params.sigma_base * (1.0 + self.params.sigma_adapt_factor * log_adapt) * bg_factor
        return float(sigma.item())

    def _compute_lightness(self, achrom: torch.Tensor, background: float, surround: str) -> torch.Tensor:
        surround_factors = {"dark": 0.8, "dim": 0.9, "average": 1.0}
        factor = surround_factors.get(surround, 0.9)
        bg_factor = 1.0 + 0.2 * torch.log10(torch.tensor(max(background, 0.01), device=self.device, dtype=self.dtype))
        return 100.0 * torch.pow(achrom * bg_factor * factor, 0.67)

    def _inverse_lightness(self, lightness: torch.Tensor, max_lum: float, surround: str) -> torch.Tensor:
        surround_factors = {"dark": 0.8, "dim": 0.9, "average": 1.0}
        factor = surround_factors.get(surround, 0.9)
        bg_factor = 1.0 + 0.2 * torch.log10(torch.tensor(max(max_lum * 0.2, 0.01), device=self.device, dtype=self.dtype))
        return torch.pow(lightness / 100.0, 1.0 / 0.67) / (bg_factor * factor)

    def _compute_colorfulness(
        self,
        red_green: torch.Tensor,
        blue_yellow: torch.Tensor,
        achrom: torch.Tensor,
    ) -> torch.Tensor:
        chroma = torch.sqrt(red_green**2 + blue_yellow**2)
        return chroma * (1.0 + 0.3 * achrom)


@dataclass
class CIECAM16ViewingConditions:
    L_A: float = 20.0
    Y_b: float = 20.0
    surround: str = "average"


class CIECAM16:
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.device = device
        self.dtype = dtype
        self.M16 = torch.tensor(
            [
                [0.401288, 0.650173, -0.051461],
                [-0.250268, 1.204414, 0.045854],
                [-0.002079, 0.048952, 0.953127],
            ],
            device=device,
            dtype=dtype,
        )
        self.M16_inv = torch.linalg.inv(self.M16)

    def forward(
        self,
        img_xyz: torch.Tensor,
        white_xyz: torch.Tensor,
        background_lum: float,
        surround: str = "average",
    ) -> Dict[str, torch.Tensor]:
        vc = self._viewing_conditions(float(img_xyz[:, 1, :, :].mean().item()), background_lum, surround)

        rgb = self._matmul(self.M16, img_xyz)
        rgb_w = self._matmul(self.M16, white_xyz)

        D = vc["D"]
        adapt_gain = D * (rgb_w / torch.clamp(rgb_w, min=1e-6)) + (1.0 - D)
        rgb_c = rgb * adapt_gain
        rgb_aw = rgb_w * adapt_gain

        rgb_a = self._post_adaptation(rgb_c, vc["F_L"])
        rgb_aw = self._post_adaptation(rgb_aw, vc["F_L"])

        a = rgb_a[:, 0, :, :] - 12.0 * rgb_a[:, 1, :, :] / 11.0 + rgb_a[:, 2, :, :] / 11.0
        b = (rgb_a[:, 0, :, :] + rgb_a[:, 1, :, :] - 2.0 * rgb_a[:, 2, :, :]) / 9.0

        h = torch.atan2(b, a)
        h = torch.where(h < 0.0, h + 2.0 * torch.pi, h)

        e_t = 0.25 * (torch.cos(h + 2.0) + 3.8)

        A = (2.0 * rgb_a[:, 0, :, :] + rgb_a[:, 1, :, :] + rgb_a[:, 2, :, :] / 20.0 - 0.305) * vc["N_bb"]
        A_w = (2.0 * rgb_aw[:, 0, :, :] + rgb_aw[:, 1, :, :] + rgb_aw[:, 2, :, :] / 20.0 - 0.305) * vc["N_bb"]

        J = 100.0 * torch.pow(A / torch.clamp(A_w, min=1e-6), vc["c"] * vc["z"])
        Q = (4.0 / vc["c"]) * torch.sqrt(J / 100.0) * (A_w + 4.0) * (vc["F_L"] ** 0.25)

        denominator = torch.clamp(rgb_a[:, 0, :, :] + rgb_a[:, 1, :, :] + 21.0 * rgb_a[:, 2, :, :] / 20.0, min=1e-6)
        t = (5e4 / 13.0) * vc["N_c"] * vc["N_cb"] * e_t * torch.sqrt(a**2 + b**2) / denominator
        C = torch.pow(t, 0.9) * torch.sqrt(J / 100.0) * torch.pow(1.64 - 0.29 ** vc["N_cb"], 0.73)
        M = C * (vc["F_L"] ** 0.25)
        s = 50.0 * torch.sqrt((vc["c"] * M) / (Q + 1e-4))

        return {
            "lightness": J,
            "brightness": Q,
            "colorfulness": M,
            "chroma": C,
            "hue": h * 180.0 / torch.pi,
            "saturation": s,
        }

    def inverse(
        self,
        lightness: torch.Tensor,
        colorfulness: torch.Tensor,
        hue: torch.Tensor,
        display_white: torch.Tensor,
        display_max_lum: float,
        surround: str = "dim",
    ) -> torch.Tensor:
        vc = self._viewing_conditions(display_max_lum / 5.0, display_max_lum * 0.2, surround)
        C = colorfulness / (vc["F_L"] ** 0.25)
        h_rad = hue * torch.pi / 180.0
        Y = display_max_lum * torch.pow(lightness / 100.0, 1.0 / (vc["c"] * vc["z"]))
        X = Y * (1.0 + C * torch.cos(h_rad) / 100.0)
        Z = Y * (1.0 + C * torch.sin(h_rad) / 100.0)
        xyz = torch.stack([X, Y, Z], dim=1)
        return xyz

    def _matmul(self, mat: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        swapped = img.permute(0, 2, 3, 1)
        result = torch.tensordot(swapped, mat.T, dims=([3], [0]))
        return result.permute(0, 3, 1, 2)

    def _post_adaptation(self, rgb: torch.Tensor, F_L: float) -> torch.Tensor:
        F_L_rgb = torch.pow(F_L * torch.abs(rgb) / 100.0, 0.42)
        rgb_a = 400.0 * torch.sign(rgb) * F_L_rgb / (27.13 + F_L_rgb) + 0.1
        return rgb_a

    def _viewing_conditions(self, L_A: float, Y_b: float, surround: str) -> Dict[str, torch.Tensor]:
        surround_params = {
            "dark": (0.8, 0.525, 0.8),
            "dim": (0.9, 0.59, 0.9),
            "average": (1.0, 0.69, 1.0),
        }
        F, c, N_c = surround_params[surround]
        k = 1.0 / (5.0 * L_A + 1.0)
        F_L = 0.2 * (k**4) * (5.0 * L_A) + 0.1 * ((1.0 - k**4) ** 2) * ((5.0 * L_A) ** (1.0 / 3.0))
        D = F * (1.0 - (1.0 / 3.6) * torch.exp(torch.tensor((-L_A - 42.0) / 92.0, device=self.device, dtype=self.dtype)))
        N_bb = 0.725 * (1.0 / Y_b) ** 0.2
        N_cb = N_bb
        z = 1.48 + torch.sqrt(torch.tensor(N_c, device=self.device, dtype=self.dtype))
        return {"F": F, "c": c, "N_c": N_c, "F_L": F_L, "D": D, "N_bb": N_bb, "N_cb": N_cb, "z": z}
