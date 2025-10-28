"""
Display adaptation utilities for the torch pipeline.
"""

from __future__ import annotations

from typing import Dict

import torch

from dtutmo.core.config import DisplayStandard


class TorchDisplayAdaptation:
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.device = device
        self.dtype = dtype
        self.specs = self._init_specs()

    def _init_specs(self) -> Dict[DisplayStandard, Dict[str, torch.Tensor | float]]:
        d65 = torch.tensor([95.047, 100.0, 108.883], device=self.device, dtype=self.dtype)
        return {
            DisplayStandard.REC_709: {
                "name": "Rec. 709",
                "max_luminance": 100.0,
                "white_point": d65,
                "gamma": 2.4,
                "eotf": "gamma",
            },
            DisplayStandard.REC_2020: {
                "name": "Rec. 2020",
                "max_luminance": 100.0,
                "white_point": d65,
                "gamma": 2.4,
                "eotf": "gamma",
            },
            DisplayStandard.DCI_P3: {
                "name": "DCI-P3",
                "max_luminance": 48.0,
                "white_point": d65,
                "gamma": 2.6,
                "eotf": "gamma",
            },
            DisplayStandard.REC_2100_PQ: {
                "name": "Rec. 2100 PQ",
                "max_luminance": 10000.0,
                "white_point": d65,
                "gamma": 2.4,
                "eotf": "pq",
            },
            DisplayStandard.REC_2100_HLG: {
                "name": "Rec. 2100 HLG",
                "max_luminance": 1000.0,
                "white_point": d65,
                "gamma": 1.2,
                "eotf": "hlg",
            },
        }

    def get_spec(self, standard: DisplayStandard) -> Dict[str, torch.Tensor | float]:
        return self.specs[standard]

    def xyz_to_display_rgb(self, xyz: torch.Tensor, standard: DisplayStandard) -> torch.Tensor:
        matrix = torch.tensor(
            [
                [3.2406, -1.5372, -0.4986],
                [-0.9689, 1.8758, 0.0415],
                [0.0557, -0.2040, 1.0570],
            ],
            device=self.device,
            dtype=xyz.dtype,
        )
        swapped = xyz.permute(0, 2, 3, 1)
        rgb = torch.tensordot(swapped, matrix.T, dims=([3], [0])).permute(0, 3, 1, 2)
        return torch.clamp(rgb, min=0.0)

    def adapt_to_display(self, img_xyz: torch.Tensor, standard: DisplayStandard) -> torch.Tensor:
        spec = self.get_spec(standard)
        Y = img_xyz[:, 1:2, :, :]
        scale = spec["max_luminance"] / torch.clamp(Y.max(), min=1e-6)
        img_xyz_scaled = img_xyz * scale
        rgb = self.xyz_to_display_rgb(img_xyz_scaled, standard)
        if spec["eotf"] == "gamma":
            encoded = torch.pow(torch.clamp(rgb, min=0.0), 1.0 / spec["gamma"])
        elif spec["eotf"] == "pq":
            encoded = self._apply_pq(rgb, spec["max_luminance"])
        elif spec["eotf"] == "hlg":
            encoded = self._apply_hlg(rgb)
        else:
            encoded = rgb
        return torch.clamp(encoded, 0.0, 1.0)

    def _apply_pq(self, rgb: torch.Tensor, max_lum: float) -> torch.Tensor:
        L = rgb / 10000.0
        m1 = 2610.0 / 16384.0
        m2 = (2523.0 / 4096.0) * 128.0
        c1 = 3424.0 / 4096.0
        c2 = (2413.0 / 4096.0) * 32.0
        c3 = (2392.0 / 4096.0) * 32.0
        L_m1 = torch.pow(torch.clamp(L, min=0.0), m1)
        numerator = c1 + c2 * L_m1
        denominator = 1.0 + c3 * L_m1
        encoded = torch.pow(numerator / denominator, m2)
        return encoded

    def _apply_hlg(self, rgb: torch.Tensor) -> torch.Tensor:
        encoded = torch.zeros_like(rgb)
        mask_low = rgb <= 1.0 / 12.0
        encoded[mask_low] = torch.sqrt(3.0 * rgb[mask_low])
        a = 0.17883277
        b = 0.28466892
        c = 0.55991073
        encoded[~mask_low] = a * torch.log(12.0 * rgb[~mask_low] - b) + c
        return encoded
