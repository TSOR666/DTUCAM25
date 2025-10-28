"""
Color space transforms implemented with torch tensors.
"""

from __future__ import annotations

import torch


class TorchColorTransform:
    """Torch equivalent of the ColorTransform utilities."""

    def __init__(self, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> None:
        self.device = device
        self.dtype = dtype

        srgb_to_xyz = torch.tensor(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            dtype=dtype,
            device=device,
        )
        self.srgb_to_xyz_matrix = srgb_to_xyz
        self.xyz_to_srgb_matrix = torch.linalg.inv(srgb_to_xyz)

        hpe = torch.tensor(
            [
                [0.38971, 0.68898, -0.07868],
                [-0.22981, 1.18340, 0.04641],
                [0.00000, 0.00000, 1.00000],
            ],
            dtype=dtype,
            device=device,
        )
        self.hpe_matrix = hpe
        self.hpe_inv = torch.linalg.inv(hpe)

    def _matmul_channel(self, mat: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Multiply a 3x3 matrix with an image tensor in NCHW format.
        """

        # Reshape: (B, C, H, W) -> (B, H, W, C)
        img_swapped = img.permute(0, 2, 3, 1)
        result = torch.tensordot(img_swapped, mat.T, dims=([3], [0]))
        return result.permute(0, 3, 1, 2)

    def srgb_to_xyz(self, rgb: torch.Tensor) -> torch.Tensor:
        return self._matmul_channel(self.srgb_to_xyz_matrix, rgb)

    def xyz_to_srgb(self, xyz: torch.Tensor) -> torch.Tensor:
        return self._matmul_channel(self.xyz_to_srgb_matrix, xyz)

    def xyz_to_lms(self, xyz: torch.Tensor) -> torch.Tensor:
        return self._matmul_channel(self.hpe_matrix, xyz)

    def lms_to_xyz(self, lms: torch.Tensor) -> torch.Tensor:
        return self._matmul_channel(self.hpe_inv, lms)

    def rgb_to_luminance(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Compute luminance from linear RGB (expects NCHW).
        """

        coeffs = torch.tensor([0.2126729, 0.7151522, 0.0721750], dtype=self.dtype, device=self.device)
        return torch.tensordot(rgb, coeffs, dims=([1], [0])).unsqueeze(1)

    def chromatic_adapt(
        self,
        lms: torch.Tensor,
        degree_adaptation: torch.Tensor,
        white_lms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Von Kries adaptation on torch tensors.

        Parameters
        ----------
        lms : torch.Tensor (N, 3, H, W)
        degree_adaptation : torch.Tensor (N, 1, H, W) or broadcastable scalar
        white_lms : torch.Tensor (1, 3, 1, 1)
        """

        D = degree_adaptation
        adapted = D * (lms / white_lms) + (1.0 - D) * lms
        return adapted
