"""
Shared helpers for the torch-based DTUTMO implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class TensorFormat:
    """Bookkeeping for tensor layout during pipeline execution."""

    original_shape: Tuple[int, ...]
    channel_first: bool


def ensure_tensor(
    data,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert input data to a torch tensor on the requested device.
    """

    if isinstance(data, torch.Tensor):
        tensor = data.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    tensor = torch.as_tensor(data, dtype=dtype, device=device)
    return tensor


def to_nchw(
    img: torch.Tensor,
) -> Tuple[torch.Tensor, TensorFormat]:
    """
    Reshape an image tensor to NCHW format with a batch dimension.
    """

    if img.dim() == 2:
        # Assume grayscale HxW; expand channel dimension.
        img = img.unsqueeze(0)

    if img.dim() == 3:
        if img.shape[0] in (1, 3):
            channel_first = True
            img_cf = img
        else:
            channel_first = False
            img_cf = img.permute(2, 0, 1)
        img_cf = img_cf.unsqueeze(0)
    elif img.dim() == 4:
        channel_first = True
        img_cf = img
    else:
        raise ValueError(f"Unsupported tensor rank {img.dim()} for image input.")

    fmt = TensorFormat(original_shape=tuple(img.shape), channel_first=channel_first)
    return img_cf, fmt


def from_nchw(
    img: torch.Tensor,
    fmt: TensorFormat,
) -> torch.Tensor:
    """
    Convert an NCHW tensor back to the original layout.
    """

    if img.dim() != 4:
        raise ValueError("Expected NCHW tensor with a batch dimension.")

    if img.shape[0] != 1:
        raise ValueError("Batch size > 1 is not supported yet.")

    img = img.squeeze(0)

    if fmt.channel_first:
        return img

    return img.permute(1, 2, 0)
