# DTUTMO GPU Acceleration Strategy

This document outlines the approach taken to create a GPU-accelerated
variant of the DTUTMO pipeline while keeping the original NumPy/SciPy
implementation intact.

## Goals

- Preserve the reference CPU implementation for validation and
  comparison.
- Provide a parallel PyTorch implementation that mirrors the original
  stages so that behaviour stays close while enabling execution on CUDA
  or other torch-supported accelerators.
- Encapsulate GPU-specific utilities in a dedicated subpackage so that
  downstream projects can choose between CPU and GPU processing.

## Design Overview

1. **New Subpackage** – All GPU-specific code lives under
   `dtutmo/torch/`. This avoids importing torch when the CPU pipeline is
   used and prevents accidental overrides of the NumPy version.

2. **Shared Configuration** – The PyTorch pipeline reuses
   `DTUTMOConfig`, `ViewingCondition`, `CAMType`, and `DisplayStandard`
   from `dtutmo.core.config` so both variants share the same API.

3. **Module Parity** – Each major component from the CPU implementation
   has a torch counterpart:

   | CPU Module                       | GPU Counterpart                       |
   | -------------------------------- | ------------------------------------- |
   | `dtutmo.utils.color`             | `dtutmo.torch.color`                  |
   | `dtutmo.optics.otf`              | `dtutmo.torch.optics`                 |
   | `dtutmo.optics.glare`            | `dtutmo.torch.optics` (same module)   |
| `dtutmo.adaptation.local_adaptation` | `dtutmo.torch.adaptation`        |
| `dtutmo.preprocessing.bilateral`     | `dtutmo.torch.preprocessing`     |
   | `dtutmo.photoreceptors.response` | `dtutmo.torch.photoreceptors`         |
   | `dtutmo.neural.castle_csf`       | `dtutmo.torch.neural`                 |
   | `dtutmo.appearance.*`            | `dtutmo.torch.appearance`             |

   Not every method is a line-by-line port, but the stage semantics are
   preserved.

4. **Torch-First Computation** – Within the torch modules no NumPy/Scipy
   calls are made. `torch.fft`, `torch.nn.functional`, and tensor
   broadcasting are used instead.

5. **Channel Ordering** – Internally tensors are processed in the common
   PyTorch `NCHW` format (batch, channels, height, width) while the
   public API accepts either `(H, W, 3)` or `(3, H, W)` tensors. Helpers
   normalise the input.

6. **Device Control** – `TorchDTUTMO` accepts an explicit `device`
   argument (defaulting to `torch.device("cuda" if available else
   "cpu")`). Intermediate tensors stay on that device to avoid host ↔
   device copies.

7. **Interoperability** – The `torch` subpackage exports its own
   convenience function `tone_map_hdr_torch` analogous to the CPU
   version. Both pipelines can co-exist in the same process.

## Future Extensions

- A hybrid execution mode where select stages (e.g. OTF + CSF) run on
  GPU while others stay on CPU for environments with limited GPU memory.
- Data-driven calibration: the modular PyTorch implementation makes it
  straightforward to fine-tune or replace stages with neural networks
  trained end-to-end.
- Automatic mixed precision and batching support for video processing.

## Testing

Unit tests for the new modules can mirror the existing ones but operate
on torch tensors. Since this environment lacks an interpreter, tests are
left for follow-up integration.
