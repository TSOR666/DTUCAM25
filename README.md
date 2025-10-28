# DTUTMO

DTUTMO (DTU Tone Mapping Operator) is a biologically inspired high dynamic range (HDR) tone mapping pipeline. It models optical, retinal, and neural stages of the human visual system to convert HDR imagery into perceptually faithful standard dynamic range (SDR) or HDR display encodings. Please refer to the [wiki](https://github.com/TSOR666/DTUCAM25/wiki) for the technical details. 

## Features

- End-to-end tone mapping via the `CompleteDTUTMO` pipeline
- DTUCAM color appearance model with CIECAM16 and XLR-CAM options
- Separate rod and cone processing with mesopic combination
- Optical blur (OTF) and CIE disability glare modelling
- Neural contrast sensitivity (CastleCSF) filtering
- Multiple display mapping strategies, including a production hybrid
- Optional PyTorch implementation for GPU acceleration

## Installation

DTUTMO targets Python 3.10+. Install runtime dependencies:

```bash
pip install numpy scipy
```

Optional GPU acceleration (recommended for large images) requires PyTorch 2.5.1+:

```bash
pip install "torch>=2.5.1" "torchvision>=0.20.1"
```

Install the project itself (editable mode shown) and include the optional GPU stack with extras:

```bash
pip install -e .[torch]
```

## Usage

```python
import numpy as np
from dtutmo import CompleteDTUTMO

hdr_image = np.random.rand(512, 512, 3) * 4000.0  # Example HDR radiance map
tmo = CompleteDTUTMO()
ldr_image = tmo.process(hdr_image)
```

Configuration is handled through the `DTUTMOConfig` dataclass:

```python
from dtutmo import CAMType, DisplayStandard, DTUTMOConfig

config = DTUTMOConfig(
    use_cam=CAMType.DTUCAM,
    target_display=DisplayStandard.REC_2100_PQ,
)
tmo = CompleteDTUTMO(config)
result = tmo.process(hdr_image, return_intermediate=True)
```

PyTorch acceleration (if installed):

```python
import torch
from dtutmo import TorchDTUTMO

hdr = torch.rand(1, 3, 512, 512, device="cuda") * 4000.0  # BCHW
tmo = TorchDTUTMO()
ldr = tmo.process(hdr)
```

See `examples/` for more detailed usage patterns.

### Display Mapping Options

`DTUTMOConfig.display_mapping` controls the final photoreceptor-to-display transform:

- `DisplayMapping.LEGACY` - original display adaptation module
- `DisplayMapping.WHITEBOARD` - fast inverse Naka-Rushton approximation
- `DisplayMapping.FULL_INVERSE` - analytical inverse of the dual-adaptation model
- `DisplayMapping.HYBRID` - automatic blend of whiteboard and full inverse
- `DisplayMapping.PRODUCTION_HYBRID` - gradient-aware hybrid mapper for production

```python
from dtutmo import DisplayMapping

config = DTUTMOConfig(display_mapping=DisplayMapping.PRODUCTION_HYBRID)
tmo = CompleteDTUTMO(config)
```

## Pipeline Overview

DTUTMO implements the following stages (see `dtutmo/core/pipeline.py`):

1. Optical transfer function (OTF) - ocular blur in frequency domain
2. CIE disability glare - wide-angle veiling glare point spread function
3. Color conversion - linear sRGB <-> XYZ <-> LMS, Von Kries adaptation
4. Local adaptation - multi-scale luminance and TVI threshold (Vangorp et al.)
5. Bilateral separation - base/detail split to preserve textures
6. Photoreceptors - corrected Hood & Finkelstein model for L/M/S cones and rods
7. Mesopic combination - luminance-dependent rod/cone blending
8. Neural CSF - CastleCSF opponent-space filtering in frequency domain
9. Color appearance (optional) - DTUCAM / CIECAM16 / XLR-CAM forward+inverse
10. Display mapping - whiteboard, full inverse, hybrid or production hybrid

Intermediate products (e.g., adaptation maps, cone/rod responses, CSF outputs) can be retrieved with `return_intermediate=True`.

## Key Equations

Below is a compact summary of the core equations implemented in DTUTMO. Symbols use cd/m^2 for luminance unless noted.

```text
Optical Transfer Function (OTF)
  age_factor = 1 + (age - 20)/100
  f_c = 60 / age_factor  # cycles/degree
  OTF(f) = exp(-(f / f_c)^2)

CIE Disability Glare PSF (piecewise; angles in degrees)
  For theta in [0.1, 1):   PSF(theta) proportional to 10*A / theta^3
      theta in [1, 30):    PSF(theta) proportional to 10*A / theta^2
      theta in [30, 100]:  PSF(theta) proportional to 5*A / theta^(1.5)
  A = age_factor; optional wavelength scaling proportional to (550 / lambda)^4
  PSF normalized; optional Purkinje reflections near ~3-3.5 deg

Von Kries Chromatic Adaptation
  LMS_adapt = D * (LMS / LMS_white) + (1 - D) * LMS
  D = F * (1 - (1/3.6) * exp(-(L_A + 42)/92)),  F set by surround

Photoreceptor Response (Corrected Hood-Finkelstein + bleaching)
  I_td = I * pupil_area * lens_factor                 # retinal trolands
  p    = B / (B + ln(I_a_td + epsilon))               # bleaching factor
  sigma_H  = k1 * ((O1 + I_a_td)/O1)^m                # semi-saturation
  sigma    = sigma_H + sigma_neural / p               # effective sigma
  R_max = k2 * ((O2 + p*I_a_td)/O2)^(-1/2)            # response ceiling
  s(I_a) = s_base + s_factor * log10(I_a_td + epsilon) # offset
  S    = p * (ln(I_td + epsilon) - s(I_a))            # modulated signal
  R    = R_max * sign(S) * |S|^n / (sigma^n + |S|^n)  # Naka-Rushton form

Inverse Photoreceptor (per channel)
  r = clip(R/R_max, 0, 0.99)
  x = [ r / (1 - r) ]^(1/n) * sigma
  E = x / p + s(I_a);  I_td = exp(E) - epsilon
  I_scene = I_td / (pupil_area * lens_factor)

Mesopic Combination (local)
  w_rod = interp(log10(L_p), [log10(0.01), log10(10)], [1, 0])
  LMS_mesopic = (1 - w_rod)*LMS_cone + w_rod*R_rod

Bilateral Base/Detail Split
  base = Gaussian(img, sigma_spatial)
  w    = exp(-|img - base|^2 / sigma_range^2)
  filtered = w*base + (1 - w)*img

Neural CSF (CastleCSF; normalized)
  Achromatic:  log-Gaussian around f_p with bandwidth b, scaled by L_A
  Chromatic:   exp(-f / f_p), with luminance-dependent gain

Whiteboard Display Mapping (fast inverse tone curve)
  R' = normalize(R) in [0, 1)
  L_d = (R' * L_mean) / (1 - R')^n  (optional blend with linear)

PQ (ST 2084) and HLG encodings
  PQ: E = [ (c1 + c2*L^m1) / (1 + c3*L^m1) ]^m2
  HLG: E = sqrt(3*L)  for L <= 1/12; else  E = a*ln(12*L - b) + c
```

Where parameters (k1, O1, m, k2, O2, sigma_neural, B, s_base, s_factor, n, epsilon) are per photoreceptor class and are documented in `dtutmo/photoreceptors/response.py` and `dtutmo/photoreceptors/inverse_complete.py`.

## Operators

- Color Appearance (`dtutmo/appearance/`)
  - `DTUCAM` - physiologically grounded opponent-space model with photoreceptor drive
  - `CIECAM16` - simplified forward/inverse of the CIE 2016 model
  - `XLR-CAM` - extended luminance-range CAM
- Optics (`dtutmo/optics/`)
  - `otf.compute_otf` / `otf.apply_otf` - ocular blur in frequency domain
  - `GlareModel` - CIE 180 veiling glare with optional spectral dependence
- Adaptation (`dtutmo/adaptation/`)
  - `LocalAdaptation` - multi-scale adaptation luminance and TVI
  - `mesopic_global` / `mesopic_local` - rod/cone blending
  - `DisplayAdaptation` - XYZ->RGB and EOTF (gamma, PQ, HLG)
- Photoreceptors (`dtutmo/photoreceptors/`)
  - `CorrectedPhotoreceptorResponse` - forward L/M/S and rod responses
  - `InversePhotoreceptorComplete` - exact analytical inverse per channel
- Neural (`dtutmo/neural/`)
  - `CastleCSF` - opponent-space CSF filter
- Display Mapping (`dtutmo/display/`)
  - `DisplayOutputMapper` - `whiteboard` | `full_inverse` | `hybrid`
  - `HybridDisplayMapper` - production-grade, gradient-aware hybrid

### Configuration Cheatsheet

Key enums exposed through `DTUTMOConfig` (see `dtutmo/core/config.py`):

- `ViewingCondition`: `DARK`, `DIM`, `AVERAGE`
- `CAMType`: `NONE`, `DTUCAM`, `XLRCAM`, `CIECAM16`
- `DisplayStandard`: `REC_709`, `REC_2020`, `DCI_P3`, `REC_2100_PQ`, `REC_2100_HLG`
- `DisplayMapping`: `LEGACY`, `WHITEBOARD`, `FULL_INVERSE`, `HYBRID`, `PRODUCTION_HYBRID`

Selected numeric parameters (defaults shown in code):

- Observer `age`, `field_diameter`, `pixels_per_degree`
- Stage toggles: `use_otf`, `use_glare`, `use_bilateral`, `use_local_adapt`, `use_cam`
- Photoreceptor timing: `cone_integration`, `rod_integration`
- Display outputs: `target_display`, `display_mapping`

## Running the Tests

```bash
pytest
```

If your environment blocks network access, pre-install `numpy` and `scipy` wheels locally before running tests.

## Project Structure

- `dtutmo/` - core optics, photoreceptors, adaptation, appearance, display
- `dtutmo/torch/` - PyTorch-accelerated implementations of key stages
- `examples/` - usage snippets illustrating the API
- `tests/` - automated regression and smoke tests for the public API
- `docs/` - supplemental documentation material

## References

- Hood, D. C., and Finkelstein, M. A. (1986). Sensitivity to light. In K. Boff et al. (Eds.), Handbook of Perception and Human Performance.
- CIE 180:2010. Disability glare.
- SMPTE ST 2084 (PQ) and ITU-R BT.2100 (HLG) EOTFs.
- Ashraf et al. (2024). CASTLE: A comprehensive CSF for natural images.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

