"""Shared photoreceptor channel definitions."""

from __future__ import annotations

from typing import Dict, Tuple

# Canonical channel ordering used throughout the pipeline. The order mirrors
# the storage layout of RGB images and the LMS photoreceptor responses (R → L,
# G → M, B → S).
PHOTORECEPTOR_CHANNELS: Tuple[str, str, str] = ("R", "G", "B")

# Mapping from rendering channel to the corresponding photoreceptor type. This
# is used by the photoreceptor response models, glare, and mesopic modules to
# ensure that the same cone type is associated with a given channel everywhere
# in the code base.
CHANNEL_TO_CONE: Dict[str, str] = {
    "R": "L_cone",
    "G": "M_cone",
    "B": "S_cone",
}

