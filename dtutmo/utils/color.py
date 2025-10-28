"""
Color space transformations and chromatic adaptation utilities.
"""

from __future__ import annotations

import numpy as np


class ColorTransform:
    """Color space transformation utilities."""

    def __init__(self) -> None:
        """Initialize color transform matrices."""

        # sRGB to XYZ (D65)
        self.srgb_to_xyz_matrix = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )

        # XYZ to sRGB
        self.xyz_to_srgb_matrix = np.linalg.inv(self.srgb_to_xyz_matrix)

        # Hunt-Pointer-Estevez (XYZ to LMS)
        self.hpe_matrix = np.array(
            [
                [0.38971, 0.68898, -0.07868],
                [-0.22981, 1.18340, 0.04641],
                [0.00000, 0.00000, 1.00000],
            ]
        )

        self.hpe_inv = np.linalg.inv(self.hpe_matrix)

    def srgb_to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        """
        Convert linear sRGB to XYZ.

        Parameters
        ----------
        rgb : np.ndarray
            Linear sRGB (cd/m^2), shape (H, W, 3)
        """

        return np.dot(rgb, self.srgb_to_xyz_matrix.T)

    def xyz_to_srgb(self, xyz: np.ndarray) -> np.ndarray:
        """Convert XYZ to linear sRGB."""

        return np.dot(xyz, self.xyz_to_srgb_matrix.T)

    def xyz_to_lms(self, xyz: np.ndarray) -> np.ndarray:
        """Convert XYZ to LMS (cone space)."""

        return np.dot(xyz, self.hpe_matrix.T)

    def lms_to_xyz(self, lms: np.ndarray) -> np.ndarray:
        """Convert LMS to XYZ."""

        return np.dot(lms, self.hpe_inv.T)

    def rgb_to_luminance(self, rgb: np.ndarray) -> np.ndarray:
        """
        Compute luminance from linear RGB.

        Parameters
        ----------
        rgb : np.ndarray
            Linear RGB, shape (H, W, 3)
        """

        return (
            0.2126729 * rgb[:, :, 0]
            + 0.7151522 * rgb[:, :, 1]
            + 0.0721750 * rgb[:, :, 2]
        )

    def chromatic_adapt(
        self,
        lms: np.ndarray,
        degree_adaptation: np.ndarray,
        white_lms: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Von Kries chromatic adaptation.

        Parameters
        ----------
        lms : np.ndarray
            LMS values, shape (H, W, 3)
        degree_adaptation : np.ndarray
            Degree of adaptation (0-1), shape (H, W)
        white_lms : np.ndarray
            White point LMS, shape (1, 1, 3)
        """

        D = degree_adaptation[:, :, np.newaxis]
        # Prevent division by zero in white_lms
        white_lms_safe = np.maximum(white_lms, 1e-10)
        adapted = D * (lms / white_lms_safe) + (1.0 - D) * lms
        return adapted
