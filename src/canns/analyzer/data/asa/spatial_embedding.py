from __future__ import annotations

"""
Spatial point-cloud builders for ASA analysis.

This module is a light organizational layer over the existing spatial-TDA
implementation. The goal is to expose the "position-indexed point cloud"
construction explicitly, without changing the underlying implementation.
"""

from .spatial_tda import (
    build_fr_tensor,
    crop_center,
    ensure_2d,
    fr_tensor_to_point_cloud,
    pca_reduce,
)

__all__ = [
    "ensure_2d",
    "crop_center",
    "build_fr_tensor",
    "fr_tensor_to_point_cloud",
    "pca_reduce",
]
