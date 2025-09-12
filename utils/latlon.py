from __future__ import annotations

import numpy as np

from auroraencoder._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auroraencoder._typing import ndarray

def reduce_lon_lat(
        patch_size: int,
        orig_lat_values: np.ndarray,
        orig_lon_values: np.ndarray,
    ) -> tuple[ndarray]:
    n_lat_patches = len(orig_lat_values) // patch_size
    n_lon_patches = len(orig_lon_values) // patch_size
    patch_center_lat = np.zeros((n_lat_patches, n_lon_patches))
    patch_center_lon = np.zeros((n_lat_patches, n_lon_patches))
    for i in range(n_lat_patches):
        for j in range(n_lon_patches):
            orig_i_start = i * patch_size
            orig_i_end = orig_i_start + patch_size
            orig_j_start = j * patch_size
            orig_j_end = orig_j_start + patch_size

            patch_lats = orig_lat_values[orig_i_start:orig_i_end]
            patch_lons = orig_lon_values[orig_j_start:orig_j_end]

            patch_center_lat[i, j] = np.mean(patch_lats)
            patch_center_lon[i, j] = np.mean(patch_lons)

    return patch_center_lat, patch_center_lon

