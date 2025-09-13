from __future__ import annotations

import pickle

import numpy as np
import xarray as xr

from auroraencoderanalysis._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auroraencoderanalysis._typing import Any, Dataset, ndarray


def get_init_dataset(
        n_embed: int,
        n_levels: int,
        lats: ndarray,
        lons: ndarray,
    ) -> Dataset:
    n_lats = len(lats)
    n_lons = len(lons)
    ds = xr.Dataset(
        {
            "surface_latent": (
                ["time", "embed", "lat", "lon"],
                np.empty((0, n_embed, n_lats, n_lons))
            ),
            "atmos_latent":   (
                ["time", "embed", "level", "lat", "lon"],
                np.empty((0, n_embed, n_levels, n_lats, n_lons))
            ),
        },
        coords={
            "embed": np.arange(n_embed),
            "level": np.arange(n_levels),
            "lat": lats,
            "lon": lons,
        }
    )
    ds = ds.chunk({"time": 1, "embed": n_embed, "lat": n_lats, "lon": n_lons})
    return ds

def get_step_dataset(
        surf_step: ndarray,
        atmos_step: ndarray,
        batch_time: str,
        n_embed: int,
        n_levels: int,
        lats: ndarray,
        lons: ndarray,
    ) -> Dataset:
    step = xr.Dataset(
        {
            "surface_latent": (
                ["time", "embed", "lat", "lon"],
                surf_step
            ),
            "atmos_latent":   (
                ["time", "embed", "level", "lat", "lon"],
                atmos_step
            ),
        },
        coords={
            "time": np.array([np.datetime64(batch_time)]),
            "embed": np.arange(n_embed),
            "level": np.arange(n_levels),
            "lat": lats,
            "lon": lons,
        }
    )
    return step

def pickle_dump(obj: Any, name: str) -> None:
    with open(name, "wb") as f:
        pickle.dump(obj, f)

def pickle_read(name: str) -> None:
    with open(name, "rb") as f:
        return pickle.load(f)

def read_edh(edh_path: str) -> Dataset:
    return xr.open_dataset(
        edh_path,
        storage_options={"client_kwargs":{"trust_env":True}},
        chunks={"time": 1},
        engine="zarr",
    )

def reduce_mask(land_sea_mask: ndarray, patch_size: int) -> ndarray:
    n_lat_patches = land_sea_mask.shape[0] // patch_size
    n_lon_patches = land_sea_mask.shape[1] // patch_size
    land_sea_mask_patched = np.zeros((n_lat_patches, n_lon_patches), dtype=np.int8)
    for i in range(n_lat_patches):
        for j in range(n_lon_patches):
            lat_slice = slice(i * patch_size, (i+1) * patch_size)
            lon_slice = slice(j * patch_size, (j+1) * patch_size)
            patch_data = land_sea_mask[lat_slice, lon_slice]

            mean_val = np.mean(patch_data)
            if mean_val >= 0.5:
                land_sea_mask_patched[i, j] = 1
            else:
                land_sea_mask_patched[i, j] = 0

    return land_sea_mask_patched

