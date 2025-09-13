from __future__ import annotations

import numpy as np
import xarray as xr

from auroraencoderanalysis._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auroraencoderanalysis._typing import Dataset, ndarray


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

def read_edh(edh_path: str) -> Dataset:
    return xr.open_dataset(
        edh_path,
        storage_options={"client_kwargs":{"trust_env":True}},
        chunks={"time": 1},
        engine="zarr",
    )

