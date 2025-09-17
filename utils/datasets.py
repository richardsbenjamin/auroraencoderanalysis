from __future__ import annotations

import pickle
import shutil

import cdsapi
import numpy as np
import xarray as xr
import zipfile
from pandas import to_datetime

from auroraencoderanalysis._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auroraencoderanalysis._typing import Any, DataArray, Dataset, ndarray


def download_europe_percentiles(output_path: str) -> Dataset:
    dataset = "sis-temperature-statistics"
    request = {
        "variable": "maximum_temperature",
        "period": "year",
        "statistic": [
            "75th_percentile",
            "90th_percentile",
            "95th_percentile",
            "99th_percentile"
        ],
        "experiment": ["rcp4_5"],
        "ensemble_statistic": ["ensemble_members_average"]
    }

    client = cdsapi.Client()
    target = f"{output_path}/temp"
    client.retrieve(dataset, request, target=target)
    
    # Unzip
    with zipfile.ZipFile(target, 'r') as zip_ref:
        zip_ref.extractall(f"{output_path}/temperature_percentiles")

    shutil.rmtree(target)

def get_init_dataset(
        n_embed: int,
        n_levels: int,
        lats: ndarray,
        lons: ndarray,
        time_encoding: dict,
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
    ds["time"].encoding["units"] = time_encoding["units"]
    ds["time"].encoding["dtype"] = time_encoding["dtype"]
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
            "time": np.array([batch_time]),
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

def reduce_field(values: ndarray, patch_size: int) -> ndarray:
    n_lat_patches = values.shape[0] // patch_size
    n_lon_patches = values.shape[1] // patch_size
    values_patched = np.zeros((n_lat_patches, n_lon_patches))
    for i in range(n_lat_patches):
        for j in range(n_lon_patches):
            lat_slice = slice(i * patch_size, (i+1) * patch_size)
            lon_slice = slice(j * patch_size, (j+1) * patch_size)
            patch_data = values[lat_slice, lon_slice]

            values_patched[i, j] = np.mean(patch_data)

    return values_patched

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

def reduce_percentiles(
        percentile_data: DataArray,
        patch_lat_bounds: ndarray,
        patch_lon_bounds: ndarray,
        h: int = 180,
        w: int = 360,
    ) -> np.ndarray:
    patch_level_percentiles = np.full((h, w), np.nan)

    pvalues = percentile_data.values
    p99_lat = percentile_data.lat.values
    p99_lon = percentile_data.lon.values

    for i in range(h):
        for j in range(w):
            lat_min, lat_max = patch_lat_bounds[i, j]
            lon_min, lon_max = patch_lon_bounds[i, j]

            in_lat = np.where((p99_lat >= lat_min) & (p99_lat <= lat_max))[0]
            in_lon = np.where((p99_lon >= lon_min) & (p99_lon <= lon_max))[0]

            if len(in_lat) > 0 and len(in_lon) > 0:
                values_in_patch = pvalues[np.ix_(in_lat, in_lon)]
                patch_level_percentiles[i, j] = np.nanmean(values_in_patch)

    return patch_level_percentiles

