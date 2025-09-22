from __future__ import annotations

import numpy as np
import xarray as xr

from auroraencoderanalysis._typing import TYPE_CHECKING
from auroraencoderanalysis.utils.constants import LEVELS, VARS
from auroraencoderanalysis.utils.datasets import get_step_dataset, get_init_dataset, read_edh
from auroraencoderanalysis.utils.latlon import reduce_lon_lat
from auroraencoderanalysis.utils.models import get_aurora_batch, get_aurora_model, run_encoder
from auroraencoderanalysis.utils.parsers import get_calc_atmos_instability_mask_parser

if TYPE_CHECKING:
    from auroraencoderanalysis._typing import Namespace


K_INDEX_LEVELS = [500, 700, 850]
K_INDEX_VARS = ["t", "r"]


def get_dewpoint(ds: xr.Dataset) -> xr.Dataset:
    # Based on https://en.wikipedia.org/wiki/Dew_point#Calculating_the_dew_point
    b = 17.625
    c = 243.04

    T = ds['t'] - 273.15 # Convert °C
    RH = ds['r']

    RH = xr.where(RH == 0, 0.001, RH)
    RH = xr.where(RH < 0, 0.001, RH)
    RH = xr.where(RH > 100, 100, RH)

    alpha = np.log(RH / 100) + (b * T) / (c + T)
    Td = (c * alpha) / (b - alpha)

    Td = xr.where(Td > T, T, Td)
    reasonable_min = T - 50
    Td = xr.where(Td < reasonable_min, reasonable_min, Td)

    Td = xr.DataArray(Td, dims=ds['t'].dims, coords=ds['t'].coords, name='dewpoint')
    return Td.compute()

def get_k_index(ds: xr.Dataset) -> xr.Dataset:
    # Based on https://en.wikipedia.org/wiki/K-index_%28meteorology%29
    T = ds['t'] - 273.15 # Convert to °C

    T850 = T.sel(isobaricInhPa=850)
    T700 = T.sel(isobaricInhPa=700)
    T500 = T.sel(isobaricInhPa=500)

    Td850 = ds['dewpoint'].sel(isobaricInhPa=850)
    Td700 = ds['dewpoint'].sel(isobaricInhPa=700)

    K = (T850 - T500) + Td850 - (T700 - Td700)
    K.name = 'K_index'
    return K.compute()

def calc_atmos_instability_mask(arg_parser: Namespace) -> None:
    edh_levels = read_edh(arg_parser.levels_path)

    edh_atmos_data = (
        edh_levels[K_INDEX_VARS]
        .sel(valid_time=slice(arg_parser.start_date, arg_parser.end_date))
        .sel(isobaricInhPa=K_INDEX_LEVELS)
        .isel(valid_time=slice(None, None, 6))
    )

    edh_atmos_data["dewpoint"] = get_dewpoint(edh_atmos_data)
    edh_atmos_data["k_index"] = get_k_index(edh_atmos_data)
    edh_atmos_data["k_gt_20"] = (edh_atmos_data["k_index"] > 20).astype(np.int8)
    edh_atmos_data["k_gt_35"] = (edh_atmos_data["k_index"] > 35).astype(np.int8)

    for var in ["k_gt_20", "k_gt_35"]:
        if var in edh_atmos_data:
            edh_atmos_data[var].encoding.clear()

    k_gt_20_clean = edh_atmos_data["k_gt_20"]
    k_gt_35_clean = edh_atmos_data["k_gt_35"]

    new_ds = xr.Dataset({
        "k_gt_20": k_gt_20_clean,
        "k_gt_35": k_gt_35_clean
    })

    for coord_name in new_ds.coords:
        if 'compressors' in new_ds[coord_name].encoding:
            new_ds[coord_name].encoding.pop('compressors', None)

    new_ds.to_zarr(arg_parser.output_zarr, mode="w")


if __name__ == "__main__":
    run_args = get_calc_atmos_instability_mask_parser()
    calc_atmos_instability_mask(run_args)
