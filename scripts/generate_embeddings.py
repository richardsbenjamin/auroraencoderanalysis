from __future__ import annotations

import numpy as np
import torch
import xarray as xr

from auroraencoderanalysis._typing import TYPE_CHECKING
from auroraencoderanalysis.utils.constants import LEVELS, VARS
from auroraencoderanalysis.utils.datasets import get_step_dataset, get_init_dataset, read_edh
from auroraencoderanalysis.utils.latlon import reduce_lon_lat
from auroraencoderanalysis.utils.models import get_aurora_batch, get_aurora_model, run_encoder
from auroraencoderanalysis.utils.parsers import get_gen_embeddings_parser

if TYPE_CHECKING:
    from auroraencoderanalysis._typing import ArgumentParser


def generate_embeddings(arg_parser: ArgumentParser) -> None:
    # Static data
    static = xr.open_dataset(arg_parser.static_path)

    # Surf/Atmos data
    edh_singles = read_edh(arg_parser.singles_path)
    edh_levels = read_edh(arg_parser.levels_path)

    # Patch and lon/lat variables
    patch_size = arg_parser.patch_size
    patch_center_lat, patch_center_lon = reduce_lon_lat(patch_size, edh_levels.latitude.values, edh_levels.longitude.values)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aurora_model = get_aurora_model(device)

    # Subeset data according to dates, variables and 6h timestep
    start_date = arg_parser.start_date
    end_date = arg_parser.end_date

    surf_var_names = VARS["edh"]["surf"]
    edh_surface_data = (
        edh_singles[surf_var_names]
        .sel(valid_time=slice(start_date, end_date))
        .isel(valid_time=slice(None, None, 6))
    )
    atmos_var_names = VARS["edh"]["atmos"]
    edh_atmos_data = (
        edh_levels[atmos_var_names]
        .sel(valid_time=slice(start_date, end_date))
        .sel(isobaricInhPa=LEVELS)
        .isel(valid_time=slice(None, None, 6))
    )

    # Initialise output dataset
    n_embed = arg_parser.embed_dim
    lats = patch_center_lat.T[0]
    lons = patch_center_lon[0]
    n_levels = arg_parser.n_embed_levels
    output_ds = get_init_dataset(n_embed, n_levels, lats, lons)
    output_ds.to_zarr(arg_parser.output_zarr_path, mode="w")

    # Main loop to download data, run encoding and export zarr
    for batch_start in edh_atmos_data.valid_time:
        batch_end = batch_start + np.timedelta64(1, "6h")

        surf_select = edh_surface_data.sel(valid_time=slice(batch_start, batch_end))
        atmos_select = edh_atmos_data.sel(valid_time=slice(batch_start, batch_end))

        batch = get_aurora_batch(
            surf_select,
            static,
            atmos_select,
        )

        full_embedding = run_encoder(aurora_model, batch)
        reshaped_embedding = full_embedding.reshape(1, 4, 64800, n_embed).squeeze()

        surf_embedding = reshaped_embedding[0].transpose(1, 0)
        atmos_embedding = reshaped_embedding[1:]
        
        surf_step = surf_embedding.reshape(1, n_embed, n_lats, n_lons)
        atmos_step = atmos_embedding.transpose(2, 0, 1).reshape(1, n_embed, n_levels, n_lats, n_lons)

        step = get_step_dataset(
            surf_step,
            atmos_step,
            batch_end.values,
            n_embed,
            n_levels,
            lats,
            lons,
        )
        step.to_zarr(arg_parser.output_zarr_path, mode="a", append_dim="time")


if __name__ == "__main__":
    run_args = get_gen_embeddings_parser()
    generate_embeddings(run_args)
