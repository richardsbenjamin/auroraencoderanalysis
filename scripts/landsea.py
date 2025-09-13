from __future__ import annotations

import numpy as np
import torch
import xarray as xr

from auroraencoderanalysis._typing import TYPE_CHECKING
from auroraencoderanalysis.utils.constants import VARS
from auroraencoderanalysis.utils.datasets import pickle_dump, reduce_mask
from auroraencoderanalysis.utils.latlon import reduce_lon_lat
from auroraencoderanalysis.utils.models import (
    get_aurora_batch,
    get_aurora_model,
    get_train_test_split,
    run_encoder,
    run_logistic_regression,
)
from auroraencoderanalysis.utils.parsers import get_land_sea_parser
from auroraencoderanalysis.utils.plotting import plot_map

if TYPE_CHECKING:
    from auroraencoderanalysis._typing import ArgumentParser


def run_land_sea_analysis(arg_parser: ArgumentParser) -> None:
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aurora_model = get_aurora_model(device)
    
    # Get land-sea mask and reduce to patches
    static = xr.open_dataset(arg_parser.static_path)
    land_sea_mask = static["lsm"].values.squeeze()
    land_sea_mask_patched = reduce_mask(land_sea_mask, arg_parser.patch_size)

    # ERA5 dataset and get initial conditions
    era5_ds = xr.open_zarr(arg_parser.era5_zarr_path)
    ic = era5_ds.sel(
        time=slice(arg_parser.start_date, arg_parser.end_date),
    )

    # Get lat/lon centres
    patch_center_lat, patch_center_lon = reduce_lon_lat(
        arg_parser.patch_size,
        era5_ds.latitude.values,
        era5_ds.longitude.values,
    )
    
    # Run encoder
    land_sea_batch = get_aurora_batch(
        ic[VARS["era5"]["surf"]],
        ic[VARS["era5"]["atmos"]],
        static,
    )
    full_embedding = run_encoder(aurora_model, land_sea_batch)

    # Reconstruct surface/atmos embeddings
    reshaped_embedding = full_embedding.reshape(1, 4, 64800, 512).squeeze()
    surf_embedding = reshaped_embedding[0].transpose(1, 0)
    
    # Get train/test split
    train_split_dict = get_train_test_split(
        arg_parser.test_lon_min,
        arg_parser.test_lon_max,
        patch_center_lon,
        land_sea_mask_patched.ravel(),
        surf_embedding,
    )

    # Run the logistic regression
    reg_res = run_logistic_regression(train_split_dict)
    pickle_dump(reg_res, arg_parser.output_path)

    # Plot classification errors
    is_misclassified = (reg_res["y_pred"] != train_split_dict["y_test"])
    region_patch_indices = np.where(train_split_dict["is_test_region"])[0]
    region_center_lons = patch_center_lon.ravel()[region_patch_indices]
    region_center_lats = patch_center_lat.ravel()[region_patch_indices]
    error_lons = region_center_lons[is_misclassified]
    error_lats = region_center_lats[is_misclassified]

    plot_map(
        error_lons,
        error_lats,
        title="Location of Classification Errors",
        color="black",
        s=50,
        alpha=0.7,
        extent=[-150, 90, -90, 90],
        export_name="landsea_log_reg_errors.jpeg",
    )




if __name__ == "__main__":
    run_args = get_land_sea_parser()
    run_land_sea_analysis(run_args)