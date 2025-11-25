from __future__ import annotations

import gcsfs
import joblib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

from auroraencoderanalysis._typing import TYPE_CHECKING
from auroraencoderanalysis.utils.cav import (
    get_concept_res,
    get_correlation_analysis,
    get_model_and_concept_vector,
)
from auroraencoderanalysis.utils.datasets import prepare_x, reduce_mask
from auroraencoderanalysis.utils.latlon import reduce_lon_lat
from auroraencoderanalysis.utils.models import (
    get_train_test_split,
    run_logistic_regression,
)
from auroraencoderanalysis.utils.parsers import get_land_sea_parser
from auroraencoderanalysis.utils.plotting import plot_map

if TYPE_CHECKING:
    from auroraencoderanalysis._typing import Namespace


def plot_pca(pca_res: dict, mask: np.ndarray, save_path: str) -> None:
    pca = pca_res["pca_model"]
    pca_transform = pca_res["pca_transform"]
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_transform[:, 0], pca_transform[:, 1],
                        c=mask, cmap='viridis', alpha=0.6, s=2)
    plt.xlabel('PC1 ({:.2f}% Var)'.format(pca.explained_variance_ratio_[0]*100))
    plt.ylabel('PC2 ({:.2f}% Var)'.format(pca.explained_variance_ratio_[1]*100))
    plt.title('Land vs. Ocean')
    plt.savefig(save_path, format="jpeg", dpi=300)

def run_pca(x: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2)
    return {
        "pca_model": pca,
        "pca_transform": pca.fit_transform(x.T),
    }

def run_land_sea_analysis(arg_parser: Namespace) -> None:
    print("STARTING PROGRAM")
    # Read data from Google bucket
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(arg_parser.embeddings_path)
    aurora_embeddings = xr.open_zarr(store, consolidated=True)
    surf_embeddings = aurora_embeddings["surface_latent"].sel(time=slice(arg_parser.start_date, arg_parser.end_date))
    print("Read embeddings")

    # Static data
    store = fs.get_mapper(arg_parser.static_path)
    static_data = xr.open_zarr(store, consolidated=True)
    print("Read static data")

    # Get land-sea mask and reduce to patches
    land_sea_mask = static_data["lsm"].squeeze().compute()
    land_sea_mask_patched = reduce_mask(land_sea_mask, arg_parser.patch_size)
    y_mask = np.tile(land_sea_mask_patched.ravel(), surf_embeddings.time.shape[0])
    print("Got LS mask")

    # Get lat/lon centres
    lat_patched, lon_patched = reduce_lon_lat(1, surf_embeddings.lat, surf_embeddings.lon)
    print("Got lat/lon centres")
    
    # Prepare X
    X_ls = prepare_x(surf_embeddings)
    print("Prepared X")

    # PCA
    pca_res = run_pca(X_ls)
    plot_pca(pca_res, y_mask, f"{arg_parser.output_dir}/land_sea_pca_plot.jpeg")
    print("PCA plot")
    
    # Logistic regression
    train_split_dict = get_train_test_split(
        arg_parser.test_lon_min,
        arg_parser.test_lon_max,
        lon_patched,
        y_mask,
        X_ls,
    )
    reg_res = run_logistic_regression(train_split_dict)
    joblib.dump(reg_res, f"{arg_parser.output_dir}/land_sea_log_res.pkl")
    print("Log reg")

    # Plot classification errors
    is_misclassified = (reg_res["y_pred"] != train_split_dict["y_test"])
    region_patch_indices = np.where(train_split_dict["is_test_region"])[0]
    region_center_lons = lon_patched.ravel()[region_patch_indices]
    region_center_lats = lat_patched.ravel()[region_patch_indices]
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

    # CAV analysis
    log_model, concept_vector = get_model_and_concept_vector(reg_res)
    ls_concept_res = get_concept_res(log_model, X_ls.T, concept_vector)
    corr_analysis = get_correlation_analysis(ls_concept_res)
    joblib.dump(corr_analysis, f"{arg_parser.output_dir}/land_sea_cav_corr_res.pkl")
    print("CAV analysis")


if __name__ == "__main__":
    run_args = get_land_sea_parser()
    run_land_sea_analysis(run_args)