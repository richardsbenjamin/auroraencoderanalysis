from __future__ import annotations

import joblib
import shutil
import zipfile

import cdsapi
import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from auroraencoderanalysis._typing import TYPE_CHECKING
from auroraencoderanalysis.utils.cav import (
    get_concept_res,
    get_correlation_analysis,
    get_model_and_concept_vector,
)
from auroraencoderanalysis.utils.datasets import (
    prepare_x,
    read_edh,
    reduce_field,
    reduce_percentiles,
)
from auroraencoderanalysis.utils.latlon import reduce_lon_lat
from auroraencoderanalysis.utils.models import run_logistic_regression
from auroraencoderanalysis.utils.parsers import get_temp_extremes_parser

if TYPE_CHECKING:
    from auroraencoderanalysis._typing import ArgumentParser, Aurora, Dataset, ndarray


COLOURS = [
    '#1E3A8A',
    '#FF6B35',
    '#FF3A00',
    '#CC0000',
    '#8B0000',
]

def get_europe_percentiles(percentiles: str, output_path: str) -> Dataset:
    dataset = "sis-temperature-statistics"
    request = {
        "variable": "maximum_temperature",
        "period": "year",
        "statistic": [
            f"{p}th_percentile" for p in percentiles
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

def get_percentile_maximums(percentiles: list, percentile_year: str, output_dir: str) -> None:
    # Extract maximums
    percentile_data = {
        f"p{p}": (
            xr.open_dataset(f"{output_dir}/temperature_percentiles/p{p}_Tmax_Yearly_rcp45_mean_v1.0.nc")
            .sel(time=percentile_year)[f"p{p}_Tmax_Yearly"]
        )
        for p in percentiles
    }
    return percentile_data

def get_regression_labels(
        era5_ds: Dataset,
        patch_level_percentiles: dict,
        start_path: str,
        end_path: str,
        is_valid_percentile: ndarray,
    ) -> dict:
    # Construct regression labels
    temp_2m = (
        era5_ds["t2m"].sel(valid_time=slice(start_path, end_path))
        .isel(valid_time=slice(None, None, 6))
    )
    temp_2m = temp_2m.assign_coords(longitude=((temp_2m.longitude % 360) - 180))

    is_extreme_all_p = {}
    for p, patch_level_percentile in patch_level_percentiles.items():
        is_extreme_all_p[p] = []
        for i, temp_2m_values in enumerate(temp_2m.values):
            temp_2m_patched = reduce_field(temp_2m_values, patch_size=4) - 273.15
            is_extreme = temp_2m_patched[is_valid_percentile] > patch_level_percentile[is_valid_percentile]
            is_extreme_all_p[p].append(is_extreme)

        is_extreme_all_p[p] = np.stack(is_extreme_all_p[p]).ravel()

    return is_extreme_all_p

def get_pca_transform(X: ndarray, is_extreme_all_p: dict) -> ndarray:
    pca = PCA(n_components=2)
    pca_p_transform = pca.fit_transform(X.T)

    is_p = np.zeros(len(is_extreme_all_p["p75"]))
    for is_extreme_p in is_extreme_all_p.values():
        is_p += is_extreme_p.astype(int)
    
    return pca_p_transform, is_p, pca.explained_variance_ratio_

def get_pca_plot(
        pca_p_transform: ndarray,
        is_p: ndarray,
        ratios: ndarray,
        save_path: str,
    ) -> None:
    custom_cmap = ListedColormap(COLOURS)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_p_transform[:, 0], pca_p_transform[:, 1],
                        c=is_p, cmap=custom_cmap, alpha=0.7, s=10, edgecolors='w', linewidth=0.2)

    cbar = plt.colorbar(scatter, ticks=[0, 1, 2, 3, 4])
    cbar.set_label('Percentile Range')
    cbar.set_ticklabels(['Below 75th', '75th-89th', '90th-94th', '95th-98th', '99th+'])

    plt.xlabel('Principal Component 1 ({:.2f}% Var)'.format(ratios[0]*100))
    plt.ylabel('Principal Component 2 ({:.2f}% Var)'.format(ratios[1]*100))
    plt.title('PCA of Surface Latent Vectors: Percentile Distribution')
    plt.grid(alpha=0.3)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOURS[i],
                                markersize=8, label=label)
                    for i, label in enumerate(['Below 75th', '75th-89th', '90th-94th', '95th-98th', '99th+'])]
    plt.legend(handles=legend_elements, title="Percentile Ranges", loc='best')
    plt.tight_layout()
    plt.savefig(save_path, format="jpeg", dpi=300)

def get_percentiles_logistic_regression(
        X: ndarray,
        is_extreme_all_p: dict,
    ) -> tuple[dict]:
    percentile_regs = {}
    percentile_splits = {}
    for p, y in is_extreme_all_p.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X.T, y,
            test_size=0.2,
            stratify=y,
        )
        train_test_split_dict = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
        percentile_splits[p] = train_test_split_dict
        percentile_regs[p] = run_logistic_regression(train_test_split_dict)

    return percentile_regs, percentile_splits

def get_scores(percentile_regs: dict, percentile_splits: dict) -> DataFrame:
    acc_df = DataFrame()
    acc_df["percentile"] = percentile_regs.keys()
    acc_df["acc"] = [percentile_regs[p]["acc"] for p in percentile_regs.keys()]
    acc_df["prec"] = [
            precision_score(percentile_splits[p]["y_test"], percentile_regs[p]["y_pred"])
            for p in percentile_regs.keys()
    ]
    acc_df["recall"] = [
            recall_score(percentile_splits[p]["y_test"], percentile_regs[p]["y_pred"])
            for p in percentile_regs.keys()
    ]

def run_temperature_extremes_analysis(arg_parser: ArgumentParser) -> None:
    percentiles = [p.strip() for p in arg_parser.percentiles.split(",") if p.strip()]

    # Read data from Google bucket
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(arg_parser.embeddings_path)
    aurora_embeddings = xr.open_zarr(store, consolidated=True)
    surf_embeddings = aurora_embeddings["surface_latent"].sel(time=slice(arg_parser.start_date, arg_parser.end_date))

    # ERA5 data
    era5_ds = read_edh(arg_parser.singles_path)

    # Lon/lat variables
    patch_center_lat, patch_center_lon = reduce_lon_lat(arg_parser.patch_size, era5_ds.latitude.values[:-1], era5_ds.longitude.values)
    patch_lat_bounds = np.stack([patch_center_lat - 0.5, patch_center_lat + 0.5], axis=-1)
    patch_lon_bounds = np.stack([patch_center_lon - 0.5, patch_center_lon + 0.5], axis=-1)

    # Download percentiles and read
    get_europe_percentiles(percentiles, arg_parser.output_dir)
    percentile_data = get_percentile_maximums(
        percentiles, arg_parser.percentile_year, arg_parser.output_dir,
    )

    # Reduce the percentiles into patches
    patch_level_percentiles = {
        p: reduce_percentiles(p_data, patch_lat_bounds, patch_lon_bounds)
        for p, p_data in percentile_data.items()
    }
    # There are many nans, only keep valid values
    # Each percentile has the same nans so just use p99's
    is_valid_percentile = ~np.isnan(patch_level_percentiles["p99"])
    
    X_temp_extremes = prepare_x(surf_embeddings, is_valid_percentile.ravel())

    # Construct regression inputs/outputs
    is_extreme_all_p = get_regression_labels(
        era5_ds, patch_level_percentiles,
        arg_parser.start_path, arg_parser.end_path,
        is_valid_percentile,
    )

    # Get PCA outputs
    pca_p_transform, is_p, ratios = get_pca_transform(X_temp_extremes, is_extreme_all_p)

    # Save PCA plot
    get_pca_plot(pca_p_transform, is_p, ratios, f"{arg_parser.output_dir}/temp_extremes_pca_plot.jpeg")

    # Run logistic regression separately for all percentiles
    percentile_regs, percentile_splits = get_percentiles_logistic_regression(
        X_temp_extremes, is_extreme_all_p, is_extreme_all_p,
    )
    scores_table = get_scores(percentile_regs, percentile_splits)
    scores_table.to_csv(f"{arg_parser.output_path}/scores_table.csv")

    # CAV analysis
    for p, reg_res in percentile_regs.items():
        log_model, concept_vector = get_model_and_concept_vector(reg_res)
        ls_concept_res = get_concept_res(log_model, X_temp_extremes.T, concept_vector)
        corr_analysis = get_correlation_analysis(ls_concept_res)
        joblib.dump(corr_analysis, f"{arg_parser.output_dir}/temp_extremes_{p}_cav_corr_res.pkl")


if __name__ == "__main__":
    run_args = get_temp_extremes_parser()
    run_temperature_extremes_analysis(run_args)

