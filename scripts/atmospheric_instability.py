from argparse import Namespace

import gcsfs
import joblib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from auroraencoderanalysis.utils.cav import (
    get_concept_res,
    get_correlation_analysis,
    get_model_and_concept_vector,
)
from auroraencoderanalysis.utils.models import run_logistic_regression
from auroraencoderanalysis.utils.parsers import get_atmos_instability_parser


COLOURS = [
    '#BFDBFE',
    '#2563EB',
    '#1E3A8A',
]
MASKS = [
    "k_gt_20",
    "k_gt_35",
]

def reduce_mask_lazy(mask: xr.DataArray, patch_size: int) -> xr.DataArray:
    coarsened = mask.coarsen(latitude=patch_size, longitude=patch_size, boundary='trim')
    reduced = (coarsened.mean() >= 0.5).astype(np.int8)
    return reduced.compute()

def run_regression(X: np.ndarray, mask: np.ndarray) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, mask,
        test_size=0.2,
        stratify=mask,
    )
    train_test_split_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
    return run_logistic_regression(train_test_split_dict)

def run_pca(X: np.ndarray) -> dict:
    pca = PCA(n_components=2)
    return {
        "model": pca,
        "res": pca.fit_transform(X.T),
    }

def visualise_pca(pca_res: dict, masks: list[np.ndarray], save_path: list) -> None:
    pca = pca_res["model"]
    atmos_insta_pca_vecs = pca_res["res"]

    is_unstable = np.zeros(len(masks[0])).astype(int)
    for mask in masks:
        is_unstable += is_unstable + mask.astype(int)

    custom_cmap = ListedColormap(COLOURS)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(atmos_insta_pca_vecs[:, 0], atmos_insta_pca_vecs[:, 1],
                        c=is_unstable, cmap=custom_cmap, alpha=0.7, s=10, edgecolors='w', linewidth=0.2)

    plt.xlabel('Principal Component 1 ({:.2f}% Var)'.format(pca.explained_variance_ratio_[0]*100))
    plt.ylabel('Principal Component 2 ({:.2f}% Var)'.format(pca.explained_variance_ratio_[1]*100))
    plt.title('PCA of Atmospheric Latent Vectors (Level 0): Atmospheric Instability')
    plt.grid(alpha=0.3)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOURS[i],
                                markersize=8, label=label)
                    for i, label in enumerate(['Below 20', 'Greater than 20', 'Greater than 35'])]
    plt.legend(handles=legend_elements, title="K-index range", loc='best')

    plt.tight_layout()
    plt.savefig(save_path, format="jpeg", dpi=300)

def main(arg_parser: Namespace) -> None:
    # Read data from Google bucket
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(arg_parser.embeddings_path)
    aurora_embeddings = xr.open_zarr(store, consolidated=True)

    atmos_embeddings = aurora_embeddings["atmos_latent"].sel(time=slice(arg_parser.start_date, arg_parser.end_date))
    atmos_embeddings_values = atmos_embeddings # Download data

    # Get atmospheric instability masks
    atmos_insta_masks = xr.open_zarr(arg_parser.mask_path)
    atmos_insta_masks = atmos_insta_masks.sel(valid_time=slice(arg_parser.start_date, arg_parser.end_date))
    patched_atmos_insta_masks = reduce_mask_lazy(atmos_insta_masks, patch_size=arg_parser.patch_size)

    # Two definitions of instability, >20 and >35
    masks = [
        patched_atmos_insta_masks[mask].data.reshape(-1) for mask in MASKS
    ]

    # Select the three different latent levels in the encoder
    X0 = atmos_embeddings_values[:, :, 0, :, :,].transpose(1, 0, 2, 3).reshape(512, -1).values
    # X1 = atmos_embeddings_values[:, :, 1, :, :,].transpose(1, 0, 2, 3).reshape(512, -1).values
    # X2 = atmos_embeddings_values[:, :, 2, :, :,].transpose(1, 0, 2, 3).reshape(512, -1).values

    # PCA
    pca_res = run_pca(X0)
    visualise_pca(
        pca_res,
        masks,
        f"{arg_parser.output_dir}/atmos_instability_pca.jpeg"
    )

    # Logistic regression for each mask
    reg_res_all = {
        mask: run_regression(X0, mask_arr) for mask, mask_arr in zip(MASKS, masks)
    }
    joblib.dump(reg_res_all, "atmos_instab_reg_res.pkl")

    # CAV analysis
    cav_corr_res = {}
    for mask, reg_res in reg_res_all.items():
        log_model, concept_vector = get_model_and_concept_vector(reg_res)
        concept_res = get_concept_res(log_model, X0.T, concept_vector)
        cav_corr_res[mask] = get_correlation_analysis(concept_res)

    joblib.dump(cav_corr_res, f"{arg_parser.output_dir}/atmos_instab_cav_corr_res.pkl")


if __name__ == "__main__":
    arg_parser = get_atmos_instability_parser()
    main(arg_parser)