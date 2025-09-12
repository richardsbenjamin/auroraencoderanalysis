from __future__ import annotations

import dataclasses

import torch
from aurora import Aurora, Batch, Metadata
from aurora import rollout as aurora_rollout

from auroraencoder._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auroraencoder._typing import Dataset, Device, ndarray


def get_aurora_batch(surface_vars: Dataset, static_vars: Dataset, atmos_vars: Dataset) -> Batch:
    return Batch(
        surf_vars={
            "2t": torch.from_numpy(surface_vars["t2m"].values[None]),
            "10u": torch.from_numpy(surface_vars["u10"].values[None]),
            "10v": torch.from_numpy(surface_vars["v10"].values[None]),
            "msl": torch.from_numpy(surface_vars["msl"].values[None]),
        },
        static_vars={
            "z": torch.from_numpy(static_vars["z"].values[0]),
            "slt": torch.from_numpy(static_vars["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars["t"].values[None]),
            "u": torch.from_numpy(atmos_vars["u"].values[None]),
            "v": torch.from_numpy(atmos_vars["v"].values[None]),
            "q": torch.from_numpy(atmos_vars["q"].values[None]),
            "z": torch.from_numpy(atmos_vars["z"].values[None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surface_vars.latitude.values),
            lon=torch.from_numpy(surface_vars.longitude.values),
            time=(surface_vars.valid_time.values.astype("datetime64[s]").tolist()[1],),
            atmos_levels=tuple(int(level) for level in atmos_vars.isobaricInhPa.values),
        ),
    )

def get_aurora_model(device: Device) -> Aurora:
    model = Aurora(use_lora=False)
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
    model.eval()
    model = model.to(device)
    return model

def run_aurora(model: Aurora, data: Batch, eval_steps: int, device: Device) -> Batch:
    with torch.inference_mode():
        preds = [pred.to(device) for pred in aurora_rollout(model, data, steps=eval_steps)]
    return preds

def run_encoder(aurora_model: Aurora, batch: Batch) -> ndarray:
    # Prepare data for encoder as per original code on GitHub
    p = next(aurora_model.parameters())
    transformed_batch = batch.type(p.dtype)
    transformed_batch = transformed_batch.normalise(surf_stats=aurora_model.surf_stats)
    transformed_batch = transformed_batch.crop(patch_size=aurora_model.patch_size)
    transformed_batch = transformed_batch.to(p.device)

    B, T = next(iter(transformed_batch.surf_vars.values())).shape[:2]
    transformed_batch = dataclasses.replace(
        transformed_batch,
        static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in transformed_batch.static_vars.items()},
    )
    # Get full embedding
    full_embedding = aurora_model.encoder(transformed_batch, aurora_model.timestep)
    return full_embedding.cpu().detach().numpy()

