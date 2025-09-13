from __future__ import annotations

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from auroraencoderanalysis._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auroraencoderanalysis._typing import ndarray


def plot_map(
        x: ndarray,
        y: ndarray,
        title: str = "",
        extent: list = [-180, 180, -90, 90],
        export_name: str | None = None,
        **kwargs: dict,
    ) -> None:
    fig = plt.figure(figsize=(10, 8))
    proj = ccrs.PlateCarree(central_longitude=165)
    ax = plt.axes(projection=proj)
    if extent:
        ax.set_extent(extent)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.coastlines()

    ax.scatter(
        x, y,  transform=ccrs.PlateCarree(), **kwargs
    )
    plt.title(title)
    
    if export_name is not None:
        plt.savefig(export_name, format="jpeg", dpi=300)

