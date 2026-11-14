"""Normalizer: raw rasters -> one processed layer on the region grid.

Takes one or more raw files of the same source (as downloaded or cropped by the
fetcher), reprojects each onto the `RegionGrid`, turns the provider's nodata into
NaN, converts units, mosaics them, writes the file with `write_layer` and registers
it in the `layers` table.

    normalize(db, GALE_MOUNT_SHARP, "elevation", "m", "hirise_dtm", [path1, path2],
              native_res_m=1.0, valid_range=(-9000, 22000), align_offsets=True)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np

from ..catalog.db import add_layer, add_region
from ..grid import RegionGrid, coverage_frac, write_layer
from ..site import DATA_DIR
from .fetch import source_crs

PROCESSED_DIR = DATA_DIR / "processed"

# Elevation on Mars is between about -8200 m (Hellas) and +21230 m (Olympus Mons).
ELEVATION_RANGE_M = (-9000.0, 22000.0)


def reproject_to_grid(path: str | Path, grid: RegionGrid, *, resampling: str | None = None,
                      valid_range: tuple[float, float] | None = None,
                      scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """One raw raster -> float32 array on the region grid, NaN where there is no data.

    `resampling` defaults to "average" when the source is finer than the grid (so a
    2 m cell is the mean of the 1 m cells in it, not a sample) and "bilinear" when it
    is coarser. `valid_range` is checked on raw values, then value = raw * scale + offset.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    out = np.full(grid.shape, np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path} has no coordinate system, cannot place it on the grid")
        if resampling is None:
            resampling = "average" if abs(src.res[0]) < grid.resolution_m * 0.75 else "bilinear"
        reproject(
            source=rasterio.band(src, 1), destination=out,
            src_crs=source_crs(src), src_nodata=src.nodata, dst_transform=grid.transform(),
            dst_crs=CRS.from_proj4(grid.proj4), dst_nodata=np.nan,
            resampling=Resampling[resampling])
    # PDS products mark gaps with huge negative floats that not every label declares
    out[~np.isfinite(out) | (np.abs(out) > 1e30)] = np.nan
    if valid_range is not None:
        lo, hi = valid_range
        out[(out < lo) | (out > hi)] = np.nan
    if scale != 1.0 or offset != 0.0:
        out = (out * scale + offset).astype(np.float32)
    return out


def mosaic(arrays: Sequence[np.ndarray], align_offsets: bool = False) -> np.ndarray:
    """First array wins, later ones only fill its gaps.

    With `align_offsets` each later array is first shifted by the median difference
    over the pixels it shares with the mosaic so far. Two DTMs of the same ground
    usually disagree by a constant few metres; without the shift every seam between
    them would look like a cliff.
    """
    out = arrays[0].copy()
    for a in arrays[1:]:
        fill = np.isnan(out) & np.isfinite(a)
        if not fill.any():
            continue
        if align_offsets:
            both = np.isfinite(out) & np.isfinite(a)
            if both.sum() >= 100:
                a = a - np.float32(np.median(a[both] - out[both]))
        out[fill] = a[fill]
    return out


def normalize(db: sqlite3.Connection, grid: RegionGrid, layer: str, unit: str, source: str,
              paths: Sequence[str | Path], native_res_m: float, *,
              resampling: str | None = None, valid_range: tuple[float, float] | None = None,
              scale: float = 1.0, offset: float = 0.0, align_offsets: bool = False,
              processed_root: str | Path = PROCESSED_DIR) -> Path | None:
    """Raw files of one source -> `<processed_root>/<region>/<layer>__<source>.tif` + `layers` row.

    `paths` in priority order (best first). The source must already be in `sources`.
    Returns the written path, or None when no file has any data inside the region
    (nothing is written or registered then).
    """
    if not paths:
        raise ValueError("normalize needs at least one raw file")
    arrays = [reproject_to_grid(p, grid, resampling=resampling, valid_range=valid_range,
                                scale=scale, offset=offset) for p in paths]
    data = mosaic(arrays, align_offsets=align_offsets)
    cover = coverage_frac(data)
    if cover == 0.0:
        return None
    add_region(db, grid)
    path = write_layer(grid.layer_path(layer, source, root=processed_root), data, grid,
                       layer=layer, unit=unit, source=source)
    add_layer(db, grid.name, layer, unit, source, native_res_m, path, cover)
    return path
