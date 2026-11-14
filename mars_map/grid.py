"""Region grid and the processed-layer file format.

This is the agreement between ingestion (A) and fusion (B):

* every processed layer of a region is a single-band GeoTIFF on the SAME grid;
* the grid is a square, local equirectangular projection centred on the region,
  so one pixel = `resolution_m` true metres;
* row 0 = north, col 0 = west (same as `Site`);
* values are in SI units; missing data = NaN (float layers) or 255 (uint8 layers);
* tags `layer`, `unit`, `source` are written into the file.

A writes with `write_layer`, B reads with `read_layer`. Nobody opens these files
any other way, so the format can be changed in one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# IAU 2000 Mars sphere radius, the one USGS / HiRISE map products use.
MARS_SPHERE_RADIUS_M = 3_396_190.0
UINT8_NODATA = 255


@dataclass(frozen=True)
class RegionGrid:
    name: str
    center_lat: float          # planetocentric, degrees
    center_lon: float          # east-positive, degrees
    size_m: float              # square side
    resolution_m: float

    @property
    def shape(self) -> tuple[int, int]:
        n = int(round(self.size_m / self.resolution_m))
        return n, n

    @property
    def proj4(self) -> str:
        """Local equirectangular: true scale at the region's latitude."""
        return (f"+proj=eqc +lat_ts={self.center_lat} +lat_0=0 +lon_0={self.center_lon} "
                f"+x_0=0 +y_0=0 +R={MARS_SPHERE_RADIUS_M} +units=m +no_defs")

    @property
    def origin(self) -> tuple[float, float]:
        """Projected (x, y) of the outer NW corner."""
        y_center = MARS_SPHERE_RADIUS_M * math.radians(self.center_lat)
        return -self.size_m / 2, y_center + self.size_m / 2

    @property
    def bounds_lonlat(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat), for catalog bbox queries."""
        dlat = math.degrees(self.size_m / 2 / MARS_SPHERE_RADIUS_M)
        dlon = dlat / math.cos(math.radians(self.center_lat))
        return (self.center_lon - dlon, self.center_lat - dlat,
                self.center_lon + dlon, self.center_lat + dlat)

    def transform(self):
        from rasterio.transform import from_origin
        x0, y0 = self.origin
        return from_origin(x0, y0, self.resolution_m, self.resolution_m)

    def layer_path(self, layer: str, source: str, root: str | Path = "data/processed") -> Path:
        """data/processed/<region>/<layer>__<source>.tif"""
        return Path(root) / self.name / f"{layer}__{source}.tif"


def write_layer(path: str | Path, array: np.ndarray, grid: RegionGrid,
                layer: str, unit: str, source: str) -> Path:
    """Write one processed layer. Returns the path."""
    import rasterio
    from rasterio.crs import CRS

    if array.shape != grid.shape:
        raise ValueError(f"array {array.shape} does not match grid {grid.shape}")
    if array.dtype == np.uint8:
        nodata = UINT8_NODATA
    else:
        array = array.astype(np.float32)
        nodata = float("nan")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype=array.dtype, crs=CRS.from_proj4(grid.proj4),
        transform=grid.transform(), nodata=nodata,
        compress="deflate", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(array, 1)
        dst.update_tags(layer=layer, unit=unit, source=source, region=grid.name)
    return path


def read_layer(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read one processed layer -> (array, tags). Missing data stays NaN / 255."""
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1), dict(src.tags())


def coverage_frac(array: np.ndarray) -> float:
    """Share of cells with real data (for the `layers.coverage_frac` column)."""
    if array.dtype == np.uint8:
        return float((array != UINT8_NODATA).mean())
    return float(np.isfinite(array).mean())


# Regions we work with. Coordinates are approximate until the analyzer picks exact ones.
GALE_MOUNT_SHARP = RegionGrid("gale_mount_sharp", center_lat=-4.75, center_lon=137.38,
                              size_m=2000.0, resolution_m=2.0)
