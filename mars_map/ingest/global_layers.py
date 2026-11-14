"""Global coarse layers: the fallback wherever high-resolution data is missing.

The mosaics are 16 MB to 23 GB each, so they are never downloaded whole: for every
region `fetch_window` reads just the pixels above it, and the normalizer puts that
crop on the region grid like any other product.

    register_global_products(db)            # once: adds them to the catalog
    ingest_global_layers(db, GALE_MOUNT_SHARP)

URLs verified 2026-11. Not available yet (no working direct URL found): the TES
thermal inertia mosaic and the TES mineral abundance maps.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ..catalog.db import add_products, add_source, find_products
from ..grid import RegionGrid
from .fetch import RAW_DIR, FetchError, fetch_window
from .normalize import ELEVATION_RANGE_M, PROCESSED_DIR, normalize

USGS = "https://planetarymaps.usgs.gov/mosaic/"
WHOLE_MARS = (-180.0, -90.0, 180.0, 90.0)


@dataclass(frozen=True)
class GlobalProduct:
    source: str
    instrument: str
    product_id: str
    kind: str
    resolution_m: float
    url: str
    layer: str                                   # processed layer name
    unit: str
    valid_range: tuple[float, float] | None = None


GLOBAL_PRODUCTS = [
    # MOLA laser altimeter DEM, metres above the areoid (int16).
    GlobalProduct("mola", "MOLA", "Mars_MGS_MOLA_DEM_mosaic_global_463m", "dtm", 463.0,
                  USGS + "Mars_MGS_MOLA_DEM_mosaic_global_463m.tif",
                  "elevation", "m", ELEVATION_RANGE_M),
    # MOLA blended with HRSC stereo, 200 m: sharper than MOLA alone where HRSC exists.
    GlobalProduct("hrsc_mola", "HRSC+MOLA", "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2", "dtm", 200.0,
                  USGS + "Mars/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif",
                  "elevation", "m", ELEVATION_RANGE_M),
    # TES bolometric albedo (bright = dust, dark = sand / rock), 7.4 km.
    GlobalProduct("tes_albedo", "TES", "Mars_MGS_TES_Albedo_mosaic_global_7410m", "thermal", 7410.0,
                  USGS + "Mars_MGS_TES_Albedo_mosaic_global_7410m.tif",
                  "albedo", "0..1", (0.0, 1.0)),
]

# THEMIS IR mosaics come as regional tiles from the STAC catalog (source "themis"):
# product id suffix -> processed layer. Values are 8-bit stretched brightness, not kelvin.
THEMIS_LAYERS = {"_nir": "themis_night_ir", "_dir": "themis_day_ir"}


def register_global_products(db: sqlite3.Connection) -> int:
    """Add the global mosaics to `sources` / `products` (bbox = whole Mars)."""
    n = 0
    for g in GLOBAL_PRODUCTS:
        add_source(db, g.source, g.instrument, "USGS Astrogeology", base_url=USGS,
                   license="NASA / USGS public domain")
        n += add_products(db, g.source, [{"product_id": g.product_id, "kind": g.kind,
                                          "url": g.url, "resolution_m": g.resolution_m,
                                          "bbox": WHOLE_MARS}])
    return n


def ingest_global_layers(db: sqlite3.Connection, grid: RegionGrid, raw_dir: str | Path = RAW_DIR,
                         processed_root: str | Path = PROCESSED_DIR,
                         verbose: bool = True) -> dict[str, Path]:
    """Crop every global layer to `grid` and normalize it. Returns {"layer__source": path}.

    A layer that cannot be fetched is reported and skipped, the others still run.
    """
    register_global_products(db)
    jobs = [(g.layer, g.unit, g.source, g.url, g.resolution_m, g.valid_range) for g in GLOBAL_PRODUCTS]
    for p in find_products(db, grid.bounds_lonlat, "thermal", source="themis"):
        for suffix, layer in THEMIS_LAYERS.items():
            if p.product_id.endswith(suffix):
                jobs.append((layer, "DN", "themis", p.url, p.resolution_m or 100.0, (1.0, 255.0)))

    out: dict[str, Path] = {}
    for layer, unit, source, url, res, valid in jobs:
        key = f"{layer}__{source}"
        try:
            raw = fetch_window(url, source, grid, raw_dir)
        except FetchError as e:
            if verbose:
                print(f"  {key}: SKIPPED, {e}")
            continue
        path = raw and normalize(db, grid, layer, unit, source, [raw], res, valid_range=valid,
                                 processed_root=processed_root)
        if path:
            out[key] = path
        if verbose:
            print(f"  {key}: {'ok' if path else 'no data over the region'}")
    return out
