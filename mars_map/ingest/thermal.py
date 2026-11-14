"""Thermal inertia from THEMIS (Mars Odyssey), 100 m, in physical units.

Source: "Mars THEMIS-Derived Global Thermal Inertia Mosaic", quantitative 32-bit
version (Fergason et al. 2006; USGS Astrogeology, 2014). Tiles of 30 x 60 degrees,
2-3 GB each, named by their south-west corner; we read only the pixels over the
region. Coverage is latitude -60..60.

Units: J m^-2 K^-1 s^-1/2. Low (< ~250) = dust and loose sand, ~400 and up = rocky
regolith, > ~1200 = bedrock. Absolute accuracy about 20 %, relative about 10 %.
At 100 m per pixel a 2 km region is only 20 x 20 values.

    ingest_thermal_inertia(db, grid)
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from ..catalog.db import add_products, add_source
from ..grid import RegionGrid
from .fetch import RAW_DIR, FetchError, fetch_window
from .normalize import PROCESSED_DIR, normalize

BASE = ("https://asc-astropedia.s3.us-west-2.amazonaws.com/Mars/Odyssey/"
        "THEMIS-Global-Thermal-Inertia-Mosaic/Quantitative-32-Bit/")
UNIT = "J m^-2 K^-1 s^-1/2"
VALID = (5.0, 5000.0)


def themis_ti_tile(lat: float, lon: float) -> str | None:
    """Tile name for a point, e.g. (-4.8, 137.4) -> "30S120E". None outside -60..60."""
    if not -60.0 <= lat < 60.0:
        return None
    south = math.floor(lat / 30.0) * 30
    west = int(math.floor((lon % 360.0) / 60.0) * 60)
    return f"{abs(south):02d}{'S' if south < 0 else 'N'}{west:03d}E"


def themis_ti_url(lat: float, lon: float) -> str | None:
    tile = themis_ti_tile(lat, lon)
    return tile and f"{BASE}THEMIS_TI_Mosaic_Quant_{tile}_100mpp.cub"


def ingest_thermal_inertia(db: sqlite3.Connection, grid: RegionGrid, raw_dir: str | Path = RAW_DIR,
                           processed_root: str | Path = PROCESSED_DIR,
                           verbose: bool = True) -> Path | None:
    """`thermal_inertia__themis_ti.tif` for the region, or None when there is no data."""
    url = themis_ti_url(grid.center_lat, grid.center_lon)
    if url is None:
        if verbose:
            print("  thermal_inertia: no THEMIS tile at this latitude")
        return None
    tile = themis_ti_tile(grid.center_lat, grid.center_lon)
    south, west = math.floor(grid.center_lat / 30.0) * 30, math.floor((grid.center_lon % 360) / 60) * 60
    lon0 = west if west < 180 else west - 360
    add_source(db, "themis_ti", "THEMIS", "USGS Astrogeology", base_url=BASE,
               license="NASA / USGS public domain")
    add_products(db, "themis_ti", [{"product_id": f"THEMIS_TI_Mosaic_Quant_{tile}_100mpp",
                                    "kind": "thermal", "url": url, "resolution_m": 100.0,
                                    "bbox": (lon0, south, lon0 + 60, south + 30)}])
    try:
        raw = fetch_window(url, "themis_ti", grid, raw_dir)
    except FetchError as e:
        if verbose:
            print(f"  thermal_inertia: SKIPPED, {e}")
        return None
    path = raw and normalize(db, grid, "thermal_inertia", UNIT, "themis_ti", [raw], 100.0,
                             valid_range=VALID, processed_root=processed_root)
    if verbose:
        print(f"  thermal_inertia__themis_ti: {'ok' if path else 'no data over the region'}")
    return path or None
