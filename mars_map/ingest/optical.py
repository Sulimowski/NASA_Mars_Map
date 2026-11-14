"""Optical layers from HiRISE orthoimages (the pictures the HiRISE DTMs were made from).

Every HiRISE DTM in the PDS comes with orthorectified JP2 images that line up with it
pixel for pixel. We use the 1 m versions:

* `..._RED_C_01_ORTHO.JP2`  grayscale, full swath      -> layer `optical`
* `..._IRB_C_01_ORTHO.JP2`  3 bands (near-IR, red, blue-green), narrower centre strip
                            -> layers `optical_ir`, `optical_red`, `optical_bg`

Values are the 8-bit stretched brightness of the product divided by 255 (0..1). They
are good for looking and for texture, not calibrated reflectance.

`optical_texture` is derived here at the native 1 m scale, before resampling: the
local contrast of the image (RMS of the high-passed brightness in a 3 m window, in
0..1 brightness units). Blocky outcrop and boulder fields are high, sand and dust low.

    load_ode_hirise_ortho(db, grid.bounds_lonlat)
    ingest_optical(db, grid)
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np

from ..catalog.db import BBox, add_products, add_source, find_products, wrap_lon
from ..grid import RegionGrid
from .fetch import RAW_DIR, FetchError, fetch, fetch_window, url_filename, window_path
from .loaders import ODE_URL, _get_json, _ode_files, _ode_products, ode_query_url
from .normalize import PROCESSED_DIR, normalize

WHOLE_FILE_MAX_BYTES = 150e6
_ORTHO = re.compile(r"_(RED|IRB)_C_01_ORTHO\.JP2$", re.I)
SOURCE_OF = {"RED": "hirise_ortho", "IRB": "hirise_color"}
COLOR_BANDS = {1: "optical_ir", 2: "optical_red", 3: "optical_bg"}
BRIGHTNESS = (1.0, 255.0)            # 0 marks "no image" in these products


def load_ode_hirise_ortho(db: sqlite3.Connection, bbox: BBox) -> int:
    """1 m orthoimages that ship with the HiRISE DTMs over `bbox` -> products (kind "optical")."""
    for src in SOURCE_OF.values():
        add_source(db, src, "HiRISE", "PDS ODE", base_url=ODE_URL, license="NASA public domain")
    rows: dict[str, list[dict]] = {s: [] for s in SOURCE_OF.values()}
    for p in _ode_products(_get_json(ode_query_url(bbox, "MRO", "HIRISE", "DTM", 100, 0))):
        wkt = p.get("Footprint_C0_geometry") or p.get("Footprint_geometry")
        for f in _ode_files(p):
            m = _ORTHO.search(f["FileName"])
            if not m:
                continue
            rows[SOURCE_OF[m.group(1).upper()]].append({
                "product_id": f["FileName"].rsplit(".", 1)[0], "kind": "optical", "url": f["URL"],
                "resolution_m": float(p["Map_scale"]) if p.get("Map_scale") else 1.0,
                # footprint of the DTM pair; the colour strip is narrower than this
                "footprint_wkt": None if not wkt or "EMPTY" in wkt else wkt,
                "bbox": (wrap_lon(float(p["Westernmost_longitude"])), float(p["Minimum_latitude"]),
                         wrap_lon(float(p["Easternmost_longitude"])), float(p["Maximum_latitude"])),
                "size_bytes": int(float(f["KBytes"]) * 1024) if f.get("KBytes") else None,
            })
    return sum(add_products(db, src, r) for src, r in rows.items())


def _box_mean(a: np.ndarray, r: int) -> np.ndarray:
    p = np.pad(a, r + 1, mode="edge").astype(np.float64)
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return ((c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k))[: a.shape[0], : a.shape[1]]


def texture_file(raw: Path) -> Path:
    """Local contrast of a raw 8-bit crop, written next to it in the same native grid."""
    import rasterio

    dest = raw.with_name(raw.stem + "__texture.tif")
    if dest.exists():
        return dest
    with rasterio.open(raw) as src:
        a = src.read(1).astype(np.float64)
        profile = src.profile
    valid = a >= BRIGHTNESS[0]
    a = np.where(valid, a, np.nan)
    filled = np.where(valid, a, np.nanmean(a) if valid.any() else 0.0) / 255.0
    high = filled - _box_mean(filled, 2)                      # remove shading broader than 5 m
    tex = np.sqrt(np.maximum(_box_mean(high * high, 1), 0))   # RMS contrast in a 3 m window
    tex[~valid] = np.nan
    profile.update(dtype="float32", nodata=float("nan"), compress="deflate", driver="GTiff")
    part = dest.with_name(dest.name + ".part")
    with rasterio.open(part, "w", **profile) as dst:
        dst.write(tex.astype(np.float32), 1)
    part.replace(dest)
    return dest


def ingest_optical(db: sqlite3.Connection, grid: RegionGrid, max_images: int = 2,
                   raw_dir: str | Path = RAW_DIR, processed_root: str | Path = PROCESSED_DIR,
                   verbose: bool = True) -> dict[str, Path]:
    """Orthoimages over the region -> `optical`, `optical_texture` and the colour bands."""
    out: dict[str, Path] = {}

    def crops(source: str, band: int = 1) -> tuple[list[Path], float]:
        found = find_products(db, grid.bounds_lonlat, "optical", source=source)
        found.sort(key=lambda p: -p.coverage)
        paths = []
        for p in found[:max_images]:
            try:
                where = p.url
                cached = window_path(p.url, source, grid, raw_dir, band).exists()
                if not cached and p.size_bytes and p.size_bytes < WHOLE_FILE_MAX_BYTES:
                    # reading JP2 windows over HTTP takes minutes; a small file is
                    # faster to download once and crop locally
                    where = fetch(p.url, Path(raw_dir) / source / url_filename(p.url)).as_posix()
                raw = fetch_window(where, source, grid, raw_dir, band=band)
            except FetchError as e:
                if verbose:
                    print(f"    {p.product_id}: SKIPPED, {e}")
                continue
            if raw:
                paths.append(raw)
                if verbose and band == 1:
                    print(f"    {p.product_id} ({p.resolution_m:.1f} m)")
        res = min((p.resolution_m for p in found[:max_images]), default=1.0)
        return paths, res

    gray, res = crops("hirise_ortho")
    if gray:
        kw = dict(valid_range=BRIGHTNESS, scale=1 / 255.0, processed_root=processed_root)
        out["optical"] = normalize(db, grid, "optical", "0..1", "hirise_ortho", gray, res, **kw)
        out["optical_texture"] = normalize(db, grid, "optical_texture", "0..1", "hirise_ortho",
                                           [texture_file(p) for p in gray], res,
                                           processed_root=processed_root)
    for band, layer in COLOR_BANDS.items():
        color, res = crops("hirise_color", band)
        if color:
            out[layer] = normalize(db, grid, layer, "0..1", "hirise_color", color, res,
                                   valid_range=BRIGHTNESS, scale=1 / 255.0,
                                   processed_root=processed_root)
    return {k: v for k, v in out.items() if v}
