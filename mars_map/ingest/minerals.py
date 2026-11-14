"""Mineral maps from CRISM (the imaging spectrometer on MRO) -> `mineral_*` layers.

Source: CRISM MTRDR "summary parameter" cubes (PDS Geosciences, found through PDS ODE).
Each cube is one map-projected scene, 18 or 36 m/pixel, with 60 bands. A band is one
spectral index: the depth of an absorption that a mineral group has (Viviano-Beck et
al. 2014, JGR 119). Higher value = stronger signature; values near or below zero =
not detected. They are unitless indices, not percent abundance.

A cube is ~130 MB but stored band after band (ENVI BSQ, float32), so we download only
the bands we need with HTTP range requests, ~2 MB each.

    load_ode_crism(db, grid.bounds_lonlat)     # catalog: which scenes exist
    ingest_minerals(db, grid)                  # fetch bands, put them on the region grid
"""

from __future__ import annotations

import re
import sqlite3
import urllib.request
from pathlib import Path

import numpy as np

from ..catalog.db import BBox, add_products, add_source, find_products, wrap_lon
from ..grid import RegionGrid
from .fetch import RAW_DIR, USER_AGENT, FetchError, fetch
from .loaders import ODE_URL, _get_json, _ode_files, _ode_products, ode_query_url
from .normalize import PROCESSED_DIR, normalize

# CRISM summary parameter -> (layer name, what a high value means)
CRISM_PARAMS = {
    "SINDEX2": ("mineral_sulfate", "hydrated sulfates (mono- and polyhydrated)"),
    "BD2100_2": ("mineral_sulfate_mono", "monohydrated sulfates such as kieserite"),
    "D2300": ("mineral_clay", "Fe/Mg clay minerals (phyllosilicates)"),
    "BD1900_2": ("mineral_hydration", "water bound in minerals (any hydrated mineral)"),
    "BD2210_2": ("mineral_silica", "Al-OH minerals or hydrated silica"),
    "BD860_2": ("mineral_hematite", "crystalline ferric oxide (hematite)"),
    "OLINDEX3": ("mineral_olivine", "olivine"),
    "LCPINDEX2": ("mineral_pyroxene", "low-calcium pyroxene"),
}
MINERAL_LAYERS = [layer for layer, _ in CRISM_PARAMS.values()]
INDEX_RANGE = (-1.0, 1.0)            # real indices are within a few percent of zero
_SU_FILE = re.compile(r"_su\d+j_mtr3\.img$", re.I)


def load_ode_crism(db: sqlite3.Connection, bbox: BBox) -> int:
    """CRISM MTRDR scenes over `bbox` -> products (kind "mineral", URL of the summary cube)."""
    add_source(db, "crism", "CRISM", "PDS ODE", base_url=ODE_URL, license="NASA public domain")
    rows = []
    for p in _ode_products(_get_json(ode_query_url(bbox, "MRO", "CRISM", "MTRDR", 100, 0))):
        su = [f for f in _ode_files(p) if _SU_FILE.search(f["FileName"])]
        if not su:
            continue
        wkt = p.get("Footprint_C0_geometry") or p.get("Footprint_geometry")
        rows.append({
            "product_id": p["pdsid"], "kind": "mineral", "url": su[0]["URL"],
            "resolution_m": float(p["Map_scale"]) if p.get("Map_scale") else None,
            "footprint_wkt": None if not wkt or "EMPTY" in wkt else wkt,
            "bbox": (wrap_lon(float(p["Westernmost_longitude"])), float(p["Minimum_latitude"]),
                     wrap_lon(float(p["Easternmost_longitude"])), float(p["Maximum_latitude"])),
            "size_bytes": int(float(su[0]["KBytes"]) * 1024) if su[0].get("KBytes") else None,
            "acquired": (p.get("Observation_time") or p.get("UTC_start_time") or "")[:10] or None,
        })
    return add_products(db, "crism", rows)


# --- ENVI header + band-wise download ------------------------------------------------

def parse_envi_header(text: str) -> dict:
    """The few ENVI header fields we need. Values in braces may span lines."""
    out = {}
    for key, braced, plain in re.findall(r"^([a-z][a-z ]*?)\s*=\s*(?:\{(.*?)\}|([^\n{]*))", text,
                                         re.M | re.S | re.I):
        out[key.strip().lower()] = (braced if braced else plain).strip()
    if out.get("file type", "ENVI Standard") != "ENVI Standard" or out.get("interleave", "").lower() != "bsq":
        raise ValueError("only band-sequential ENVI Standard files are supported")
    if int(out["data type"]) != 4:
        raise ValueError(f"expected float32 (ENVI data type 4), got {out['data type']}")
    info = [v.strip() for v in out["map info"].split(",")]
    px, py, x, y, dx, dy = (float(v) for v in info[1:7])
    return {
        "samples": int(out["samples"]), "lines": int(out["lines"]), "bands": int(out["bands"]),
        "offset": int(out.get("header offset", 0)),
        "dtype": "<f4" if int(out.get("byte order", 0)) == 0 else ">f4",
        "nodata": float(out["data ignore value"]) if "data ignore value" in out else None,
        "band_names": [n.strip() for n in out["band names"].replace("\n", " ").split(",")],
        "wkt": out.get("coordinate system string"),
        # upper-left corner of the upper-left pixel, pixel size
        "x0": x - (px - 1) * dx, "y0": y + (py - 1) * dy, "dx": dx, "dy": dy,
    }


def _get_range(url: str, start: int, length: int, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Range": f"bytes={start}-{start + length - 1}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 206:
                raise FetchError(f"{url}: server ignored the range request (status {resp.status})")
            data = resp.read()
    except OSError as e:
        raise FetchError(f"could not read {url}: {e}") from e
    if len(data) != length:
        raise FetchError(f"{url}: got {len(data)} of {length} bytes")
    return data


def fetch_crism_params(url: str, params=tuple(CRISM_PARAMS), raw_dir: str | Path = RAW_DIR) -> dict[str, Path]:
    """Download the given bands of one summary cube. Returns {parameter: single-band GeoTIFF}.

    Files are the whole scene in its own projection, `data/raw/crism/<scene>__<PARAM>.tif`.
    Cached: bands already on disk are not requested again.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    folder = Path(raw_dir) / "crism"
    stem = url.rsplit("/", 1)[1].rsplit(".", 1)[0]
    out = {p: folder / f"{stem}__{p}.tif" for p in params}
    if all(p.exists() for p in out.values()):
        return out

    hdr_path = fetch(url.rsplit(".", 1)[0] + ".hdr", folder / f"{stem}.hdr")
    h = parse_envi_header(hdr_path.read_text(encoding="latin-1"))
    if not h["wkt"]:
        raise ValueError(f"{hdr_path} has no coordinate system string")
    band_bytes = h["samples"] * h["lines"] * 4
    profile = dict(driver="GTiff", height=h["lines"], width=h["samples"], count=1, dtype="float32",
                   crs=CRS.from_wkt(h["wkt"]), nodata=h["nodata"], compress="deflate",
                   transform=from_origin(h["x0"], h["y0"], h["dx"], h["dy"]))
    for param, dest in out.items():
        if dest.exists():
            continue
        if param not in h["band_names"]:
            raise KeyError(f"{stem} has no band {param!r}")
        i = h["band_names"].index(param)
        raw = _get_range(url, h["offset"] + i * band_bytes, band_bytes)
        data = np.frombuffer(raw, dtype=h["dtype"]).reshape(h["lines"], h["samples"]).astype(np.float32)
        part = dest.with_name(dest.name + ".part")
        with rasterio.open(part, "w", **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(crism_parameter=param, cropped_from=url)
        part.replace(dest)
    return out


def ingest_minerals(db: sqlite3.Connection, grid: RegionGrid, max_scenes: int = 3,
                    min_coverage: float = 0.2, raw_dir: str | Path = RAW_DIR,
                    processed_root: str | Path = PROCESSED_DIR, verbose: bool = True) -> dict[str, Path]:
    """CRISM scenes over the region -> one `mineral_*` layer per parameter.

    Scenes are used finest first, then by coverage; the first one wins and later ones
    only fill its gaps. Returns {layer: path}.
    """
    scenes = find_products(db, grid.bounds_lonlat, "mineral", source="crism", min_coverage=min_coverage)
    scenes = scenes[:max_scenes]
    if not scenes:
        if verbose:
            print("  crism: no scene covers the region")
        return {}
    per_param: dict[str, list[Path]] = {p: [] for p in CRISM_PARAMS}
    for s in scenes:
        try:
            files = fetch_crism_params(s.url, raw_dir=raw_dir)
        except (FetchError, KeyError, ValueError) as e:
            if verbose:
                print(f"    {s.product_id}: SKIPPED, {e}")
            continue
        for p, path in files.items():
            per_param[p].append(path)
        if verbose:
            print(f"    {s.product_id} ({s.resolution_m:.0f} m, cover {s.coverage:.0%}, {s.acquired})")
    out = {}
    res = min(s.resolution_m for s in scenes)
    for param, paths in per_param.items():
        if not paths:
            continue
        layer = CRISM_PARAMS[param][0]
        path = normalize(db, grid, layer, "index", "crism", paths, res, valid_range=INDEX_RANGE,
                         processed_root=processed_root)
        if path:
            out[layer] = path
    return out
