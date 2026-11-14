"""Fusion: processed layers of one region -> one `Site`.

For every layer name the catalog may hold several files from different sources
(elevation from HiRISE, CTX and MOLA). Per pixel we take the finest source that
has data there. `source_res` records which resolution each pixel came from, and
`nodata` marks pixels where the best source was missing.

Derived here (never taken from files): slope, roughness, terrain_class.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..grid import UINT8_NODATA, RegionGrid, read_layer
from ..site import MissionParams, Site, Target, resolve_path, slope_deg

# Terrain class thresholds (working assumptions, tune on real data).
SAND_THERMAL_INERTIA_MAX = 250.0    # below this the ground is loose sand / dust
BOULDER_ROCK_MIN = 0.30             # above this share of rocks it is a boulder field
ROUGHNESS_WINDOW_M = 7.0
SEAM_FEATHER_M = 60.0               # distance over which a coarse source is blended in


@dataclass
class LayerFile:
    name: str
    source: str
    native_res_m: float
    path: Path


def region_grid(db: sqlite3.Connection, region: str) -> RegionGrid:
    row = db.execute("SELECT name, center_lat, center_lon, size_m, resolution_m FROM regions "
                     "WHERE name=?", (region,)).fetchone()
    if row is None:
        known = [r[0] for r in db.execute("SELECT name FROM regions")]
        raise KeyError(f"region {region!r} not in catalog, known: {known}")
    return RegionGrid(*row)


def region_layer_files(db: sqlite3.Connection, region: str) -> dict[str, list[LayerFile]]:
    """{layer name: files sorted finest first}."""
    out: dict[str, list[LayerFile]] = {}
    for name, source, res, path in db.execute(
            "SELECT l.name, s.name, l.native_res_m, l.path FROM layers l "
            "JOIN regions r ON r.id = l.region_id JOIN sources s ON s.id = l.source_id "
            "WHERE r.name=? ORDER BY l.name, l.native_res_m", (region,)):
        out.setdefault(name, []).append(LayerFile(name, source, res, resolve_path(path)))
    return out


def box_mean(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)x(2r+1) window, edges padded."""
    p = np.pad(a, r + 1, mode="edge").astype(np.float64)
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s / (k * k))[: a.shape[0], : a.shape[1]]


def seam_correction(values: np.ndarray, coarse: np.ndarray, feather_px: int) -> np.ndarray:
    """Offset to add to `coarse` so it meets `values` at the seam without a step.

    Where both exist we know the mismatch (values - coarse). It is spread outward
    from the known area and fades linearly to zero over `feather_px` pixels.
    """
    known = np.isfinite(values) & np.isfinite(coarse)
    delta = np.where(known, values - coarse, 0.0)
    dist = np.where(known, 0, feather_px).astype(np.float32)
    step = 2
    for i in range(0, feather_px, step):
        num = box_mean(delta * known, step)
        den = box_mean(known.astype(np.float64), step)
        new = ~known & (den > 1e-9)
        if not new.any():
            break
        delta[new] = num[new] / den[new]
        dist[new] = i + step
        known = known | new
    return (delta * np.clip(1 - dist / feather_px, 0, 1)).astype(np.float32)


def merge_sources(files: list[LayerFile], shape,
                  feather_px: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Finest source wins per pixel. Returns (values, native resolution used per pixel).

    With `feather_px` > 0 a coarser source is shifted to meet the finer one at the
    seam (see `seam_correction`); use it for continuous layers like elevation.
    Pixels no source covers stay NaN in both arrays.
    """
    values = np.full(shape, np.nan, np.float32)
    used_res = np.full(shape, np.nan, np.float32)
    for i, f in enumerate(files):                    # already sorted finest first
        arr, _ = read_layer(f.path)
        if arr.shape != tuple(shape):
            raise ValueError(f"{f.path} is {arr.shape}, region grid is {shape}")
        valid = (arr != UINT8_NODATA) if arr.dtype == np.uint8 else np.isfinite(arr)
        take = valid & np.isnan(values)
        if i > 0 and feather_px > 0 and take.any():
            arr = arr + seam_correction(values, arr.astype(np.float32), feather_px)
        values[take] = arr[take]
        used_res[take] = f.native_res_m
    return values, used_res


def roughness(elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    """Std of elevation in a ~7 m window."""
    r = max(1, int(ROUGHNESS_WINDOW_M / 2 / resolution_m))
    e = elevation.astype(np.float64)
    e = e - e.mean()                                 # keep squares small, avoids float error
    var = box_mean(e * e, r) - box_mean(e, r) ** 2
    return np.sqrt(np.maximum(var, 0)).astype(np.float32)


def terrain_class(slope, rock, thermal_inertia, mission: MissionParams) -> np.ndarray:
    """0 bedrock, 1 sand, 2 boulders, 3 impassable. Later rules override earlier ones."""
    cls = np.zeros(slope.shape, np.uint8)
    cls[thermal_inertia < SAND_THERMAL_INERTIA_MAX] = 1
    cls[rock > BOULDER_ROCK_MIN] = 2
    cls[slope > mission.hard_max_slope_deg] = 3
    return cls


def _fill(values: np.ndarray, default: float) -> np.ndarray:
    """Replace leftover NaN (no source at all) with the layer median, or a default."""
    missing = np.isnan(values)
    if missing.any():
        values = values.copy()
        values[missing] = np.nanmedian(values) if (~missing).any() else default
    return values


CORE_LAYERS = {"elevation", "rock_abundance", "thermal_inertia", "science_score"}


def _common_series(db: sqlite3.Connection, region: str) -> dict[str, list[float]]:
    """Per-sol series of a region, cut to the sols every series has.

    Sources cover different sols (weather: the whole mission, radiation: the sols we
    downloaded). `Site` needs equal-length series, so we keep the sols common to all.
    A series sharing no sol with the others is dropped rather than emptying the rest.
    """
    by_name: dict[str, dict[int, float]] = {}
    for name, sol, value in db.execute(
            "SELECT s.name, s.sol, s.value FROM series s JOIN regions r ON r.id = s.region_id "
            "WHERE r.name=?", (region,)):
        by_name.setdefault(name, {})[sol] = value
    if not by_name:
        return {}
    # start from the shortest series: it limits the common range
    order = sorted(by_name, key=lambda n: len(by_name[n]))
    common = set(by_name[order[0]])
    kept = [order[0]]
    for name in order[1:]:
        both = common & set(by_name[name])
        if both:
            common, kept = both, kept + [name]
    sols = sorted(common)
    out: dict[str, list[float]] = {"sol": sols}
    for name in sorted(kept):
        out[name] = [by_name[name][s] for s in sols]
    return out


def fuse_region(db_path: str | Path, region: str, start: tuple[float, float],
                targets: list[Target], no_go=(), mission: MissionParams | None = None) -> Site:
    """Build a `Site` for a region from whatever processed layers the catalog has."""
    mission = mission or MissionParams()
    db = sqlite3.connect(resolve_path(db_path))
    try:
        grid = region_grid(db, region)
        files = region_layer_files(db, region)
        if "elevation" not in files:
            raise ValueError(f"region {region!r} has no elevation layer, cannot build a site")

        feather_px = max(2, int(round(SEAM_FEATHER_M / grid.resolution_m)))
        elev, elev_res = merge_sources(files["elevation"], grid.shape, feather_px)
        best_res = files["elevation"][0].native_res_m
        nodata = ~(elev_res == best_res)             # not from the best source (or from none)
        elev = _fill(elev, 0.0)

        def layer(name: str, default: float) -> np.ndarray:
            if name not in files:
                return np.full(grid.shape, default, np.float32)
            return _fill(merge_sources(files[name], grid.shape)[0], default)

        rock = layer("rock_abundance", 0.05)
        ti = layer("thermal_inertia", 350.0)
        sci = layer("science_score", 0.0)
        slope = slope_deg(elev, grid.resolution_m)

        # Optional layers the catalog has but the contract does not require
        # (albedo, THEMIS brightness, minerals...): carried through unchanged.
        extras = {name: _fill(merge_sources(fs, grid.shape)[0], 0.0)
                  for name, fs in files.items() if name not in CORE_LAYERS}

        series = _common_series(db, region)
        used = sorted({f.source for fs in files.values() for f in fs})
    finally:
        db.close()

    site = Site(
        name=grid.name,
        source="fused from: " + ", ".join(used),
        resolution_m=grid.resolution_m,
        center_lat=grid.center_lat,
        center_lon=grid.center_lon,
        layers={
            "elevation": elev,
            "slope": slope,
            "roughness": roughness(elev, grid.resolution_m),
            "rock_abundance": np.clip(rock, 0, 1),
            "thermal_inertia": ti,
            "terrain_class": terrain_class(slope, rock, ti, mission),
            "science_score": np.clip(sci, 0, 1),
            "nodata": nodata,
            "source_res": np.where(np.isnan(elev_res), np.float32(np.inf), elev_res),
            **extras,
        },
        series=series,
        start=tuple(start),
        targets=list(targets),
        no_go=[list(p) for p in no_go],
        mission=mission,
    )
    site.validate()
    return site
