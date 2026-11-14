

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mars_map.site import Site
else:
    from .site import Site


@dataclass
class AstronautTerrainModel:
    """Physical raster maps of terrain passability the dimension matches that of site.shape"""
    sinkage: np.ndarray            # 0..1: risk of getting stuck in loose sand/dust (from thermal_inertia)
    rock_impedance: np.ndarray     # 0..1: obstruction to movement caused by boulders and micro-topography (rock_abundance + roughness)
    looseness: np.ndarray          #  0..1: integral index of soil looseness and resistance
    terramech_factor: np.ndarray   # 1.0..2.35: Pandolf-Becker metabolic cost multiplier
    speed_multiplier: np.ndarray   # 0.45..1.0: coefficient of walking speed reduction on the ground
    impassable: np.ndarray         # bool: cliffs > hard_max_slope_deg, terrain_class == 3, and no-go polygons
    nodata_penalty: np.ndarray     # 1.0 or 1.25: penalty for lack of high-resolution imagery


def rasterize_nogo(
    shape: tuple[int, int],
    resolution_m: float,
    no_go: list[list[tuple[float, float]]],
) -> np.ndarray:
    """boolean mask"""
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=bool)
    if not no_go:
        return mask

    xs = (np.arange(cols, dtype=np.float32) + 0.5) * resolution_m
    ys = (np.arange(rows, dtype=np.float32) + 0.5) * resolution_m

    for poly in no_go:
        if len(poly) < 3:
            continue
        px = np.array([p[0] for p in poly], dtype=np.float32)
        py = np.array([p[1] for p in poly], dtype=np.float32)

        c0 = max(0, int(px.min() / resolution_m) - 1)
        c1 = min(cols, int(px.max() / resolution_m) + 2)
        r0 = max(0, int(py.min() / resolution_m) - 1)
        r1 = min(rows, int(py.max() / resolution_m) + 2)
        if c0 >= c1 or r0 >= r1:
            continue

        gx, gy = np.meshgrid(xs[c0:c1], ys[r0:r1])
        inside = np.zeros(gx.shape, dtype=bool)
        j = len(poly) - 1
        for i in range(len(poly)):
            yi, yj = py[i], py[j]
            xi, xj = px[i], px[j]
            cond = ((yi > gy) != (yj > gy)) & (
                gx < (xj - xi) * (gy - yi) / (yj - yi + 1e-12) + xi
            )
            inside ^= cond
            j = i
        mask[r0:r1, c0:c1] |= inside

    return mask


def build_astronaut_terrain(site: Site, nodata_risk_weight: float = 0.25) -> AstronautTerrainModel:
    """looseness of soil"""
    ti = site.layers["thermal_inertia"].astype(np.float32)
    rock = site.layers["rock_abundance"].astype(np.float32)
    rough = site.layers["roughness"].astype(np.float32)
    slope = site.layers["slope"].astype(np.float32)
    tclass = site.layers["terrain_class"]
    nodata = site.layers["nodata"].astype(bool)

    # 1. Sinkage
    # TI <= 100 J m^-2 K^-1 s^-1/2 -> non likely (1.0), TI >= 450 ->= solid (0.0)
    sinkage = np.clip((450.0 - ti) / 350.0, 0.0, 1.0)
    sinkage = np.where(tclass == 1, np.maximum(sinkage, 0.65), sinkage)  # класс "sand"

    # 2. Stones and resistance
    rough_norm = np.clip(rough / 0.45, 0.0, 1.0)
    rock_impedance = np.clip(0.65 * rock + 0.35 * rough_norm, 0.0, 1.0)
    rock_impedance = np.where(tclass == 2, np.maximum(rock_impedance, 0.55), rock_impedance)  # класс "boulders"

   
    looseness = np.clip(0.58 * sinkage + 0.42 * rock_impedance, 0.0, 1.0).astype(np.float32)

    # Coefficients for biomechanics
    terramech_factor = (1.0 + 1.35 * looseness).astype(np.float32)

    speed_multiplier = np.clip(1.0 - 0.50 * looseness, 0.45, 1.0).astype(np.float32)

    # 5. NO-GO mask
    nogo_mask = rasterize_nogo(site.shape, site.resolution_m, site.no_go)
    impassable = (
        (tclass == 3)
        | (slope > site.mission.hard_max_slope_deg)
        | nogo_mask
    )

    nodata_penalty = np.where(nodata, 1.0 + nodata_risk_weight, 1.0).astype(np.float32)

    return AstronautTerrainModel(
        sinkage=sinkage.astype(np.float32),
        rock_impedance=rock_impedance.astype(np.float32),
        looseness=looseness,
        terramech_factor=terramech_factor,
        speed_multiplier=speed_multiplier,
        impassable=impassable,
        nodata_penalty=nodata_penalty,
    )