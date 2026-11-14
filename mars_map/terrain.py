"""Terrain geometry shared by the radiation and temperature maps.

Everything here is derived from the elevation layer alone: how high the horizon
is in each direction, how much sky a cell sees, and which way the ground faces.
"""

from __future__ import annotations

import numpy as np


def _ray_steps(max_px: int) -> list[float]:
    """Sample distances along a ray, in pixels: every pixel nearby, sparser far away."""
    steps, d = [], 1.0
    while d <= max_px:
        steps.append(d)
        d = max(d + 1.0, d * 1.2)
    return steps


def horizon_angles(elevation: np.ndarray, resolution_m: float,
                   n_azimuth: int = 16, max_distance_m: float = 400.0) -> np.ndarray:
    """Horizon elevation angle seen from every cell, in radians (>= 0).

    Returns float32 (n_azimuth, rows, cols). Slice k looks toward azimuth
    360*k/n_azimuth degrees, clockwise from north (0 = N, 90 = E, 180 = S).
    Terrain beyond max_distance_m and outside the grid is ignored.
    """
    z = np.asarray(elevation, dtype=np.float32)
    rows, cols = z.shape
    r_idx, c_idx = np.arange(rows), np.arange(cols)
    steps = _ray_steps(max(1, int(max_distance_m / resolution_m)))

    out = np.zeros((n_azimuth, rows, cols), np.float32)
    for k in range(n_azimuth):
        az = 2 * np.pi * k / n_azimuth
        d_row, d_col = -np.cos(az), np.sin(az)        # north = toward row 0
        tan_h = out[k]
        seen = {(0, 0)}
        for d in steps:
            dr, dc = int(round(d_row * d)), int(round(d_col * d))
            if (dr, dc) in seen:
                continue
            seen.add((dr, dc))
            rr = np.clip(r_idx + dr, 0, rows - 1)
            cc = np.clip(c_idx + dc, 0, cols - 1)
            dist = float(np.hypot(dr, dc)) * resolution_m
            np.maximum(tan_h, (z[np.ix_(rr, cc)] - z) / dist, out=tan_h)
        np.arctan(tan_h, out=tan_h)
    return out


def sky_view_factor(horizons: np.ndarray) -> np.ndarray:
    """Share of the sky a cell sees, 0..1 (1 = open plain).

    Cosine-weighted: sky near the horizon counts less than sky overhead.
    """
    return (1.0 - np.mean(np.sin(horizons) ** 2, axis=0)).astype(np.float32)


def horizon_toward(horizons: np.ndarray, azimuth_rad: float) -> np.ndarray:
    """Horizon angle in an arbitrary direction, interpolated between the stored azimuths."""
    n = horizons.shape[0]
    pos = (azimuth_rad % (2 * np.pi)) / (2 * np.pi) * n
    k0 = int(np.floor(pos)) % n
    w = pos - np.floor(pos)
    return (1 - w) * horizons[k0] + w * horizons[(k0 + 1) % n]


def surface_normal(elevation: np.ndarray, resolution_m: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit normal of the ground as (east, north, up) components, float32."""
    dz_dsouth, dz_deast = np.gradient(np.asarray(elevation, dtype=np.float64), resolution_m)
    norm = np.sqrt(dz_deast ** 2 + dz_dsouth ** 2 + 1.0)
    return ((-dz_deast / norm).astype(np.float32),
            (dz_dsouth / norm).astype(np.float32),     # -dz/dnorth = +dz/dsouth
            (1.0 / norm).astype(np.float32))
