"""UHF radio link between the base and an astronaut: is the terrain in the way?

For every receiver point the straight ray from the transmitter antenna is
compared with the ground under it:

* shadow    -- the ground cuts the ray: no line of sight;
* degraded  -- line of sight exists, but the ground intrudes into the first
               Fresnel zone (less than `min_fresnel` of its radius is free),
               so the signal is weakened by diffraction;
* clear     -- the Fresnel zone is free enough.

With antennas a few metres above the ground a long link is almost always
"degraded" (the ground itself sits inside the Fresnel zone), which is normal for
surface UHF. The state that matters for planning is "shadow".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0
MARS_RADIUS_M = 3_389_500.0


@dataclass
class RadioParams:
    """Working assumptions for the surface radio link."""
    frequency_mhz: float = 400.0
    base_height_m: float = 10.0        # antenna mast at the base (site.start)
    astronaut_height_m: float = 1.8    # antenna on the backpack
    relay_height_m: float = 1.5        # portable relay dropped on the route
    min_fresnel: float = 0.6           # share of the first Fresnel radius that must be free
    step_m: float | None = None        # ground sampling along the ray (default: 2 cells)
    merge_gap_m: float = 30.0          # shadow stretches closer than this count as one
    min_relay_zone_m: float = 20.0     # shorter shadow stretches get no relay of their own


def fresnel_clearance(elevation: np.ndarray, resolution_m: float, tx_xy: tuple[float, float],
                      rx_xy: np.ndarray, tx_height_m: float, rx_height_m: float,
                      frequency_mhz: float = 400.0, step_m: float | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Clearance of the ray from one transmitter to N receivers.

    rx_xy is (N, 2) in metres. Returns two arrays of length N:
    margin_m -- lowest height of the ray above the ground; negative = blocked;
    ratio    -- lowest (height above ground / first Fresnel radius) along the ray.
    """
    rows, cols = elevation.shape
    flat = np.ascontiguousarray(elevation).ravel()
    rx = np.atleast_2d(np.asarray(rx_xy, dtype=np.float64))
    step = step_m or 2.0 * resolution_m
    wavelength = SPEED_OF_LIGHT / (frequency_mhz * 1e6)

    def ground(x, y):
        c = np.clip((x / resolution_m).astype(np.intp), 0, cols - 1)
        r = np.clip((y / resolution_m).astype(np.intp), 0, rows - 1)
        return flat[r * cols + c].astype(np.float64)

    tx = np.array(tx_xy, dtype=np.float64)
    z_tx = float(ground(tx[:1], tx[1:])[0]) + tx_height_m
    z_rx = ground(rx[:, 0], rx[:, 1]) + rx_height_m
    dist = np.hypot(rx[:, 0] - tx[0], rx[:, 1] - tx[1])

    margin = np.full(len(rx), np.inf)
    ratio = np.full(len(rx), np.inf)
    n_steps = max(2, math.ceil(float(dist.max(initial=0.0)) / step))
    for s in range(1, n_steps):
        frac = s / n_steps
        d1, d2 = frac * dist, (1.0 - frac) * dist
        ray = z_tx + frac * (z_rx - z_tx)
        bulge = d1 * d2 / (2.0 * MARS_RADIUS_M)                      # planet curvature
        clear = ray - ground(tx[0] + frac * (rx[:, 0] - tx[0]),
                             tx[1] + frac * (rx[:, 1] - tx[1])) - bulge
        inside = (d1 > resolution_m) & (d2 > resolution_m)           # skip the antenna cells
        np.minimum(margin, np.where(inside, clear, np.inf), out=margin)
        fresnel = np.sqrt(wavelength * d1 * d2 / np.maximum(dist, 1e-9))
        np.minimum(ratio, np.where(inside, clear / np.maximum(fresnel, 1e-6), np.inf), out=ratio)
    return margin, ratio


def link_quality(elevation: np.ndarray, resolution_m: float, tx_xy: tuple[float, float],
                 rx_xy: np.ndarray, params: RadioParams = RadioParams(),
                 tx_height_m: float | None = None) -> list[str]:
    """'clear', 'degraded' or 'shadow' for every receiver point."""
    margin, ratio = fresnel_clearance(
        elevation, resolution_m, tx_xy, rx_xy,
        params.base_height_m if tx_height_m is None else tx_height_m,
        params.astronaut_height_m, params.frequency_mhz, params.step_m)
    return np.where(margin < 0, "shadow",
                    np.where(ratio < params.min_fresnel, "degraded", "clear")).tolist()


def coverage_map(elevation: np.ndarray, resolution_m: float, tx_xy: tuple[float, float],
                 params: RadioParams = RadioParams(), cell_m: float = 10.0) -> np.ndarray:
    """Radio shadow of a transmitter over the whole site, on a grid of about cell_m.

    Returns uint8 (rows, cols) of the reduced grid: 0 shadow, 1 degraded, 2 clear.
    """
    rows, cols = elevation.shape
    f = max(1, int(round(cell_m / resolution_m)))
    ys = (np.arange(0, rows, f) + f / 2) * resolution_m
    xs = (np.arange(0, cols, f) + f / 2) * resolution_m
    gx, gy = np.meshgrid(xs, ys)
    margin, ratio = fresnel_clearance(
        elevation, resolution_m, tx_xy, np.column_stack([gx.ravel(), gy.ravel()]),
        params.base_height_m, params.astronaut_height_m, params.frequency_mhz,
        params.step_m or f * resolution_m)
    out = np.where(margin < 0, 0, np.where(ratio < params.min_fresnel, 1, 2))
    return out.reshape(gx.shape).astype(np.uint8)
