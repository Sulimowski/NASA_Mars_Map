"""Height of the obstacles in every cell, for vehicles that must clear them.

Two things stand in a rover's way, and the layer holds the taller of them:

Rocks
    From the rock abundance ``k`` (share of the surface covered by rocks) with
    the size-frequency model of Golombek and Rapp (1997), which was used to
    certify the landing sites of the Mars rovers. The share of the surface
    covered by rocks of diameter ``D`` or larger is::

        F(D) = k * exp(-q(k) * D),      q(k) = 1.79 + 0.152 / k

    The obstacle diameter is the one whose rocks cover ``hazard_area_fraction``
    of the cell, ``D = ln(k / f) / q(k)``, and a rock stands about half its
    diameter high. Where ``k <= f`` there are no obstacle rocks.

Micro-relief
    Steps and ledges resolved by the elevation data: the difference between a
    cell and the mean of its neighbourhood (a plane fitted to the surroundings),
    which is zero on any even slope. It is only meaningful where the grid is
    about as fine as a rover; on a coarser grid the same difference measures the
    curvature of hills, so it is skipped there.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.config import ObstacleConfig

FloatGrid = NDArray[np.float64]


def rock_obstacle_height_m(rock_abundance: FloatGrid,
                           cfg: ObstacleConfig = ObstacleConfig()) -> FloatGrid:
    """Height of the obstacle rocks for a rock abundance raster (0..1), m."""
    k = np.clip(np.asarray(rock_abundance, dtype=np.float64), 0.0, 0.999)
    out = np.zeros(k.shape, dtype=np.float64)
    rocky = k > cfg.hazard_area_fraction
    q = cfg.rock_q_intercept + cfg.rock_q_slope / k[rocky]
    out[rocky] = cfg.rock_height_to_diameter * np.log(k[rocky] / cfg.hazard_area_fraction) / q
    return out


def relief_step_m(elevation: FloatGrid, cfg: ObstacleConfig = ObstacleConfig()) -> FloatGrid:
    """Height of the steps the elevation data resolves, m.

    ``relief_amplitude_factor * |elevation - mean of the surrounding window|``.
    The grid border is extended by odd reflection, which continues a slope
    instead of flattening it, so an even slope gives zero everywhere.
    """
    z = np.asarray(elevation, dtype=np.float64)
    w = cfg.relief_window_cells
    padded = np.pad(z, w, mode="reflect", reflect_type="odd") if min(z.shape) > w else np.pad(
        z, w, mode="edge")
    size = 2 * w + 1
    total = np.zeros(z.shape, dtype=np.float64)
    for d_row in range(size):
        for d_col in range(size):
            total += padded[d_row:d_row + z.shape[0], d_col:d_col + z.shape[1]]
    return cfg.relief_amplitude_factor * np.abs(z - total / (size * size))


def default_obstacle_height_m(elevation: FloatGrid, cell_size_m: float, rock_hazard: FloatGrid,
                              cfg: ObstacleConfig = ObstacleConfig()) -> FloatGrid:
    """Obstacle height when a provider has no layer of its own.

    Rocks are estimated from ``rock_hazard`` (abundance = hazard *
    ``hazard_to_abundance``); the relief term is added on grids no coarser than
    ``relief_max_cell_m``.
    """
    height = rock_obstacle_height_m(np.asarray(rock_hazard) * cfg.hazard_to_abundance, cfg)
    if cell_size_m <= cfg.relief_max_cell_m:
        height = np.maximum(height, relief_step_m(elevation, cfg))
    return height
