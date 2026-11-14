"""H2: what an external router needs from the astronaut model.

The static edge-cost cubes (``shelter_maps.build_edge_cost_cubes``) are computed
for one reference state: a given fatigue, daylight, one instant. A router that
searches on them can correct for the state it is actually in with two scalars::

    speed_multiplier = visibility * (1 - penalty * F) / (1 - penalty * F_ref)
    time_actual      = time_cube / speed_multiplier
    o2_actual        = o2_cube  * o2_multiplier,     o2_multiplier = 1 / speed_multiplier

These are first-order corrections: they are exact where the walking speed is set
by the power cap and conservative where it is set by the preferred speed (the
crew member then simply walks at the preferred speed). ``evaluate_route`` gives
the exact figures for a finished route.

``export_cost_cubes`` packs the cubes with their direction table into plain
arrays, so another team can read them without importing this package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..core.config import SimConfig
from ..core.exceptions import ValidationError
from ..providers.interfaces import NEIGHBOR_OFFSETS
from .shelter_maps import EdgeCostCubes
from .transition import visibility_factor

CUBE_SCHEMA = "eva-cost-cubes/1"


def speed_multiplier(sun_elevation_rad: float, fatigue: float, cfg: SimConfig) -> float:
    """Walking speed relative to the reference state of the cost cubes, (0, 1].

    Args:
        sun_elevation_rad: Sun elevation now (night halves the speed).
        fatigue: whole-body fatigue F now, 0..1.

    Raises:
        ValidationError: ``fatigue`` is outside 0..1.
    """
    if not 0.0 <= fatigue <= 1.0:
        raise ValidationError(f"fatigue={fatigue} outside [0, 1]")
    penalty = cfg.capacity.fatigue_power_penalty
    reference = 1.0 - penalty * cfg.routing.standard_fatigue
    relative = (1.0 - penalty * fatigue) / reference
    return min(1.0, relative) * visibility_factor(sun_elevation_rad, cfg.routing)


def o2_multiplier(sun_elevation_rad: float, fatigue: float, cfg: SimConfig) -> float:
    """Oxygen per metre relative to the reference state of the cost cubes, >= 1."""
    return 1.0 / speed_multiplier(sun_elevation_rad, fatigue, cfg)


def export_cost_cubes(cubes: EdgeCostCubes, cell_size_m: float) -> dict[str, NDArray[np.float64]]:
    """Cost cubes as plain arrays for another team's router.

    Returns:
        ``time_s``, ``o2_kg``, ``energy_j``, ``risk``, ``length_m``: ``[8, rows, cols]``,
        the cost of leaving cell ``(r, c)`` in direction ``d``; ``inf`` = impassable.
        ``directions``: ``[8, 2]`` as ``(d_row, d_col)`` per direction index.
        ``cell_size_m``: scalar array.
    """
    return {
        "time_s": cubes.time_s, "o2_kg": cubes.o2_kg, "energy_j": cubes.energy_j,
        "risk": cubes.risk, "length_m": cubes.length_m,
        "directions": np.array([[d_row, d_col] for d_row, d_col, _ in NEIGHBOR_OFFSETS],
                               dtype=np.float64),
        "cell_size_m": np.array(cell_size_m, dtype=np.float64),
    }


def save_cost_cubes(path: str | Path, cubes: EdgeCostCubes, cell_size_m: float) -> Path:
    """Write the cost cubes to a compressed ``.npz`` file and return its path."""
    out = Path(path)
    if out.suffix != ".npz":
        out = out.with_suffix(".npz")
    np.savez_compressed(out, schema=np.array(CUBE_SCHEMA), **export_cost_cubes(cubes, cell_size_m))
    return out
