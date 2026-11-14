"""Rover traverses and rover maps in the units a map or a report shows.

Output boundary for the rover side, like ``map_layers`` for the astronaut:
inside everything is SI; here values become km, m/h, minutes, hours, sols,
degrees, percent, watt-hours.

    rover_route_profile   one row per action of a traverse
    rover_plan_summary    totals and verdict of a traverse
    rover_comparison      the same traverse for several rovers, one row each
    build_rover_layers    rasters: cost per km, slip, what blocks each cell, reach
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..core.constants import HOUR_S, MINUTE_S, SOL_S
from ..routing.reachability import ReachabilityResult
from ..routing.rover_planner import RoverPlan
from ..routing.rover_transition import RoverEdgeEvaluator
from ..routing.shelter_maps import build_metric_cubes
from ..rovers.types import RoverBlock, RoverLabel

BLOCK_CODES: Mapping[RoverBlock, int] = {
    RoverBlock.NONE: 0, RoverBlock.FORBIDDEN: 1, RoverBlock.IMPASSABLE: 2,
    RoverBlock.CLEARANCE: 3, RoverBlock.TILT: 4,
}
"""Values of the ``cell_block`` raster."""


@dataclass(frozen=True, slots=True)
class RoverProfileRow:
    """State after one action of a traverse, in report units."""

    index: int
    row: int
    col: int
    action: str
    mode: str
    elapsed_h: float
    distance_m: float
    speed_m_h: float
    slope_deg: float
    slip_pct: float
    load_w: float
    generation_w: float
    heater_w: float
    battery_pct: float
    upset_risk: float

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary, ready for JSON."""
        return asdict(self)


def rover_route_profile(final_label: RoverLabel, battery_capacity_j: float
                        ) -> list[RoverProfileRow]:
    """Action-by-action profile of the traverse that ends at ``final_label``."""
    chain = final_label.chain()
    t0 = chain[0].t_s
    rows: list[RoverProfileRow] = []
    for index, label in enumerate(chain):
        tr = label.transition
        rows.append(RoverProfileRow(
            index=index, row=label.row, col=label.col, action=label.action,
            mode=tr.mode.value if tr else "",
            elapsed_h=(label.t_s - t0) / HOUR_S, distance_m=label.distance_m,
            speed_m_h=tr.speed_m_s * HOUR_S if tr else 0.0,
            slope_deg=math.degrees(tr.slope_rad) if tr else 0.0,
            slip_pct=100.0 * tr.slip if tr else 0.0,
            load_w=tr.load_w if tr else 0.0, generation_w=tr.generation_w if tr else 0.0,
            heater_w=tr.heater_w if tr else 0.0,
            battery_pct=100.0 * label.battery_j / battery_capacity_j, upset_risk=label.risk))
    return rows


def rover_plan_summary(plan: RoverPlan, battery_capacity_j: float) -> dict[str, Any]:
    """Totals and verdict of a rover traverse, in report units."""
    summary: dict[str, Any] = {
        "rover": plan.rover,
        "found": plan.found,
        "message": plan.message,
        "broken_constraint": plan.bottleneck.value if plan.bottleneck else None,
        "blocked_steps": dict(plan.blocked_steps),
    }
    final = plan.final_label
    if final is None:
        return summary
    summary.update({
        "distance_m": final.distance_m,
        "duration_h": plan.duration_s / HOUR_S,
        "sols": plan.sols,
        "drive_time_h": plan.drive_time_s / HOUR_S,
        "parked_h": final.wait_s / HOUR_S,
        "mean_drive_speed_m_h": (final.distance_m / plan.drive_time_s * HOUR_S
                                 if plan.drive_time_s > 0.0 else 0.0),
        "energy_used_wh": final.energy_used_j / HOUR_S,
        "battery_end_pct": 100.0 * final.battery_j / battery_capacity_j,
        "battery_min_pct": 100.0 * min(final.min_battery_j, final.battery_j) / battery_capacity_j,
        "max_slip_pct": 100.0 * final.max_slip,
        "wheel_walking_m": final.wheel_walk_m,
        "upset_risk": final.risk,
        "dose_mgy": 1000.0 * final.dose_gy,
        "cells": len(plan.path),
    })
    return summary


def rover_comparison(plans: Mapping[str, RoverPlan | None],
                     capacities_j: Mapping[str, float]) -> list[dict[str, Any]]:
    """One row per rover for the same traverse.

    Args:
        plans: ``{rover key: plan or None}`` as ``compare_rovers`` returns it.
        capacities_j: battery capacity of each rover, by the same keys.
    """
    rows: list[dict[str, Any]] = []
    for key, plan in plans.items():
        if plan is None:
            rows.append({"rover": key, "found": False, "message": "not on Mars in this year"})
            continue
        summary = rover_plan_summary(plan, capacities_j[key])
        rows.append({name: summary.get(name) for name in (
            "rover", "found", "message", "distance_m", "sols", "drive_time_h", "parked_h",
            "battery_min_pct", "max_slip_pct", "wheel_walking_m")})
    return rows


def _mean_over_directions(cube: NDArray[np.float64]) -> NDArray[np.float64]:
    finite = np.isfinite(cube)
    count = finite.sum(axis=0)
    out = np.full(count.shape, np.nan)
    np.divide(np.where(finite, cube, 0.0).sum(axis=0), count, out=out, where=count > 0)
    return out


def build_rover_layers(evaluator: RoverEdgeEvaluator, t_s: float,
                       reach: ReachabilityResult | None = None,
                       battery_fraction: float = 1.0) -> dict[str, NDArray[Any]]:
    """Raster layers of a rover on a site, each ``[rows, cols]``.

    Layers:
        cell_block            what stops the rover from entering the cell (``BLOCK_CODES``)
        passable_directions   number of the 8 moves out of the cell the rover can make
        time_min_per_km, energy_wh_per_km, slip_pct
            mean over the passable moves out of the cell (NaN if none)
        wheel_walking_share   share of the passable moves that need wheel-walking
        obstacle_margin_m     rover obstacle limit minus the obstacle height of the cell
        arrival_time_h, reachable, isochrone_sol
            only when ``reach`` is given

    Args:
        t_s: mission time the environment is sampled at; pick a daylight time,
            because at night no move is passable.
    """
    shape = (evaluator.rows, evaluator.cols)
    block = np.zeros(shape, dtype=np.int8)
    for row in range(shape[0]):
        for col in range(shape[1]):
            block[row, col] = BLOCK_CODES.get(evaluator.cell_block((row, col)), 2)
    energy = battery_fraction * evaluator.profile.battery_j
    state = RoverLabel(row=0, col=0, t_s=t_s, battery_j=energy, min_battery_j=energy)
    cubes = build_metric_cubes(evaluator, state)
    empty = np.full((8, *shape), np.inf)
    time_s, length = cubes.get("time_s", empty), cubes.get("length_m", empty)
    passable = np.isfinite(time_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_km = np.where(passable, 1000.0 / length, np.inf)
        layers: dict[str, NDArray[Any]] = {
            "cell_block": block,
            "passable_directions": passable.sum(axis=0).astype(np.int8),
            "time_min_per_km": _mean_over_directions(time_s * per_km) / MINUTE_S,
            "energy_wh_per_km": _mean_over_directions(cubes.get("energy_j", empty) * per_km) / HOUR_S,
            "slip_pct": 100.0 * _mean_over_directions(cubes.get("slip", empty)),
            "wheel_walking_share": _mean_over_directions(cubes.get("wheel_walking", empty)),
            "obstacle_margin_m": (evaluator.profile.max_obstacle_height_m
                                  - np.asarray(evaluator.terrain.obstacle_height_m)),
        }
    if reach is not None:
        isochrone = np.full(shape, np.nan)
        for limit in sorted(reach.isochrones, reverse=True):
            isochrone[reach.isochrones[limit]] = limit / SOL_S
        layers["arrival_time_h"] = reach.arrival_time_s / HOUR_S
        layers["reachable"] = reach.reachable
        layers["isochrone_sol"] = isochrone
    return layers
