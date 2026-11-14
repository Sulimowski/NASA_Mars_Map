"""B5: where the crew member can go and still come back.

A goal-free run of the label search from the base (one label per cell, cheapest
first) gives, for every cell, the state on arrival. A cell counts as a valid
round trip when, after working the given task there, every constraint still
holds, including the oxygen and time needed to walk back.

Outputs:
    arrival_time_s        time from leaving the base to reaching the cell
    distance_m            distance walked to the cell
    reachable             cell can be reached and left again at once
    round_trip_ok         cell can be reached, the task done, and the base regained
    isochrones            round-trip cells reachable within 15 / 30 / 60 / 120 minutes
    point_of_no_return    outer edge of the round-trip zone
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.schemas import Cell, PhysState, SuitResources, TaskType
from ..physiology import radiation
from .constraints import ConstraintChecker
from .pareto_astar import ParetoAStar, SearchDiagnostics
from .shelter_maps import ShelterMaps, compute_shelter_maps
from .transition import EdgeEvaluator


@dataclass(slots=True)
class ReachabilityResult:
    """Reachability rasters, all ``[rows, cols]``."""

    arrival_time_s: NDArray[np.float64]
    distance_m: NDArray[np.float64]
    reachable: NDArray[np.bool_]
    round_trip_ok: NDArray[np.bool_]
    isochrones: dict[float, NDArray[np.bool_]]
    point_of_no_return: NDArray[np.bool_]
    max_round_trip_distance_m: float
    diagnostics: SearchDiagnostics


def _outer_edge(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Cells of ``mask`` with at least one 8-neighbour outside it (grid border excluded)."""
    padded = np.pad(mask, 1, mode="edge")
    rows, cols = mask.shape
    all_inside = np.ones_like(mask)
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if d_row or d_col:
                all_inside &= padded[1 + d_row:1 + d_row + rows, 1 + d_col:1 + d_col + cols]
    return mask & ~all_inside


def compute_reachability(evaluator: EdgeEvaluator, base: Cell, *, start_time_s: float,
                         maps: ShelterMaps | None = None, task: TaskType = TaskType.OBSERVATION,
                         task_duration_s: float = 0.0, dose_budget_sv: float | None = None,
                         spe_watch: bool = False, resources: SuitResources | None = None,
                         phys: PhysState | None = None) -> ReachabilityResult:
    """Reachability of every cell for an EVA leaving ``base`` at ``start_time_s``.

    Args:
        task, task_duration_s: work assumed at each cell before turning back
            (``task_duration_s = 0`` means touch and go).
        dose_budget_sv: dose allowed on the EVA; the per-EVA limit when omitted.

    Raises:
        InfeasibleStateError: the base cell is not walkable.
    """
    cfg = evaluator.config
    maps = maps if maps is not None else compute_shelter_maps(evaluator, base, start_time_s)
    budget = dose_budget_sv if dose_budget_sv is not None else radiation.eva_dose_budget_sv(
        cfg.radiation, None, start_time_s)
    checker = ConstraintChecker(evaluator, maps, dose_budget_sv=budget, spe_watch=spe_watch)
    start = evaluator.initial_label(base, start_time_s, resources, phys)
    best, diagnostics = ParetoAStar(evaluator, checker).explore(start, max_labels_per_cell=1)

    shape = (evaluator.rows, evaluator.cols)
    arrival = np.full(shape, math.inf)
    distance = np.full(shape, math.inf)
    reachable = np.zeros(shape, dtype=np.bool_)
    round_trip = np.zeros(shape, dtype=np.bool_)
    farthest = 0.0
    for cell, label in best.items():
        arrival[cell] = label.t_s - start_time_s
        distance[cell] = label.distance_m
        reachable[cell] = True
        after = label
        if task_duration_s > 0.0:
            work = evaluator.stationary(cell, label, task_duration_s, task)
            after = evaluator.advance(label, cell, work, "task")
        if checker.violated(after) is None:
            round_trip[cell] = True
            farthest = max(farthest, label.distance_m)
    isochrones = {limit: round_trip & (arrival <= limit) for limit in cfg.routing.isochrones_s}
    return ReachabilityResult(arrival_time_s=arrival, distance_m=distance, reachable=reachable,
                              round_trip_ok=round_trip, isochrones=isochrones,
                              point_of_no_return=_outer_edge(round_trip),
                              max_round_trip_distance_m=farthest, diagnostics=diagnostics)
