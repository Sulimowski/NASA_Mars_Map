"""B2: static cost cubes and "how far is safety" maps.

Edge-cost cubes
---------------
``cube[d, r, c]`` is the cost of walking from cell ``(r, c)`` to its neighbour in
direction ``d`` (index into ``NEIGHBOR_OFFSETS``), for a reference state: full
suit, fixed fatigue, one instant. Impassable edges are ``inf``. Cubes exist for
time, oxygen, metabolic energy and fall risk.

Maps
----
Multi-source Dijkstra over **reversed** edges: the value at a cell is the cheapest
cost of walking from that cell to the nearest source. Edges are reversed because
uphill and downhill cost differently.

    return_o2_kg      least oxygen to walk back to the base, nominal pace
    return_time_s     least time to walk back to the base, nominal pace
    shelter_time_s    least time to the nearest shelter, nominal pace
    emergency_time_s  least time to the nearest shelter, emergency pace
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from ..core.exceptions import InfeasibleStateError
from ..core.schemas import Cell, Pace
from ..providers.interfaces import NEIGHBOR_OFFSETS
from .transition import EdgeEvaluator

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class EdgeCostCubes:
    """Per-edge costs for a reference state, each of shape ``[8, rows, cols]``."""

    time_s: FloatArray
    o2_kg: FloatArray
    energy_j: FloatArray
    risk: FloatArray
    length_m: FloatArray


@dataclass(slots=True)
class ShelterMaps:
    """Rasters of the cost of reaching safety from every cell (``inf`` = unreachable)."""

    return_o2_kg: FloatArray
    return_time_s: FloatArray
    shelter_time_s: FloatArray
    emergency_time_s: FloatArray
    nominal: EdgeCostCubes
    base: Cell


def build_metric_cubes(model: Any, state: Any, **metric_args: Any) -> dict[str, FloatArray]:
    """Evaluate every edge of the grid once and collect the named costs.

    Works for any ``TraversalModel``: the model's ``edge_metrics`` decides which
    costs exist. Each returned cube is ``[8, rows, cols]``; refused edges are ``inf``.

    Args:
        model: an astronaut's or a rover's evaluator.
        state: the reference state every edge is priced for.
        metric_args: extra keyword arguments passed to ``edge_metrics``.
    """
    rows, cols = model.rows, model.cols
    shape = (len(NEIGHBOR_OFFSETS), rows, cols)
    cubes: dict[str, FloatArray] = {}
    for row in range(rows):
        for col in range(cols):
            if not model.is_walkable((row, col)):
                continue
            for direction, (d_row, d_col, factor) in enumerate(NEIGHBOR_OFFSETS):
                nb = (row + d_row, col + d_col)
                if not model.in_bounds(nb):
                    continue
                metrics = model.edge_metrics((row, col), nb, factor * model.cell_size_m, state,
                                             **metric_args)
                if metrics is None:
                    continue
                for name, value in metrics.items():
                    if name not in cubes:
                        cubes[name] = np.full(shape, math.inf)
                    cubes[name][direction, row, col] = value
    return cubes


def build_edge_cost_cubes(evaluator: EdgeEvaluator, pace: Pace, t_s: float,
                          fatigue_level: float) -> EdgeCostCubes:
    """Evaluate every edge of the grid once for a reference state.

    Args:
        pace: nominal or emergency.
        t_s: mission time the environment is sampled at.
        fatigue_level: whole-body fatigue of the reference crew member.
    """
    state = evaluator.standard_label(t_s, fatigue_level)
    cubes = build_metric_cubes(evaluator, state, pace=pace)
    shape = (len(NEIGHBOR_OFFSETS), evaluator.rows, evaluator.cols)

    def cube(name: str) -> FloatArray:
        return cubes[name] if name in cubes else np.full(shape, math.inf)

    return EdgeCostCubes(time_s=cube("time_s"), o2_kg=cube("o2_kg"), energy_j=cube("energy_j"),
                         risk=cube("risk"), length_m=cube("length_m"))


def reverse_dijkstra(weight: FloatArray, sources: Iterable[Cell]) -> FloatArray:
    """Cheapest cost from every cell to the nearest source.

    Args:
        weight: ``[8, rows, cols]`` edge costs, ``inf`` for impassable edges.
        sources: destination cells (cost 0).

    Returns:
        ``[rows, cols]`` array; ``inf`` where no source can be reached.
    """
    _, rows, cols = weight.shape
    w: list[list[list[float]]] = weight.tolist()
    dist = [[math.inf] * cols for _ in range(rows)]
    heap: list[tuple[float, int, int]] = []
    for row, col in sources:
        dist[row][col] = 0.0
        heap.append((0.0, row, col))
    heapq.heapify(heap)
    inf = math.inf
    while heap:
        d, row, col = heapq.heappop(heap)
        if d > dist[row][col]:
            continue
        for direction, (d_row, d_col, _) in enumerate(NEIGHBOR_OFFSETS):
            u_row, u_col = row - d_row, col - d_col      # the cell that steps INTO (row, col)
            if 0 <= u_row < rows and 0 <= u_col < cols:
                step = w[direction][u_row][u_col]
                if step == inf:
                    continue
                candidate = d + step
                if candidate < dist[u_row][u_col]:
                    dist[u_row][u_col] = candidate
                    heapq.heappush(heap, (candidate, u_row, u_col))
    return np.asarray(dist, dtype=np.float64)


def compute_shelter_maps(evaluator: EdgeEvaluator, base: Cell, t_s: float) -> ShelterMaps:
    """Build the four safety maps for an EVA starting near mission time ``t_s``.

    Shelters are the cells flagged ``is_shelter`` in the terrain plus the base.

    Raises:
        InfeasibleStateError: the base cell is not walkable.
    """
    evaluator.require_walkable(base, "base")
    fatigue_level = evaluator.config.routing.standard_fatigue
    nominal = build_edge_cost_cubes(evaluator, Pace.NOMINAL, t_s, fatigue_level)
    emergency = build_edge_cost_cubes(evaluator, Pace.EMERGENCY, t_s, fatigue_level)
    shelter_rows, shelter_cols = np.nonzero(evaluator.terrain.is_shelter)
    shelters: set[Cell] = {(int(r), int(c)) for r, c in zip(shelter_rows, shelter_cols)
                           if evaluator.is_walkable((int(r), int(c)))}
    shelters.add(base)
    if not shelters:
        raise InfeasibleStateError("no walkable shelter on the site")
    return ShelterMaps(
        return_o2_kg=reverse_dijkstra(nominal.o2_kg, [base]),
        return_time_s=reverse_dijkstra(nominal.time_s, [base]),
        shelter_time_s=reverse_dijkstra(nominal.time_s, shelters),
        emergency_time_s=reverse_dijkstra(emergency.time_s, shelters),
        nominal=nominal, base=base)
