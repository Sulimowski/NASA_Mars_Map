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
from typing import Iterable

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


def build_edge_cost_cubes(evaluator: EdgeEvaluator, pace: Pace, t_s: float,
                          fatigue_level: float) -> EdgeCostCubes:
    """Evaluate every edge of the grid once for a reference state.

    Args:
        pace: nominal or emergency.
        t_s: mission time the environment is sampled at.
        fatigue_level: whole-body fatigue of the reference crew member.
    """
    rows, cols = evaluator.rows, evaluator.cols
    shape = (len(NEIGHBOR_OFFSETS), rows, cols)
    time_s = np.full(shape, math.inf)
    o2_kg = np.full(shape, math.inf)
    energy_j = np.full(shape, math.inf)
    risk = np.full(shape, math.inf)
    length_m = np.full(shape, math.inf)
    state = evaluator.standard_label(t_s, fatigue_level)
    for row in range(rows):
        for col in range(cols):
            if not evaluator.is_walkable((row, col)):
                continue
            for direction, (d_row, d_col, factor) in enumerate(NEIGHBOR_OFFSETS):
                nb = (row + d_row, col + d_col)
                if not evaluator.in_bounds(nb):
                    continue
                tr = evaluator.walk_edge((row, col), nb, factor * evaluator.cell_size_m, state, pace)
                if tr is None:
                    continue
                time_s[direction, row, col] = tr.dt_s
                o2_kg[direction, row, col] = tr.d_o2_kg
                energy_j[direction, row, col] = tr.energy_j
                risk[direction, row, col] = tr.risk
                length_m[direction, row, col] = tr.distance_m
    return EdgeCostCubes(time_s=time_s, o2_kg=o2_kg, energy_j=energy_j, risk=risk,
                         length_m=length_m)


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
