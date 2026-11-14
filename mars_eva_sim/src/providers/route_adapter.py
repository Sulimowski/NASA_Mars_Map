"""Input boundary for routes produced outside the simulator.

A route may come from the routing team's planner (a ``MissionPlan`` object or
the ``eva-plan/1`` JSON it writes), from the data team (a track of points such
as the Curiosity waypoints) or from a plain list of points. All of them become
one ``RouteInput``: a polyline in site metres plus the stops on it.

``rasterize_route`` then turns the polyline into what the simulator walks: a
chain of 8-connected cells of the simulator's own grid, which may be coarser
than the grid the route was planned on.

Site metres: ``x`` east and ``y`` south of the north-west corner
(``DATA_CONTRACT.md``). Durations are converted to seconds here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..core.config import RouteAdapterConfig
from ..core.constants import HOUR_S, MINUTE_S, O2_DENSITY_STP_KG_M3
from ..core.exceptions import ValidationError
from ..core.schemas import Cell, TaskType
from .interfaces import TerrainProvider

PointM = tuple[float, float]
TaskChooser = Callable[[str], TaskType]

EVA_PLAN_SCHEMA = "eva-plan/1"


@dataclass(frozen=True, slots=True)
class RouteStop:
    """A place on the route where the crew works.

    Attributes:
        name: label of the stop (target name).
        x_m, y_m: position in site metres.
        dwell_s: working time, s.
        task: kind of work, which sets the metabolic power and the hand load.
    """

    name: str
    x_m: float
    y_m: float
    dwell_s: float
    task: TaskType = TaskType.SAMPLING


@dataclass(frozen=True, slots=True)
class RouteInput:
    """A route to evaluate.

    Attributes:
        points_m: polyline in site metres, in travel order.
        stops: work stops, in travel order.
        source: where the route came from (for reports).
        reference: totals claimed by the route's author, in SI, for comparison:
            any of ``time_s``, ``distance_m``, ``o2_kg``, ``dose_sv``, ``energy_j``.
    """

    points_m: tuple[PointM, ...]
    stops: tuple[RouteStop, ...] = ()
    source: str = "points"
    reference: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.points_m) < 1:
            raise ValidationError("a route needs at least one point")
        for x_m, y_m in self.points_m:
            if not (math.isfinite(x_m) and math.isfinite(y_m)):
                raise ValidationError(f"route point ({x_m}, {y_m}) is not finite")
        for stop in self.stops:
            if stop.dwell_s < 0.0 or not math.isfinite(stop.dwell_s):
                raise ValidationError(f"stop '{stop.name}' has an invalid dwell time")


def default_task(name: str) -> TaskType:
    """Task assumed at a stop when the route does not say: sample collection."""
    return TaskType.SAMPLING


def _points(raw: Iterable[Any]) -> tuple[PointM, ...]:
    """Points from pairs, mappings or objects with ``x_m`` / ``y_m``; repeats dropped."""
    out: list[PointM] = []
    for item in raw:
        if isinstance(item, Mapping):
            point = (float(item["x_m"]), float(item["y_m"]))
        elif hasattr(item, "x_m") and hasattr(item, "y_m"):
            point = (float(item.x_m), float(item.y_m))
        else:
            x_m, y_m = item
            point = (float(x_m), float(y_m))
        if not out or out[-1] != point:
            out.append(point)
    return tuple(out)


def route_from_points(points_m: Iterable[Any], stops: Sequence[RouteStop] = (),
                      source: str = "points") -> RouteInput:
    """Route from a list of ``(x_m, y_m)`` pairs (or objects / dicts with ``x_m``, ``y_m``)."""
    return RouteInput(points_m=_points(points_m), stops=tuple(stops), source=source)


def route_from_track(waypoints: Iterable[Any], source: str = "track") -> RouteInput:
    """Route from a recorded track, for example the data team's Curiosity waypoints.

    Each waypoint needs ``x_m`` and ``y_m`` (attribute or key). No stops are assumed.
    """
    return RouteInput(points_m=_points(waypoints), source=source)


def _stops_from_dicts(stops: Iterable[Mapping[str, Any]], task_for: TaskChooser
                      ) -> tuple[RouteStop, ...]:
    out: list[RouteStop] = []
    for stop in stops:
        if stop.get("kind", "target") != "target":
            continue
        name = str(stop.get("name", "stop"))
        out.append(RouteStop(name=name, x_m=float(stop["x_m"]), y_m=float(stop["y_m"]),
                             dwell_s=float(stop.get("dwell_min", 0.0)) * MINUTE_S,
                             task=task_for(name)))
    return tuple(out)


def route_from_mission_plan(plan: Any, task_for: TaskChooser = default_task) -> RouteInput:
    """Route from the routing team's ``MissionPlan`` object.

    Reads ``segments[*].points_m`` (the walked samples), ``stops`` and the totals.

    Raises:
        ValidationError: the plan has no segments.
    """
    segments = list(getattr(plan, "segments", []) or [])
    if not segments:
        raise ValidationError("the mission plan has no route segments")
    raw: list[Any] = []
    for segment in segments:
        raw.extend(segment.points_m)
    reference = {
        "time_s": float(plan.total_time_hours) * HOUR_S,
        "distance_m": float(plan.total_distance_km) * 1000.0,
        "dose_sv": float(plan.total_dose_msv) * 1.0e-3,
        "o2_kg": float(getattr(plan, "total_o2_liters", 0.0)) * 1.0e-3 * O2_DENSITY_STP_KG_M3,
        "energy_j": float(getattr(plan, "total_energy_wh", 0.0)) * HOUR_S,
    }
    return RouteInput(points_m=_points(raw),
                      stops=_stops_from_dicts(getattr(plan, "stops", []) or [], task_for),
                      source=f"mission_plan:{getattr(plan, 'strategy', 'custom')}",
                      reference=reference)


def route_from_eva_plans(data: Mapping[str, Any] | str | Path, strategy: str,
                         task_for: TaskChooser = default_task) -> RouteInput:
    """Route of one strategy from an ``eva-plan/1`` document (a dict or a JSON file).

    Raises:
        ValidationError: wrong schema, unknown strategy, or a plan without waypoints.
    """
    if not isinstance(data, Mapping):
        try:
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValidationError(f"cannot read EVA plans from {data}: {error}") from None
    assert isinstance(data, Mapping)
    if data.get("schema") != EVA_PLAN_SCHEMA:
        raise ValidationError(f"expected schema '{EVA_PLAN_SCHEMA}', got {data.get('schema')!r}")
    plans = data.get("strategies", {})
    if strategy not in plans:
        raise ValidationError(f"strategy '{strategy}' not in the document "
                              f"(available: {sorted(plans)})")
    plan = plans[strategy]
    waypoints = plan.get("waypoints") or []
    if not waypoints:
        raise ValidationError(f"strategy '{strategy}' has no waypoints (nothing to walk)")
    summary = plan.get("summary", {})
    reference = {
        "time_s": float(summary.get("total_time_hours", 0.0)) * HOUR_S,
        "distance_m": float(summary.get("total_distance_km", 0.0)) * 1000.0,
        "dose_sv": float(summary.get("total_dose_msv", 0.0)) * 1.0e-3,
        "o2_kg": float(summary.get("o2_used_liters", 0.0)) * 1.0e-3 * O2_DENSITY_STP_KG_M3,
        "energy_j": float(summary.get("energy_used_wh", 0.0)) * HOUR_S,
    }
    return RouteInput(points_m=_points(waypoints),
                      stops=_stops_from_dicts(plan.get("stops") or [], task_for),
                      source=f"eva_plans:{strategy}", reference=reference)


def cell_of(terrain: TerrainProvider, x_m: float, y_m: float) -> Cell:
    """Cell holding a point in site metres.

    Raises:
        ValidationError: the point lies outside the terrain.
    """
    rows, cols = terrain.shape
    size = terrain.cell_size_m
    row, col = math.floor(y_m / size), math.floor(x_m / size)
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValidationError(f"point ({x_m:.1f}, {y_m:.1f}) m is outside the terrain "
                              f"({cols * size:.0f} x {rows * size:.0f} m)")
    return row, col


def polyline_length_m(points_m: Sequence[PointM]) -> float:
    """Horizontal length of a polyline, m."""
    return sum(math.hypot(x1 - x0, y1 - y0)
               for (x0, y0), (x1, y1) in zip(points_m, points_m[1:]))


def simplify_polyline(points_m: Sequence[PointM], tolerance_m: float) -> list[PointM]:
    """Drop the points that lie within ``tolerance_m`` of the line through their neighbours.

    Ramer-Douglas-Peucker, iterative. A densely sampled route that runs along a
    cell boundary would otherwise rasterize into a chain hopping between two
    rows of cells, which is longer than the route.
    """
    count = len(points_m)
    if count < 3 or tolerance_m <= 0.0:
        return list(points_m)
    keep = [False] * count
    keep[0] = keep[-1] = True
    stack = [(0, count - 1)]
    while stack:
        first, last = stack.pop()
        (x0, y0), (x1, y1) = points_m[first], points_m[last]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        worst, worst_dist = -1, tolerance_m
        for index in range(first + 1, last):
            px, py = points_m[index]
            if length > 0.0:
                dist = abs(dy * (px - x0) - dx * (py - y0)) / length
            else:
                dist = math.hypot(px - x0, py - y0)
            if dist > worst_dist:
                worst, worst_dist = index, dist
        if worst >= 0:
            keep[worst] = True
            stack.append((first, worst))
            stack.append((worst, last))
    return [point for point, kept in zip(points_m, keep) if kept]


def _line(a: Cell, b: Cell) -> list[Cell]:
    """8-connected cells from ``a`` (excluded) to ``b`` (included), Bresenham."""
    (r0, c0), (r1, c1) = a, b
    d_row, d_col = abs(r1 - r0), abs(c1 - c0)
    s_row, s_col = (1 if r1 >= r0 else -1), (1 if c1 >= c0 else -1)
    cells: list[Cell] = []
    error = d_col - d_row
    row, col = r0, c0
    while (row, col) != (r1, c1):
        twice = 2 * error
        if twice > -d_row:
            error -= d_row
            col += s_col
        if twice < d_col:
            error += d_col
            row += s_row
        cells.append((row, col))
    return cells


def rasterize_route(terrain: TerrainProvider, points_m: Sequence[PointM]) -> list[Cell]:
    """Chain of 8-connected cells that follows a polyline.

    The polyline is first simplified to within half a cell, then each straight
    piece becomes a straight line of cells. Consecutive repeats collapse.

    Raises:
        ValidationError: the polyline is empty or leaves the terrain.
    """
    if not points_m:
        raise ValidationError("cannot rasterize an empty route")
    for x_m, y_m in points_m:
        cell_of(terrain, x_m, y_m)                       # every point must be on the terrain
    points_m = simplify_polyline(points_m, 0.5 * terrain.cell_size_m)
    cells: list[Cell] = [cell_of(terrain, *points_m[0])]
    for x_m, y_m in points_m[1:]:
        target = cell_of(terrain, x_m, y_m)
        if target != cells[-1]:
            cells.extend(_line(cells[-1], target))
    return cells


def stop_indices(terrain: TerrainProvider, cells: Sequence[Cell], stops: Sequence[RouteStop],
                 cfg: RouteAdapterConfig) -> dict[int, tuple[TaskType, float]]:
    """Index on the cell chain where each stop is worked.

    Stops are matched in travel order: each one to the nearest cell at or after
    the previous stop. Two stops landing on the same cell are merged (durations
    add, the first task wins).

    Returns:
        ``{index in cells: (task, dwell in s)}`` as ``simulate_path`` expects.

    Raises:
        ValidationError: a stop is farther than ``stop_snap_distance_m`` from the route.
    """
    size = terrain.cell_size_m
    out: dict[int, tuple[TaskType, float]] = {}
    search_from = 0
    for stop in stops:
        best_index, best_dist = -1, math.inf
        for index in range(search_from, len(cells)):
            row, col = cells[index]
            dist = math.hypot((col + 0.5) * size - stop.x_m, (row + 0.5) * size - stop.y_m)
            if dist < best_dist:
                best_index, best_dist = index, dist
        if best_index < 0 or best_dist > max(cfg.stop_snap_distance_m, size):
            raise ValidationError(f"stop '{stop.name}' is {best_dist:.0f} m from the route "
                                  f"(limit {max(cfg.stop_snap_distance_m, size):.0f} m)")
        if best_index in out:
            task, dwell = out[best_index]
            out[best_index] = (task, dwell + stop.dwell_s)
        else:
            out[best_index] = (stop.task, stop.dwell_s)
        search_from = best_index
    return out
