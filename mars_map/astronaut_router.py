"""EVA route planner for an astronaut on foot.

    router = EvaRouter(site, terrain)          # terrain from build_astronaut_terrain()
    plan   = router.plan("science")            # one mission plan
    plans  = router.plan_pareto()              # shielded / eco / science
    escape = router.shelter((x_m, y_m))        # nearest natural radiation shelter

How a plan is made:

1. Coarse graph. The site is reduced to ~200 x 200 cells. An edge exists only if
   the straight line between two coarse cells crosses no impassable fine cell.
   One Dijkstra per waypoint gives the cost of every leg (time, distance, O2,
   battery, dose) for all waypoint pairs.
2. Target choice. Every order of every subset of targets that fits the budgets,
   return leg included, is scored; the best one wins. Priority dominates, then
   science score, then the resource the strategy wants to save.
3. Fine legs. Each chosen leg is searched again at full resolution with weighted
   A*, inside a corridor around its coarse path.
4. String pulling. Grid corners are replaced by straight lines wherever the line
   stays on passable ground and does not cost more than the piece it replaces.
5. Report. The smoothed route is walked sample by sample with the same physics
   the search used (biomechanics.locomotion).

Units: metres, seconds inside; mSv and mSv/h for radiation; litres of O2; Wh.
"""

from __future__ import annotations

import heapq
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mars_map.astronaut_terrain import AstronautTerrainModel, build_astronaut_terrain
    from mars_map.biomechanics import (EARTH_GRAVITY, MARS_GRAVITY, SuitParams, dwell,
                                       locomotion)
    from mars_map.radio import RadioParams, link_quality
    from mars_map.site import MissionParams, Site, Target
else:
    from .astronaut_terrain import AstronautTerrainModel, build_astronaut_terrain
    from .biomechanics import EARTH_GRAVITY, MARS_GRAVITY, SuitParams, dwell, locomotion
    from .radio import RadioParams, link_quality
    from .site import MissionParams, Site, Target

# Kept for code that imported these from here; the values live in SuitParams.
ASTRONAUT_MASS_KG = SuitParams.body_mass_kg
SUIT_MASS_KG = SuitParams.suit_mass_kg
O2_TANK_LITERS = SuitParams.o2_tank_l
BATTERY_WH = SuitParams.battery_wh

INF = math.inf
_SQRT2 = math.sqrt(2.0)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
_STEP = tuple(_SQRT2 if dr and dc else 1.0 for dr, dc in _DIRS)
# At a typical 300 W one litre of O2 lasts about 67 s; this puts oxygen on the time scale.
_O2_SECONDS_PER_LITRE = 67.0
RESOURCES = ("time_s", "distance_m", "o2_l", "energy_wh", "dose_msv")


# ----------------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Strategy:
    """What a route tries to minimise and how much of the limits a plan may use."""
    name: str
    title: str = ""
    time_weight: float = 1.0        # cost per second of walking
    o2_weight: float = 0.0          # cost per litre of O2 (on the time scale)
    looseness_weight: float = 0.3   # extra cost on loose or rocky ground
    steep_weight: float = 0.15      # extra cost per (degree above max_slope_deg)^2
    nodata_weight: float = 1.0      # how much the coarse-data penalty counts
    rad_power: float = 0.0          # cost *= (dose rate / open-ground rate) ** rad_power
    rad_linear: float = 0.0         # cost += rad_linear * dose rate (mSv/h) * seconds
    budget_use: float = 1.0         # share of every limit the plan may use
    saving: str = ""                # "", "o2" or "dose": the resource the plan saves
    saving_weight: float = 0.0      # value an optional target must bring per 100 % of it
    science_weight: float = 1.0     # weight of science_score next to the target priority

    @property
    def routing(self) -> tuple:
        """The part that shapes a route; strategies equal here share searches and legs."""
        return (self.time_weight, self.o2_weight, self.looseness_weight, self.steep_weight,
                self.nodata_weight, self.rad_power, self.rad_linear)


STRATEGIES: dict[str, Strategy] = {
    "shielded": Strategy("shielded", "Maximum radiation shelter", rad_power=4.0,
                         budget_use=0.85, saving="dose", saving_weight=40.0),
    "eco": Strategy("eco", "Minimum oxygen and energy", time_weight=0.25, o2_weight=1.0,
                    looseness_weight=1.0, steep_weight=0.3, budget_use=0.85,
                    saving="o2", saving_weight=40.0),
    "science": Strategy("science", "Maximum science", science_weight=5.0),
}
# Leg costs for target choice are measured on the fastest sensible route.
_NOMINAL = Strategy("nominal")
_FASTEST = Strategy("fastest", looseness_weight=0.0, steep_weight=0.0, nodata_weight=0.0)


def priority_value(priority: int) -> float:
    """1 -> 100, 2 -> 10, 3 -> 1: a higher priority always outweighs lower ones."""
    return 100.0 * 0.1 ** (max(int(priority), 1) - 1)


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
@dataclass
class RouteSegment:
    points_m: list[tuple[float, float]]   # route samples [(x_m, y_m), ...], about one per cell
    pixels: list[tuple[int, int]]         # the same samples as [(row, col), ...]
    distance_m: float
    elevation_gain_m: float
    elevation_drop_m: float
    max_slope_deg: float
    mean_looseness: float
    duration_min: float
    o2_liters: float
    energy_wh: float
    dose_msv: float
    profile: list[dict] = field(default_factory=list)
    name_from: str = ""
    name_to: str = ""
    vertices_m: list[tuple[float, float]] = field(default_factory=list)   # corners after smoothing

    def resources(self) -> list[float]:
        return [self.duration_min * 60.0, self.distance_m, self.o2_liters, self.energy_wh,
                self.dose_msv]

    def to_dict(self) -> dict:
        return {
            "from": self.name_from, "to": self.name_to,
            "distance_m": round(self.distance_m, 1),
            "duration_min": round(self.duration_min, 1),
            "elevation_gain_m": round(self.elevation_gain_m, 1),
            "elevation_drop_m": round(self.elevation_drop_m, 1),
            "max_slope_deg": round(self.max_slope_deg, 1),
            "mean_looseness": round(self.mean_looseness, 3),
            "o2_liters": round(self.o2_liters, 2),
            "energy_wh": round(self.energy_wh, 1),
            "dose_msv": round(self.dose_msv, 4),
            "vertices_m": [[round(x, 1), round(y, 1)] for x, y in self.vertices_m],
        }


@dataclass
class MissionPlan:
    feasible: bool
    warnings: list[str]
    visited_targets: list[str]
    segments: list[RouteSegment]
    total_distance_km: float
    total_elevation_gain_m: float
    max_slope_deg: float
    total_time_hours: float
    o2_remaining_pct: float
    battery_remaining_pct: float
    total_dose_msv: float
    # --- added with target choice, strategies and the web contract ---
    strategy: str = "custom"
    title: str = ""
    skipped_targets: list[dict] = field(default_factory=list)
    stops: list[dict] = field(default_factory=list)
    science_value: float = 0.0
    total_o2_liters: float = 0.0
    total_energy_wh: float = 0.0
    limits: dict = field(default_factory=dict)
    usage_pct: dict = field(default_factory=dict)
    radio: dict = field(default_factory=dict)
    compute_s: float = 0.0
    site: Site | None = field(default=None, repr=False, compare=False)
    suit: SuitParams | None = field(default=None, repr=False, compare=False)

    def waypoints(self, step_m: float = 6.0) -> list[dict]:
        """Route samples of the whole plan with running totals, thinned to about step_m."""
        suit = self.suit or SuitParams()
        out: list[dict] = []
        base = dict.fromkeys(("dist_m", "t_min", "o2_l", "energy_wh", "dose_msv"), 0.0)
        radio = self.radio.get("samples", [])
        for leg_i, seg in enumerate(self.segments):
            last_kept = -INF
            for i, p in enumerate(seg.profile):
                is_end = i in (0, len(seg.profile) - 1)
                if not is_end and p["dist_m"] - last_kept < step_m:
                    continue
                last_kept = p["dist_m"]
                o2 = base["o2_l"] + p["o2_l"]
                wh = base["energy_wh"] + p["energy_wh"]
                w = {
                    "leg": leg_i,
                    "x_m": p["x_m"], "y_m": p["y_m"], "elev_m": p["elev_m"],
                    "dist_m": round(base["dist_m"] + p["dist_m"], 1),
                    "t_min": round(base["t_min"] + p["t_min"], 2),
                    "slope_deg": p["slope_deg"],
                    "looseness": p["looseness"],
                    "sinkage_cm": p["sinkage_cm"],
                    "terrain_class": p["terrain_class"],
                    "dose_rate_usv_h": p["dose_rate_usv_h"],
                    "dose_msv": round(base["dose_msv"] + p["dose_msv"], 4),
                    "temp_c": p["temp_c"],
                    "o2_pct": round(100.0 * (1.0 - o2 / suit.o2_tank_l), 1),
                    "battery_pct": round(100.0 * (1.0 - wh / suit.battery_wh), 1),
                }
                if self.site is not None:
                    lat, lon = self.site.to_latlon(p["x_m"], p["y_m"])
                    w["lat"], w["lon"] = round(lat, 6), round(lon, 6)
                if radio:
                    w["radio"] = _radio_at(radio, w["dist_m"])
                out.append(w)
            # Carry the totals over the leg and the work at its end.
            stop = self.stops[leg_i + 1] if leg_i + 1 < len(self.stops) else None
            base["dist_m"] += seg.distance_m
            if stop is not None:
                base["t_min"] = stop["depart_min"]
                base["o2_l"] = stop["o2_l"]
                base["energy_wh"] = stop["energy_wh"]
                base["dose_msv"] = stop["dose_msv"]
            else:
                base["t_min"] += seg.duration_min
                base["o2_l"] += seg.o2_liters
                base["energy_wh"] += seg.energy_wh
                base["dose_msv"] += seg.dose_msv
        return out

    def to_dict(self, waypoint_step_m: float = 6.0) -> dict:
        return {
            "strategy": self.strategy,
            "title": self.title,
            "feasible": self.feasible,
            "warnings": self.warnings,
            "visited_targets": self.visited_targets,
            "skipped_targets": self.skipped_targets,
            "summary": {
                "total_distance_km": round(self.total_distance_km, 3),
                "total_elevation_gain_m": round(self.total_elevation_gain_m, 1),
                "max_slope_deg": round(self.max_slope_deg, 1),
                "total_time_hours": round(self.total_time_hours, 2),
                "o2_used_liters": round(self.total_o2_liters, 1),
                "o2_remaining_pct": round(self.o2_remaining_pct, 1),
                "energy_used_wh": round(self.total_energy_wh, 1),
                "battery_remaining_pct": round(self.battery_remaining_pct, 1),
                "total_dose_msv": round(self.total_dose_msv, 3),
                "mean_dose_rate_usv_h": round(self.total_dose_msv * 1000.0
                                              / max(self.total_time_hours, 1e-9), 2),
                "science_value": round(self.science_value, 2),
                "targets_visited": len(self.visited_targets),
            },
            "limits": self.limits,
            "usage_pct": self.usage_pct,
            "stops": self.stops,
            "legs": [s.to_dict() for s in self.segments],
            "radio": {k: v for k, v in self.radio.items() if k != "samples"},
            "waypoints": self.waypoints(waypoint_step_m),
            "compute_s": round(self.compute_s, 3),
        }

    def to_json(self, path: str | Path, waypoint_step_m: float = 6.0) -> Path:
        """Write the plan as JSON for the frontend."""
        out = Path(path)
        out.write_text(json.dumps(self.to_dict(waypoint_step_m), indent=2, ensure_ascii=False),
                       encoding="utf-8")
        return out


@dataclass
class ShelterRoute:
    """Escape route to the best natural radiation shelter within reach."""
    segment: RouteSegment
    shelter_m: tuple[float, float]
    eta_min: float
    dose_rate_here_msv_h: float
    dose_rate_shelter_msv_h: float
    reduction_pct: float            # how much lower the dose rate is in the shelter
    worthwhile: bool                # False: nothing within reach is meaningfully better
    shelter_hours: float
    dose_if_stay_msv: float         # staying put for shelter_hours
    dose_if_shelter_msv: float      # walking there, then waiting the rest of shelter_hours
    compute_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "shelter": {"x_m": round(self.shelter_m[0], 1), "y_m": round(self.shelter_m[1], 1)},
            "eta_min": round(self.eta_min, 1),
            "distance_m": round(self.segment.distance_m, 1),
            "dose_rate_here_usv_h": round(self.dose_rate_here_msv_h * 1000, 2),
            "dose_rate_shelter_usv_h": round(self.dose_rate_shelter_msv_h * 1000, 2),
            "reduction_pct": round(self.reduction_pct, 1),
            "worthwhile": self.worthwhile,
            "shelter_hours": self.shelter_hours,
            "dose_if_stay_msv": round(self.dose_if_stay_msv, 4),
            "dose_if_shelter_msv": round(self.dose_if_shelter_msv, 4),
            "vertices_m": [[round(x, 1), round(y, 1)] for x, y in self.segment.vertices_m],
            "compute_s": round(self.compute_s, 3),
        }


def _radio_at(samples: list[dict], dist_m: float) -> str:
    """Radio status of the checked sample nearest to a distance along the route."""
    lo, hi = 0, len(samples) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if samples[mid]["dist_m"] < dist_m:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0 and dist_m - samples[lo - 1]["dist_m"] < samples[lo]["dist_m"] - dist_m:
        lo -= 1
    return samples[lo]["status"]


# ----------------------------------------------------------------------------
# Graph: edge costs and searches
# ----------------------------------------------------------------------------
class _Grid:
    """Rasters of one search grid (the whole site, a window of it, or the coarse copy)."""
    __slots__ = ("elev", "eta", "speed", "loose", "pen", "rate", "heater", "imp", "res")

    def __init__(self, elev, eta, speed, loose, pen, rate, heater, imp, res):
        self.elev, self.eta, self.speed, self.loose = elev, eta, speed, loose
        self.pen, self.rate, self.heater, self.imp, self.res = pen, rate, heater, imp, res

    def window(self, r0, r1, c0, c1, extra_block=None) -> "_Grid":
        s = (slice(r0, r1), slice(c0, c1))
        imp = self.imp[s] if extra_block is None else self.imp[s] | extra_block
        return _Grid(self.elev[s], self.eta[s], self.speed[s], self.loose[s], self.pen[s],
                     self.rate[s], self.heater[s], imp, self.res)


def _generalized_cost(strat: Strategy, t, o2, loose, pen, rate, slope_abs_deg, rate_ref: float,
                      mission: MissionParams):
    """What the search minimises for one step; the single definition of route quality."""
    cost = strat.time_weight * t + strat.o2_weight * _O2_SECONDS_PER_LITRE * o2
    cost = cost * (1.0 + strat.looseness_weight * loose) * (1.0 + strat.nodata_weight * (pen - 1.0))
    over = np.maximum(slope_abs_deg - mission.max_slope_deg, 0.0)
    cost = cost * (1.0 + strat.steep_weight * over * over)
    if strat.rad_power:
        cost = cost * (rate / rate_ref) ** strat.rad_power
    if strat.rad_linear:
        cost = cost + strat.rad_linear * rate * t
    return cost


def _edge_arrays(g: _Grid, strat: Strategy, mission: MissionParams, suit: SuitParams,
                 rate_ref: float, edge_blocked: np.ndarray | None = None,
                 resources: bool = False):
    """Cost of stepping from every cell to each of its 8 neighbours.

    Returns cost (8, R, C) with inf for forbidden steps, and when asked the
    physical resources (5, 8, R, C) in the order of RESOURCES.
    """
    R, C = g.elev.shape
    cost = np.full((8, R, C), INF)
    res = np.zeros((5, 8, R, C), np.float32) if resources else None
    hard = mission.hard_max_slope_deg
    for k, (dr, dc) in enumerate(_DIRS):
        rs, rd = slice(max(0, -dr), R - max(0, dr)), slice(max(0, dr), R + min(0, dr))
        cs, cd = slice(max(0, -dc), C - max(0, dc)), slice(max(0, dc), C + min(0, dc))
        src, dst = (rs, cs), (rd, cd)
        d = g.res * _STEP[k]
        dh = g.elev[dst] - g.elev[src]
        t, _, o2, wh = locomotion(d, dh, g.eta[dst], g.speed[dst], mission, suit, g.heater[dst])
        slope = np.degrees(np.arctan(np.abs(dh) / d))
        c = _generalized_cost(strat, t, o2, g.loose[dst], g.pen[dst], g.rate[dst], slope,
                              rate_ref, mission)
        bad = slope > hard
        if edge_blocked is not None:
            bad |= edge_blocked[k][src]
        else:
            bad |= g.imp[dst]
            if dr and dc:                       # no cutting corners past a blocked cell
                bad |= g.imp[rd, cs] | g.imp[rs, cd]
        c[bad] = INF
        cost[k][src] = c
        if resources:
            res[0, k][src] = t
            res[1, k][src] = d
            res[2, k][src] = o2
            res[3, k][src] = wh
            res[4, k][src] = g.rate[dst] * t / 3600.0
    return cost, res


def _min_cost_per_cell(cost: np.ndarray) -> float:
    """Cheapest cost of moving one cell width anywhere in the grid (for the A* heuristic)."""
    best = INF
    for k in range(8):
        finite = cost[k][np.isfinite(cost[k])]
        if finite.size:
            best = min(best, float(finite.min()) / _STEP[k])
    return best


def _as_lists(cost: np.ndarray) -> tuple:
    """Python lists index several times faster than numpy arrays inside the search loop."""
    return tuple(c.ravel().tolist() for c in cost)


def _dijkstra(costs: tuple, offsets: tuple, start: int, stop_at=None):
    """Cheapest cost from start to every node. Returns (cost list, parent list)."""
    n = len(costs[0])
    g = [INF] * n
    parent = [-1] * n
    g[start] = 0.0
    heap = [(0.0, start)]
    pop, push = heapq.heappop, heapq.heappush
    pairs = tuple(zip(costs, offsets))
    left = set(stop_at) if stop_at else None
    while heap:
        d, u = pop(heap)
        if d > g[u]:
            continue
        if left is not None and u in left:
            left.discard(u)
            if not left:
                break
        for ck, off in pairs:
            c = ck[u]
            if c < INF:
                v = u + off
                nd = d + c
                if nd < g[v]:
                    g[v] = nd
                    parent[v] = u
                    push(heap, (nd, v))
    return g, parent


def _octile(rows: int, cols: int, goal: int, scale: float) -> list:
    """Heuristic for every node: 8-connected distance to the goal in cells, times scale."""
    gr, gc = divmod(goal, cols)
    a = np.abs(np.arange(rows) - gr)[:, None]
    b = np.abs(np.arange(cols) - gc)[None, :]
    return ((np.maximum(a, b) + 0.41421356 * np.minimum(a, b)) * scale).ravel().tolist()


def _astar(costs: tuple, offsets: tuple, start: int, goal: int, h: list):
    """A* on an 8-connected grid with heuristic h. Returns the node list start..goal, or None.

    A node is expanded once: with an inflated heuristic, reopening nodes can blow
    up next to obstacles, and the result stays within the same bound without it.
    """
    if start == goal:
        return [start]
    n = len(costs[0])
    g = [INF] * n
    parent = [-1] * n
    closed = bytearray(n)
    g[start] = 0.0
    heap = [(0.0, 0.0, start)]
    pop, push = heapq.heappop, heapq.heappush
    pairs = tuple(zip(costs, offsets))
    found = False
    while heap:
        _, d, u = pop(heap)
        if closed[u]:
            continue
        if u == goal:
            found = True
            break
        closed[u] = 1
        for ck, off in pairs:
            c = ck[u]
            if c < INF:
                v = u + off
                nd = d + c
                if nd < g[v] and not closed[v]:
                    g[v] = nd
                    parent[v] = u
                    push(heap, (nd + h[v], nd, v))
    if not found:
        return None
    return _backtrack(parent, start, goal)


def _backtrack(parent: list, start: int, goal: int):
    path = [goal]
    while path[-1] != start:
        p = parent[path[-1]]
        if p < 0:
            return None
        path.append(p)
    path.reverse()
    return path


def _snap_to_walkable(row: int, col: int, impassable: np.ndarray, max_radius: int = 15
                      ) -> tuple[int, int]:
    """Nearest passable cell to (row, col), for points placed on a cliff edge or a no-go line."""
    if not impassable[row, col]:
        return row, col
    rows, cols = impassable.shape
    r0, r1 = max(0, row - max_radius), min(rows, row + max_radius + 1)
    c0, c1 = max(0, col - max_radius), min(cols, col + max_radius + 1)
    free = np.argwhere(~impassable[r0:r1, c0:c1])
    if free.size == 0:
        raise ValueError(f"Точка ({row}, {col}) полностью заблокирована непроходимой зоной.")
    best = free[np.argmin((free[:, 0] + r0 - row) ** 2 + (free[:, 1] + c0 - col) ** 2)]
    return int(best[0] + r0), int(best[1] + c0)


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
class EvaRouter:
    """Prepared site: build once, then ask for legs, plans and shelters."""

    def __init__(self, site: Site, terrain: AstronautTerrainModel | None = None,
                 suit: SuitParams | None = None, radio: RadioParams | None = None,
                 coarse_cells: int = 200, corridor_cells: int = 2,
                 heuristic_weight: float = 1.1, smooth_tolerance: float = 0.03,
                 max_span_m: float = 300.0, coarse_margin: float = 1.05):
        self.site, self.mission = site, site.mission
        self.suit = suit or SuitParams()
        self.radio_params = radio or RadioParams()
        self.terrain = terrain or build_astronaut_terrain(site, suit=self.suit)
        self.corridor_cells = corridor_cells
        self.heuristic_weight = heuristic_weight
        self.smooth_tolerance = smooth_tolerance
        self.coarse_margin = coarse_margin

        t = self.terrain
        self.rows, self.cols = site.shape
        self.res = float(site.resolution_m)
        self.max_span = max(2, int(max_span_m / self.res))
        elev = np.ascontiguousarray(site.layers["elevation"], dtype=np.float32)
        self.fine = _Grid(elev, t.terramech_factor, t.speed_multiplier, t.looseness,
                          t.nodata_penalty, t.dose_rate_msv_h, t.heater_w, t.impassable, self.res)
        self.rate_ref = float(np.median(t.dose_rate_msv_h))
        # Flat views for sampling routes.
        self._flat = {k: np.ascontiguousarray(getattr(self.fine, k)).ravel()
                      for k in ("elev", "eta", "speed", "loose", "pen", "rate", "heater", "imp")}
        self._sinkage_cm = (t.sinkage_cm if t.sinkage_cm is not None else t.sinkage * 5.0).ravel()
        self._tclass = site.layers["terrain_class"].ravel()
        self._temp = None if t.ground_temp_c is None else t.ground_temp_c.ravel()
        # Straight shortcuts keep one cell away from anything impassable.
        fat = t.impassable.copy()
        fat[1:] |= t.impassable[:-1]
        fat[:-1] |= t.impassable[1:]
        fat[:, 1:] |= t.impassable[:, :-1]
        fat[:, :-1] |= t.impassable[:, 1:]
        self._flat["fat"] = fat.ravel()

        self._build_coarse(coarse_cells)
        self._coarse_cost: dict[tuple, tuple] = {}
        self._trees: dict[tuple, tuple] = {}
        self._legs: dict[tuple, RouteSegment] = {}
        self._matrices: dict[tuple, list] = {}

    # ---- coarse graph ----------------------------------------------------
    def _build_coarse(self, coarse_cells: int) -> None:
        f = max(1, math.ceil(max(self.rows, self.cols) / coarse_cells))
        self.f = f
        Rc, Cc = math.ceil(self.rows / f), math.ceil(self.cols / f)
        self.Rc, self.Cc = Rc, Cc
        pad = ((0, Rc * f - self.rows), (0, Cc * f - self.cols))

        def block(a, how):
            a = np.pad(a, pad, mode="edge").reshape(Rc, f, Cc, f)
            return getattr(a, how)(axis=(1, 3))

        F = self.fine
        imp = F.imp
        # One fine cell stands for each coarse cell; edges are checked between them.
        rep_r = np.minimum(np.arange(Rc) * f + f // 2, self.rows - 1)
        rep_c = np.minimum(np.arange(Cc) * f + f // 2, self.cols - 1)
        self._rep_r, self._rep_c = rep_r, rep_c
        imp_c = imp[np.ix_(rep_r, rep_c)]

        padded = np.pad(imp, f, mode="constant", constant_values=True)
        R0, C0 = rep_r + f, rep_c + f
        blocked = np.zeros((8, Rc, Cc), bool)
        for k, (dr, dc) in enumerate(_DIRS):
            b = imp_c.copy()
            for s in range(1, f + 1):
                b |= padded[np.ix_(R0 + dr * s, C0 + dc * s)]
                if dr and dc:
                    b |= padded[np.ix_(R0 + dr * s, C0 + dc * (s - 1))]
                    b |= padded[np.ix_(R0 + dr * (s - 1), C0 + dc * s)]
            blocked[k] = b
        self._edge_blocked = blocked
        self._c_valid = ~imp_c & ~blocked.all(axis=0)

        self.coarse = _Grid(
            block(F.elev, "mean").astype(np.float32), block(F.eta, "mean"), block(F.speed, "mean"),
            block(F.loose, "mean"), block(F.pen, "mean"), block(F.rate, "mean"),
            block(F.heater, "mean"), imp_c, self.res * f)
        # Lowest dose rate on passable ground inside each coarse cell (shelter search).
        self._c_rate_min = block(np.where(imp, np.float32(np.inf), F.rate), "min")
        self._c_offsets = tuple(dr * Cc + dc for dr, dc in _DIRS)
        self._off_to_k = {off: k for k, off in enumerate(self._c_offsets)}

    def _coarse(self, strat: Strategy) -> tuple:
        """(cost lists, heuristic scale, resources (5, 8, n)) of the coarse graph for a strategy."""
        if strat.routing not in self._coarse_cost:
            cost, res = _edge_arrays(self.coarse, strat, self.mission, self.suit, self.rate_ref,
                                     edge_blocked=self._edge_blocked, resources=True)
            self._coarse_cost[strat.routing] = (
                _as_lists(cost), _min_cost_per_cell(cost), res.reshape(5, 8, -1))
        return self._coarse_cost[strat.routing]

    def _coarse_node(self, r: int, c: int, radius: int = 4) -> int | None:
        """Coarse node for a fine cell; the nearest usable one if its own is blocked."""
        cr, cc = min(r // self.f, self.Rc - 1), min(c // self.f, self.Cc - 1)
        if self._c_valid[cr, cc]:
            return cr * self.Cc + cc
        r0, r1 = max(0, cr - radius), min(self.Rc, cr + radius + 1)
        c0, c1 = max(0, cc - radius), min(self.Cc, cc + radius + 1)
        ok = np.argwhere(self._c_valid[r0:r1, c0:c1])
        if ok.size == 0:
            return None
        best = ok[np.argmin((ok[:, 0] + r0 - cr) ** 2 + (ok[:, 1] + c0 - cc) ** 2)]
        return int(best[0] + r0) * self.Cc + int(best[1] + c0)

    def _tree(self, strat: Strategy, node: int) -> tuple:
        key = (strat.routing, node)
        if key not in self._trees:
            self._trees[key] = _dijkstra(self._coarse(strat)[0], self._c_offsets, node)
        return self._trees[key]

    def _coarse_path(self, strat: Strategy, a: int, b: int):
        """Cheapest coarse path a -> b and its cost per fine cell of length, or (None, 0)."""
        lists, hscale, _ = self._coarse(strat)
        if (strat.routing, a) in self._trees:
            path = _backtrack(self._trees[(strat.routing, a)][1], a, b)
        else:       # admissible heuristic: the coarse path has to be the true optimum
            path = _astar(lists, self._c_offsets, a, b, _octile(self.Rc, self.Cc, b, hscale))
        if path is None or len(path) < 2:
            return path, 0.0
        cost = length = 0.0
        for u, v in zip(path[:-1], path[1:]):
            k = self._off_to_k[v - u]
            cost += lists[k][u]
            length += _STEP[k]
        return path, cost / (length * self.f)

    def _path_resources(self, strat: Strategy, path: list[int]) -> list[float]:
        """Sum of the physical resources along a coarse path."""
        if len(path) < 2:
            return [0.0] * 5
        res = self._coarse(strat)[2]
        src = np.asarray(path[:-1])
        k = np.fromiter((self._off_to_k[b - a] for a, b in zip(path[:-1], path[1:])),
                        dtype=np.intp, count=len(path) - 1)
        return res[:, k, src].sum(axis=1).tolist()

    # ---- waypoints -------------------------------------------------------
    def _place(self, xy: tuple[float, float]):
        """(fine cell, coarse node) for a point in metres; (None, None) if it is walled in."""
        try:
            cell = _snap_to_walkable(*self.site.to_pixel(*xy), self.fine.imp)
        except ValueError:
            return None, None
        return cell, self._coarse_node(*cell)

    def _matrix(self, points: list[tuple[float, float]]):
        """Leg resources between all points on the nominal route, inflated by coarse_margin.

        Returns (places, M) where M[i][j] is a list of RESOURCES or None if unreachable.
        """
        key = tuple(points)
        if key in self._matrices:
            return self._matrices[key]
        places = [self._place(p) for p in points]
        n = len(points)
        M: list[list] = [[None] * n for _ in range(n)]
        for i, (_, node_i) in enumerate(places):
            if node_i is None:
                continue
            _, parent = self._tree(_NOMINAL, node_i)
            for j, (_, node_j) in enumerate(places):
                if i == j or node_j is None:
                    continue
                path = _backtrack(parent, node_i, node_j)
                if path is not None:
                    M[i][j] = [v * self.coarse_margin for v in self._path_resources(_NOMINAL, path)]
        self._matrices[key] = (places, M)
        return places, M

    # ---- fine legs -------------------------------------------------------
    def _corridor(self, nodes: list[int], width: int):
        """Fine window (r0, r1, c0, c1) and mask of the corridor around coarse nodes."""
        f = self.f
        m = np.zeros((self.Rc, self.Cc), bool)
        m.ravel()[nodes] = True
        for _ in range(width):
            m[1:] |= m[:-1].copy()
            m[:-1] |= m[1:].copy()
            m[:, 1:] |= m[:, :-1].copy()
            m[:, :-1] |= m[:, 1:].copy()
        rr, cc = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        r0c, r1c, c0c, c1c = rr[0], rr[-1] + 1, cc[0], cc[-1] + 1
        r0, r1 = r0c * f, min(r1c * f, self.rows)
        c0, c1 = c0c * f, min(c1c * f, self.cols)
        mask = np.repeat(np.repeat(m[r0c:r1c, c0c:c1c], f, axis=0), f, axis=1)
        return (r0, r1, c0, c1), mask[:r1 - r0, :c1 - c0]

    def _fine_path(self, a: tuple[int, int], b: tuple[int, int], coarse_nodes: list[int],
                   strat: Strategy, cost_per_cell: float = 0.0):
        """Full-resolution path of cells [(row, col), ...] from a to b, or None.

        The heuristic assumes the rest of the way costs per cell what the coarse
        path costs on average (cost_per_cell), times heuristic_weight. That keeps
        the search on the axis of the corridor instead of flooding all of it.
        """
        f = self.f
        ends = [min(p[0] // f, self.Rc - 1) * self.Cc + min(p[1] // f, self.Cc - 1) for p in (a, b)]
        nodes = list(coarse_nodes) + ends
        for width in (self.corridor_cells, self.corridor_cells * 4, None):
            if width is None:
                r0, r1, c0, c1, outside = 0, self.rows, 0, self.cols, None
            else:
                (r0, r1, c0, c1), mask = self._corridor(nodes, width)
                outside = ~mask
            win = self.fine.window(r0, r1, c0, c1, outside)
            cost, _ = _edge_arrays(win, strat, self.mission, self.suit, self.rate_ref)
            scale = cost_per_cell or _min_cost_per_cell(cost)
            if not math.isfinite(scale):
                continue
            wc = c1 - c0
            offsets = tuple(dr * wc + dc for dr, dc in _DIRS)
            goal = (b[0] - r0) * wc + (b[1] - c0)
            path = _astar(_as_lists(cost), offsets, (a[0] - r0) * wc + (a[1] - c0), goal,
                          _octile(r1 - r0, wc, goal, self.heuristic_weight * scale))
            if path is not None:
                return [(p // wc + r0, p % wc + c0) for p in path]
        return None

    def _elevation_at(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Elevation interpolated between cell centres (exact at the centres)."""
        e, cols = self._flat["elev"], self.cols
        u, v = xs / self.res - 0.5, ys / self.res - 0.5
        c0 = np.clip(np.floor(u).astype(np.intp), 0, cols - 2)
        r0 = np.clip(np.floor(v).astype(np.intp), 0, self.rows - 2)
        fu, fv = np.clip(u - c0, 0.0, 1.0), np.clip(v - r0, 0.0, 1.0)
        i = r0 * cols + c0
        return ((e[i] * (1 - fu) + e[i + 1] * fu) * (1 - fv)
                + (e[i + cols] * (1 - fu) + e[i + cols + 1] * fu) * fv)

    def _walk(self, xs: np.ndarray, ys: np.ndarray, strat: Strategy | None = None,
              margin: bool = False) -> dict:
        """Walk through sample points in order; per-step physics as arrays.

        margin=True also counts cells next to impassable ones as blocked.
        """
        res, fl = self.res, self._flat
        cc = np.clip((xs / res).astype(np.intp), 0, self.cols - 1)
        rr = np.clip((ys / res).astype(np.intp), 0, self.rows - 1)
        idx = rr * self.cols + cc
        d = np.maximum(np.hypot(np.diff(xs), np.diff(ys)), 1e-6)
        z = self._elevation_at(xs, ys)
        dh = np.diff(z)
        j = idx[1:]
        t, met, o2, wh = locomotion(d, dh, fl["eta"][j], fl["speed"][j], self.mission, self.suit,
                                    fl["heater"][j])
        slope = np.degrees(np.arctan(dh / d))
        out = {"idx": idx, "rr": rr, "cc": cc, "z": z, "d": d, "dh": dh, "t": t, "met": met,
               "o2": o2, "wh": wh, "slope": slope, "dose": fl["rate"][j] * t / 3600.0,
               "blocked": bool(fl["fat" if margin else "imp"][idx].any())
               or bool((np.abs(slope) > self.mission.hard_max_slope_deg).any())}
        if strat is not None:
            out["cost"] = _generalized_cost(strat, t, o2, fl["loose"][j], fl["pen"][j],
                                            fl["rate"][j], np.abs(slope), self.rate_ref,
                                            self.mission)
        return out

    def _line(self, x0, y0, x1, y1):
        """Sample points along a straight line, about one per cell, both ends included."""
        n = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / self.res))
        s = np.arange(n + 1) / n
        return x0 + s * (x1 - x0), y0 + s * (y1 - y0)

    def _smooth(self, cells: list[tuple[int, int]], strat: Strategy) -> list[int]:
        """String pulling. Returns the indices of the cells kept as corners.

        A straight line replaces a piece of the grid path when it crosses no
        impassable cell, never exceeds the hard slope, and its cost is not above
        the cost of that piece: a line through sand or a hot spot is rejected.
        """
        m = len(cells)
        if m < 3:
            return list(range(m))
        arr = np.asarray(cells, dtype=np.float64)
        xs, ys = (arr[:, 1] + 0.5) * self.res, (arr[:, 0] + 0.5) * self.res
        cum = np.concatenate(([0.0], np.cumsum(self._walk(xs, ys, strat)["cost"])))
        tol = 1.0 + self.smooth_tolerance

        def ok(i: int, j: int) -> bool:
            if j == i + 1:
                return True
            w = self._walk(*self._line(xs[i], ys[i], xs[j], ys[j]), strat, margin=True)
            return not w["blocked"] and float(w["cost"].sum()) <= (cum[j] - cum[i]) * tol + 1e-9

        keep, i, last = [0], 0, m - 1
        while i < last:
            j = min(i + self.max_span, last)
            if not ok(i, j):
                lo, hi = i + 1, j
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if ok(i, mid):
                        lo = mid
                    else:
                        hi = mid
                j = lo
            keep.append(j)
            i = j
        # Second pass: drop corners that are there only because a span was capped.
        i = 1
        while i < len(keep) - 1:
            if ok(keep[i - 1], keep[i + 1]):
                del keep[i]
            else:
                i += 1
        return keep

    def leg(self, start_m: tuple[float, float], goal_m: tuple[float, float],
            strategy: Strategy | str = "science", name_from: str = "", name_to: str = ""
            ) -> RouteSegment:
        """Route between two points in metres. Raises RuntimeError if there is none."""
        strat = self._strategy(strategy)
        (a, na), (b, nb) = self._place(start_m), self._place(goal_m)
        if a is None or b is None or na is None or nb is None:
            raise RuntimeError(f"Маршрут от {start_m} до {goal_m} не найден (точка заблокирована).")
        key = (strat.routing, a, b)
        if key not in self._legs:
            coarse, per_cell = self._coarse_path(strat, na, nb)
            cells = self._fine_path(a, b, coarse, strat, per_cell) if coarse is not None else None
            if cells is None:
                raise RuntimeError(f"Маршрут от {start_m} до {goal_m} не найден "
                                   f"(цель изолирована обрывами или no_go).")
            corners = [cells[i] for i in self._smooth(cells, strat)]
            self._legs[key] = self._segment(corners)
        return replace(self._legs[key], name_from=name_from, name_to=name_to)

    def _segment(self, corners: list[tuple[int, int]]) -> RouteSegment:
        """Walk the smoothed route and report it."""
        res = self.res
        vx = [(c + 0.5) * res for _, c in corners]
        vy = [(r + 0.5) * res for r, _ in corners]
        px, py = [np.array(vx[:1])], [np.array(vy[:1])]
        for i in range(len(corners) - 1):
            lx, ly = self._line(vx[i], vy[i], vx[i + 1], vy[i + 1])
            px.append(lx[1:])
            py.append(ly[1:])
        xs, ys = np.concatenate(px), np.concatenate(py)
        if len(xs) == 1:                                   # start and goal in the same cell
            xs, ys = np.repeat(xs, 2), np.repeat(ys, 2)
        w = self._walk(xs, ys)
        idx, fl = w["idx"], self._flat

        def lead(a):                                       # per-step values -> per-sample
            return np.concatenate(([0.0], a))

        cum_d, cum_t = lead(np.cumsum(w["d"])), lead(np.cumsum(w["t"]))
        cum_o2, cum_wh = lead(np.cumsum(w["o2"])), lead(np.cumsum(w["wh"]))
        cum_dose = lead(np.cumsum(w["dose"]))
        temp = self._temp[idx] if self._temp is not None else None
        cols = {
            "x_m": np.round(xs, 1), "y_m": np.round(ys, 1),
            "elev_m": np.round(w["z"], 2),
            "dist_m": np.round(cum_d, 1),
            "slope_deg": np.round(lead(w["slope"]), 1),
            "looseness": np.round(fl["loose"][idx].astype(np.float64), 3),
            "sinkage_cm": np.round(self._sinkage_cm[idx].astype(np.float64), 2),
            "terrain_class": self._tclass[idx].astype(int),
            "dose_rate_usv_h": np.round(fl["rate"][idx].astype(np.float64) * 1000.0, 2),
            "t_min": np.round(cum_t / 60.0, 3),
            "o2_l": np.round(cum_o2, 3), "energy_wh": np.round(cum_wh, 3),
            "dose_msv": np.round(cum_dose, 5),
        }
        lists = {k: v.tolist() for k, v in cols.items()}
        lists["temp_c"] = (np.round(temp.astype(np.float64), 1).tolist() if temp is not None
                           else [None] * len(xs))
        keys = list(lists)
        profile = [dict(zip(keys, vals)) for vals in zip(*lists.values())]
        dh = w["dh"]
        return RouteSegment(
            points_m=list(zip(xs.tolist(), ys.tolist())),
            pixels=list(zip(w["rr"].tolist(), w["cc"].tolist())),
            distance_m=float(w["d"].sum()),
            elevation_gain_m=float(dh[dh > 0].sum()),
            elevation_drop_m=float(-dh[dh < 0].sum()),
            max_slope_deg=float(np.abs(w["slope"]).max()),
            mean_looseness=float(fl["loose"][idx].mean()),
            duration_min=float(w["t"].sum()) / 60.0,
            o2_liters=float(w["o2"].sum()),
            energy_wh=float(w["wh"].sum()),
            dose_msv=float(w["dose"].sum()),
            profile=profile,
            vertices_m=list(zip(vx, vy)),
        )

    # ---- mission planning --------------------------------------------------
    def _strategy(self, strategy: Strategy | str) -> Strategy:
        if isinstance(strategy, Strategy):
            return strategy
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy '{strategy}', expected one of {sorted(STRATEGIES)}")
        return STRATEGIES[strategy]

    def budgets(self, use: float = 1.0) -> list[float]:
        """Limits in the order of RESOURCES, scaled by the share a plan may use."""
        m, s = self.mission, self.suit
        return [m.eva_max_hours * 3600.0 * (1.0 - m.return_reserve_frac) * use,
                m.max_distance_km * 1000.0 * use,
                s.o2_tank_l * (1.0 - s.o2_reserve_frac) * use,
                s.battery_wh * (1.0 - s.battery_reserve_frac) * use,
                m.max_dose_msv * use]

    def _stop_cost(self, target: Target, cell: tuple[int, int]) -> list[float]:
        """Resources spent working at a target, in the order of RESOURCES."""
        t, o2, wh = dwell(target.dwell_min, float(self.fine.heater[cell]), self.suit)
        return [t, 0.0, o2, wh, float(self.fine.rate[cell]) * t / 3600.0]

    def _choose(self, strat: Strategy, values: list[float], stop: list, M: list, budget: list,
                inflate: float, max_nodes: int = 200_000) -> list[int]:
        """Best order of target indices (1..n) to visit from node 0 and back within budget."""
        n = len(values)
        time_budget = budget[0]
        save_i = {"o2": 2, "dose": 4}.get(strat.saving)
        save_ref = budget[2] if save_i == 2 else self.rate_ref * time_budget / 3600.0
        best_score, best_order, nodes = -INF, [], 0

        def visit(cur: int, used: list, mask: int, value: float, order: list) -> None:
            nonlocal best_score, best_order, nodes
            nodes += 1
            back = M[cur][0] if cur else [0.0] * 5
            total = [u + b * inflate for u, b in zip(used, back)]
            score = value - 0.5 * total[0] / time_budget
            if save_i is not None:
                score -= strat.saving_weight * total[save_i] / save_ref
            if score > best_score:
                best_score, best_order = score, order
            if nodes > max_nodes:
                return
            for j in range(1, n):
                if mask >> j & 1 or M[cur][j] is None or M[j][0] is None or stop[j] is None:
                    continue
                there = [u + a * inflate + w for u, a, w in zip(used, M[cur][j], stop[j])]
                if all(t + b * inflate <= lim for t, b, lim in zip(there, M[j][0], budget)):
                    visit(j, there, mask | 1 << j, value + values[j], order + [j])

        visit(0, [0.0] * 5, 1, 0.0, [])
        return best_order

    def plan(self, strategy: Strategy | str = "science", targets: list[Target] | None = None,
             fixed_order: bool = False, check_radio: bool = True) -> MissionPlan:
        """Plan one EVA: start -> chosen targets -> back to start.

        targets     -- candidates (default: all site.targets).
        fixed_order -- visit exactly the given targets in the given order, no choice.
        """
        t_start = time.perf_counter()
        strat = self._strategy(strategy)
        cands = list(self.site.targets if targets is None else targets)
        points = [tuple(self.site.start)] + [(t.x_m, t.y_m) for t in cands]
        places, M = self._matrix(points)
        if places[0][0] is None or places[0][1] is None:
            raise RuntimeError("Старт site.start заблокирован непроходимой зоной.")

        sci = self.site.layers["science_score"]
        scores = [0.0] + [float(sci[self.site.to_pixel(t.x_m, t.y_m)]) for t in cands]
        values = [0.0] + [priority_value(t.priority) + strat.science_weight * s
                          for t, s in zip(cands, scores[1:])]
        stop = [None] + [self._stop_cost(t, places[i + 1][0]) if places[i + 1][0] else None
                         for i, t in enumerate(cands)]
        budget = self.budgets(strat.budget_use)

        inflate, order, legs = 1.0, [], []
        for _ in range(4):
            if fixed_order:
                order = [i for i in range(1, len(points)) if stop[i] is not None]
            else:
                order = self._choose(strat, values, stop, M, budget, inflate)
            legs = self._plan_legs(strat, order, points, cands)
            used = self._totals(legs, order, stop)
            over = max(u / lim for u, lim in zip(used, budget))
            if over <= 1.0 or not order or fixed_order:
                break
            inflate *= over * 1.02            # real legs cost more than estimated: tighten
        plan = self._assemble(strat, cands, order, legs, stop, scores, values, M, places, budget)
        if check_radio:
            plan.radio = self._radio(plan)
            if plan.radio.get("shadow_m", 0) > 0:
                plan.warnings.append(
                    f"Радиотень на {plan.radio['shadow_m']:.0f} м маршрута "
                    f"(точек сброса ретранслятора: {len(plan.radio['relay_points'])})")
        plan.compute_s = time.perf_counter() - t_start
        return plan

    def plan_pareto(self, targets: list[Target] | None = None, check_radio: bool = True
                    ) -> dict[str, MissionPlan]:
        """The three alternative plans: shielded, eco, science."""
        return {name: self.plan(name, targets, check_radio=check_radio) for name in STRATEGIES}

    def _plan_legs(self, strat, order, points, cands) -> list[RouteSegment]:
        if not order:
            return []
        seq = [0] + order + [0]
        names = ["Start"] + [t.name for t in cands]
        return [self.leg(points[a], points[b], strat, names[a],
                         "Return Base" if i == len(seq) - 2 else names[b])
                for i, (a, b) in enumerate(zip(seq[:-1], seq[1:]))]

    @staticmethod
    def _totals(legs, order, stop) -> list[float]:
        used = [0.0] * 5
        for leg in legs:
            used = [u + v for u, v in zip(used, leg.resources())]
        for j in order:
            used = [u + v for u, v in zip(used, stop[j])]
        return used

    def _assemble(self, strat, cands, order, legs, stop, scores, values, M, places, budget
                  ) -> MissionPlan:
        m, suit, site = self.mission, self.suit, self.site
        hard = self.budgets(1.0)
        elev = site.layers["elevation"]

        def stop_entry(name, kind, xy, used, arrive_s, dwell_min=0.0, **extra):
            lat, lon = site.to_latlon(*xy)
            return {
                "name": name, "kind": kind,
                "x_m": round(xy[0], 1), "y_m": round(xy[1], 1),
                "elev_m": round(float(elev[site.to_pixel(*xy)]), 2),
                "lat": round(lat, 6), "lon": round(lon, 6),
                "arrive_min": round(arrive_s / 60.0, 1),
                "dwell_min": dwell_min,
                "depart_min": round(used[0] / 60.0, 1),
                "o2_l": round(used[2], 2), "energy_wh": round(used[3], 1),
                "dose_msv": round(used[4], 4),
                "o2_pct": round(100.0 * (1.0 - used[2] / suit.o2_tank_l), 1),
                "battery_pct": round(100.0 * (1.0 - used[3] / suit.battery_wh), 1),
                **extra,
            }

        used = [0.0] * 5
        stops = [stop_entry("Start", "start", tuple(site.start), used, 0.0)]
        for leg, j in zip(legs, order + [None]):
            used = [u + v for u, v in zip(used, leg.resources())]
            arrive = used[0]
            if j is None:
                stops.append(stop_entry("Return Base", "return", tuple(site.start), used, arrive))
                break
            t = cands[j - 1]
            used = [u + v for u, v in zip(used, stop[j])]
            stops.append(stop_entry(t.name, "target", (t.x_m, t.y_m), used, arrive, t.dwell_min,
                                    priority=t.priority, science_score=round(scores[j], 3),
                                    note=t.note))

        o2_left = 100.0 * (1.0 - used[2] / suit.o2_tank_l)
        bat_left = 100.0 * (1.0 - used[3] / suit.battery_wh)
        hours, km = used[0] / 3600.0, used[1] / 1000.0
        warnings: list[str] = []
        if used[0] > hard[0] + 1e-6:
            warnings.append(f"Превышен лимит времени с учетом резерва: {hours:.2f} ч > "
                            f"{hard[0] / 3600:.2f} ч")
        if used[1] > hard[1] + 1e-6:
            warnings.append(f"Превышена макс. дистанция EVA: {km:.2f} км > {m.max_distance_km:.2f} км")
        if used[4] > hard[4] + 1e-9:
            warnings.append(f"Превышен лимит радиации: {used[4]:.2f} мЗв > {m.max_dose_msv:.2f} мЗв")
        if used[2] > hard[2] + 1e-6 or used[3] > hard[3] + 1e-6:
            warnings.append(f"Низкий остаток СЖО после возврата (O2: {o2_left:.1f}%, "
                            f"BAT: {bat_left:.1f}%)")
        feasible = not warnings

        skipped = []
        for j, t in enumerate(cands, start=1):
            if j in order:
                continue
            reason = self._skip_reason(strat, j, order, stop, M, budget, hard)
            skipped.append({"name": t.name, "priority": t.priority, "reason": reason})
            if t.priority == 1:
                warnings.append(f"Не посещена обязательная цель «{t.name}»: {reason}")
        if not order and cands:
            warnings.append("Ни одна цель не укладывается в лимиты: выход не планируется.")

        return MissionPlan(
            feasible=feasible, warnings=warnings,
            visited_targets=[cands[j - 1].name for j in order],
            segments=legs,
            total_distance_km=km,
            total_elevation_gain_m=sum(s.elevation_gain_m for s in legs),
            max_slope_deg=max((s.max_slope_deg for s in legs), default=0.0),
            total_time_hours=hours,
            o2_remaining_pct=max(0.0, o2_left),
            battery_remaining_pct=max(0.0, bat_left),
            total_dose_msv=used[4],
            strategy=strat.name, title=strat.title,
            skipped_targets=skipped, stops=stops,
            science_value=sum(values[j] for j in order),
            total_o2_liters=used[2], total_energy_wh=used[3],
            limits={
                "eva_max_hours": m.eva_max_hours,
                "return_reserve_frac": m.return_reserve_frac,
                "usable_hours": round(hard[0] / 3600.0, 2),
                "max_distance_km": m.max_distance_km,
                "max_dose_msv": m.max_dose_msv,
                "o2_tank_l": suit.o2_tank_l, "o2_reserve_frac": suit.o2_reserve_frac,
                "battery_wh": suit.battery_wh, "battery_reserve_frac": suit.battery_reserve_frac,
                "max_slope_deg": m.max_slope_deg, "hard_max_slope_deg": m.hard_max_slope_deg,
                "strategy_budget_use": strat.budget_use,
            },
            usage_pct={k: round(100.0 * u / lim, 1) for k, u, lim in zip(RESOURCES, used, hard)},
            site=site, suit=suit,
        )

    def _skip_reason(self, strat, j, order, stop, M, budget, hard) -> str:
        if stop[j] is None or M[0][j] is None or M[j][0] is None:
            return "недостижима пешком (обрывы или запретная зона)"
        alone = [a + w + b for a, w, b in zip(M[0][j], stop[j], M[j][0])]
        for name, u, lim in zip(("времени", "дистанции", "O2", "батареи", "дозы"), alone, hard):
            if u > lim:
                return f"вне радиуса действия: не хватает {name} даже на одиночный выход"
        seq = [0] + order + [0]
        for pos in range(1, len(seq)):                 # would it fit anywhere in the tour?
            trial = seq[:pos] + [j] + seq[pos:]
            if any(M[a][b] is None for a, b in zip(trial[:-1], trial[1:])):
                continue
            total = [0.0] * 5
            for a, b in zip(trial[:-1], trial[1:]):
                total = [u + v for u, v in zip(total, M[a][b])]
            for s in trial[1:-1]:
                total = [u + v for u, v in zip(total, stop[s])]
            if all(u <= lim for u, lim in zip(total, budget)):
                saved = {"o2": "кислорода", "dose": "дозы"}.get(strat.saving, "ресурсов")
                return f"необязательная, пропущена ради экономии {saved}"
        return "не помещается в лимиты вместе с более приоритетными целями"

    # ---- radio -------------------------------------------------------------
    def _radio(self, plan: MissionPlan, step_m: float = 10.0) -> dict:
        """Line of sight from the base to the astronaut along the route."""
        pts, dist, base = [], [], 0.0
        for seg in plan.segments:
            last = -INF
            for i, p in enumerate(seg.profile):
                if p["dist_m"] - last >= step_m or i == len(seg.profile) - 1:
                    last = p["dist_m"]
                    pts.append((p["x_m"], p["y_m"]))
                    dist.append(base + p["dist_m"])
            base += seg.distance_m
        if not pts:
            return {"shadow_m": 0.0, "degraded_m": 0.0, "shadow_zones": [], "relay_points": [],
                    "samples": []}
        rp, elev = self.radio_params, self.site.layers["elevation"]
        pts_a = np.asarray(pts)
        status = link_quality(elev, self.res, tuple(self.site.start), pts_a, rp)
        samples = [{"dist_m": round(d, 1), "status": s} for d, s in zip(dist, status)]

        # Shadow stretches; breaks shorter than merge_gap_m do not split a stretch.
        n, spans, i = len(status), [], 0
        while i < n:
            if status[i] != "shadow":
                i += 1
                continue
            j = i
            while j + 1 < n and status[j + 1] == "shadow":
                j += 1
            if spans and dist[i] - dist[spans[-1][1]] <= rp.merge_gap_m:
                spans[-1][1] = j
            else:
                spans.append([i, j])
            i = j + 1

        zones, relays = [], []
        for i, j in spans:
            zone = {"from_dist_m": round(dist[i], 1), "to_dist_m": round(dist[j], 1),
                    "from": [round(v, 1) for v in pts[i]], "to": [round(v, 1) for v in pts[j]]}
            # A relay is worth dropping only before a stretch of some length.
            if i > 0 and dist[j] - dist[i] >= rp.min_relay_zone_m:
                rx = pts_a[i:j + 1]
                seen = link_quality(elev, self.res, pts[i - 1], rx, rp,
                                    tx_height_m=rp.relay_height_m)
                covered = float(np.mean([s != "shadow" for s in seen])) * 100.0
                relay = {"x_m": round(pts[i - 1][0], 1), "y_m": round(pts[i - 1][1], 1),
                         "elev_m": round(float(elev[self.site.to_pixel(*pts[i - 1])]), 2),
                         "dist_m": round(dist[i - 1], 1), "covers_zone_pct": round(covered, 0)}
                relays.append(relay)
                zone["relay"] = relay
            zones.append(zone)

        def length(kind):
            return float(sum(dist[k + 1] - dist[k] for k in range(n - 1) if status[k + 1] == kind))

        return {"frequency_mhz": rp.frequency_mhz, "base_antenna_m": rp.base_height_m,
                "shadow_m": round(length("shadow"), 1), "degraded_m": round(length("degraded"), 1),
                "shadow_zones": zones, "relay_points": relays, "samples": samples}

    # ---- emergency -------------------------------------------------------------
    def shelter(self, current_pos_m: tuple[float, float], shelter_hours: float = 2.0,
                max_walk_min: float = 30.0, min_reduction_pct: float = 3.0) -> ShelterRoute:
        """Fastest way to the natural shelter that minimises the dose of a particle event.

        Every reachable spot is scored by the dose of walking there plus waiting
        there for the rest of shelter_hours; the lowest total wins.
        """
        t0 = time.perf_counter()
        cell, node = self._place(current_pos_m)
        if cell is None or node is None:
            raise RuntimeError(f"Точка {current_pos_m} заблокирована непроходимой зоной.")
        g, _ = self._tree(_FASTEST, node)
        walk_s = np.asarray(g).reshape(self.Rc, self.Cc)          # cost of _FASTEST is seconds
        here = float(self.fine.rate[cell])
        wait_s = np.maximum(shelter_hours * 3600.0 - walk_s, 0.0)
        with np.errstate(invalid="ignore"):                       # inf * 0 in cells masked below
            total = (here * walk_s + self._c_rate_min * wait_s) / 3600.0
        total[(walk_s > max_walk_min * 60.0) | ~self._c_valid] = INF
        if np.isfinite(total).any():
            best = np.unravel_index(int(np.argmin(total)), total.shape)
        else:                                          # nowhere to go: stay where you are
            best = divmod(node, self.Cc)

        # The best fine cell inside that coarse cell.
        f = self.f
        r0, c0 = int(best[0]) * f, int(best[1]) * f
        block_rate = np.where(self.fine.imp[r0:r0 + f, c0:c0 + f], np.inf,
                              self.fine.rate[r0:r0 + f, c0:c0 + f])
        br, bc = np.unravel_index(int(np.argmin(block_rate)), block_rate.shape)
        spot = (r0 + int(br), c0 + int(bc))
        spot_m = self.site.to_metres(*spot)
        seg = self.leg(tuple(current_pos_m), spot_m, _FASTEST, "Current position", "Shelter")
        there = float(self.fine.rate[spot])
        reduction = 100.0 * (1.0 - there / here) if here > 0 else 0.0
        walk_h = seg.duration_min / 60.0
        return ShelterRoute(
            segment=seg, shelter_m=spot_m, eta_min=seg.duration_min,
            dose_rate_here_msv_h=here, dose_rate_shelter_msv_h=there,
            reduction_pct=reduction, worthwhile=reduction >= min_reduction_pct,
            shelter_hours=shelter_hours, dose_if_stay_msv=here * shelter_hours,
            dose_if_shelter_msv=seg.dose_msv + there * max(shelter_hours - walk_h, 0.0),
            compute_s=time.perf_counter() - t0)


# ----------------------------------------------------------------------------
# Function API (kept from the first version of this module)
# ----------------------------------------------------------------------------
def step_biomechanics(dist_m: float, delta_h_m: float, terramech_factor: float,
                      speed_mult: float, mission: MissionParams) -> tuple[float, float, float, float]:
    """One step on foot. Returns (time_sec, o2_liters, energy_wh, slope_deg)."""
    t, _, o2, wh = locomotion(dist_m, delta_h_m, terramech_factor, speed_mult, mission)
    return float(t), float(o2), float(wh), math.degrees(math.atan2(delta_h_m, dist_m))


def _router(site: Site, terrain: AstronautTerrainModel | None,
            radiation_rate_msv_h: np.ndarray | None) -> EvaRouter:
    if terrain is None:
        terrain = build_astronaut_terrain(site, dose_rate_msv_h=radiation_rate_msv_h)
    elif radiation_rate_msv_h is not None:
        terrain = replace(terrain, radiation_is_map=True,
                          dose_rate_msv_h=np.asarray(radiation_rate_msv_h, dtype=np.float32))
    return EvaRouter(site, terrain)


def _custom(strategy: Strategy | str, looseness_weight, radiation_weight) -> Strategy:
    strat = STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    if looseness_weight is not None:
        strat = replace(strat, name=f"{strat.name}+", looseness_weight=looseness_weight)
    if radiation_weight:
        strat = replace(strat, name=f"{strat.name}+", rad_linear=radiation_weight)
    return strat


def route_segment(
    site: Site,
    start_m: tuple[float, float],
    goal_m: tuple[float, float],
    terrain: AstronautTerrainModel | None = None,
    radiation_rate_msv_h: np.ndarray | None = None,
    looseness_weight: float | None = None,
    radiation_weight: float = 0.0,
    strategy: Strategy | str = "science",
    router: EvaRouter | None = None,
) -> RouteSegment:
    """Best route between two points (x_m, y_m).

    radiation_rate_msv_h -- dose-rate map in mSv per Earth hour.
    Pass a prepared `router` when asking for many routes on the same site.
    """
    router = router or _router(site, terrain, radiation_rate_msv_h)
    return router.leg(start_m, goal_m, _custom(strategy, looseness_weight, radiation_weight))


def plan_eva_mission(
    site: Site,
    targets: list[Target] | None = None,
    radiation_rate_msv_h: np.ndarray | None = None,
    looseness_weight: float | None = None,
    radiation_weight: float = 0.0,
    strategy: Strategy | str = "science",
    fixed_order: bool = False,
    router: EvaRouter | None = None,
) -> MissionPlan:
    """Plan a round trip: start -> targets -> start, within every limit of site.mission.

    Targets and their order are chosen automatically among `targets` (default: all
    of site.targets). With fixed_order=True the given list is visited as is.
    """
    router = router or _router(site, None, radiation_rate_msv_h)
    return router.plan(_custom(strategy, looseness_weight, radiation_weight), targets, fixed_order)


def plan_pareto(site: Site, terrain: AstronautTerrainModel | None = None,
                radiation_rate_msv_h: np.ndarray | None = None,
                router: EvaRouter | None = None) -> dict[str, MissionPlan]:
    """Three alternative plans: shielded, eco, science."""
    return (router or _router(site, terrain, radiation_rate_msv_h)).plan_pareto()


def find_nearest_shelter(site: Site, current_pos_m: tuple[float, float],
                         radiation_grid: np.ndarray | None = None,
                         router: EvaRouter | None = None, **kwargs) -> ShelterRoute:
    """Escape route to the nearest natural radiation shelter (solar particle event).

    radiation_grid -- dose-rate map during the event, mSv per Earth hour.
    Pass a prepared `router` (built with that map) to answer in a fraction of a second.
    kwargs: shelter_hours, max_walk_min, min_reduction_pct.
    """
    router = router or _router(site, None, radiation_grid)
    return router.shelter(current_pos_m, **kwargs)


def render_route_preview(site: Site, plan: MissionPlan, out_path: str | Path) -> Path:
    """Looseness map, route and elevation profile of one plan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    terrain = build_astronaut_terrain(site)
    w, h = site.extent_m
    extent = (0, w, h, 0)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    im0 = axes[0].imshow(terrain.looseness, cmap="YlOrBr", vmin=0, vmax=1, extent=extent)
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    axes[0].set_title("Soil Looseness (0 = Bedrock, 1 = Loose Sand)")

    im1 = axes[1].imshow(site.layers["elevation"], cmap="terrain", extent=extent)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    for poly in site.no_go:
        pxs, pys = zip(*(poly + poly[:1]))
        axes[1].plot(pxs, pys, color="red", lw=1.2)
    colors = ["cyan", "lime", "orange", "magenta", "white", "yellow"]
    for idx, seg in enumerate(plan.segments):
        xs, ys = zip(*seg.points_m)
        axes[1].plot(xs, ys, color=colors[idx % len(colors)], lw=2.2, label=f"Leg {idx + 1}")
    axes[1].plot(*site.start, marker="^", color="white", mec="k", ms=9)
    for t in site.targets:
        axes[1].plot(t.x_m, t.y_m, marker="*", color="yellow", mec="k", ms=10)
    if plan.segments:
        axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_title(f"EVA Route ({plan.total_distance_km:.2f} km, {plan.total_time_hours:.1f} h)")

    pts = plan.waypoints(step_m=0.0)
    ax2 = axes[2]
    ax2.plot([p["dist_m"] for p in pts], [p["elev_m"] for p in pts], color="#fb923c", lw=2)
    ax2.set_xlabel("Cumulative distance, m")
    ax2.set_ylabel("Elevation, m", color="#fb923c")
    twin = ax2.twinx()
    twin.fill_between([p["dist_m"] for p in pts], 0, [p["looseness"] for p in pts],
                      color="#38bdf8", alpha=0.25)
    twin.set_ylabel("Looseness (0..1)", color="#38bdf8")
    twin.set_ylim(0, 1.1)
    ax2.set_title(f"O2 left: {plan.o2_remaining_pct:.0f}% | Max slope: {plan.max_slope_deg:.1f}°")

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    from mars_map import STUB_DIR, load_stub

    site = load_stub()
    plan = plan_eva_mission(site)
    print(f"Цели: {plan.visited_targets}")
    print(f"Дистанция: {plan.total_distance_km:.2f} км | Время: {plan.total_time_hours:.2f} ч")
    print(f"Набор высоты: +{plan.total_elevation_gain_m:.1f} м | Макс. уклон: {plan.max_slope_deg:.1f}°")
    print(f"Остаток O2: {plan.o2_remaining_pct:.1f}% | Батарея: {plan.battery_remaining_pct:.1f}%")
    print(f"Статус миссии: {'OK' if plan.feasible else plan.warnings}")

    img_path = render_route_preview(site, plan, STUB_DIR / "astronaut_route_preview.png")
    plan.to_json(STUB_DIR / "astronaut_plan.json")
    print(f"Сохранено: {img_path}")
