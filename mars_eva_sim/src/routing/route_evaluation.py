"""H10: verdict on a route produced outside the simulator.

``evaluate_route`` takes a ``RouteInput`` (from the routing team's planner, the
data team's track, or a list of points) and runs it through the same physics
and the same eight constraints as the simulator's own search:

    1. rasterize   polyline in site metres -> chain of 8-connected cells; step lengths
                   are scaled so the chain is as long as the polyline
    2. replay      walk the chain, work at the stops, insert rests (``simulate_path``)
    3. contact     radio contact along the walked route (H9), when coverage is given
    4. report      legs, totals, what is left, the limiting resource, the verdict,
                   and how the totals differ from the ones the author claimed

Verdict: ``OK``; ``MARGINAL`` when the route completes but is out of contact
for too long or crosses cells that are not emergency-safe; ``FAIL`` when a
step is impassable or a constraint breaks, with the place and the reason.

The first cell of the route is taken as the base (start and safe haven).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..comms.coverage import ContactReport, CoverageProvider, contact_along
from ..core.schemas import Cell, Label, PruneRule, SafetyFlag
from ..physiology import emergency, life_support, radiation
from ..physiology.radiation import DoseLedger
from ..providers.route_adapter import (RouteInput, polyline_length_m, rasterize_route,
                                       stop_indices)
from .constraints import ConstraintChecker
from .mission_planner import (LegSummary, MissionStatus, cooling_tolerance_s, simulate_path,
                              summarize_leg)
from .shelter_maps import ShelterMaps, compute_shelter_maps
from .transition import EdgeEvaluator


@dataclass(slots=True)
class RouteReport:
    """Result of ``evaluate_route`` (SI units).

    Attributes:
        status: OK, MARGINAL or FAIL.
        completed: the whole route was walked.
        message: "ok" or why and where the route stopped.
        rule: the constraint that stopped it (``None`` for a terrain or power stop).
        source: where the route came from.
        cells: the rasterized route.
        start_label, final_label: state at the start and at the last point reached.
        legs: walking legs and work stops, in order.
        contact: radio contact along the route (``None`` without coverage data).
        limiting_resource, limiting_time_left_s: consumable that would run out first.
        safety_counts: number of route cells per emergency safety class.
        warnings: what makes the route MARGINAL.
        reference_delta: simulator total minus the author's claimed total, per quantity.
    """

    status: MissionStatus
    completed: bool
    message: str
    rule: PruneRule | None
    source: str
    cells: list[Cell]
    start_label: Label
    final_label: Label
    legs: list[LegSummary] = field(default_factory=list)
    contact: ContactReport | None = None
    limiting_resource: str = "none"
    limiting_time_left_s: float = math.inf
    safety_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    reference_delta: dict[str, float] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Time from the start to the last point reached, s."""
        return self.final_label.t_s - self.start_label.t_s

    @property
    def distance_m(self) -> float:
        """Distance walked, m."""
        return self.final_label.distance_m - self.start_label.distance_m


def split_legs(labels: list[Label], evaluator: EdgeEvaluator) -> list[LegSummary]:
    """Cut a label chain into walking legs and work stops.

    A new piece starts whenever the activity switches between moving (walk, rest)
    and working (task).
    """
    legs: list[LegSummary] = []
    start = 0
    walk_count = task_count = 0

    def kind(label: Label) -> str:
        return "task" if label.action == "task" else "move"

    for index in range(2, len(labels) + 1):
        at_end = index == len(labels)
        if not at_end and kind(labels[index]) == kind(labels[index - 1]):
            continue
        piece = labels[start:index]
        if len(piece) > 1:
            if kind(piece[-1]) == "task":
                task_count += 1
                name = f"task {task_count}"
            else:
                walk_count += 1
                name = f"leg {walk_count}"
            legs.append(summarize_leg(name, piece, evaluator))
        start = index - 1
    return legs


def chain_scale(cell_size_m: float, cells: list[Cell],
                points_m: tuple[tuple[float, float], ...]) -> float:
    """Ratio of the polyline length to the length of the cell chain, limited to [0.7, 1]."""
    chain = sum(cell_size_m * math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(cells, cells[1:]))
    if chain <= 0.0:
        return 1.0
    return min(1.0, max(0.7, polyline_length_m(points_m) / chain))


def evaluate_route(evaluator: EdgeEvaluator, route: RouteInput, *, start_time_s: float,
                   maps: ShelterMaps | None = None, coverage: CoverageProvider | None = None,
                   ledger: DoseLedger | None = None, spe_watch: bool = False) -> RouteReport:
    """Walk an externally planned route through the simulator and judge it.

    Args:
        evaluator: edge physics for this crew member, suit and site.
        route: the route and its stops.
        start_time_s: mission time at which the suit is sealed.
        maps: precomputed safety maps for the route's first cell; computed when omitted.
        coverage: radio coverage; the contact rule is skipped when omitted.
        ledger: personal dose record; limits the dose budget when given.
        spe_watch: a solar particle event is forecast.

    Raises:
        ValidationError: the route leaves the terrain or a stop is too far from it.
        InfeasibleStateError: the first cell of the route is not walkable.
    """
    cfg = evaluator.config
    cells = rasterize_route(evaluator.terrain, route.points_m)
    stops = stop_indices(evaluator.terrain, cells, route.stops, cfg.route_adapter)
    base = cells[0]
    start = evaluator.initial_label(base, start_time_s)
    maps = maps if maps is not None else compute_shelter_maps(evaluator, base, start_time_s)
    budget = radiation.eva_dose_budget_sv(cfg.radiation, ledger, start_time_s)
    checker = ConstraintChecker(evaluator, maps, dose_budget_sv=budget, spe_watch=spe_watch)

    replay = simulate_path(evaluator, checker, cells, start, stops,
                           distance_scale=chain_scale(evaluator.cell_size_m, cells,
                                                      route.points_m))
    final = replay.final_label
    labels = final.chain()
    report = RouteReport(
        status=MissionStatus.OK if replay.completed else MissionStatus.FAIL,
        completed=replay.completed, message=replay.message, rule=replay.rule,
        source=route.source, cells=cells, start_label=start, final_label=final,
        legs=split_legs(labels, evaluator))

    elapsed = final.t_s - start.t_s
    spent = life_support.ResourceDelta(
        o2_kg=start.o2_kg - final.o2_kg,
        co2_kg=start.co2_capacity_kg - final.co2_capacity_kg,
        cooling_water_kg=start.cooling_water_kg - final.cooling_water_kg,
        battery_j=start.battery_j - final.battery_j,
        eva_s=start.eva_remaining_s - final.eva_remaining_s)
    report.limiting_resource, report.limiting_time_left_s = life_support.limiting_resource(
        final.resources(), spent, elapsed, evaluator.suit)

    tolerance = cooling_tolerance_s(evaluator, base, start_time_s)
    counts = {flag.name: 0 for flag in SafetyFlag}
    for cell in dict.fromkeys(label.cell for label in labels):
        flag = emergency.classify_cell(
            float(maps.emergency_time_s[cell]), float(maps.shelter_time_s[cell]), tolerance,
            evaluator.suit, cfg.emergency)
        counts[flag.name] += 1
    report.safety_counts = counts
    if counts[SafetyFlag.UNSAFE.name]:
        report.warnings.append(f"{counts[SafetyFlag.UNSAFE.name]} route cells are beyond the "
                               "reach of the secondary oxygen")
    elif counts[SafetyFlag.MARGINAL.name]:
        report.warnings.append(f"{counts[SafetyFlag.MARGINAL.name]} route cells are "
                               "emergency-safe only without the reaction margin")

    if coverage is not None:
        report.contact = contact_along(labels, coverage, cfg.comms)
        if report.contact.violates:
            report.warnings.append(
                f"out of radio contact for {report.contact.longest_gap_s / 60.0:.0f} min at a "
                f"stretch (limit {cfg.comms.max_no_contact_s / 60.0:.0f} min)")

    if replay.completed and report.warnings:
        report.status = MissionStatus.MARGINAL

    ours = {"time_s": elapsed, "distance_m": final.distance_m - start.distance_m,
            "o2_kg": spent.o2_kg, "dose_sv": final.dose_sv - start.dose_sv,
            "energy_j": spent.battery_j}
    report.reference_delta = {key: ours[key] - float(value)
                              for key, value in route.reference.items() if key in ours}
    return report
