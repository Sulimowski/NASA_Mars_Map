"""B4: a complete EVA, base -> target -> task -> base, and the replay of a given route.

``plan_round_trip`` runs, in order:

    pre-EVA checks   first-sol rule (A1), daily limits (A7), dose budget (A6)
    prebreathe       nitrogen washout time (A11); adds to the timeline only
    outbound         Pareto A* from the base to the target (B3)
    task             stationary work in steps of ``task_step_s``, constraints checked each step
    return           Pareto A* from the state after the task back to the base
    post-EVA         dust cleaning time and perchlorate flag (A12), limiting resource (A4),
                     safety class of every cell on the route (A10)

Status: ``OK``; ``MARGINAL`` when the route is feasible but the task was cut
short, a cell on it is not emergency-safe, or glove contact is time-limited;
``FAIL`` when a pre-check fails or a leg has no feasible route.

``simulate_path`` replays a route produced elsewhere (for example by another
team's router) through the same physics and the same constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from ..core.exceptions import ValidationError
from ..core.schemas import (Cell, HabitatAtmosphere, Label, Pace, PhysState, PruneRule,
                            SafetyFlag, SuitResources, TaskType)
from ..physiology import (adaptation, decompression, dust, emergency, fatigue, life_support,
                          metabolism, radiation, thermal)
from ..physiology.fatigue import DailyEvaRecord
from ..physiology.radiation import DoseLedger
from .constraints import ConstraintChecker
from .pareto_astar import ParetoAStar, SearchDiagnostics
from .shelter_maps import ShelterMaps, compute_shelter_maps
from .transition import EdgeEvaluator


class MissionStatus(str, Enum):
    """Verdict of a mission plan."""

    OK = "OK"
    MARGINAL = "MARGINAL"
    FAIL = "FAIL"


@dataclass(slots=True)
class LegSummary:
    """Totals of one leg of the EVA (SI units)."""

    name: str
    start_cell: Cell
    end_cell: Cell
    distance_m: float
    duration_s: float
    elevation_gain_m: float
    elevation_loss_m: float
    max_slope_rad: float
    o2_kg: float
    energy_j: float
    dose_sv: float
    rest_s: float
    steps: int


@dataclass(slots=True)
class MissionPlan:
    """Result of ``plan_round_trip``.

    Attributes:
        status: OK, MARGINAL or FAIL.
        reasons: why the plan failed (empty unless FAIL).
        warnings: what makes the plan MARGINAL, and other notes.
        legs: outbound, task and return summaries.
        start_label, final_label: state at the start and at the end of the EVA.
        prebreathe_s: nitrogen washout before the EVA.
        cleaning_s: airlock cleaning after the EVA.
        task_completed_s: task time actually worked.
        limiting_resource, limiting_time_left_s: consumable that would run out first.
        perchlorate_flag: dust brought in exceeds the contamination threshold.
        safety_counts: number of route cells per safety class.
        diagnostics: search telemetry of the outbound and return legs.
    """

    status: MissionStatus
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    legs: list[LegSummary] = field(default_factory=list)
    start_label: Label | None = None
    final_label: Label | None = None
    prebreathe_s: float = 0.0
    cleaning_s: float = 0.0
    task_completed_s: float = 0.0
    limiting_resource: str = "none"
    limiting_time_left_s: float = math.inf
    perchlorate_flag: bool = False
    safety_counts: dict[str, int] = field(default_factory=dict)
    diagnostics: list[SearchDiagnostics] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        """True unless the plan failed."""
        return self.status is not MissionStatus.FAIL

    @property
    def eva_duration_s(self) -> float:
        """Time spent in the suit, s."""
        if self.start_label is None or self.final_label is None:
            return 0.0
        return self.final_label.t_s - self.start_label.t_s

    @property
    def total_distance_m(self) -> float:
        """Distance walked, m."""
        return 0.0 if self.final_label is None else self.final_label.distance_m

    def labels(self) -> list[Label]:
        """Every state of the EVA in time order (empty when the plan failed early)."""
        return [] if self.final_label is None else self.final_label.chain()


@dataclass(slots=True)
class PathReplay:
    """Result of ``simulate_path``.

    Attributes:
        completed: the whole path was walked without breaking a constraint.
        final_label: state at the last point reached.
        failed_index: index in the path of the cell that could not be entered.
        rule: constraint that stopped the replay (``None`` for a terrain or power stop).
        message: readable reason, "ok" when completed.
    """

    completed: bool
    final_label: Label
    failed_index: int | None = None
    rule: PruneRule | None = None
    message: str = "ok"


def cooling_tolerance_s(evaluator: EdgeEvaluator, cell: Cell, t_s: float) -> float:
    """Time a crew member can keep walking at ``cell`` after the cooling fails, s."""
    cfg = evaluator.config
    heat = evaluator.heat_at(cell, t_s)
    walking = metabolism.walking_power_w(
        evaluator.astronaut.body_mass_kg, evaluator.suit.mass_kg,
        evaluator.astronaut.preferred_speed_m_s, 0.0, 1.2, evaluator.suit.stiffness_factor,
        cfg.metabolism)
    resting = cfg.metabolism.power_of(TaskType.REST)
    return emergency.cooling_failure_tolerance_s(walking, resting, heat.q_fixed_w,
                                                 evaluator.heat_storage_limit_j)


def summarize_leg(name: str, labels: list[Label], evaluator: EdgeEvaluator) -> LegSummary:
    """Totals of a leg from its labels (the first label is the state before the leg)."""
    first, last = labels[0], labels[-1]
    gain = loss = rest = 0.0
    max_slope = 0.0
    for previous, current in zip(labels, labels[1:]):
        tr = current.transition
        if tr is None:
            continue
        rise = (evaluator.terrain.elevation[current.row, current.col]
                - evaluator.terrain.elevation[previous.row, previous.col])
        if rise > 0.0:
            gain += float(rise)
        else:
            loss -= float(rise)
        max_slope = max(max_slope, abs(tr.slope_rad))
        if current.action == "rest":
            rest += tr.dt_s
    return LegSummary(name=name, start_cell=first.cell, end_cell=last.cell,
                      distance_m=last.distance_m - first.distance_m,
                      duration_s=last.t_s - first.t_s, elevation_gain_m=gain,
                      elevation_loss_m=loss, max_slope_rad=max_slope,
                      o2_kg=first.o2_kg - last.o2_kg, energy_j=last.energy_j - first.energy_j,
                      dose_sv=last.dose_sv - first.dose_sv, rest_s=rest, steps=len(labels) - 1)


def run_task(evaluator: EdgeEvaluator, checker: ConstraintChecker, label: Label, task: TaskType,
             duration_s: float) -> tuple[Label, float, str | None]:
    """Work on ``task`` at the label's cell for up to ``duration_s``.

    The task stops early when the next step would break a constraint, or when the
    hands are too tired for a task that needs grip.

    Returns:
        ``(state after the task, time worked in s, stop reason or None)``.
    """
    cfg = evaluator.config
    step = cfg.routing.task_step_s
    needs_grip = cfg.metabolism.grip_load_of(task) > 0.0
    done = 0.0
    while done < duration_s - 1.0e-9:
        if needs_grip and not fatigue.tool_work_allowed(label.hand_fatigue, cfg.fatigue):
            return label, done, "hand fatigue limit"
        dt = min(step, duration_s - done)
        new = evaluator.advance(label, label.cell,
                                evaluator.stationary(label.cell, label, dt, task), "task")
        violation = checker.violated(new)
        if violation is not None:
            return label, done, f"{violation[0].value} ({violation[1]})"
        label, done = new, done + dt
    return label, done, None


def simulate_path(evaluator: EdgeEvaluator, checker: ConstraintChecker, path: list[Cell],
                  start: Label, stops: dict[int, tuple[TaskType, float]] | None = None,
                  distance_scale: float = 1.0) -> PathReplay:
    """Replay a given route through the edge physics and the constraints.

    A rest of ``rest_duration_s`` is inserted whenever fatigue is above the rest trigger.

    Args:
        path: cells in travel order; consecutive cells must be 8-neighbours.
            ``path[0]`` must be the start label's cell.
        start: state at ``path[0]``.
        stops: optional ``{path index: (task, duration in s)}`` worked on arrival.
        distance_scale: multiplier of every step length, in (0, 1]. A route drawn as
            a polyline is shorter than the chain of cells that follows it; the
            scale restores its true length. Rises are kept, so slopes get
            slightly steeper, which errs on the safe side.

    Raises:
        ValidationError: the path is empty, does not begin at the start label, or
            holds two consecutive cells that are not neighbours, or the scale is invalid.
    """
    if not 0.0 < distance_scale <= 1.0:
        raise ValidationError(f"distance_scale={distance_scale} must be in (0, 1]")
    if not path:
        raise ValidationError("path is empty")
    if path[0] != start.cell:
        raise ValidationError(f"path starts at {path[0]} but the start label is at {start.cell}")
    cfg = evaluator.config
    stops = stops or {}
    label = start
    violation = checker.violated(label)
    if violation is not None:
        return PathReplay(False, label, 0, violation[0],
                          f"start state breaks {violation[0].value} ({violation[1]})")
    for index in range(len(path)):
        if index > 0:
            a, b = path[index - 1], path[index]
            d_row, d_col = b[0] - a[0], b[1] - a[1]
            if max(abs(d_row), abs(d_col)) != 1:
                raise ValidationError(f"path cells {a} and {b} (index {index}) are not neighbours")
            if not evaluator.in_bounds(b):
                raise ValidationError(f"path cell {b} (index {index}) is outside the grid")
            if label.fatigue > cfg.fatigue.rest_trigger:
                rest = evaluator.stationary(a, label, cfg.fatigue.rest_duration_s, TaskType.REST)
                label = evaluator.advance(label, a, rest, "rest")
            step_m = evaluator.cell_size_m * math.hypot(d_row, d_col) * distance_scale
            tr = evaluator.walk_edge(a, b, step_m, label, Pace.NOMINAL)
            if tr is None:
                reason = evaluator.explain_blocked_edge(a, b, step_m, label, Pace.NOMINAL)
                return PathReplay(False, label, index, None,
                                  f"step into {b} after {label.distance_m / 1000.0:.2f} km is "
                                  f"blocked: {reason}")
            new = evaluator.advance(label, b, tr, "walk")
            violation = checker.violated(new, a)
            if violation is not None:
                return PathReplay(False, label, index, violation[0],
                                  f"{violation[0].value} ({violation[1]}) at {b} after "
                                  f"{new.distance_m / 1000.0:.2f} km")
            label = new
        if index in stops:
            task, duration = stops[index]
            label, worked, reason = run_task(evaluator, checker, label, task, duration)
            if reason is not None:
                return PathReplay(False, label, index, None,
                                  f"task at {path[index]} stopped after {worked:.0f} s: {reason}")
    return PathReplay(True, label)


def plan_round_trip(evaluator: EdgeEvaluator, base: Cell, target: Cell, *, start_time_s: float,
                    task: TaskType = TaskType.SAMPLING, task_duration_s: float = 1_200.0,
                    habitat: HabitatAtmosphere | None = None, ledger: DoseLedger | None = None,
                    daily: DailyEvaRecord | None = None, maps: ShelterMaps | None = None,
                    spe_watch: bool = False, resources: SuitResources | None = None,
                    phys: PhysState | None = None) -> MissionPlan:
    """Plan an EVA from ``base`` to ``target``, work there, and come back.

    Args:
        evaluator: edge physics for this crew member, suit and site.
        base: habitat cell; start and finish of the EVA.
        target: cell of the work site.
        start_time_s: mission time at which the suit is sealed (after the prebreathe).
        task, task_duration_s: work at the target.
        habitat: cabin atmosphere, for the prebreathe (default: exploration atmosphere).
        ledger: personal dose record; limits the dose budget when given.
        daily: EVA history for the daily limits.
        maps: precomputed safety maps; computed here when omitted.
        spe_watch: a solar particle event is forecast.
        resources, phys: non-default start state.

    Returns:
        The plan. Infeasibility is reported through ``status`` and ``reasons``, not raised.

    Raises:
        TerrainValidationError: base or target lies outside the grid.
        InfeasibleStateError: the base cell is not walkable.
    """
    cfg = evaluator.config
    suit = evaluator.suit
    evaluator.require_walkable(base, "base")
    if not evaluator.in_bounds(target):
        evaluator.require_walkable(target, "target")
    daily = daily if daily is not None else DailyEvaRecord(sleep_debt_s=evaluator.sleep_debt_s)

    reasons: list[str] = []
    if not adaptation.eva_permitted(evaluator.time_on_surface_s, cfg.adaptation):
        reasons.append("no EVA is allowed during the first sol after landing")
    reasons.extend(fatigue.daily_eva_violations(daily, 0.0, cfg.fatigue, cfg.capacity))
    dose_budget = radiation.eva_dose_budget_sv(cfg.radiation, ledger, start_time_s)
    if dose_budget <= 0.0:
        reasons.append("no radiation dose budget is left for an EVA")
    prebreathe = decompression.prebreathe_time_s(habitat or HabitatAtmosphere(), suit.pressure_pa,
                                                 cfg.decompression)
    if reasons:
        return MissionPlan(status=MissionStatus.FAIL, reasons=reasons, prebreathe_s=prebreathe)

    maps = maps if maps is not None else compute_shelter_maps(evaluator, base, start_time_s)
    checker = ConstraintChecker(evaluator, maps, dose_budget_sv=dose_budget, spe_watch=spe_watch)
    astar = ParetoAStar(evaluator, checker)
    start = evaluator.initial_label(base, start_time_s, resources, phys)
    plan = MissionPlan(status=MissionStatus.OK, start_label=start, prebreathe_s=prebreathe)

    outbound = astar.search(start, target)
    plan.diagnostics.append(outbound.diagnostics)
    if not outbound.found or outbound.goal_label is None:
        plan.status = MissionStatus.FAIL
        plan.reasons.append(f"outbound leg: {outbound.message}")
        return plan
    at_target = outbound.goal_label
    plan.legs.append(summarize_leg("outbound", at_target.chain(), evaluator))

    after_task, worked, stop_reason = run_task(evaluator, checker, at_target, task, task_duration_s)
    plan.task_completed_s = worked
    plan.legs.append(summarize_leg("task", after_task.chain()[len(at_target.chain()) - 1:],
                                   evaluator))
    if stop_reason is not None:
        plan.warnings.append(f"task cut short after {worked:.0f} s of {task_duration_s:.0f} s: "
                             f"{stop_reason}")
    contact = cfg.metabolism.contact_fraction_of(task)
    touch_limit = thermal.touch_contact_limit_s(
        evaluator.env_at(target, at_target.t_s).t_ground_k, cfg.thermal)
    if contact > 0.0 and math.isfinite(touch_limit):
        plan.warnings.append(f"glove contact with the ground is limited to {touch_limit:.0f} s "
                             "at a time at this ground temperature")

    homeward = astar.search(after_task, base)
    plan.diagnostics.append(homeward.diagnostics)
    if not homeward.found or homeward.goal_label is None:
        plan.status = MissionStatus.FAIL
        plan.final_label = after_task
        plan.reasons.append(f"return leg: {homeward.message}")
        return plan
    final = homeward.goal_label
    plan.final_label = final
    plan.legs.append(summarize_leg("return", final.chain()[len(after_task.chain()) - 1:],
                                   evaluator))

    plan.cleaning_s = dust.cleaning_time_s(final.dust_load_kg, cfg.dust)
    plan.perchlorate_flag = dust.perchlorate_flag(final.dust_load_kg, cfg.dust)
    spent = life_support.ResourceDelta(
        o2_kg=start.o2_kg - final.o2_kg,
        co2_kg=start.co2_capacity_kg - final.co2_capacity_kg,
        cooling_water_kg=start.cooling_water_kg - final.cooling_water_kg,
        battery_j=start.battery_j - final.battery_j,
        eva_s=start.eva_remaining_s - final.eva_remaining_s)
    plan.limiting_resource, plan.limiting_time_left_s = life_support.limiting_resource(
        final.resources(), spent, final.t_s - start.t_s, suit)

    tolerance = cooling_tolerance_s(evaluator, base, start_time_s)
    counts = {flag.name: 0 for flag in SafetyFlag}
    visited: set[Cell] = set()
    for label in final.chain():
        if label.cell in visited:
            continue
        visited.add(label.cell)
        flag = emergency.classify_cell(
            float(maps.emergency_time_s[label.row, label.col]),
            float(maps.shelter_time_s[label.row, label.col]), tolerance, suit, cfg.emergency)
        counts[flag.name] += 1
    plan.safety_counts = counts
    if counts[SafetyFlag.UNSAFE.name]:
        plan.warnings.append(f"{counts[SafetyFlag.UNSAFE.name]} route cells are beyond the reach "
                             "of the secondary oxygen")
    elif counts[SafetyFlag.MARGINAL.name]:
        plan.warnings.append(f"{counts[SafetyFlag.MARGINAL.name]} route cells are emergency-safe "
                             "only without the reaction margin")

    late = fatigue.daily_eva_violations(daily, plan.eva_duration_s, cfg.fatigue, cfg.capacity)
    if late:
        plan.status = MissionStatus.FAIL
        plan.reasons.extend(late)
    elif plan.warnings:
        plan.status = MissionStatus.MARGINAL
    return plan


def record_dose(plan: MissionPlan, ledger: DoseLedger) -> None:
    """Add the dose of a finished EVA to the personal ledger.

    Raises:
        ValidationError: the plan has no final state.
    """
    if plan.final_label is None:
        raise ValidationError("the plan has no final state to record")
    ledger.add(plan.final_label.t_s, plan.final_label.dose_sv)
