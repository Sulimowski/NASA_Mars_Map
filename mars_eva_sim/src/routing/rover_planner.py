"""Rover traverses: one rover from a start to a goal, the reach of a rover, and the
same task for every rover side by side.

All three run on the generic graph algorithms (``pareto_astar``,
``reachability``) with the rover's physics and constraints plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.config import DEFAULT_CONFIG, SimConfig
from ..core.constants import SOL_S
from ..core.schemas import Cell
from ..providers.interfaces import EnvironmentProvider, TerrainProvider
from ..rovers.profiles import ROVERS, RoverProfile
from ..rovers.types import RoverBlock, RoverLabel, RoverRule
from .pareto_astar import ParetoAStar, SearchDiagnostics
from .reachability import ReachabilityResult, explore_reachability
from .rover_constraints import RoverConstraintChecker
from .rover_transition import RoverEdgeEvaluator


@dataclass(slots=True)
class RoverPlan:
    """Result of ``plan_rover_traverse`` (SI units).

    Attributes:
        rover: name of the rover.
        found: a feasible traverse exists.
        message: "ok" or why there is none.
        bottleneck: the constraint that blocked the search (``None`` for terrain).
        blocked_steps: how many steps each vehicle limit refused during the search.
        path: cells of the traverse, in order.
        final_label: state at the goal (``None`` when not found).
        start_label: state at the start.
        diagnostics: search telemetry.
    """

    rover: str
    found: bool
    message: str
    start_label: RoverLabel
    final_label: RoverLabel | None = None
    path: list[Cell] = field(default_factory=list)
    bottleneck: RoverRule | None = None
    blocked_steps: dict[str, int] = field(default_factory=dict)
    diagnostics: SearchDiagnostics | None = None

    @property
    def duration_s(self) -> float:
        """Time from the start to the goal, parked intervals included, s."""
        return 0.0 if self.final_label is None else self.final_label.t_s - self.start_label.t_s

    @property
    def sols(self) -> float:
        """Duration of the traverse in sols."""
        return self.duration_s / SOL_S

    @property
    def distance_m(self) -> float:
        """Ground distance driven, m."""
        return 0.0 if self.final_label is None else self.final_label.distance_m

    @property
    def drive_time_s(self) -> float:
        """Time spent moving, s."""
        return 0.0 if self.final_label is None else self.duration_s - self.final_label.wait_s


def _terrain_message(profile: RoverProfile, blocks: dict[RoverBlock, int]) -> str:
    """Sentence naming the vehicle limits that cut the goal off."""
    reasons = {
        RoverBlock.CLEARANCE: f"obstacles taller than {profile.max_obstacle_height_m:.2f} m",
        RoverBlock.TILT: "ground steeper than the tilt limit",
        RoverBlock.CLIMB: "steps steeper than the climb limit",
        RoverBlock.SLIP: "wheel slip beyond the embedding limit",
        RoverBlock.FORBIDDEN: "forbidden zones",
        RoverBlock.IMPASSABLE: "impassable terrain",
    }
    seen = sorted(((count, block) for block, count in blocks.items() if block in reasons),
                  key=lambda item: -item[0])
    if not seen:
        return "the goal is not connected to the start"
    listed = ", ".join(f"{reasons[block]} ({count} steps)" for count, block in seen)
    return f"the goal is cut off by: {listed}"


def plan_rover_traverse(evaluator: RoverEdgeEvaluator, start: Cell, goal: Cell, *,
                        start_time_s: float, battery_fraction: float = 1.0,
                        max_duration_s: float | None = None) -> RoverPlan:
    """Cheapest feasible traverse of a rover from ``start`` to ``goal``.

    Args:
        evaluator: the rover's physics on the site.
        start, goal: cells.
        start_time_s: mission time at the start.
        battery_fraction: state of charge at the start, 0..1.
        max_duration_s: longest allowed traverse; the configured default when omitted.

    Returns:
        The plan. An impossible traverse is reported through ``found`` and
        ``message``, not raised.

    Raises:
        TerrainValidationError: start or goal lies outside the grid.
        InfeasibleStateError: the rover cannot stand in the start cell.
        ValidationError: ``battery_fraction`` is outside 0..1.
    """
    evaluator.block_counts.clear()
    label = evaluator.initial_label(start, start_time_s, battery_fraction)
    checker = RoverConstraintChecker(evaluator, start_time_s=start_time_s,
                                     max_duration_s=max_duration_s)
    result = ParetoAStar(evaluator, checker).search(label, goal)
    blocked = {block.value: count for block, count in evaluator.block_counts.items()}
    plan = RoverPlan(rover=evaluator.profile.name, found=result.found, message=result.message,
                     start_label=label, final_label=result.goal_label, path=result.path,
                     bottleneck=result.bottleneck, blocked_steps=blocked,
                     diagnostics=result.diagnostics)
    if not result.found and result.bottleneck is None and not result.diagnostics.budget_exhausted:
        goal_block = evaluator.cell_block(goal)
        if goal_block is not RoverBlock.NONE:
            plan.message = f"the rover cannot stand in the goal cell: {goal_block.value}"
        else:
            plan.message = _terrain_message(evaluator.profile, evaluator.block_counts)
    return plan


def compute_rover_reachability(evaluator: RoverEdgeEvaluator, start: Cell, *, start_time_s: float,
                               battery_fraction: float = 1.0,
                               max_duration_s: float | None = None,
                               isochrones_s: tuple[float, ...] = (SOL_S / 4, SOL_S / 2, SOL_S,
                                                                  2 * SOL_S)
                               ) -> ReachabilityResult:
    """Cells a rover can reach from ``start`` within its limits, with arrival times.

    A rover does not come back, so ``round_trip_ok`` equals ``reachable`` here.
    Isochrones default to a quarter, half, one and two sols.
    """
    label = evaluator.initial_label(start, start_time_s, battery_fraction)
    checker = RoverConstraintChecker(evaluator, start_time_s=start_time_s,
                                     max_duration_s=max_duration_s)
    return explore_reachability(evaluator, checker, label, isochrones_s)


def compare_rovers(terrain: TerrainProvider, environment: EnvironmentProvider, start: Cell,
                   goal: Cell, *, start_time_s: float, mission_year: float,
                   config: SimConfig = DEFAULT_CONFIG, battery_fraction: float = 1.0,
                   max_duration_s: float | None = None) -> dict[str, RoverPlan | None]:
    """The same traverse for every modelled rover.

    Returns:
        ``{rover key: plan}``; ``None`` for a rover that has not landed by ``mission_year``.
    """
    out: dict[str, RoverPlan | None] = {}
    for key, profile in ROVERS.items():
        if profile.landing_year > mission_year:
            out[key] = None
            continue
        evaluator = RoverEdgeEvaluator(terrain, environment, profile, config,
                                       mission_year=mission_year)
        out[key] = plan_rover_traverse(evaluator, start, goal, start_time_s=start_time_s,
                                       battery_fraction=battery_fraction,
                                       max_duration_s=max_duration_s)
    return out
