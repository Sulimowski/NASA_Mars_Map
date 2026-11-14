"""B3: resource-constrained route search, A* with Pareto labels.

A plain shortest-path search keeps one best cost per cell. That is wrong here:
the cheapest way to reach a cell may have used more oxygen than a slightly longer
one, and only the second can still get home. So each cell keeps up to ``K`` labels
that do not dominate one another.

    label   position, time, every consumable, fatigue, dose, stored heat, risk, cost
    order   priority = cost + heuristic,  heuristic = w_t * straight distance / V_pref
    prune   the eight rules of ``ConstraintChecker``
    dominance   Y dominates X when Y is no worse in cost and in every state variable
    trim    more than K labels in a cell: the most expensive one is dropped

The heuristic is admissible because every cost term is non-negative and no step
is faster than the preferred speed. With ``K`` finite the search is a heuristic:
it can miss a feasible route whose label was trimmed.

A crew member whose fatigue is above the rest trigger may also rest in place,
which is how a route gets past the fatigue rule.

Every rejected state is counted per rule, together with the farthest distance
reached before the rule fired. When no route exists, the rule that fired most
often is reported as the bottleneck.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

from ..core.config import RoutingConfig
from ..core.schemas import Cell, Label, Pace, PruneRule, TaskType
from .constraints import ConstraintChecker
from .transition import EdgeEvaluator

_BOTTLENECK_TEXT: dict[PruneRule, str] = {
    PruneRule.RESOURCE_RESERVE: "{detail} reserve reached after {km:.1f} km",
    PruneRule.RETURN_O2: "insufficient O₂ for the return trip after {km:.1f} km",
    PruneRule.RETURN_TIME: "insufficient EVA time for the return trip after {km:.1f} km",
    PruneRule.EMERGENCY_RADIUS: "no shelter within secondary-oxygen range beyond {km:.1f} km",
    PruneRule.SPE_SHELTER: ("solar particle event: no shelter within the warning time "
                            "beyond {km:.1f} km"),
    PruneRule.FATIGUE: "fatigue limit ({detail}) reached after {km:.1f} km",
    PruneRule.DOSE_LIMIT: "radiation dose budget exhausted after {km:.1f} km",
    PruneRule.HEAT_STORAGE: "body heat storage limit ({detail}) reached after {km:.1f} km",
}


@dataclass(slots=True)
class SearchDiagnostics:
    """Telemetry of one search.

    Attributes:
        expanded: labels taken from the queue and expanded.
        generated: labels created.
        infeasible_edges: moves rejected by the edge physics (terrain or power).
        dominated: labels discarded because another label was no worse in everything.
        trimmed: labels discarded because a cell already held K cheaper ones.
        prune_counts: how often each constraint rejected a label.
        prune_max_distance_m: farthest distance walked by a label rejected by each rule.
        prune_details: per rule, how often each detail (for example the resource) occurred.
        budget_exhausted: the search stopped at ``max_expansions``.
    """

    expanded: int = 0
    generated: int = 0
    infeasible_edges: int = 0
    dominated: int = 0
    trimmed: int = 0
    prune_counts: dict[PruneRule, int] = field(default_factory=dict)
    prune_max_distance_m: dict[PruneRule, float] = field(default_factory=dict)
    prune_details: dict[PruneRule, dict[str, int]] = field(default_factory=dict)
    budget_exhausted: bool = False

    def record(self, rule: PruneRule, detail: str, distance_m: float) -> None:
        """Count one rejection by ``rule``."""
        self.prune_counts[rule] = self.prune_counts.get(rule, 0) + 1
        if distance_m > self.prune_max_distance_m.get(rule, -1.0):
            self.prune_max_distance_m[rule] = distance_m
        details = self.prune_details.setdefault(rule, {})
        details[detail] = details.get(detail, 0) + 1

    def bottleneck(self) -> tuple[PruneRule | None, str]:
        """The constraint that blocked the search most often, with a readable message.

        Returns:
            ``(rule, message)``. ``rule`` is ``None`` when no constraint fired: the
            goal is then cut off by terrain, or the search budget ran out.
        """
        if self.prune_counts:
            rule = max(self.prune_counts,
                       key=lambda r: (self.prune_counts[r], self.prune_max_distance_m.get(r, 0.0)))
            details = self.prune_details.get(rule, {})
            detail = max(details, key=lambda d: details[d]) if details else ""
            km = self.prune_max_distance_m.get(rule, 0.0) / 1000.0
            return rule, _BOTTLENECK_TEXT[rule].format(detail=detail, km=km)
        if self.budget_exhausted:
            return None, "search budget exhausted before the goal was reached"
        if self.infeasible_edges:
            return None, "the goal is cut off by impassable, forbidden or too steep terrain"
        return None, "the goal is not connected to the start"


@dataclass(slots=True)
class SearchResult:
    """Outcome of a route search.

    Attributes:
        found: a feasible route exists.
        goal_label: final state at the goal (``None`` when not found).
        path: cells of the route in travel order, rests collapsed (empty when not found).
        diagnostics: telemetry of the search.
        message: "ok" or the bottleneck description.
        bottleneck: the blocking constraint when not found.
    """

    found: bool
    goal_label: Label | None
    path: list[Cell]
    diagnostics: SearchDiagnostics
    message: str
    bottleneck: PruneRule | None = None


def dominates(y: Label, x: Label, tol: float) -> bool:
    """True when ``y`` is at least as good as ``x`` in cost and every state variable.

    ``tol`` is a relative tolerance: two paths of equal length that differ only by
    floating-point noise then dominate each other, instead of both being kept.
    """
    up, down = 1.0 + tol, 1.0 - tol
    return (y.cost <= x.cost * up + tol
            and y.o2_kg >= x.o2_kg * down - tol
            and y.eva_remaining_s >= x.eva_remaining_s * down - tol
            and y.battery_j >= x.battery_j * down - tol
            and y.co2_capacity_kg >= x.co2_capacity_kg * down - tol
            and y.cooling_water_kg >= x.cooling_water_kg * down - tol
            and y.fatigue <= x.fatigue * up + tol
            and y.dose_sv <= x.dose_sv * up + tol
            and abs(y.heat_storage_j) <= abs(x.heat_storage_j) * up + tol
            and y.risk <= x.risk * up + tol
            and y.hand_fatigue <= x.hand_fatigue * up + tol
            and y.hydration_deficit_kg <= x.hydration_deficit_kg * up + tol
            and y.t_s <= x.t_s * up + tol)


def path_of(label: Label) -> list[Cell]:
    """Cells visited on the way to ``label``, consecutive repeats (rests, tasks) collapsed."""
    cells: list[Cell] = []
    for node in label.chain():
        if not cells or cells[-1] != node.cell:
            cells.append(node.cell)
    return cells


class ParetoAStar:
    """Pareto-label A* over the 8-connected grid."""

    def __init__(self, evaluator: EdgeEvaluator, checker: ConstraintChecker) -> None:
        self._evaluator = evaluator
        self._checker = checker
        self._cfg: RoutingConfig = evaluator.config.routing
        self._rest_trigger = evaluator.config.fatigue.rest_trigger
        self._rest_duration = evaluator.config.fatigue.rest_duration_s
        self._v_max = evaluator.astronaut.preferred_speed_m_s

    def search(self, start: Label, goal: Cell) -> SearchResult:
        """Cheapest feasible route from the start label to ``goal``.

        Never raises for an unreachable goal: the result then has ``found=False``
        and names the bottleneck.

        Raises:
            TerrainValidationError: the goal lies outside the grid.
        """
        if not self._evaluator.in_bounds(goal):
            self._evaluator.require_walkable(goal, "goal")
        goal_label, _, diagnostics = self._run(start, goal, self._cfg.max_labels_per_cell)
        if goal_label is not None:
            return SearchResult(True, goal_label, path_of(goal_label), diagnostics, "ok")
        rule, message = diagnostics.bottleneck()
        return SearchResult(False, None, [], diagnostics, message, rule)

    def explore(self, start: Label, max_labels_per_cell: int = 1
                ) -> tuple[dict[Cell, Label], SearchDiagnostics]:
        """Cheapest feasible label for every cell reachable from the start (no goal).

        Returns:
            ``(best label per reachable cell, diagnostics)``.
        """
        _, labels, diagnostics = self._run(start, None, max_labels_per_cell)
        best: dict[Cell, Label] = {}
        for cell, bucket in labels.items():
            alive = [label for label in bucket if label.alive]
            if alive:
                best[cell] = min(alive, key=lambda label: label.cost)
        return best, diagnostics

    # ---- core loop ------------------------------------------------------
    def _run(self, start: Label, goal: Cell | None, k: int
             ) -> tuple[Label | None, dict[Cell, list[Label]], SearchDiagnostics]:
        evaluator, checker, cfg = self._evaluator, self._checker, self._cfg
        diagnostics = SearchDiagnostics()
        labels: dict[Cell, list[Label]] = {}
        start_violation = checker.violated(start)
        if start_violation is not None:
            diagnostics.record(start_violation[0], start_violation[1], start.distance_m)
            return None, labels, diagnostics

        if goal is None:
            def heuristic(row: int, col: int) -> float:
                return 0.0
        else:
            scale = cfg.weight_time_per_s * evaluator.cell_size_m / self._v_max
            goal_row, goal_col = goal

            def heuristic(row: int, col: int) -> float:
                return scale * math.hypot(row - goal_row, col - goal_col)

        counter = itertools.count()
        labels[start.cell] = [start]
        heap: list[tuple[float, int, Label]] = [
            (start.cost + heuristic(start.row, start.col), next(counter), start)]
        tol = cfg.dominance_tolerance

        while heap:
            _, _, label = heapq.heappop(heap)
            if not label.alive:
                continue
            cell = (label.row, label.col)
            if goal is not None and cell == goal:
                return label, labels, diagnostics
            diagnostics.expanded += 1
            if diagnostics.expanded > cfg.max_expansions:
                diagnostics.budget_exhausted = True
                break

            candidates: list[tuple[Cell, Label]] = []
            for nb, dist in evaluator.neighbors(cell):
                tr = evaluator.walk_edge(cell, nb, dist, label, Pace.NOMINAL)
                if tr is None:
                    diagnostics.infeasible_edges += 1
                    continue
                candidates.append((nb, evaluator.advance(label, nb, tr, "walk")))
            if label.fatigue > self._rest_trigger:
                rest = evaluator.stationary(cell, label, self._rest_duration, TaskType.REST)
                candidates.append((cell, evaluator.advance(label, cell, rest, "rest")))

            for nb, new in candidates:
                diagnostics.generated += 1
                violation = checker.violated(new, cell)
                if violation is not None:
                    diagnostics.record(violation[0], violation[1], new.distance_m)
                    continue
                bucket = labels.setdefault(nb, [])
                if self._insert(bucket, new, k, tol, diagnostics):
                    heapq.heappush(heap, (new.cost + heuristic(new.row, new.col),
                                          next(counter), new))
        return None, labels, diagnostics

    @staticmethod
    def _insert(bucket: list[Label], new: Label, k: int, tol: float,
                diagnostics: SearchDiagnostics) -> bool:
        """Add ``new`` to the Pareto set of its cell. False when it is not kept."""
        for old in bucket:
            if dominates(old, new, tol):
                diagnostics.dominated += 1
                return False
        kept: list[Label] = []
        for old in bucket:
            if dominates(new, old, tol):
                old.alive = False
                diagnostics.dominated += 1
            else:
                kept.append(old)
        kept.append(new)
        accepted = True
        if len(kept) > k:
            worst = max(kept, key=lambda label: label.cost)
            worst.alive = False
            kept.remove(worst)
            diagnostics.trimmed += 1
            accepted = worst is not new
        bucket[:] = kept
        return accepted
