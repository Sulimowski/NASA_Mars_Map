"""B3: resource-constrained route search, A* with Pareto labels.

A plain shortest-path search keeps one best cost per cell. That is wrong here:
the cheapest way to reach a cell may have used more of a resource than a slightly
longer one, and only the second can still finish. So each cell keeps up to ``K``
labels that do not dominate one another.

    label       position, time, the traveller's resources and risks, cost
    order       priority = cost + heuristic,  heuristic = w_t * straight distance / V_max
    prune       the rules of the traveller's constraint checker
    dominance   Y dominates X when Y is no worse in cost and in every state variable
    trim        more than K labels in a cell: the most expensive one is dropped

The search does not know who travels. Moves, stays in place (an astronaut
resting, a rover recharging), dominance and the constraints all come from the
``TraversalModel`` and ``Constraints`` it is given (see ``agent``).

The heuristic is admissible because every cost term is non-negative and no step
is faster than the model's top speed. With ``K`` finite the search is a
heuristic: it can miss a feasible route whose label was trimmed.

Every rejected state is counted per rule, together with the farthest distance
reached before the rule fired. When no route exists, the rule that fired most
often is reported as the bottleneck.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic

from ..core.schemas import Cell, PruneRule
from .agent import Constraints, L, TraversalModel
from .transition import dominates

__all__ = ["ParetoAStar", "SearchDiagnostics", "SearchResult", "dominates", "path_of",
           "describe_astronaut_rule"]

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


def describe_astronaut_rule(rule: Enum, detail: str, distance_km: float) -> str:
    """Readable sentence for an astronaut constraint that blocked the search."""
    template = _BOTTLENECK_TEXT.get(rule)  # type: ignore[call-overload]
    if template is None:
        return f"{rule.value} ({detail}) after {distance_km:.1f} km"
    return template.format(detail=detail, km=distance_km)


@dataclass(slots=True)
class SearchDiagnostics:
    """Telemetry of one search.

    Attributes:
        expanded: labels taken from the queue and expanded.
        generated: labels created.
        infeasible_edges: moves rejected by the physics (terrain, power, vehicle limits).
        dominated: labels discarded because another label was no worse in everything.
        trimmed: labels discarded because a cell already held K cheaper ones.
        prune_counts: how often each constraint rejected a label.
        prune_max_distance_m: farthest distance travelled by a label rejected by each rule.
        prune_details: per rule, how often each detail (for example the resource) occurred.
        budget_exhausted: the search stopped at ``max_expansions``.
        describe: turns a rule into a sentence; supplied by the constraint checker.
    """

    expanded: int = 0
    generated: int = 0
    infeasible_edges: int = 0
    dominated: int = 0
    trimmed: int = 0
    prune_counts: dict[Any, int] = field(default_factory=dict)
    prune_max_distance_m: dict[Any, float] = field(default_factory=dict)
    prune_details: dict[Any, dict[str, int]] = field(default_factory=dict)
    budget_exhausted: bool = False
    describe: Callable[[Enum, str, float], str] = describe_astronaut_rule

    def record(self, rule: Enum, detail: str, distance_m: float) -> None:
        """Count one rejection by ``rule``."""
        self.prune_counts[rule] = self.prune_counts.get(rule, 0) + 1
        if distance_m > self.prune_max_distance_m.get(rule, -1.0):
            self.prune_max_distance_m[rule] = distance_m
        details = self.prune_details.setdefault(rule, {})
        details[detail] = details.get(detail, 0) + 1

    def bottleneck(self) -> tuple[Any, str]:
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
            return rule, self.describe(rule, detail, km)
        if self.budget_exhausted:
            return None, "search budget exhausted before the goal was reached"
        if self.infeasible_edges:
            return None, "the goal is cut off by impassable, forbidden or too steep terrain"
        return None, "the goal is not connected to the start"


@dataclass(slots=True)
class SearchResult(Generic[L]):
    """Outcome of a route search.

    Attributes:
        found: a feasible route exists.
        goal_label: final state at the goal (``None`` when not found).
        path: cells of the route in travel order, stays in place collapsed.
        diagnostics: telemetry of the search.
        message: "ok" or the bottleneck description.
        bottleneck: the blocking constraint when not found.
    """

    found: bool
    goal_label: L | None
    path: list[Cell]
    diagnostics: SearchDiagnostics
    message: str
    bottleneck: Any = None


def path_of(label: Any) -> list[Cell]:
    """Cells visited on the way to ``label``, consecutive repeats (stays in place) collapsed."""
    cells: list[Cell] = []
    for node in label.chain():
        if not cells or cells[-1] != node.cell:
            cells.append(node.cell)
    return cells


class ParetoAStar(Generic[L]):
    """Pareto-label A* over the 8-connected grid, for any traveller."""

    def __init__(self, model: TraversalModel[L], checker: Constraints[L]) -> None:
        """
        Args:
            model: physics of the traveller (an astronaut's or a rover's evaluator).
            checker: the traveller's hard constraints.
        """
        self._model = model
        self._checker = checker
        self._cfg = model.config.routing

    def search(self, start: L, goal: Cell) -> SearchResult[L]:
        """Cheapest feasible route from the start label to ``goal``.

        Never raises for an unreachable goal: the result then has ``found=False``
        and names the bottleneck.

        Raises:
            TerrainValidationError: the goal lies outside the grid.
        """
        if not self._model.in_bounds(goal):
            self._model.require_walkable(goal, "goal")
        goal_label, _, diagnostics = self._run(start, goal, self._cfg.max_labels_per_cell)
        if goal_label is not None:
            return SearchResult(True, goal_label, path_of(goal_label), diagnostics, "ok")
        rule, message = diagnostics.bottleneck()
        return SearchResult(False, None, [], diagnostics, message, rule)

    def explore(self, start: L, max_labels_per_cell: int = 1
                ) -> tuple[dict[Cell, L], SearchDiagnostics]:
        """Cheapest feasible label for every cell reachable from the start (no goal).

        Returns:
            ``(best label per reachable cell, diagnostics)``.
        """
        _, labels, diagnostics = self._run(start, None, max_labels_per_cell)
        best: dict[Cell, L] = {}
        for cell, bucket in labels.items():
            alive = [label for label in bucket if label.alive]
            if alive:
                best[cell] = min(alive, key=lambda label: label.cost)
        return best, diagnostics

    # ---- core loop ------------------------------------------------------
    def _run(self, start: L, goal: Cell | None, k: int
             ) -> tuple[L | None, dict[Cell, list[L]], SearchDiagnostics]:
        model, checker, cfg = self._model, self._checker, self._cfg
        diagnostics = SearchDiagnostics(describe=checker.describe)
        labels: dict[Cell, list[L]] = {}
        start_violation = checker.violated(start)
        if start_violation is not None:
            diagnostics.record(start_violation[0], start_violation[1], start.distance_m)
            return None, labels, diagnostics

        if goal is None:
            def heuristic(row: int, col: int) -> float:
                return 0.0
        else:
            scale = model.time_weight_per_s * model.cell_size_m / model.max_speed_m_s
            goal_row, goal_col = goal

            def heuristic(row: int, col: int) -> float:
                return scale * math.hypot(row - goal_row, col - goal_col)

        counter = itertools.count()
        labels[start.cell] = [start]
        heap: list[tuple[float, int, L]] = [
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

            candidates, refused = model.successors(label)
            diagnostics.infeasible_edges += refused
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

    def _insert(self, bucket: list[L], new: L, k: int, tol: float,
                diagnostics: SearchDiagnostics) -> bool:
        """Add ``new`` to the Pareto set of its cell. False when it is not kept."""
        dominates_fn = self._model.dominates
        for old in bucket:
            if dominates_fn(old, new, tol):
                diagnostics.dominated += 1
                return False
        kept: list[L] = []
        for old in bucket:
            if dominates_fn(new, old, tol):
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
