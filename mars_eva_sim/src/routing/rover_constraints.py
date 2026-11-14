"""Hard limits on a rover's state during a traverse.

A state is rejected when any rule holds:

    BATTERY_RESERVE   stored energy fell below the safe-mode threshold at any moment
    THERMAL_WINDOW    the heaters cannot keep the rover inside its temperature window
    TIME_BUDGET       the traverse has lasted longer than allowed
    RISK_BUDGET       accumulated probability of a radiation upset above the budget

A rover needs no shelter and has no way home to protect, so none of the
astronaut's return and shelter rules apply.
"""

from __future__ import annotations

from enum import Enum

from ..core.schemas import Cell
from ..rovers.types import RoverLabel, RoverRule
from .rover_transition import RoverEdgeEvaluator

_TEXT: dict[RoverRule, str] = {
    RoverRule.BATTERY_RESERVE: "battery at the safe-mode threshold after {km:.2f} km",
    RoverRule.THERMAL_WINDOW: "too cold for the heaters ({detail}) after {km:.2f} km",
    RoverRule.TIME_BUDGET: "traverse time budget used up after {km:.2f} km",
    RoverRule.RISK_BUDGET: "radiation upset risk budget used up after {km:.2f} km",
}


class RoverConstraintChecker:
    """Evaluates the rover rules for a state."""

    def __init__(self, evaluator: RoverEdgeEvaluator, *, start_time_s: float,
                 max_duration_s: float | None = None) -> None:
        """
        Args:
            evaluator: the rover's physics (gives the profile and the configuration).
            start_time_s: mission time at which the traverse starts.
            max_duration_s: longest allowed traverse; the configured default when omitted.
        """
        cfg = evaluator.config.rover
        self._reserve = evaluator.profile.battery_reserve_j
        self._deadline = start_time_s + (max_duration_s if max_duration_s is not None
                                         else cfg.max_traverse_s)
        self._risk_budget = cfg.risk_budget

    def violated(self, label: RoverLabel, prev_cell: Cell | None = None
                 ) -> tuple[RoverRule, str] | None:
        """First rule the state breaks, with a short detail, or ``None`` when it is feasible."""
        if min(label.battery_j, label.min_battery_j) < self._reserve:
            return RoverRule.BATTERY_RESERVE, "battery"
        if label.heater_deficit_w > 1.0e-9:
            return RoverRule.THERMAL_WINDOW, f"{label.heater_deficit_w:.0f} W short"
        if label.t_s > self._deadline:
            return RoverRule.TIME_BUDGET, "time"
        if label.risk > self._risk_budget:
            return RoverRule.RISK_BUDGET, "upset risk"
        return None

    def describe(self, rule: Enum, detail: str, distance_km: float) -> str:
        """Readable sentence for a rule that blocked the search."""
        template = _TEXT.get(rule)  # type: ignore[call-overload]
        if template is None:
            return f"{rule.value} ({detail}) after {distance_km:.2f} km"
        return template.format(detail=detail, km=distance_km)
