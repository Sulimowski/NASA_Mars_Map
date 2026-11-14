"""The eight hard constraints of the route search, in one reusable checker.

A state is rejected when any rule holds:

    1 RESOURCE_RESERVE   a consumable is below its reserve share of a full suit
    2 RETURN_O2          oxygen left - margin * oxygen to walk back < oxygen reserve
    3 RETURN_TIME        EVA time left - margin * time to walk back < time reserve
    4 EMERGENCY_RADIUS   (strict mode) no shelter within the secondary oxygen
    5 SPE_SHELTER        particle event forecast or in progress, and no shelter
                         within the warning time
    6 FATIGUE            F above the stop level, or above the return-only level
                         while moving away from the base
    7 DOSE_LIMIT         trip dose above the dose budget
    8 HEAT_STORAGE       stored body heat beyond its limit

The search, the mission planner and the route replay all use this class, so a
route accepted by one is accepted by the others.
"""

from __future__ import annotations

from ..core.schemas import Cell, CoolingMode, Label, PruneRule
from ..physiology import emergency
from .shelter_maps import ShelterMaps
from .transition import EdgeEvaluator


class ConstraintChecker:
    """Evaluates the pruning rules for a label."""

    def __init__(self, evaluator: EdgeEvaluator, maps: ShelterMaps, *, dose_budget_sv: float,
                 spe_watch: bool = False) -> None:
        """
        Args:
            evaluator: edge physics (gives the suit, the configuration and the environment).
            maps: safety maps of the site.
            dose_budget_sv: dose the crew member may receive on this EVA.
            spe_watch: a solar particle event is forecast, so rule 5 applies everywhere.
        """
        cfg = evaluator.config
        suit = evaluator.suit
        share = cfg.routing.reserve_fraction
        self._evaluator = evaluator
        self._o2_reserve = share * suit.o2_primary_kg
        self._co2_reserve = share * suit.co2_capacity_kg
        self._battery_reserve = share * suit.battery_j
        self._time_reserve = share * suit.max_eva_s
        self._water_reserve = (share * suit.cooling_water_kg
                               if suit.cooling_mode is CoolingMode.MEMBRANE_EVAPORATOR else None)
        self._margin = cfg.routing.return_margin
        self._strict = cfg.routing.strict_emergency
        self._emergency_limit = emergency.emergency_time_limit_s(suit, cfg.emergency)
        self._spe_watch = spe_watch
        self._spe_warning = cfg.radiation.spe_warning_time_s
        self._fatigue_stop = cfg.fatigue.stop
        self._fatigue_return = cfg.fatigue.return_only
        self._dose_budget = dose_budget_sv
        self._heat_limit = evaluator.heat_storage_limit_j
        self._return_o2: list[list[float]] = maps.return_o2_kg.tolist()
        self._return_time: list[list[float]] = maps.return_time_s.tolist()
        self._shelter_time: list[list[float]] = maps.shelter_time_s.tolist()
        self._emergency_time: list[list[float]] = maps.emergency_time_s.tolist()

    @property
    def dose_budget_sv(self) -> float:
        """Dose allowed on this EVA, Sv."""
        return self._dose_budget

    def return_time_s(self, cell: Cell) -> float:
        """Nominal-pace walking time from ``cell`` back to the base, s."""
        return self._return_time[cell[0]][cell[1]]

    def violated(self, label: Label, prev_cell: Cell | None = None
                 ) -> tuple[PruneRule, str] | None:
        """First rule the label breaks, with a short detail, or ``None`` when it is feasible.

        Args:
            label: state to test.
            prev_cell: cell the crew member came from; needed by the fatigue rule
                to tell moving away from the base from moving toward it.
        """
        row, col = label.row, label.col
        # 1. reserves
        if label.o2_kg < self._o2_reserve:
            return PruneRule.RESOURCE_RESERVE, "O2"
        if label.eva_remaining_s < self._time_reserve:
            return PruneRule.RESOURCE_RESERVE, "EVA time"
        if label.battery_j < self._battery_reserve:
            return PruneRule.RESOURCE_RESERVE, "battery"
        if label.co2_capacity_kg < self._co2_reserve:
            return PruneRule.RESOURCE_RESERVE, "CO2 scrubber"
        if self._water_reserve is not None and label.cooling_water_kg < self._water_reserve:
            return PruneRule.RESOURCE_RESERVE, "cooling water"
        # 2. oxygen for the walk back
        if label.o2_kg - self._margin * self._return_o2[row][col] < self._o2_reserve:
            return PruneRule.RETURN_O2, "O2"
        # 3. time for the walk back
        if label.eva_remaining_s - self._margin * self._return_time[row][col] < self._time_reserve:
            return PruneRule.RETURN_TIME, "EVA time"
        # 4. emergency radius
        if self._strict and self._emergency_time[row][col] > self._emergency_limit:
            return PruneRule.EMERGENCY_RADIUS, "secondary O2"
        # 5. solar particle event
        if self._shelter_time[row][col] > self._spe_warning and (
                self._spe_watch or self._evaluator.env_at((row, col), label.t_s).spe_active):
            return PruneRule.SPE_SHELTER, "shelter"
        # 6. fatigue
        if label.fatigue > self._fatigue_stop:
            return PruneRule.FATIGUE, "stop level"
        if (label.fatigue > self._fatigue_return and prev_cell is not None
                and label.action == "walk"
                and self._return_time[row][col] >= self._return_time[prev_cell[0]][prev_cell[1]]):
            return PruneRule.FATIGUE, "return-only level"
        # 7. dose
        if label.dose_sv > self._dose_budget:
            return PruneRule.DOSE_LIMIT, "dose"
        # 8. heat
        if abs(label.heat_storage_j) > self._heat_limit:
            return PruneRule.HEAT_STORAGE, "overheating" if label.heat_storage_j > 0 else "cold"
        return None
