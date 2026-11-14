"""H6: spreading the surface dose budget over the sols that are left.

A per-EVA dose limit says little on Mars. The suit barely shields, so the dose
on an EVA is close to the dose in an unshielded habitat, and a quiet sol gives
about 0.67 mSv whatever the crew does. What can be managed is the mission
budget and which sols are spent outside::

    allocation = (budget - received) / sols_remaining            [Sv per sol]
    a sol is acceptable when   dose_sol <= allocation * (1 + margin)

and, independently, the rolling limits of the personal dose ledger (A6) must hold.

The EVA share of a sol's dose is ``dose_sol * eva_duration / sol``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.config import SolSelectionConfig
from ..core.constants import SOL_S
from ..core.exceptions import ValidationError
from ..physiology.radiation import DoseLedger


@dataclass(frozen=True, slots=True)
class DoseDecision:
    """Outcome of the dose check for one sol.

    Attributes:
        allowed: the sol fits the allocation and the ledger limits.
        allocation_sv: dose the budget allows for this sol.
        expected_sol_sv: dose expected over the whole sol.
        expected_eva_sv: part of it received during the EVA.
        reasons: why the sol is not allowed (empty when it is).
    """

    allowed: bool
    allocation_sv: float
    expected_sol_sv: float
    expected_eva_sv: float
    reasons: tuple[str, ...]


def sol_allocation_sv(budget_sv: float, received_sv: float, sols_remaining: int) -> float:
    """Dose the mission budget allows per remaining sol, Sv (never negative).

    Raises:
        ValidationError: ``sols_remaining`` is not positive or a dose is negative.
    """
    if sols_remaining <= 0:
        raise ValidationError(f"sols_remaining={sols_remaining} must be positive")
    if budget_sv < 0.0 or received_sv < 0.0:
        raise ValidationError("budget_sv and received_sv must be non-negative")
    return max(0.0, budget_sv - received_sv) / sols_remaining


def eva_dose_sv(dose_sol_sv: float, eva_duration_s: float) -> float:
    """Part of a sol's dose received during an EVA of ``eva_duration_s``, Sv."""
    return dose_sol_sv * max(0.0, eva_duration_s) / SOL_S


def assess_dose(dose_sol_sv: float, *, sols_remaining: int, received_sv: float,
                eva_duration_s: float, cfg: SolSelectionConfig,
                ledger: DoseLedger | None = None, t_s: float = 0.0) -> DoseDecision:
    """Check one sol against the mission allocation and the ledger limits.

    Args:
        dose_sol_sv: dose expected over the sol.
        sols_remaining: surface sols left in the mission, this one included.
        received_sv: surface dose received so far.
        eva_duration_s: planned time outside.
        ledger: personal dose record; its rolling limits are checked when given.
        t_s: mission time of the sol, for the ledger windows.
    """
    allocation = sol_allocation_sv(cfg.mission_dose_budget_sv, received_sv, sols_remaining)
    reasons: list[str] = []
    if dose_sol_sv > allocation * (1.0 + cfg.dose_margin):
        reasons.append(f"sol dose {dose_sol_sv * 1e3:.2f} mSv exceeds the allocation "
                       f"{allocation * 1e3:.2f} mSv by more than {cfg.dose_margin:.0%}")
    if ledger is not None:
        for limit in ledger.violations(t_s, dose_sol_sv):
            reasons.append(f"dose limit '{limit}' would be exceeded")
    return DoseDecision(allowed=not reasons, allocation_sv=allocation,
                        expected_sol_sv=dose_sol_sv,
                        expected_eva_sv=eva_dose_sv(dose_sol_sv, eva_duration_s),
                        reasons=tuple(reasons))
