"""A7: whole-body fatigue, hand fatigue, daily EVA limits.

Whole-body fatigue ``F`` in [0, 1], driven by ``r = M / P_sustain``::

    dF/dt = k_up * (r - growth_threshold)^2        r > growth_threshold
    dF/dt = 0                                      rest_ratio <= r <= growth_threshold
    dF/dt = -k_rec * F                             r < rest_ratio      (exact: F * exp(-k_rec * dt))

Hand fatigue ``H`` in [0, 1]: gripping a pressurised glove is the usual limiter of
real EVAs::

    dH/dt = k_h * grip_load * (P_suit / P_ref)     grip_load > 0
    dH/dt = -k_hr * H                              hands free

Thresholds: rest when ``F > rest_trigger``; only moves toward the base when
``F > return_only``; no movement when ``F > stop``; no tool work when ``H > hand_limit``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.config import CapacityConfig, FatigueConfig
from ..core.exceptions import ValidationError


def update_fatigue(fatigue: float, metabolic_w: float, sustainable_w: float, dt_s: float,
                   cfg: FatigueConfig) -> float:
    """Whole-body fatigue after working at ``metabolic_w`` for ``dt_s``.

    Raises:
        ValidationError: ``sustainable_w`` is not positive or ``dt_s`` is negative.
    """
    if sustainable_w <= 0.0 or dt_s < 0.0:
        raise ValidationError(f"need sustainable_w > 0 and dt_s >= 0, got {sustainable_w}, {dt_s}")
    ratio = metabolic_w / sustainable_w
    if ratio < cfg.rest_ratio:
        return fatigue * math.exp(-cfg.recovery_rate_s * dt_s)
    excess = ratio - cfg.growth_threshold
    if excess <= 0.0:
        return fatigue
    return min(1.0, fatigue + cfg.growth_rate_s * excess * excess * dt_s)


def time_to_fatigue_s(fatigue: float, target: float, metabolic_w: float, sustainable_w: float,
                      cfg: FatigueConfig) -> float:
    """Time of continuous work at ``metabolic_w`` until fatigue reaches ``target``, s.

    ``math.inf`` when the workload does not build fatigue; 0 when already there.
    """
    if fatigue >= target:
        return 0.0
    excess = metabolic_w / sustainable_w - cfg.growth_threshold
    if excess <= 0.0:
        return math.inf
    return (target - fatigue) / (cfg.growth_rate_s * excess * excess)


def update_hand_fatigue(hand_fatigue: float, grip_load: float, suit_pressure_pa: float,
                        dt_s: float, cfg: FatigueConfig) -> float:
    """Hand fatigue after ``dt_s`` at the given grip load (0 = hands free)."""
    if grip_load <= 0.0:
        return hand_fatigue * math.exp(-cfg.hand_recovery_rate_s * dt_s)
    growth = (cfg.hand_growth_rate_s * grip_load
              * (suit_pressure_pa / cfg.hand_reference_pressure_pa) * dt_s)
    return min(1.0, hand_fatigue + growth)


def tool_work_allowed(hand_fatigue: float, cfg: FatigueConfig) -> bool:
    """False once the hands are too tired for tool work."""
    return hand_fatigue <= cfg.hand_limit


@dataclass(frozen=True, slots=True)
class DailyEvaRecord:
    """EVA history used by the daily limits.

    Attributes:
        evas_this_sol: EVAs already done in the current sol.
        suit_time_this_sol_s: time already spent in the suit in the current sol.
        consecutive_eva_sols: sols in a row with an EVA, not counting the current one.
        sleep_debt_s: accumulated sleep debt.
    """

    evas_this_sol: int = 0
    suit_time_this_sol_s: float = 0.0
    consecutive_eva_sols: int = 0
    sleep_debt_s: float = 0.0


def daily_eva_violations(record: DailyEvaRecord, planned_suit_time_s: float,
                         cfg: FatigueConfig, capacity: CapacityConfig) -> list[str]:
    """Daily rules a planned EVA would break (empty list = the EVA may start).

    Rules: one EVA per sol, at most ``max_suit_time_per_sol_s`` in the suit per sol,
    a rest sol after ``max_consecutive_eva_sols`` EVA sols, sleep debt below the limit.
    """
    out: list[str] = []
    if record.evas_this_sol >= cfg.max_evas_per_sol:
        out.append("an EVA was already done this sol")
    if record.suit_time_this_sol_s + planned_suit_time_s > cfg.max_suit_time_per_sol_s:
        out.append("planned suit time exceeds the limit per sol")
    if record.consecutive_eva_sols >= cfg.max_consecutive_eva_sols:
        out.append("a rest sol is required after consecutive EVA sols")
    if record.sleep_debt_s > capacity.max_sleep_debt_s:
        out.append("sleep debt is above the limit")
    return out
