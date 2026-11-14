"""H3.1: which sols are fit for an EVA, and which is best.

Inputs are the per-sol series every site carries: the radiation dose and,
when present, the daily ground temperature extremes. There is no dust or wind
criterion because the sites have no such data.

A sol is rejected when any rule holds:

    particle event    dose above ``spe_dose_factor`` times the median dose
    event guard       within ``spe_guard_sols`` sols of a particle event
    dose allocation   dose above the per-sol allocation of the mission budget (optional)
    too cold / hot    afternoon ground temperature outside the allowed band
    short daylight    less than ``min_window_s`` between sunrise + margin and sunset - margin

Accepted sols are ranked by::

    score = w_dose * (1 - dose / median dose) + w_warmth * warmth

where ``warmth`` is the sol's afternoon ground temperature scaled to 0..1 over
the series.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable

from ..core.config import SolSelectionConfig
from ..core.constants import SOL_S, ZERO_CELSIUS_K
from ..core.exceptions import ValidationError
from ..providers.site_adapter import site_sols
from .diurnal import day_length_s


@dataclass(frozen=True, slots=True)
class SolSeries:
    """Per-sol inputs of the selection, in SI.

    Attributes:
        sols: sol numbers.
        dose_sv: dose equivalent of each sol.
        ground_temp_min_k, ground_temp_max_k: daily ground temperature extremes,
            or ``None`` when the site has no temperature series.
    """

    sols: tuple[int, ...]
    dose_sv: tuple[float, ...]
    ground_temp_min_k: tuple[float, ...] | None = None
    ground_temp_max_k: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        count = len(self.sols)
        if count == 0:
            raise ValidationError("the sol series is empty")
        if len(set(self.sols)) != count:
            raise ValidationError("sol numbers must be unique")
        for name in ("dose_sv", "ground_temp_min_k", "ground_temp_max_k"):
            values = getattr(self, name)
            if values is not None and len(values) != count:
                raise ValidationError(f"series '{name}' has {len(values)} entries, expected {count}")
        if any(not math.isfinite(d) or d < 0.0 for d in self.dose_sv):
            raise ValidationError("dose_sv must be finite and non-negative")
        if (self.ground_temp_min_k is None) != (self.ground_temp_max_k is None):
            raise ValidationError("give both ground temperature series or neither")


@dataclass(frozen=True, slots=True)
class SolAssessment:
    """Verdict for one sol.

    Attributes:
        sol: sol number.
        allowed: an EVA may be planned on this sol.
        score: ranking value; higher is better (only meaningful among allowed sols).
        reasons: why the sol is rejected (empty when allowed).
        dose_sv: dose of the sol.
        particle_event: the sol itself carries a particle event.
        window_s: EVA window as (start, end) seconds after local midnight, or ``None``.
    """

    sol: int
    allowed: bool
    score: float
    reasons: tuple[str, ...]
    dose_sv: float
    particle_event: bool
    window_s: tuple[float, float] | None


def sol_series_from_site(site: Any, cfg: SolSelectionConfig) -> SolSeries:
    """Read the selection inputs from a ``Site`` of the data contract.

    ``radiation_msv_per_sol`` becomes the dose (the typical dose when absent).
    Temperatures come from ``ground_temp_min_c`` / ``ground_temp_max_c`` when the
    site has them, else from ``temp_min_c`` / ``temp_max_c``, else there are none.

    Raises:
        ValidationError: the site has no series at all.
    """
    sols = site_sols(site)
    if not sols:
        raise ValidationError(f"site '{getattr(site, 'name', '?')}' has no per-sol series")
    series = site.series
    doses = series.get("radiation_msv_per_sol")
    dose_sv = (tuple(float(d) * 1.0e-3 for d in doses) if doses
               else tuple(cfg.typical_dose_sv_per_sol for _ in sols))
    t_min = t_max = None
    for low, high in (("ground_temp_min_c", "ground_temp_max_c"), ("temp_min_c", "temp_max_c")):
        if series.get(low) and series.get(high):
            t_min = tuple(float(v) + ZERO_CELSIUS_K for v in series[low])
            t_max = tuple(float(v) + ZERO_CELSIUS_K for v in series[high])
            break
    return SolSeries(sols=tuple(sols), dose_sv=dose_sv, ground_temp_min_k=t_min,
                     ground_temp_max_k=t_max)


def detect_particle_events(dose_sv: tuple[float, ...], cfg: SolSelectionConfig) -> list[bool]:
    """Flag the sols whose dose is far above the median of the series."""
    median = statistics.median(dose_sv)
    if median <= 0.0:
        return [d > 0.0 for d in dose_sv]
    return [d > cfg.spe_dose_factor * median for d in dose_sv]


def eva_window_s(latitude_rad: float, declination: float,
                 cfg: SolSelectionConfig) -> tuple[float, float] | None:
    """Daylight window for an EVA as (start, end) seconds after local midnight.

    From ``daylight_margin_s`` after sunrise to the same before sunset; ``None``
    when that leaves less than ``min_window_s``.
    """
    day = day_length_s(latitude_rad, declination)
    start = (SOL_S - day) / 2.0 + cfg.daylight_margin_s
    end = (SOL_S + day) / 2.0 - cfg.daylight_margin_s
    if end - start < cfg.min_window_s:
        return None
    return start, end


def assess_sols(series: SolSeries, cfg: SolSelectionConfig, *, latitude_rad: float = 0.0,
                declination_of_sol: Callable[[int], float] | None = None,
                dose_allocation_sv: float | None = None) -> list[SolAssessment]:
    """Assess every sol of the series, in series order.

    Args:
        latitude_rad: site latitude.
        declination_of_sol: solar declination (rad) for a sol number; equinox when omitted.
        dose_allocation_sv: per-sol dose the mission budget allows (see
            ``dose_budget.sol_allocation_sv``); the allocation rule is skipped when omitted.
    """
    events = detect_particle_events(series.dose_sv, cfg)
    median = statistics.median(series.dose_sv)
    count = len(series.sols)
    t_max = series.ground_temp_max_k
    warm_low = min(t_max) if t_max else 0.0
    warm_span = (max(t_max) - warm_low) if t_max else 0.0

    out: list[SolAssessment] = []
    for index, sol in enumerate(series.sols):
        dose = series.dose_sv[index]
        reasons: list[str] = []
        if events[index]:
            reasons.append("solar particle event")
        else:
            near = range(max(0, index - cfg.spe_guard_sols),
                         min(count, index + cfg.spe_guard_sols + 1))
            if any(events[i] and abs(series.sols[i] - sol) <= cfg.spe_guard_sols for i in near):
                reasons.append("within the guard interval of a solar particle event")
        if dose_allocation_sv is not None and dose > dose_allocation_sv * (1.0 + cfg.dose_margin):
            reasons.append("dose above the mission allocation for one sol")
        warmth = 0.0
        if t_max is not None:
            if t_max[index] < cfg.min_daytime_ground_temp_k:
                reasons.append("ground too cold even in the afternoon")
            elif t_max[index] > cfg.max_daytime_ground_temp_k:
                reasons.append("ground too hot in the afternoon")
            warmth = (t_max[index] - warm_low) / warm_span if warm_span > 0.0 else 0.5
        declination = declination_of_sol(sol) if declination_of_sol is not None else 0.0
        window = eva_window_s(latitude_rad, declination, cfg)
        if window is None:
            reasons.append("daylight window too short")
        dose_term = 1.0 - dose / median if median > 0.0 else 0.0
        score = cfg.weight_dose * dose_term + cfg.weight_warmth * warmth
        out.append(SolAssessment(sol=sol, allowed=not reasons, score=score,
                                 reasons=tuple(reasons), dose_sv=dose,
                                 particle_event=events[index], window_s=window))
    return out


def rank_sols(assessments: list[SolAssessment]) -> list[SolAssessment]:
    """Allowed sols, best first (ties broken by the earlier sol)."""
    allowed = [a for a in assessments if a.allowed]
    return sorted(allowed, key=lambda a: (-a.score, a.sol))


def best_sol(assessments: list[SolAssessment]) -> SolAssessment | None:
    """The best allowed sol, or ``None`` when every sol is rejected."""
    ranked = rank_sols(assessments)
    return ranked[0] if ranked else None
