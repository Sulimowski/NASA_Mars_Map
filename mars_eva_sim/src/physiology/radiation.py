"""A6: radiation dose accounting, limits, solar particle events, ultraviolet exposure.

Dose over a step::

    rate = dose_rate_gcr * suit.dose_factor + spe_dose_rate * [spe_active]      [Sv/s]
    d_dose = rate * dt

Limits (NASA-STD-3001)::

    career                       0.6 Sv effective dose
    blood-forming organs         0.25 Gy-Eq in any 30 days, 0.5 Gy-Eq in any year

The 30-day and 1-year limits are true rolling windows: ``DoseLedger`` keeps the
dose in time bins on two deques and drops bins as they leave each window. The
effective dose is converted to blood-forming-organ dose with ``bfo_gyeq_per_sv``.

Solar particle event: a cell is allowed only if the nearest shelter can be reached
within the warning time between the flare and the arrival of the protons.

The suit is opaque to ultraviolet, so UV does not harm the crew; the exposure
is accumulated only as wear of the visor and outer fabric.
"""

from __future__ import annotations

from collections import deque

from ..core.config import RadiationConfig
from ..core.exceptions import ValidationError
from ..core.schemas import EnvSample


def dose_rate_sv_s(env: EnvSample, suit_dose_factor: float) -> float:
    """Dose-equivalent rate received inside the suit, Sv/s."""
    rate = env.dose_rate_gcr_sv_s * suit_dose_factor
    if env.spe_active:
        rate += env.spe_dose_rate_sv_s
    return rate


def dose_increment_sv(env: EnvSample, dt_s: float, suit_dose_factor: float) -> float:
    """Dose received over ``dt_s``, Sv."""
    return dose_rate_sv_s(env, suit_dose_factor) * dt_s


def uv_exposure_increment(env: EnvSample, dt_s: float, cfg: RadiationConfig) -> float:
    """Ultraviolet wear accumulated over ``dt_s``: (uv_index / reference) * seconds."""
    return env.uv_index / cfg.uv_reference_index * dt_s


def spe_shelter_reachable(shelter_time_s: float, cfg: RadiationConfig) -> bool:
    """True when the nearest shelter is within the solar particle event warning time."""
    return shelter_time_s <= cfg.spe_warning_time_s


class _RollingWindow:
    """Sum of binned doses over the last ``window_s`` seconds."""

    __slots__ = ("window_s", "bins", "total")

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.bins: deque[list[float]] = deque()   # [bin_start_s, dose_sv]
        self.total = 0.0

    def add(self, bin_start_s: float, dose_sv: float) -> None:
        if self.bins and self.bins[-1][0] == bin_start_s:
            self.bins[-1][1] += dose_sv
        else:
            self.bins.append([bin_start_s, dose_sv])
        self.total += dose_sv

    def evict(self, now_s: float) -> None:
        horizon = now_s - self.window_s
        while self.bins and self.bins[0][0] <= horizon:
            self.total -= self.bins.popleft()[1]
        if not self.bins:
            self.total = 0.0   # remove accumulated rounding error


class DoseLedger:
    """Personal dose record with rolling 30-day and 1-year windows.

    Doses are stored in bins of ``cfg.ledger_bin_s``; a bin leaves a window once
    its start is older than the window length. Time must not go backwards.
    """

    def __init__(self, cfg: RadiationConfig, career_dose_sv: float = 0.0) -> None:
        """
        Args:
            cfg: A6 parameters.
            career_dose_sv: dose received before the ledger starts (transit included).

        Raises:
            ValidationError: the initial career dose is negative.
        """
        if career_dose_sv < 0.0:
            raise ValidationError("career_dose_sv must be non-negative")
        self._cfg = cfg
        self._career_sv = career_dose_sv
        self._w30 = _RollingWindow(cfg.window_30d_s)
        self._w1y = _RollingWindow(cfg.window_1y_s)
        self._last_t_s = float("-inf")

    @property
    def career_sv(self) -> float:
        """Total career dose, Sv."""
        return self._career_sv

    def add(self, t_s: float, dose_sv: float) -> None:
        """Record ``dose_sv`` received at mission time ``t_s``.

        Raises:
            ValidationError: the dose is negative or the time is earlier than the last entry.
        """
        if dose_sv < 0.0:
            raise ValidationError("dose_sv must be non-negative")
        if t_s < self._last_t_s:
            raise ValidationError(f"time went backwards: {t_s} < {self._last_t_s}")
        self._last_t_s = t_s
        bin_start = (t_s // self._cfg.ledger_bin_s) * self._cfg.ledger_bin_s
        for window in (self._w30, self._w1y):
            window.evict(t_s)
            window.add(bin_start, dose_sv)
        self._career_sv += dose_sv

    def dose_30d_sv(self, t_s: float) -> float:
        """Dose received in the 30 days before ``t_s``, Sv."""
        self._w30.evict(t_s)
        return self._w30.total

    def dose_1y_sv(self, t_s: float) -> float:
        """Dose received in the year before ``t_s``, Sv."""
        self._w1y.evict(t_s)
        return self._w1y.total

    def remaining_budget_sv(self, t_s: float) -> float:
        """Largest additional dose at ``t_s`` that keeps every limit, Sv (never negative)."""
        k = self._cfg.bfo_gyeq_per_sv
        return max(0.0, min(
            self._cfg.career_limit_sv - self._career_sv,
            self._cfg.bfo_30d_limit_gyeq / k - self.dose_30d_sv(t_s),
            self._cfg.bfo_1y_limit_gyeq / k - self.dose_1y_sv(t_s),
        ))

    def violations(self, t_s: float, extra_dose_sv: float = 0.0) -> list[str]:
        """Names of the limits that ``extra_dose_sv`` received at ``t_s`` would break."""
        k = self._cfg.bfo_gyeq_per_sv
        out: list[str] = []
        if self._career_sv + extra_dose_sv > self._cfg.career_limit_sv:
            out.append("career")
        if (self.dose_30d_sv(t_s) + extra_dose_sv) * k > self._cfg.bfo_30d_limit_gyeq:
            out.append("bfo_30d")
        if (self.dose_1y_sv(t_s) + extra_dose_sv) * k > self._cfg.bfo_1y_limit_gyeq:
            out.append("bfo_1y")
        return out


def eva_dose_budget_sv(cfg: RadiationConfig, ledger: DoseLedger | None, t_s: float) -> float:
    """Dose the crew member may receive during one EVA starting at ``t_s``, Sv.

    The smaller of the per-EVA allowance and what the ledger still permits.
    """
    if ledger is None:
        return cfg.per_eva_limit_sv
    return min(cfg.per_eva_limit_sv, ledger.remaining_budget_sv(t_s))
