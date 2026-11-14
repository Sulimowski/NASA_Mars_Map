"""A3: how much power the crew member can hold, the walking speed that fits it, heart rate.

Power::

    P_aerobic = VO2max_earth * f_aerobic * W * E_O2          [W], 100 % of VO2max now
    P_sustain = sustained_fraction * P_aerobic               hours
    P_peak    = peak_fraction * P_aerobic                    10-15 minutes
    P_cap     = min(P_base, P_thermal) * (1 - penalty * F) * readiness * hydration * fed

where ``E_O2`` is the energy equivalent of oxygen (J/m^3) and ``P_thermal`` is the
largest metabolic power the suit cooling can carry away (from A5).

Speed: the highest ``V <= V_upper`` with ``M(V) <= P_cap``.

* Level and uphill (``grade >= 0``): ``M(V)`` grows with ``V``, so a binary search
  finds the limit.
* Downhill (``grade < 0``): the linear term is negative, ``M(V)`` is not monotonic,
  so the speeds are scanned from ``V_upper`` down in ``downhill_speed_step_m_s`` steps.

Heart rate (Karvonen)::

    HR = HR_rest + (M / P_aerobic) * (HR_max - HR_rest) + offset * (1 - f_orthostatic)
"""

from __future__ import annotations

from ..core.config import CapacityConfig
from ..core.constants import O2_ENERGY_EQUIVALENT_J_M3
from ..core.schemas import AstronautProfile, Fitness
from .metabolism import power_from_coefficients


def aerobic_power_w(astronaut: AstronautProfile, fitness: Fitness) -> float:
    """Metabolic power at 100 % of the current VO2max, W."""
    return (astronaut.vo2max_earth_m3_kg_s * fitness.aerobic * astronaut.body_mass_kg
            * O2_ENERGY_EQUIVALENT_J_M3)


def sustainable_power_w(astronaut: AstronautProfile, fitness: Fitness,
                        cfg: CapacityConfig) -> float:
    """Power the crew member can hold for hours, W."""
    return cfg.sustained_fraction * aerobic_power_w(astronaut, fitness)


def peak_power_w(astronaut: AstronautProfile, fitness: Fitness, cfg: CapacityConfig) -> float:
    """Power the crew member can hold for 10-15 minutes, W."""
    return cfg.peak_fraction * aerobic_power_w(astronaut, fitness)


def readiness_factor(sleep_debt_s: float, cfg: CapacityConfig) -> float:
    """Multiplier of the power cap from accumulated sleep debt, 0..1."""
    return min(1.0, max(0.0, 1.0 - cfg.readiness_loss_per_sleep_debt_s * max(sleep_debt_s, 0.0)))


def power_cap_w(base_power_w: float, thermal_power_w: float, fatigue: float, readiness: float,
                hydration_factor: float, energy_factor: float, cfg: CapacityConfig) -> float:
    """Largest metabolic power the crew member may work at right now, W.

    Args:
        base_power_w: ``P_sustain`` (nominal pace) or ``P_peak`` (emergency pace).
        thermal_power_w: thermal limit from A5; pass ``math.inf`` to ignore it.
        fatigue: whole-body fatigue F, 0..1.
        readiness: sleep-debt multiplier.
        hydration_factor, energy_factor: multipliers from A8.
    """
    fatigue = min(max(fatigue, 0.0), 1.0)
    cap = min(base_power_w, thermal_power_w) * (1.0 - cfg.fatigue_power_penalty * fatigue)
    return max(0.0, cap * readiness * hydration_factor * energy_factor)


def select_speed(c0: float, c1: float, c2: float, floor: float, grade: float,
                 upper_speed_m_s: float, power_cap: float, cfg: CapacityConfig) -> float | None:
    """Highest speed not above ``upper_speed_m_s`` whose walking power fits ``power_cap``.

    Args:
        c0, c1, c2, floor: walking power quadratic from ``pandolf_coefficients``.
        grade: rise over run of the edge; selects the search method.
        upper_speed_m_s: preferred speed already reduced by terrain limits.
        power_cap: ``P_cap`` in watts.

    Returns:
        Speed in m/s, or ``None`` when even ``min_speed_m_s`` needs more than the cap
        (the edge is physiologically impassable).
    """
    v_min = cfg.min_speed_m_s
    if upper_speed_m_s < v_min:
        return None
    if power_from_coefficients(c0, c1, c2, floor, upper_speed_m_s) <= power_cap:
        return upper_speed_m_s
    if grade >= 0.0:
        if power_from_coefficients(c0, c1, c2, floor, v_min) > power_cap:
            return None
        low, high = v_min, upper_speed_m_s
        while high - low > cfg.speed_tolerance_m_s:
            mid = 0.5 * (low + high)
            if power_from_coefficients(c0, c1, c2, floor, mid) <= power_cap:
                low = mid
            else:
                high = mid
        return low
    step = cfg.downhill_speed_step_m_s
    speed = upper_speed_m_s - step
    while speed > v_min:
        if power_from_coefficients(c0, c1, c2, floor, speed) <= power_cap:
            return speed
        speed -= step
    if power_from_coefficients(c0, c1, c2, floor, v_min) <= power_cap:
        return v_min
    return None


def heart_rate_hz(metabolic_w: float, astronaut: AstronautProfile, fitness: Fitness,
                  cfg: CapacityConfig) -> float:
    """Heart rate by the Karvonen relation, Hz, limited to the maximal heart rate."""
    reserve_share = min(1.0, max(0.0, metabolic_w / aerobic_power_w(astronaut, fitness)))
    rate = (astronaut.hr_rest_hz + reserve_share * (astronaut.hr_max_hz - astronaut.hr_rest_hz)
            + cfg.hr_orthostatic_offset_hz * (1.0 - fitness.orthostatic))
    return min(rate, astronaut.hr_max_hz)


def heart_rate_alarm(heart_rate: float, astronaut: AstronautProfile, cfg: CapacityConfig) -> bool:
    """True when the heart rate is above the level that forces a rest if held too long."""
    return heart_rate > cfg.hr_alarm_fraction * astronaut.hr_max_hz
