"""The walking model of A2-A4 on whole arrays, for a router that prices millions of steps.

Same formulas and the same configuration as ``metabolism``, ``capacity`` and
``life_support``; nothing is re-tuned here. Inputs are NumPy arrays (or scalars)
that broadcast together, one element per step.

For one step with horizontal length ``d``, rise ``h``, terrain factor ``eta``::

    M(V)  = max(floor, c0 + c1 * V + c2 * V^2)                 Pandolf on Mars (A2)
    V     = highest V <= V_upper with M(V) <= P_cap            (A3)
    t     = sqrt(d^2 + h^2) / V
    O2    = M * t / E_O2 * rho_O2 + leak * t                   (A4)
    E_bat = (P_base + P_fan * f(M) + P_heater) * t             (A4)

``M(V)`` is a quadratic, so the speed has a closed form: with the roots
``V1 <= V2`` of ``c2 V^2 + c1 V + (c0 - P_cap) = 0`` the feasible speeds are
``[V1, V2]``, and ``V = min(V_upper, V2)``. A step is infeasible when there are
no real roots, when ``V2`` is below the minimum speed, or when ``V_upper < V1``
(a descent so steep that braking at any allowed speed exceeds the cap).

The scalar model finds the same uphill speed by bisection and scans descents in
0.02 m/s steps, so the two agree to the bisection tolerance uphill and to one
scan step downhill.

State that builds up along a route (fatigue, stored heat, water) cannot be part
of a static step price. The cap is therefore taken for a fixed fatigue level,
and the finished route is checked by the sequential simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.config import SimConfig
from ..core.constants import (G_EARTH_M_S2, G_MARS_M_S2, O2_DENSITY_STP_KG_M3,
                              O2_ENERGY_EQUIVALENT_J_M3, ZERO_CELSIUS_K)
from ..core.exceptions import ValidationError
from ..core.schemas import AstronautProfile, SuitProfile, TaskType
from . import adaptation, capacity, metabolism

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StepArrays:
    """Result of ``walking_steps``; every array has the broadcast shape of the inputs.

    Infeasible steps have ``feasible`` False, zero speed and infinite time,
    oxygen and battery energy.
    """

    time_s: FloatArray
    speed_m_s: FloatArray
    metabolic_w: FloatArray
    o2_kg: FloatArray
    battery_j: FloatArray
    feasible: NDArray[np.bool_]


def planning_power_cap_w(astronaut: AstronautProfile, cfg: SimConfig, time_on_surface_s: float,
                         fatigue: float | None = None) -> float:
    """Power cap used to price steps before the route is known, W.

    ``P_sustain * (1 - penalty * F)`` with ``F`` fixed at the rest trigger by
    default: the simulator rests the crew member at that level, so a step priced
    with this cap stays walkable along the whole route.
    """
    level = cfg.fatigue.rest_trigger if fatigue is None else fatigue
    fitness = adaptation.compute_fitness(time_on_surface_s, astronaut.transit_duration_s,
                                         cfg.adaptation)
    return capacity.power_cap_w(capacity.sustainable_power_w(astronaut, fitness, cfg.capacity),
                                np.inf, level, 1.0, 1.0, 1.0, cfg.capacity)


def effective_grade_percent(grade: ArrayLike, cfg: SimConfig) -> FloatArray:
    """Array form of ``metabolism.effective_grade_percent``."""
    percent = 100.0 * np.asarray(grade, dtype=np.float64)
    m = cfg.metabolism
    braking = np.maximum(0.0, -percent - 100.0 * m.braking_onset_grade)
    return np.where(percent >= 0.0, percent, m.downhill_recovery * percent + m.braking_factor * braking)


def electric_power_w(metabolic_w: ArrayLike, heater_w: ArrayLike, suit: SuitProfile,
                     cfg: SimConfig, lights_on: bool = False) -> FloatArray:
    """Array form of ``life_support.electric_power_w``."""
    ls = cfg.life_support
    fan = np.maximum(ls.fan_min_factor, np.asarray(metabolic_w, dtype=np.float64)
                     / ls.fan_reference_metabolic_w)
    lights = ls.lights_power_w if lights_on else 0.0
    return (suit.base_power_w + suit.fan_power_w * fan
            + np.maximum(0.0, np.asarray(heater_w, dtype=np.float64)) + lights)


def extremity_heater_w(ground_temp_k: ArrayLike, suit: SuitProfile, cfg: SimConfig) -> FloatArray:
    """Glove and boot heater power for a ground temperature, W (array form of A5)."""
    th = cfg.thermal
    power = th.extremity_heater_gain_w_k * (th.extremity_heater_threshold_k
                                            - np.asarray(ground_temp_k, dtype=np.float64))
    return np.clip(power, 0.0, suit.heater_max_w)


def celsius_to_kelvin(temp_c: ArrayLike) -> FloatArray:
    """Degrees Celsius -> kelvin for arrays."""
    return np.asarray(temp_c, dtype=np.float64) + ZERO_CELSIUS_K


def walking_steps(dist_m: ArrayLike, rise_m: ArrayLike, eta: ArrayLike, speed_factor: ArrayLike,
                  heater_w: ArrayLike, *, astronaut: AstronautProfile, suit: SuitProfile,
                  cfg: SimConfig, power_cap_w: float) -> StepArrays:
    """Time, speed, power and consumables of walking steps.

    Args:
        dist_m: horizontal length of each step, m (> 0).
        rise_m: height change of each step, m (negative downhill).
        eta: Pandolf terrain factor of the ground stepped onto (>= 1).
        speed_factor: terrain speed limit as a share of the preferred speed, (0, 1].
        heater_w: electric heater power while on the step, W.
        power_cap_w: metabolic power the crew member may hold, W.

    Raises:
        ValidationError: the power cap is not positive.
    """
    if not power_cap_w > 0.0:
        raise ValidationError(f"power_cap_w={power_cap_w} must be positive")
    dist = np.asarray(dist_m, dtype=np.float64)
    rise = np.asarray(rise_m, dtype=np.float64)
    eta_a = np.asarray(eta, dtype=np.float64)
    m, cap_cfg = cfg.metabolism, cfg.capacity
    body, load = astronaut.body_mass_kg, suit.mass_kg
    total, stiffness = body + load, suit.stiffness_factor
    k_g = G_MARS_M_S2 / G_EARTH_M_S2
    k_level = m.level_scale_intercept + m.level_scale_slope * k_g

    grade = rise / dist
    floor = stiffness * m.standing_coeff * body
    c0 = floor + stiffness * k_level * m.load_coeff * total * (load / body) ** 2
    c1 = stiffness * k_g * eta_a * total * m.grade_coeff * effective_grade_percent(grade, cfg)
    c2 = stiffness * k_level * eta_a * total * m.speed_coeff

    v_min = cap_cfg.min_speed_m_s
    upper = np.maximum(v_min, astronaut.preferred_speed_m_s
                       * np.asarray(speed_factor, dtype=np.float64))
    disc = c1 * c1 - 4.0 * c2 * (c0 - power_cap_w)
    root = np.sqrt(np.maximum(disc, 0.0))
    v_low, v_high = (-c1 - root) / (2.0 * c2), (-c1 + root) / (2.0 * c2)
    speed = np.minimum(upper, v_high)
    feasible = (disc >= 0.0) & (speed >= v_min) & (speed >= v_low - 1.0e-12) & (floor <= power_cap_w)
    feasible = feasible & np.isfinite(grade)
    speed = np.where(feasible, speed, 0.0)

    metabolic = np.maximum(floor, c0 + c1 * speed + c2 * speed * speed)
    metabolic = np.where(feasible, metabolic, power_cap_w)
    length = np.hypot(dist, rise)
    with np.errstate(divide="ignore", invalid="ignore"):
        time_s = np.where(feasible, length / np.where(feasible, speed, 1.0), np.inf)
        o2 = (metabolic * time_s / O2_ENERGY_EQUIVALENT_J_M3 * O2_DENSITY_STP_KG_M3
              + suit.leak_rate_kg_s * time_s)
        battery = electric_power_w(metabolic, heater_w, suit, cfg) * time_s
    shape = np.broadcast(time_s, o2, battery).shape
    return StepArrays(time_s=np.broadcast_to(time_s, shape).astype(np.float64),
                      speed_m_s=np.broadcast_to(speed, shape).astype(np.float64),
                      metabolic_w=np.broadcast_to(metabolic, shape).astype(np.float64),
                      o2_kg=np.broadcast_to(o2, shape).astype(np.float64),
                      battery_j=np.broadcast_to(battery, shape).astype(np.float64),
                      feasible=np.broadcast_to(feasible, shape).astype(np.bool_))


def stationary_work(duration_s: float, task: TaskType, heater_w: float, *, suit: SuitProfile,
                    cfg: SimConfig, fatigue: float | None = None) -> tuple[float, float, float]:
    """Metabolic power, oxygen and battery energy of working in place.

    Returns:
        ``(metabolic_w, o2_kg, battery_j)`` for ``duration_s`` of ``task``.

    Raises:
        ValidationError: the duration is negative.
    """
    if duration_s < 0.0:
        raise ValidationError(f"duration_s={duration_s} must be non-negative")
    level = cfg.fatigue.rest_trigger if fatigue is None else fatigue
    power = metabolism.task_power_w(task, level, cfg.metabolism)
    o2 = (power * duration_s / O2_ENERGY_EQUIVALENT_J_M3 * O2_DENSITY_STP_KG_M3
          + suit.leak_rate_kg_s * duration_s)
    battery = float(electric_power_w(power, heater_w, suit, cfg)) * duration_s
    return power, o2, battery


def o2_kg_to_litres(o2_kg: Any) -> Any:
    """Oxygen mass -> litres at standard conditions (the unit of the routing team)."""
    return o2_kg / (O2_DENSITY_STP_KG_M3 * 1.0e-3)
