"""A4: what the suit spends, and which consumable runs out first.

For metabolic power ``M`` (W) held for ``dt`` (s)::

    V_O2      = M * dt / E_O2                               [m^3 at STP]
    dO2       = V_O2 * rho_O2 + leak_rate * dt               [kg]
    dCO2      = V_O2 * RQ * rho_CO2                          [kg absorbed by the scrubber]
    dWater    = cooling_w * dt / h_evap                      [kg]  membrane evaporator
              = 0                                            radiator
    dBattery  = (P_base + P_fan * f(M) + P_heater + P_lights) * dt     [J]
    f(M)      = max(f_min, M / M_ref)
    dEVA      = dt

Cooling on Mars: the ambient pressure (about 600 Pa) is at the triple point of
water (611.657 Pa), so an ISS-style sublimator cannot hold ice. The two modes
modelled here are a membrane evaporator, which spends water, and a radiator
with a thermal buffer, which spends none but depends on the sky temperature
(its capacity is computed in A5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.config import LifeSupportConfig
from ..core.constants import (CO2_DENSITY_STP_KG_M3, O2_DENSITY_STP_KG_M3,
                              O2_ENERGY_EQUIVALENT_J_M3)
from ..core.schemas import CoolingMode, EnvSample, SuitProfile, SuitResources


@dataclass(frozen=True, slots=True)
class ResourceDelta:
    """Consumables spent over a step (all >= 0)."""

    o2_kg: float
    co2_kg: float
    cooling_water_kg: float
    battery_j: float
    eva_s: float


def o2_volume_m3(metabolic_w: float, dt_s: float) -> float:
    """Oxygen consumed by metabolism, m^3 at STP."""
    return metabolic_w * dt_s / O2_ENERGY_EQUIVALENT_J_M3


def o2_consumption_kg(metabolic_w: float, dt_s: float, suit: SuitProfile) -> float:
    """Oxygen drawn from the tank: metabolism plus suit leakage, kg."""
    return o2_volume_m3(metabolic_w, dt_s) * O2_DENSITY_STP_KG_M3 + suit.leak_rate_kg_s * dt_s


def co2_production_kg(metabolic_w: float, dt_s: float, cfg: LifeSupportConfig) -> float:
    """Carbon dioxide the scrubber must absorb, kg."""
    return o2_volume_m3(metabolic_w, dt_s) * cfg.respiratory_quotient * CO2_DENSITY_STP_KG_M3


def cooling_water_kg(cooling_w: float, dt_s: float, suit: SuitProfile,
                     cfg: LifeSupportConfig) -> float:
    """Feedwater evaporated to reject ``cooling_w`` for ``dt_s``, kg (0 for a radiator)."""
    if suit.cooling_mode is CoolingMode.RADIATOR_STORAGE:
        return 0.0
    return max(0.0, cooling_w) * dt_s / cfg.membrane_latent_heat_j_kg


def electric_power_w(metabolic_w: float, heater_w: float, suit: SuitProfile, env: EnvSample,
                     cfg: LifeSupportConfig) -> float:
    """Electric load of the suit, W: avionics, fan (scaled by workload), heaters, lights."""
    fan_factor = max(cfg.fan_min_factor, metabolic_w / cfg.fan_reference_metabolic_w)
    lights = cfg.lights_power_w if env.sun_elevation_rad <= 0.0 else 0.0
    return suit.base_power_w + suit.fan_power_w * fan_factor + max(0.0, heater_w) + lights


def consume(metabolic_w: float, cooling_w: float, heater_w: float, dt_s: float,
            suit: SuitProfile, env: EnvSample, cfg: LifeSupportConfig) -> ResourceDelta:
    """Consumables spent over one step.

    Args:
        metabolic_w: metabolic power.
        cooling_w: heat rejected by the cooling loop (from A5).
        heater_w: total heater power, torso plus extremities (from A5).
        dt_s: step length, s.
    """
    return ResourceDelta(
        o2_kg=o2_consumption_kg(metabolic_w, dt_s, suit),
        co2_kg=co2_production_kg(metabolic_w, dt_s, cfg),
        cooling_water_kg=cooling_water_kg(cooling_w, dt_s, suit, cfg),
        battery_j=electric_power_w(metabolic_w, heater_w, suit, env, cfg) * dt_s,
        eva_s=dt_s,
    )


def limiting_resource(resources: SuitResources, spent: ResourceDelta, elapsed_s: float,
                      suit: SuitProfile) -> tuple[str, float]:
    """The consumable that would run out first at the average rate seen so far.

    Args:
        resources: what is left now.
        spent: what was spent during ``elapsed_s``.
        elapsed_s: duration over which ``spent`` was measured, s.

    Returns:
        ``(name, time_left_s)``. ``time_left_s`` is ``math.inf`` when nothing was spent.
    """
    if elapsed_s <= 0.0:
        return "none", math.inf
    pairs: list[tuple[str, float, float]] = [
        ("o2", resources.o2_kg, spent.o2_kg),
        ("co2_scrubber", resources.co2_capacity_kg, spent.co2_kg),
        ("battery", resources.battery_j, spent.battery_j),
        ("eva_time", resources.eva_remaining_s, spent.eva_s),
    ]
    if suit.cooling_mode is CoolingMode.MEMBRANE_EVAPORATOR:
        pairs.append(("cooling_water", resources.cooling_water_kg, spent.cooling_water_kg))
    best_name, best_time = "none", math.inf
    for name, left, used in pairs:
        if used <= 0.0:
            continue
        time_left = max(0.0, left) / (used / elapsed_s)
        if time_left < best_time:
            best_name, best_time = name, time_left
    return best_name, best_time
