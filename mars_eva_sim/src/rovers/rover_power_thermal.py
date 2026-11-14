"""R2: what the rover's systems draw, what its generator gives, what is left in the battery.

Load::

    P_total   = P_drive + P_base + P_heaters
    P_drive   = thrust * v_wheel / efficiency + P_drive_idle        while moving, else 0
    P_base    = avionics_w while awake, sleep_w while parked

``thrust * v_wheel`` is the mechanical power at the wheel rims; with slip the rim
moves faster than the rover, so slip costs energy. Thrust is never negative:
there is no energy recovery on a descent.

Heaters::

    P_body      = max(0, UA * (T_set - T_ground) - passive_heat_w - P_base)
    P_actuators = k * max(0, T_actuator_min - T_ground)             while moving
    P_heaters   = min(heater_max_w, P_body + P_actuators)

The warm body loses ``UA * dT`` to the cold ground; the heat of the MMRTG loop or
of the radioisotope heater units and the dissipation of the electronics come
first, electric heaters cover the rest. Demand beyond ``heater_max_w`` is
reported as a deficit: the rover is outside its thermal window.

Generation::

    MMRTG   P = P_bol * exp(-decay * years since landing)
    solar   P = area * efficiency * dust * (S_direct * sin(elevation) + S_diffuse)
            dust = exp(-loss_per_sol * sols since the arrays were last cleaned)

The MMRTG decay rate is the one NASA publishes for the generator as a whole
(110 W at the start, about 72 W after 17 years), which is faster than the decay
of the plutonium alone because the thermocouples age too.

Battery::

    dE/dt = eta_charge * (P_gen - P_total)        when generating more than the load
          = (P_gen - P_total) / eta_discharge     otherwise
    0 <= E <= capacity                            a full battery sheds the surplus
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.config import RoverConfig
from ..core.exceptions import ValidationError
from ..core.schemas import EnvSample
from .profiles import RoverProfile
from .rover_mobility import MobilityResult
from .types import PowerSource


@dataclass(frozen=True, slots=True)
class HeaterDemand:
    """Electric heater power used and the part of the demand that could not be met."""

    power_w: float
    deficit_w: float


def mmrtg_power_w(bol_w: float, age_years: float, cfg: RoverConfig) -> float:
    """Electric output of an MMRTG ``age_years`` after landing, W.

    Raises:
        ValidationError: the age is negative.
    """
    if age_years < 0.0:
        raise ValidationError(f"age_years={age_years} must be non-negative")
    return bol_w * math.exp(-cfg.mmrtg_decay_per_year * age_years)


def panel_dust_factor(sols_since_cleaning: float, cfg: RoverConfig) -> float:
    """Share of the solar array output left after dust has settled on it, (0, 1]."""
    return math.exp(-cfg.panel_dust_loss_per_sol * max(0.0, sols_since_cleaning))


def solar_power_w(profile: RoverProfile, env: EnvSample, dust_factor: float) -> float:
    """Electric output of a horizontal solar array, W (0 with the Sun below the horizon)."""
    if env.sun_elevation_rad <= 0.0:
        return 0.0
    irradiance = (env.solar_direct_w_m2 * math.sin(env.sun_elevation_rad)
                  + env.solar_diffuse_w_m2)
    return profile.solar_area_m2 * profile.solar_efficiency * dust_factor * irradiance


def generation_w(profile: RoverProfile, env: EnvSample, age_years: float, dust_factor: float,
                 cfg: RoverConfig) -> float:
    """Electric power the rover generates now, W."""
    if profile.power_source is PowerSource.MMRTG:
        return mmrtg_power_w(profile.generator_bol_w, age_years, cfg)
    return solar_power_w(profile, env, dust_factor)


def drive_power_w(profile: RoverProfile, mobility: MobilityResult) -> float:
    """Electric power of the drive for a step, W (0 when the step is refused)."""
    if not mobility.passable or mobility.wheel_speed_m_s <= 0.0:
        return 0.0
    mechanical = mobility.thrust_n * mobility.wheel_speed_m_s
    return mechanical / profile.drivetrain_efficiency + profile.drive_idle_w


def heater_demand(profile: RoverProfile, env: EnvSample, base_load_w: float,
                  moving: bool) -> HeaterDemand:
    """Electric heater power for the ambient temperature.

    Args:
        base_load_w: electric load dissipated inside the warm body (avionics or sleep).
        moving: the drive actuators must be warm.
    """
    body = max(0.0, profile.thermal_conductance_w_k
               * (profile.internal_setpoint_k - env.t_ground_k)
               - profile.passive_heat_w - base_load_w)
    actuators = (profile.actuator_heater_w_per_k
                 * max(0.0, profile.actuator_min_temp_k - env.t_ground_k) if moving else 0.0)
    demand = body + actuators
    used = min(profile.heater_max_w, demand)
    return HeaterDemand(power_w=used, deficit_w=demand - used)


def battery_after(battery_j: float, generation: float, load_w: float, dt_s: float,
                  profile: RoverProfile, cfg: RoverConfig) -> float:
    """Stored energy after ``dt_s`` at a constant generation and load, J.

    The result is capped at the capacity. It is not floored at zero: a negative
    value means the step cannot be powered, which the constraints reject.
    """
    net = generation - load_w
    if net >= 0.0:
        delta = cfg.battery_charge_efficiency * net * dt_s
    else:
        delta = net * dt_s / cfg.battery_discharge_efficiency
    return min(profile.battery_j, battery_j + delta)
