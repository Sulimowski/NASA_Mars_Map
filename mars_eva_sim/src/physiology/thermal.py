"""A5: heat balance of the suited astronaut.

Outer surface of the suit
-------------------------
The shell temperature ``T_s`` settles where absorbed and emitted fluxes balance
(per square metre of suit surface)::

    q_abs + h_c * (T_air - T_s) + (T_in - T_s) / R = eps * sigma * T_s^4

    q_abs = alpha * (S_direct * f_proj + S_diffuse / 2)
            + eps * sigma * (T_sky^4 + T_ground^4) / 2
    f_proj = f_side * cos(el) + f_top * sin(el)        (0 when the Sun is down)
    h_c    = h_ref * sqrt(p / p_ref),  at least h_min

``R`` is the thermal resistance of the shell (m^2 K / W) and ``T_in`` the suit
interior. The equation is solved by Newton iterations. Temperatures are kelvin,
which the T^4 terms require.

Heat reaching the body from outside
-----------------------------------
::

    Q_shell   = A * (T_s - T_in) / R                    [W], negative = heat lost
    Q_boots   = (T_ground - T_in) / R_boot
    Q_gloves  = (T_ground - T_in) / R_glove * contact_fraction
    Q_env     = Q_shell + Q_boots + Q_gloves

Balance of the body
-------------------
::

    Q_met  = M * (1 - mechanical_efficiency)
    Q_need = Q_met + Q_env
    cooling = clip(Q_need + stored heat to remove, 0, capacity)
    heater  = clip(-(Q_need + stored heat to add), 0, heater available)
    dS/dt   = Q_need - cooling + heater

``S`` (J) is heat stored in the body. Spare cooling or heating capacity relaxes it
toward zero with time constant ``storage_relax_tau_s``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.config import LifeSupportConfig, ThermalConfig
from ..core.constants import STEFAN_BOLTZMANN_W_M2_K4, ZERO_CELSIUS_K
from ..core.schemas import AstronautProfile, CoolingMode, EnvSample, SuitProfile


@dataclass(frozen=True, slots=True)
class EnvHeat:
    """Heat exchange with the environment that does not depend on the workload.

    Attributes:
        q_fixed_w: heat into the body through the shell and the boots (negative = loss).
        q_glove_w: extra heat into the body if the gloves touched the ground all the time.
        capacity_w: heat the cooling system can reject in these conditions.
        extremity_heater_w: electric power the glove and boot heaters draw.
        surface_temp_k: temperature of the suit outer surface.
    """

    q_fixed_w: float
    q_glove_w: float
    capacity_w: float
    extremity_heater_w: float
    surface_temp_k: float


@dataclass(frozen=True, slots=True)
class ThermalResult:
    """Outcome of one heat-balance step."""

    q_metabolic_w: float
    q_need_w: float
    cooling_w: float
    heater_w: float
    storage_rate_w: float
    heat_storage_j: float


def saturation_pressure_pa(temp_k: float) -> float:
    """Saturation vapour pressure of water over liquid (Magnus formula), Pa."""
    temp_c = temp_k - ZERO_CELSIUS_K
    return 611.2 * math.exp(17.62 * temp_c / (243.12 + temp_c))


def body_surface_area_m2(body_mass_kg: float, height_m: float) -> float:
    """Body surface area by DuBois and DuBois: 0.007184 * W^0.425 * H_cm^0.725, m^2."""
    return 0.007184 * body_mass_kg ** 0.425 * (100.0 * height_m) ** 0.725


def suit_surface_area_m2(astronaut: AstronautProfile, cfg: ThermalConfig) -> float:
    """Outer area of the suit, m^2: body surface area times ``suit_area_factor``."""
    return cfg.suit_area_factor * body_surface_area_m2(astronaut.body_mass_kg, astronaut.height_m)


def convection_coefficient_w_m2_k(env: EnvSample, cfg: ThermalConfig) -> float:
    """Convective coefficient in the thin CO2 atmosphere, W/(m^2 K).

    Convection scales as the square root of the Reynolds number, hence of the gas
    density, hence of the pressure. No wind data is available for the sites, so
    the coefficient is for the air flow of a walking pace.
    """
    scale = env.pressure_pa / cfg.convection_ref_pressure_pa
    return max(cfg.convection_min_w_m2_k, cfg.convection_ref_w_m2_k * math.sqrt(scale))


def suit_surface_temperature_k(suit: SuitProfile, env: EnvSample, cfg: ThermalConfig) -> float:
    """Equilibrium temperature of the suit outer surface, K (Newton iterations)."""
    sigma_eps = STEFAN_BOLTZMANN_W_M2_K4 * suit.epsilon_ir
    elevation = env.sun_elevation_rad
    if elevation > 0.0:
        projected = (cfg.projected_side_fraction * math.cos(elevation)
                     + cfg.projected_top_fraction * math.sin(elevation))
    else:
        projected = 0.0
    q_abs = (suit.alpha_solar * (env.solar_direct_w_m2 * projected + 0.5 * env.solar_diffuse_w_m2)
             + 0.5 * sigma_eps * (env.t_sky_k ** 4 + env.t_ground_k ** 4))
    h_c = convection_coefficient_w_m2_k(env, cfg)
    conductance = 1.0 / cfg.shell_resistance_m2_k_w
    temp = max(env.t_air_k, 100.0)
    for _ in range(cfg.newton_iterations):
        residual = (q_abs + h_c * (env.t_air_k - temp)
                    + conductance * (cfg.suit_internal_temp_k - temp) - sigma_eps * temp ** 4)
        derivative = -h_c - conductance - 4.0 * sigma_eps * temp ** 3
        temp -= residual / derivative
    return temp


def cooling_capacity_w(suit: SuitProfile, env: EnvSample, cfg: ThermalConfig,
                       life_support: LifeSupportConfig) -> float:
    """Heat the suit cooling system can reject in the given conditions, W.

    Membrane evaporator: the vapour leaves through a membrane into the atmosphere,
    driven by the difference between the saturation pressure of the loop water and
    the ambient pressure. ``cooling_max_w`` is the capacity at the design ambient
    pressure; it shrinks as the ambient pressure approaches the saturation pressure::

        capacity = cooling_max * clip((p_sat - p) / (p_sat - p_design), 0, 1)

    Radiator: grey-body emission toward the sky and the ground::

        capacity = min(cooling_max, eps_r * sigma * A_r * (T_r^4 - T_sink^4))
        T_sink^4 = view_sky * T_sky^4 + (1 - view_sky) * T_ground^4
    """
    if suit.cooling_mode is CoolingMode.MEMBRANE_EVAPORATOR:
        p_sat = saturation_pressure_pa(life_support.membrane_loop_temp_k)
        span = p_sat - life_support.membrane_design_pressure_pa
        if span <= 0.0:
            return 0.0
        return suit.cooling_max_w * min(1.0, max(0.0, (p_sat - env.pressure_pa) / span))
    sink4 = (cfg.radiator_sky_view * env.t_sky_k ** 4
             + (1.0 - cfg.radiator_sky_view) * env.t_ground_k ** 4)
    emitted = (suit.radiator_emissivity * STEFAN_BOLTZMANN_W_M2_K4 * suit.radiator_area_m2
               * (cfg.radiator_temp_k ** 4 - sink4))
    return min(suit.cooling_max_w, max(0.0, emitted))


def extremity_heater_w(env: EnvSample, cfg: ThermalConfig) -> float:
    """Electric power of the glove and boot heaters, W: gain * (threshold - T_ground)."""
    return max(0.0, cfg.extremity_heater_gain_w_k
               * (cfg.extremity_heater_threshold_k - env.t_ground_k))


def environmental_heat(astronaut: AstronautProfile, suit: SuitProfile, env: EnvSample,
                       cfg: ThermalConfig, life_support: LifeSupportConfig) -> EnvHeat:
    """Workload-independent part of the heat balance for one place and time."""
    surface = suit_surface_temperature_k(suit, env, cfg)
    area = suit_surface_area_m2(astronaut, cfg)
    q_shell = area * (surface - cfg.suit_internal_temp_k) / cfg.shell_resistance_m2_k_w
    q_boots = (env.t_ground_k - cfg.suit_internal_temp_k) / cfg.boot_resistance_k_w
    q_glove = (env.t_ground_k - cfg.suit_internal_temp_k) / cfg.glove_resistance_k_w
    return EnvHeat(q_fixed_w=q_shell + q_boots, q_glove_w=q_glove,
                   capacity_w=cooling_capacity_w(suit, env, cfg, life_support),
                   extremity_heater_w=min(suit.heater_max_w, extremity_heater_w(env, cfg)),
                   surface_temp_k=surface)


def thermal_limit_power_w(capacity_w: float, q_env_w: float, mech_efficiency: float) -> float:
    """Largest metabolic power whose heat the suit can carry away, W.

    From ``M * (1 - eff) + Q_env <= capacity``.
    """
    return max(0.0, capacity_w - q_env_w) / (1.0 - mech_efficiency)


def heat_storage_limit_j(suit: SuitProfile, cfg: ThermalConfig) -> float:
    """Largest tolerated magnitude of stored heat, J.

    A radiator suit adds its phase-change buffer to the body limit.
    """
    if suit.cooling_mode is CoolingMode.RADIATOR_STORAGE:
        return cfg.heat_storage_limit_j + suit.thermal_storage_capacity_j
    return cfg.heat_storage_limit_j


def thermal_balance(metabolic_w: float, mech_efficiency: float, q_env_w: float,
                    capacity_w: float, heat_storage_j: float, dt_s: float,
                    heater_available_w: float, cfg: ThermalConfig) -> ThermalResult:
    """Advance the heat balance of the body by ``dt_s``.

    Args:
        metabolic_w: metabolic power M.
        mech_efficiency: share of M leaving as external work.
        q_env_w: heat entering the body from the environment (negative = loss).
        capacity_w: cooling capacity available.
        heat_storage_j: heat stored in the body at the start of the step.
        dt_s: step length, s (> 0).
        heater_available_w: electric heater power left for the torso.

    Returns:
        Cooling and heater power used over the step and the stored heat at its end.
        Unclipped, the stored heat decays exactly as ``S * exp(-dt / tau)``, so the
        step is stable for any ``dt_s``.
    """
    q_met = metabolic_w * (1.0 - mech_efficiency)
    q_need = q_met + q_env_w
    target = heat_storage_j * math.exp(-dt_s / cfg.storage_relax_tau_s)
    demand = q_need + (heat_storage_j - target) / dt_s
    cooling = min(capacity_w, max(0.0, demand))
    heater = min(max(0.0, heater_available_w), max(0.0, -demand))
    rate = q_need - cooling + heater
    return ThermalResult(q_metabolic_w=q_met, q_need_w=q_need, cooling_w=cooling, heater_w=heater,
                         storage_rate_w=rate, heat_storage_j=heat_storage_j + rate * dt_s)


def touch_contact_limit_s(surface_temp_k: float, cfg: ThermalConfig) -> float:
    """Longest continuous glove contact with a surface at ``surface_temp_k``, s.

    ``math.inf`` inside the comfortable band; shorter the colder (or hotter) the surface.
    """
    if surface_temp_k >= cfg.touch_hot_k:
        return cfg.touch_time_hot_s
    if surface_temp_k >= cfg.touch_unlimited_min_k:
        return math.inf
    if surface_temp_k >= cfg.touch_cold_k:
        return cfg.touch_time_cold_s
    if surface_temp_k >= cfg.touch_very_cold_k:
        return cfg.touch_time_very_cold_s
    return cfg.touch_time_extreme_s
