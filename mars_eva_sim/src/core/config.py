"""Calibrated parameters and thresholds of every algorithm, in one place.

Each algorithm has its own frozen dataclass; ``SimConfig`` groups them. Every
default is the value from the algorithm specification. Values the specification
marks as "to verify" are engineering estimates and are flagged ``[verify]`` in the
comment next to them: replace them when the reference data is confirmed.

All values are SI (seconds, kelvin, radians, pascal, sievert, joule, watt, kg, m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .constants import (EARTH_DAY_S, EARTH_MONTH_S, EARTH_YEAR_S, HOUR_S,
                        MARS_REFERENCE_PRESSURE_PA, SOL_S)
from .exceptions import ConfigurationError
from .schemas import TaskType


@dataclass(frozen=True, slots=True)
class AdaptationConfig:
    """A1: deconditioning after the transit and re-adaptation on the surface."""

    aerobic_loss_per_transit_s: float = 0.0012 / EARTH_DAY_S   # VO2max share lost per second of transit
    aerobic_loss_max: float = 0.25
    aerobic_plateau_loss: float = 0.05          # permanent deficit at 0.38 g  [verify]
    aerobic_recovery_tau_s: float = 14.0 * EARTH_DAY_S
    strength_loss_per_transit_s: float = 0.0010 / EARTH_DAY_S
    strength_loss_max: float = 0.20
    strength_plateau_loss: float = 0.05         # [verify]
    strength_recovery_tau_s: float = 21.0 * EARTH_DAY_S
    sensorimotor_initial_deficit: float = 0.5
    sensorimotor_tau_s: float = 3.0 * EARTH_DAY_S
    orthostatic_initial_deficit: float = 0.4
    orthostatic_tau_s: float = 2.0 * EARTH_DAY_S
    bone_fracture_gain_per_transit_s: float = 0.0125 / EARTH_MONTH_S
    min_surface_time_for_eva_s: float = SOL_S   # no EVA during the first sol
    sensorimotor_slope_threshold: float = 0.7   # below it the reduced slope limit applies


@dataclass(frozen=True, slots=True)
class MetabolismConfig:
    """A2: Pandolf load-carriage equation rescaled to Mars, and stationary tasks."""

    standing_coeff: float = 1.5                 # W per kg of body mass
    load_coeff: float = 2.0
    speed_coeff: float = 1.5
    grade_coeff: float = 0.35                   # W per (kg * m/s * % grade)
    level_scale_intercept: float = 0.3          # k_lvl = intercept + slope * g/g_earth  [verify]
    level_scale_slope: float = 0.65
    downhill_recovery: float = 0.5              # share of the grade term recovered downhill
    braking_onset_grade: float = 0.10           # rise/run where braking starts to cost energy
    braking_factor: float = 0.6
    mech_efficiency_per_grade: float = 2.0      # external work share per unit of uphill grade
    mech_efficiency_max: float = 0.2
    task_fatigue_penalty: float = 0.3           # tired crew spends more on the same task
    task_power_w: tuple[tuple[TaskType, float], ...] = (
        (TaskType.REST, 110.0), (TaskType.OBSERVATION, 175.0), (TaskType.SAMPLING, 300.0),
        (TaskType.HAMMERING, 400.0), (TaskType.DIGGING, 475.0))
    task_grip_load: tuple[tuple[TaskType, float], ...] = (
        (TaskType.REST, 0.0), (TaskType.OBSERVATION, 0.2), (TaskType.SAMPLING, 1.0),
        (TaskType.HAMMERING, 1.5), (TaskType.DIGGING, 1.2))
    task_contact_fraction: tuple[tuple[TaskType, float], ...] = (
        (TaskType.REST, 0.0), (TaskType.OBSERVATION, 0.0), (TaskType.SAMPLING, 0.5),
        (TaskType.HAMMERING, 0.3), (TaskType.DIGGING, 0.4))

    def power_of(self, task: TaskType) -> float:
        """Metabolic power of a stationary task for a fresh crew member, W."""
        return _lookup(self.task_power_w, task)

    def grip_load_of(self, task: TaskType) -> float:
        """Relative hand load of a task (0 = hands free)."""
        return _lookup(self.task_grip_load, task)

    def contact_fraction_of(self, task: TaskType) -> float:
        """Share of the task time the gloves touch the ground or rocks."""
        return _lookup(self.task_contact_fraction, task)


@dataclass(frozen=True, slots=True)
class CapacityConfig:
    """A3: sustainable power, speed selection and heart rate."""

    sustained_fraction: float = 0.45            # share of VO2max held for hours
    peak_fraction: float = 0.75                 # share held for 10-15 minutes
    fatigue_power_penalty: float = 0.3          # P_cap multiplier is 1 - penalty * F
    min_speed_m_s: float = 0.2
    speed_tolerance_m_s: float = 1.0e-3         # binary search stop for uphill / level
    downhill_speed_step_m_s: float = 0.02       # grid step for downhill
    emergency_speed_m_s: float = 1.25
    hr_orthostatic_offset_hz: float = 10.0 / 60.0
    hr_alarm_fraction: float = 0.85
    hr_alarm_duration_s: float = 300.0
    readiness_loss_per_sleep_debt_s: float = 0.05 / HOUR_S
    max_sleep_debt_s: float = 6.0 * HOUR_S


@dataclass(frozen=True, slots=True)
class LifeSupportConfig:
    """A4: consumables of the suit."""

    respiratory_quotient: float = 0.85          # CO2 produced / O2 consumed, by volume
    membrane_latent_heat_j_kg: float = 2.4e6
    membrane_loop_temp_k: float = 288.15        # water temperature at the evaporator
    membrane_design_pressure_pa: float = MARS_REFERENCE_PRESSURE_PA
    fan_reference_metabolic_w: float = 300.0
    fan_min_factor: float = 0.5
    lights_power_w: float = 30.0                # helmet lights, on when the Sun is down


@dataclass(frozen=True, slots=True)
class ThermalConfig:
    """A5: heat balance of the suited astronaut."""

    suit_internal_temp_k: float = 293.15
    shell_resistance_m2_k_w: float = 1.0        # insulation of the suit shell in CO2  [verify]
    suit_area_factor: float = 1.3               # suit outer area / body surface area
    convection_ref_w_m2_k: float = 1.0          # at the reference pressure
    convection_ref_pressure_pa: float = MARS_REFERENCE_PRESSURE_PA
    convection_min_w_m2_k: float = 0.3
    projected_side_fraction: float = 0.25       # area seen by a Sun on the horizon / total
    projected_top_fraction: float = 0.08        # area seen by a Sun at zenith / total
    boot_resistance_k_w: float = 2.0
    glove_resistance_k_w: float = 4.0
    heat_storage_limit_j: float = 300.0e3       # about 1 K of core temperature  [verify]
    storage_relax_tau_s: float = 600.0
    extremity_heater_threshold_k: float = 233.15
    extremity_heater_gain_w_k: float = 1.5
    radiator_temp_k: float = 300.0
    radiator_sky_view: float = 0.7
    newton_iterations: int = 6
    # Glove contact limits by surface temperature (NASA-STD-3001 touch limits)  [verify]
    touch_unlimited_min_k: float = 253.15
    touch_cold_k: float = 213.15
    touch_very_cold_k: float = 153.15
    touch_hot_k: float = 336.15
    touch_time_cold_s: float = 300.0
    touch_time_very_cold_s: float = 60.0
    touch_time_extreme_s: float = 10.0
    touch_time_hot_s: float = 60.0


@dataclass(frozen=True, slots=True)
class RadiationConfig:
    """A6: dose limits (NASA-STD-3001) and solar particle event handling."""

    career_limit_sv: float = 0.6
    bfo_30d_limit_gyeq: float = 0.25
    bfo_1y_limit_gyeq: float = 0.5
    bfo_gyeq_per_sv: float = 0.5                # effective dose -> blood-forming-organ dose  [verify]
    window_30d_s: float = 30.0 * EARTH_DAY_S
    window_1y_s: float = EARTH_YEAR_S
    ledger_bin_s: float = HOUR_S
    spe_warning_time_s: float = 900.0           # flare to proton arrival  [verify]
    per_eva_limit_sv: float = 1.0e-3
    uv_reference_index: float = 10.0


@dataclass(frozen=True, slots=True)
class FatigueConfig:
    """A7: whole-body fatigue F, hand fatigue H, daily limits."""

    growth_rate_s: float = 1.9 / HOUR_S
    growth_threshold: float = 0.5               # M / P_sustain above which F grows
    rest_ratio: float = 0.4                     # M / P_sustain below which F recovers
    recovery_rate_s: float = 1.0 / HOUR_S
    rest_trigger: float = 0.5                   # insert a rest when F exceeds it
    return_only: float = 0.7                    # above it: only moves toward the base
    stop: float = 0.9                           # above it: no movement at all
    rest_duration_s: float = 300.0
    hand_growth_rate_s: float = 0.8 / (3.0 * HOUR_S)
    hand_recovery_rate_s: float = 1.0 / (2.0 * HOUR_S)
    hand_reference_pressure_pa: float = 29_600.0
    hand_limit: float = 0.8
    max_evas_per_sol: int = 1
    max_suit_time_per_sol_s: float = 8.0 * HOUR_S
    max_consecutive_eva_sols: int = 2           # [verify]


@dataclass(frozen=True, slots=True)
class NutritionConfig:
    """A8: water and energy balance."""

    sweat_base_kg_s: float = 0.1 / HOUR_S
    sweat_gain_kg_s: float = 0.6 / HOUR_S
    sweat_onset_w: float = 150.0
    sweat_span_w: float = 400.0
    respiratory_loss_kg_s: float = 0.04 / HOUR_S
    drink_trigger_kg: float = 0.25
    dehydration_fraction: float = 0.02          # of body mass
    dehydration_power_factor: float = 0.85
    fasting_limit_s: float = 6.0 * HOUR_S
    fasting_power_factor: float = 0.9


@dataclass(frozen=True, slots=True)
class InjuryConfig:
    """A9: logistic fall model. Coefficients are calibration parameters, not NASA data."""

    intercept: float = -2.0
    slope_per_rad: float = 0.08 * 180.0 / math.pi
    rock_hazard: float = 2.0
    sensorimotor_deficit: float = 2.0
    fatigue: float = 1.5
    visibility_deficit: float = 1.5
    steep_downhill: float = 0.8
    speed_ratio: float = 0.5
    steep_downhill_grade: float = -0.15
    base_rate_per_m: float = 2.0e-4
    injury_given_fall: float = 0.02
    breach_given_fall: float = 0.005
    weight_injury: float = 1.0
    weight_breach: float = 10.0


@dataclass(frozen=True, slots=True)
class EmergencyConfig:
    """A10: contingency scenarios."""

    reaction_time_s: float = 300.0
    battery_failure_tolerance_s: float = 1_800.0   # covered by the secondary oxygen purge flow


@dataclass(frozen=True, slots=True)
class DecompressionConfig:
    """A11: nitrogen washout before the EVA."""

    allowed_tissue_ratio: float = 1.4           # [verify]
    n2_half_time_s: float = 360.0 * 60.0        # slowest tissue compartment used by NASA


@dataclass(frozen=True, slots=True)
class DustConfig:
    """A12: dust carried back to the airlock."""

    load_per_m_kg: float = 1.0e-6
    cleaning_time_per_kg_s: float = 6.0e5
    base_cleaning_time_s: float = 300.0
    perchlorate_mass_fraction: float = 0.006
    perchlorate_flag_kg: float = 5.0e-6


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """B1-B5: edge rules, cost weights, pruning thresholds."""

    max_labels_per_cell: int = 8
    weight_time_per_s: float = 1.0
    weight_o2_per_kg: float = 2.0e3
    weight_dose_per_sv: float = 1.0e7
    weight_fatigue: float = 600.0
    weight_risk: float = 1.0e6
    reserve_fraction: float = 0.2
    return_margin: float = 1.2
    strict_emergency: bool = False
    slope_up_limit_rad: float = math.radians(25.0)
    slope_down_limit_rad: float = math.radians(20.0)
    reduced_slope_limit_rad: float = math.radians(10.0)
    cell_slope_hard_limit_rad: float = math.radians(25.0)
    standard_fatigue: float = 0.3               # state used for the static shelter maps
    max_expansions: int = 2_000_000
    dominance_tolerance: float = 1.0e-9
    visibility_night_factor: float = 0.5
    rock_speed_penalty: float = 0.4
    rock_speed_min_factor: float = 0.3
    env_cache_bucket_s: float = 600.0
    task_step_s: float = 60.0
    isochrones_s: tuple[float, ...] = (900.0, 1_800.0, 3_600.0, 7_200.0)


@dataclass(frozen=True, slots=True)
class SolSelectionConfig:
    """H3 / H6: choosing the sol of an EVA and budgeting the mission dose."""

    spe_dose_factor: float = 1.5                # dose above factor * median = particle event
    spe_guard_sols: int = 1                     # sols kept clear before and after an event
    min_daytime_ground_temp_k: float = 190.0    # colder afternoons: no EVA  [verify]
    max_daytime_ground_temp_k: float = 320.0
    daylight_margin_s: float = HOUR_S           # after sunrise and before sunset
    min_window_s: float = 3.0 * HOUR_S          # shortest useful EVA window
    weight_dose: float = 1.0
    weight_warmth: float = 0.5
    mission_dose_budget_sv: float = 0.3         # surface share of the career limit  [verify]
    dose_margin: float = 0.25                   # tolerated excess over the per-sol allocation
    typical_dose_sv_per_sol: float = 0.67e-3    # MSL/RAD average in Gale
    default_ground_temp_min_k: float = 195.0
    default_ground_temp_max_k: float = 261.0
    peak_lag_s: float = 3_700.0                 # warmest ground this long after noon
    night_decay: float = 3.0
    beam_transmittance: float = 0.6             # direct beam through a clear sky at zenith
    diffuse_share: float = 0.25


@dataclass(frozen=True, slots=True)
class CommsConfig:
    """H9: radio contact with the base."""

    base_antenna_height_m: float = 10.0
    astronaut_antenna_height_m: float = 1.8
    max_no_contact_s: float = 900.0             # longest tolerated stretch without contact
    planet_radius_m: float = 3_389_500.0


@dataclass(frozen=True, slots=True)
class RouteAdapterConfig:
    """Input boundary for routes produced by other teams."""

    stop_snap_distance_m: float = 60.0          # a stop farther than this from the route is an error
    default_dwell_s: float = 1_200.0


@dataclass(frozen=True, slots=True)
class ObstacleConfig:
    """Height of the obstacles a vehicle meets in a cell, from rocks and micro-relief.

    Rocks follow the size-frequency model of Golombek and Rapp (1997): the share of
    the surface covered by rocks of diameter D or larger is ``k * exp(-q(k) * D)``
    with ``q(k) = q_intercept + q_slope / k`` for a total rock abundance ``k``.
    """

    rock_q_intercept: float = 1.79
    rock_q_slope: float = 0.152
    hazard_area_fraction: float = 0.01          # obstacles are the rocks covering this share
    rock_height_to_diameter: float = 0.5        # a rock stands about half its diameter high
    hazard_to_abundance: float = 0.4            # rock abundance assumed at rock_hazard = 1
    relief_window_cells: int = 1                # half-width of the detrending window
    relief_amplitude_factor: float = 2.0        # step height = factor * |residual|
    relief_max_cell_m: float = 3.0              # coarser grids resolve hills, not steps


@dataclass(frozen=True, slots=True)
class RoverConfig:
    """Rover mobility, power and radiation (algorithms R1-R3) and rover routing.

    Soil parameters of the Bekker-Wong model are interpolated between ``firm``
    (terrain factor 1.0) and ``loose`` (terrain factor ``eta_loose``). They are
    engineering estimates for Martian regolith  [verify].
    """

    # --- soil: firm end / loose end -------------------------------------
    eta_firm: float = 1.0
    eta_loose: float = 2.1
    cohesion_firm_pa: float = 5_000.0
    cohesion_loose_pa: float = 500.0
    friction_firm_rad: float = math.radians(38.0)
    friction_loose_rad: float = math.radians(30.0)
    k_phi_firm: float = 6.0e6                   # N/m^(n+2)
    k_phi_loose: float = 4.0e5
    k_c: float = 1.0e3                          # N/m^(n+1), nearly cohesionless regolith
    sinkage_exponent_firm: float = 1.0
    sinkage_exponent_loose: float = 1.1
    shear_modulus_firm_m: float = 0.005
    shear_modulus_loose_m: float = 0.025
    contact_length_floor: float = 0.10          # of the wheel diameter
    # --- slip ---------------------------------------------------------------
    slip_critical: float = 0.60                 # above it the wheels dig in
    slip_critical_wheel_walking: float = 0.90
    wheel_walking_thrust_factor: float = 1.6
    slip_tolerance: float = 1.0e-4
    # --- power --------------------------------------------------------------
    mmrtg_decay_per_year: float = math.log(110.0 / 72.0) / 17.0   # NASA: 110 W -> 72 W in 17 years
    panel_dust_loss_per_sol: float = 0.002      # share of output lost per sol without cleaning
    battery_charge_efficiency: float = 0.95
    battery_discharge_efficiency: float = 0.95
    recharge_trigger_fraction: float = 0.75     # below it a parked rover may wait to recharge
    # --- operations ---------------------------------------------------------
    min_sun_elevation_drive_rad: float = math.radians(5.0)   # rovers drive in daylight
    wait_step_s: float = 1_800.0
    max_traverse_s: float = 5.0 * SOL_S
    # --- radiation ----------------------------------------------------------
    spe_speed_factor: float = 0.5               # autonomous navigation slows down in a particle event
    spe_hold: bool = False                      # True: no driving at all during an event
    seu_rate_per_s: float = 1.0e-7              # upset probability per second at the reference event
    seu_reference_dose_rate_sv_s: float = 3.0 * 0.64e-3 / SOL_S
    electronics_gy_per_sv: float = 1.0 / 3.05   # MSL/RAD mean quality factor in Gale
    risk_budget: float = 0.05
    # --- route cost (seconds-equivalent) ------------------------------------
    weight_time_per_s: float = 1.0
    weight_energy_per_j: float = 2.0e-3
    weight_risk: float = 1.0e6
    weight_slip_per_m: float = 20.0             # cost of a metre driven at slip 1


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Every tunable parameter of the simulator."""

    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    metabolism: MetabolismConfig = field(default_factory=MetabolismConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    life_support: LifeSupportConfig = field(default_factory=LifeSupportConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    radiation: RadiationConfig = field(default_factory=RadiationConfig)
    fatigue: FatigueConfig = field(default_factory=FatigueConfig)
    nutrition: NutritionConfig = field(default_factory=NutritionConfig)
    injury: InjuryConfig = field(default_factory=InjuryConfig)
    emergency: EmergencyConfig = field(default_factory=EmergencyConfig)
    decompression: DecompressionConfig = field(default_factory=DecompressionConfig)
    dust: DustConfig = field(default_factory=DustConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    sol_selection: SolSelectionConfig = field(default_factory=SolSelectionConfig)
    comms: CommsConfig = field(default_factory=CommsConfig)
    route_adapter: RouteAdapterConfig = field(default_factory=RouteAdapterConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    rover: RoverConfig = field(default_factory=RoverConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject parameter sets that would make the model meaningless.

        Raises:
            ConfigurationError: a value is outside its valid range.
        """
        checks: tuple[tuple[bool, str], ...] = (
            (0.0 < self.capacity.sustained_fraction <= self.capacity.peak_fraction <= 1.0,
             "capacity: need 0 < sustained_fraction <= peak_fraction <= 1"),
            (self.capacity.min_speed_m_s > 0.0, "capacity: min_speed_m_s must be positive"),
            (self.capacity.downhill_speed_step_m_s > 0.0,
             "capacity: downhill_speed_step_m_s must be positive"),
            (0.0 <= self.routing.reserve_fraction < 1.0,
             "routing: reserve_fraction must be in [0, 1)"),
            (self.routing.return_margin >= 1.0, "routing: return_margin must be >= 1"),
            (self.routing.max_labels_per_cell >= 1, "routing: max_labels_per_cell must be >= 1"),
            (self.fatigue.rest_ratio <= self.fatigue.growth_threshold,
             "fatigue: rest_ratio must not exceed growth_threshold"),
            (self.fatigue.rest_trigger <= self.fatigue.return_only <= self.fatigue.stop <= 1.0,
             "fatigue: need rest_trigger <= return_only <= stop <= 1"),
            (self.thermal.shell_resistance_m2_k_w > 0.0,
             "thermal: shell_resistance_m2_k_w must be positive"),
            (0.0 <= self.metabolism.mech_efficiency_max < 1.0,
             "metabolism: mech_efficiency_max must be in [0, 1)"),
            (self.decompression.allowed_tissue_ratio > 0.0,
             "decompression: allowed_tissue_ratio must be positive"),
            (self.radiation.bfo_gyeq_per_sv > 0.0, "radiation: bfo_gyeq_per_sv must be positive"),
            (self.sol_selection.spe_dose_factor > 1.0,
             "sol_selection: spe_dose_factor must be above 1"),
            (self.comms.max_no_contact_s >= 0.0, "comms: max_no_contact_s must be non-negative"),
            (self.route_adapter.stop_snap_distance_m > 0.0,
             "route_adapter: stop_snap_distance_m must be positive"),
            (0.0 < self.rover.slip_critical < 1.0, "rover: slip_critical must be in (0, 1)"),
            (self.rover.slip_critical <= self.rover.slip_critical_wheel_walking < 1.0,
             "rover: need slip_critical <= slip_critical_wheel_walking < 1"),
            (self.rover.mmrtg_decay_per_year >= 0.0,
             "rover: mmrtg_decay_per_year must be non-negative"),
            (0.0 < self.obstacles.hazard_area_fraction < 1.0,
             "obstacles: hazard_area_fraction must be in (0, 1)"),
        )
        for ok, message in checks:
            if not ok:
                raise ConfigurationError(message)


def _lookup(table: tuple[tuple[TaskType, float], ...], task: TaskType) -> float:
    for key, value in table:
        if key is task:
            return value
    raise ConfigurationError(f"task {task!r} is missing from the configuration table")


DEFAULT_CONFIG: SimConfig = SimConfig()
