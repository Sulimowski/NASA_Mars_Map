"""Data structures shared by the physiology and routing layers. All fields are SI.

Static inputs      AstronautProfile, SuitProfile, HabitatAtmosphere
Environment        EnvSample
Derived            Fitness
Dynamic state      PhysState, SuitResources, Label (flat search state)
Edge result        Transition
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TypeAlias

from .constants import EARTH_DAY_S, HOUR_S
from .exceptions import EnvironmentValidationError, ProfileValidationError

Cell: TypeAlias = tuple[int, int]
"""Grid cell as (row, col)."""


class CoolingMode(str, Enum):
    """How the suit rejects heat (A4)."""

    MEMBRANE_EVAPORATOR = "membrane_evaporator"   # spends water, works near the triple point
    RADIATOR_STORAGE = "radiator_storage"         # no water, capacity depends on the sky temperature


class TaskType(str, Enum):
    """Stationary activities (A2)."""

    REST = "rest"
    OBSERVATION = "observation"
    SAMPLING = "sampling"
    HAMMERING = "hammering"
    DIGGING = "digging"


class Pace(str, Enum):
    """Walking regime."""

    NOMINAL = "nominal"       # sustainable power, preferred speed
    EMERGENCY = "emergency"   # peak power, thermal limit ignored


class SafetyFlag(IntEnum):
    """Cell classification by the contingency scenarios (A10)."""

    UNSAFE = 0
    MARGINAL = 1
    SAFE_EMERGENCY = 2


class PruneRule(str, Enum):
    """The eight hard constraints of the route search (B3)."""

    RESOURCE_RESERVE = "resource_reserve"
    RETURN_O2 = "return_o2"
    RETURN_TIME = "return_time"
    EMERGENCY_RADIUS = "emergency_radius"
    SPE_SHELTER = "spe_shelter"
    FATIGUE = "fatigue"
    DOSE_LIMIT = "dose_limit"
    HEAT_STORAGE = "heat_storage"


def _require(condition: bool, message: str, error: type[Exception]) -> None:
    if not condition:
        raise error(message)


@dataclass(frozen=True, slots=True)
class AstronautProfile:
    """Crew member, fixed before the mission.

    Attributes:
        body_mass_kg: body mass.
        height_m: stature, used for the body surface area.
        vo2max_earth_m3_kg_s: pre-flight maximal oxygen uptake, m^3 O2 per kg per second
            (40 ml/kg/min = 6.667e-7).
        hr_rest_hz, hr_max_hz: resting and maximal heart rate, Hz.
        grip_strength_n: maximal grip force, N.
        sweat_rate_coef: individual multiplier of the sweat rate.
        preferred_speed_m_s: unconstrained suited walking speed on firm level ground.
        transit_duration_s: time spent in microgravity before landing.
    """

    body_mass_kg: float = 80.0
    height_m: float = 1.78
    vo2max_earth_m3_kg_s: float = 40.0e-6 / 60.0
    hr_rest_hz: float = 1.0
    hr_max_hz: float = 190.0 / 60.0
    grip_strength_n: float = 450.0
    sweat_rate_coef: float = 1.0
    preferred_speed_m_s: float = 3.0 / 3.6
    transit_duration_s: float = 180.0 * EARTH_DAY_S

    def __post_init__(self) -> None:
        err = ProfileValidationError
        _require(20.0 <= self.body_mass_kg <= 200.0, "body_mass_kg out of range [20, 200]", err)
        _require(1.0 <= self.height_m <= 2.5, "height_m out of range [1.0, 2.5]", err)
        _require(self.vo2max_earth_m3_kg_s > 0.0, "vo2max_earth_m3_kg_s must be positive", err)
        _require(0.0 < self.hr_rest_hz < self.hr_max_hz, "need 0 < hr_rest_hz < hr_max_hz", err)
        _require(self.preferred_speed_m_s > 0.0, "preferred_speed_m_s must be positive", err)
        _require(self.sweat_rate_coef > 0.0, "sweat_rate_coef must be positive", err)
        _require(self.transit_duration_s >= 0.0, "transit_duration_s must be non-negative", err)


@dataclass(frozen=True, slots=True)
class SuitProfile:
    """Spacesuit with its portable life support system.

    Attributes:
        mass_kg: suit plus life support mass carried by the astronaut.
        pressure_pa: operating pressure.
        o2_primary_kg: usable primary oxygen.
        o2_secondary_duration_s: duration of the emergency oxygen supply.
        co2_capacity_kg: CO2 the scrubber can absorb.
        cooling_water_kg: feedwater of the evaporator.
        drink_water_kg: in-suit drink bag.
        battery_j: battery energy.
        cooling_max_w: heat the cooling loop can reject at the design point.
        heater_max_w: total electric heater power.
        stiffness_factor: metabolic penalty of moving a pressurised suit.
        leak_rate_kg_s: oxygen leakage.
        dose_factor: transmitted share of the ambient radiation dose (1 = no shielding).
        max_eva_s: certified EVA duration.
        alpha_solar, epsilon_ir: solar absorptance and infrared emittance of the outer layer.
        base_power_w, fan_power_w: avionics and ventilation electric loads.
        cooling_mode: heat rejection technology.
        radiator_area_m2, radiator_emissivity: radiator used by ``RADIATOR_STORAGE``.
        thermal_storage_capacity_j: phase-change buffer used by ``RADIATOR_STORAGE``.
    """

    mass_kg: float = 60.0
    pressure_pa: float = 29_600.0
    o2_primary_kg: float = 0.55
    o2_secondary_duration_s: float = 1_800.0
    co2_capacity_kg: float = 0.8
    cooling_water_kg: float = 4.0
    drink_water_kg: float = 0.95
    battery_j: float = 700.0 * HOUR_S
    cooling_max_w: float = 550.0
    heater_max_w: float = 150.0
    stiffness_factor: float = 1.25
    leak_rate_kg_s: float = 0.005 / HOUR_S
    dose_factor: float = 1.0
    max_eva_s: float = 8.0 * HOUR_S
    alpha_solar: float = 0.18
    epsilon_ir: float = 0.85
    base_power_w: float = 60.0
    fan_power_w: float = 40.0
    cooling_mode: CoolingMode = CoolingMode.MEMBRANE_EVAPORATOR
    radiator_area_m2: float = 1.2
    radiator_emissivity: float = 0.9
    thermal_storage_capacity_j: float = 200.0e3

    def __post_init__(self) -> None:
        err = ProfileValidationError
        _require(self.mass_kg >= 0.0, "suit mass_kg must be non-negative", err)
        _require(self.pressure_pa > 0.0, "suit pressure_pa must be positive", err)
        for name in ("o2_primary_kg", "co2_capacity_kg", "battery_j", "max_eva_s",
                     "cooling_max_w", "o2_secondary_duration_s"):
            _require(getattr(self, name) > 0.0, f"suit {name} must be positive", err)
        for name in ("cooling_water_kg", "drink_water_kg", "heater_max_w", "leak_rate_kg_s",
                     "base_power_w", "fan_power_w", "thermal_storage_capacity_j"):
            _require(getattr(self, name) >= 0.0, f"suit {name} must be non-negative", err)
        _require(self.stiffness_factor >= 1.0, "suit stiffness_factor must be >= 1", err)
        _require(0.0 <= self.dose_factor <= 1.0, "suit dose_factor must be in [0, 1]", err)
        _require(0.0 < self.alpha_solar <= 1.0, "suit alpha_solar must be in (0, 1]", err)
        _require(0.0 < self.epsilon_ir <= 1.0, "suit epsilon_ir must be in (0, 1]", err)


@dataclass(frozen=True, slots=True)
class HabitatAtmosphere:
    """Cabin atmosphere the crew is saturated with before the EVA (A11).

    Default is the NASA exploration atmosphere: 56.5 kPa (8.2 psi), 34 % oxygen.
    """

    pressure_pa: float = 56_500.0
    o2_fraction: float = 0.34

    def __post_init__(self) -> None:
        err = ProfileValidationError
        _require(self.pressure_pa > 0.0, "habitat pressure_pa must be positive", err)
        _require(0.0 < self.o2_fraction <= 1.0, "habitat o2_fraction must be in (0, 1]", err)


@dataclass(frozen=True, slots=True)
class EnvSample:
    """Environment at one cell and one instant.

    Attributes:
        t_ground_k, t_air_k, t_sky_k: ground, near-surface air and effective sky temperature.
        pressure_pa: atmospheric pressure.
        solar_direct_w_m2: direct beam irradiance on a plane normal to the Sun.
        solar_diffuse_w_m2: diffuse irradiance on a horizontal plane.
        sun_elevation_rad: Sun elevation above the horizon (negative at night).
        dose_rate_gcr_sv_s: galactic cosmic ray dose-equivalent rate.
        spe_active: a solar particle event is in progress.
        spe_dose_rate_sv_s: extra dose-equivalent rate of the event.
        uv_index: relative ultraviolet irradiance, 0 at night.
    """

    t_ground_k: float = 260.0
    t_air_k: float = 245.0
    t_sky_k: float = 170.0
    pressure_pa: float = 700.0
    solar_direct_w_m2: float = 400.0
    solar_diffuse_w_m2: float = 80.0
    sun_elevation_rad: float = math.radians(60.0)
    dose_rate_gcr_sv_s: float = 0.64e-3 / 88_775.244
    spe_active: bool = False
    spe_dose_rate_sv_s: float = 0.0
    uv_index: float = 5.0

    def validate(self) -> None:
        """Check physical plausibility.

        Raises:
            EnvironmentValidationError: a field is non-finite or outside its physical range.
        """
        err = EnvironmentValidationError
        for name in ("t_ground_k", "t_air_k", "t_sky_k"):
            value = getattr(self, name)
            _require(math.isfinite(value) and 20.0 <= value <= 400.0,
                     f"{name}={value} K outside [20, 400]", err)
        _require(math.isfinite(self.pressure_pa) and 1.0 <= self.pressure_pa <= 5_000.0,
                 f"pressure_pa={self.pressure_pa} outside [1, 5000]", err)
        for name in ("solar_direct_w_m2", "solar_diffuse_w_m2",
                     "dose_rate_gcr_sv_s", "spe_dose_rate_sv_s", "uv_index"):
            value = getattr(self, name)
            _require(math.isfinite(value) and value >= 0.0, f"{name}={value} must be >= 0", err)
        _require(-math.pi / 2 <= self.sun_elevation_rad <= math.pi / 2,
                 f"sun_elevation_rad={self.sun_elevation_rad} outside [-pi/2, pi/2]", err)


@dataclass(frozen=True, slots=True)
class Fitness:
    """Condition of the crew member relative to pre-flight (A1). 1.0 = pre-flight level."""

    aerobic: float
    strength: float
    sensorimotor: float
    orthostatic: float
    bone_fracture_mult: float


@dataclass(slots=True)
class PhysState:
    """Physiological state of the crew member at one instant."""

    t_s: float = 0.0
    fatigue: float = 0.0
    hand_fatigue: float = 0.0
    hydration_deficit_kg: float = 0.0
    energy_deficit_j: float = 0.0
    heat_storage_j: float = 0.0
    dose_trip_sv: float = 0.0
    dust_load_kg: float = 0.0
    uv_exposure: float = 0.0


@dataclass(slots=True)
class SuitResources:
    """Consumables left in the suit."""

    o2_kg: float
    co2_capacity_kg: float
    cooling_water_kg: float
    drink_water_kg: float
    battery_j: float
    eva_remaining_s: float

    @classmethod
    def full(cls, suit: SuitProfile) -> SuitResources:
        """Resources of a freshly serviced suit."""
        return cls(o2_kg=suit.o2_primary_kg, co2_capacity_kg=suit.co2_capacity_kg,
                   cooling_water_kg=suit.cooling_water_kg, drink_water_kg=suit.drink_water_kg,
                   battery_j=suit.battery_j, eva_remaining_s=suit.max_eva_s)


@dataclass(slots=True)
class Transition:
    """Result of one move or one stationary step: durations, rates and state increments.

    ``d_*`` fields are signed increments of the matching state variable, except the
    consumables (``d_o2_kg``, ``d_co2_kg``, ``d_cooling_water_kg``, ``d_drink_kg``,
    ``d_battery_j``) which are amounts consumed (>= 0).
    """

    dt_s: float
    distance_m: float
    speed_m_s: float
    grade: float                 # rise / run
    slope_rad: float
    metabolic_w: float
    energy_j: float
    heart_rate_hz: float
    cooling_w: float
    heater_w: float
    d_o2_kg: float
    d_co2_kg: float
    d_cooling_water_kg: float
    d_drink_kg: float
    d_battery_j: float
    d_dose_sv: float
    d_fatigue: float
    d_hand_fatigue: float
    d_heat_j: float
    d_hydration_kg: float
    d_dust_kg: float
    d_uv: float
    risk: float                  # expected harm of the step (A9)
    p_fall: float


@dataclass(slots=True, eq=False)
class Label:
    """Flat search state of the route search (B3): position, time, resources, body state.

    Flat rather than nested so that a Pareto set of up to K labels per cell stays
    cheap. ``eq=False`` gives identity comparison, which the search relies on.
    """

    row: int
    col: int
    t_s: float
    o2_kg: float
    co2_capacity_kg: float
    cooling_water_kg: float
    drink_water_kg: float
    battery_j: float
    eva_remaining_s: float
    fatigue: float
    hand_fatigue: float
    hydration_deficit_kg: float
    heat_storage_j: float
    dose_sv: float
    risk: float = 0.0
    cost: float = 0.0
    distance_m: float = 0.0
    energy_j: float = 0.0
    dust_load_kg: float = 0.0
    uv_exposure: float = 0.0
    action: str = "start"
    transition: Transition | None = None
    parent: Label | None = None
    alive: bool = True

    def __repr__(self) -> str:
        """Short form: the default would print the whole parent chain."""
        return (f"Label(cell=({self.row}, {self.col}), t_s={self.t_s:.0f}, "
                f"dist_m={self.distance_m:.0f}, o2_kg={self.o2_kg:.4f}, "
                f"fatigue={self.fatigue:.3f}, cost={self.cost:.1f}, action={self.action!r})")

    @property
    def cell(self) -> Cell:
        """Grid cell of the label."""
        return (self.row, self.col)

    @classmethod
    def from_states(cls, cell: Cell, phys: PhysState, resources: SuitResources) -> Label:
        """Build the start label of a search from the two state records."""
        return cls(row=cell[0], col=cell[1], t_s=phys.t_s, o2_kg=resources.o2_kg,
                   co2_capacity_kg=resources.co2_capacity_kg,
                   cooling_water_kg=resources.cooling_water_kg,
                   drink_water_kg=resources.drink_water_kg, battery_j=resources.battery_j,
                   eva_remaining_s=resources.eva_remaining_s, fatigue=phys.fatigue,
                   hand_fatigue=phys.hand_fatigue,
                   hydration_deficit_kg=phys.hydration_deficit_kg,
                   heat_storage_j=phys.heat_storage_j, dose_sv=phys.dose_trip_sv,
                   energy_j=phys.energy_deficit_j, dust_load_kg=phys.dust_load_kg,
                   uv_exposure=phys.uv_exposure)

    def phys_state(self) -> PhysState:
        """Physiological part of the label."""
        return PhysState(t_s=self.t_s, fatigue=self.fatigue, hand_fatigue=self.hand_fatigue,
                         hydration_deficit_kg=self.hydration_deficit_kg,
                         energy_deficit_j=self.energy_j, heat_storage_j=self.heat_storage_j,
                         dose_trip_sv=self.dose_sv, dust_load_kg=self.dust_load_kg,
                         uv_exposure=self.uv_exposure)

    def resources(self) -> SuitResources:
        """Consumables part of the label."""
        return SuitResources(o2_kg=self.o2_kg, co2_capacity_kg=self.co2_capacity_kg,
                             cooling_water_kg=self.cooling_water_kg,
                             drink_water_kg=self.drink_water_kg, battery_j=self.battery_j,
                             eva_remaining_s=self.eva_remaining_s)

    def chain(self) -> list[Label]:
        """Labels from the start of the search to this one, in travel order."""
        out: list[Label] = []
        node: Label | None = self
        while node is not None:
            out.append(node)
            node = node.parent
        out.reverse()
        return out
