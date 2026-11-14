"""The rovers: what each one is and what it can do.

Only vehicles that are operating on Mars or are officially planned to land are
listed. Retired and cancelled rovers are deliberately absent.

Where a number comes from is noted next to it:

    [NASA] / [JPL] / [ESA]   published by the agency
    [derived]                computed here from published numbers
    [verify]                 an engineering estimate; no agency figure was found

Sources (checked November 2026):
    Curiosity      MSL landing press kit (JPL), NASA Science mission pages, MMRTG fact sheet
    Perseverance   NASA Science "Rover Components", JPL Robotics "Mars 2020 Mobility",
                   NASA Science blog of October 2026 (tilt limit 24 deg, belly clearance 20 cm)
    Rosalind Franklin   ESA Exploration Science and Cosmos mission pages, CNES project page,
                   "The ExoMars rover locomotion subsystem" (J. Terramechanics, 2010)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from ..core.constants import HOUR_S
from ..core.exceptions import ProfileValidationError, ValidationError
from .types import PowerSource


@dataclass(frozen=True, slots=True)
class RoverProfile:
    """Engineering description of one rover (SI units).

    Attributes:
        name, agency: identification.
        landing_year: year of landing on Mars; a rover cannot be simulated before it.
        planned: True while the rover has not landed yet.
        mass_kg: rover mass.
        ground_clearance_m: height of the belly above level ground.
        max_obstacle_height_m: tallest obstacle the rover may pass over.
        wheel_count, wheel_diameter_m, wheel_width_m: running gear.
        contact_length_factor: ground contact length relative to a rigid wheel
            (above 1 for flexible wheels).
        traction_factor: thrust the wheel tread develops relative to a plain wheel.
        max_speed_m_s: top speed on flat hard ground.
        autonav_speed_factor: share of the top speed held when driving autonomously.
        max_tilt_rad: steepest ground the rover may stand on.
        max_climb_rad: steepest slope along the direction of travel.
        wheel_walking: the rover can walk its wheels through loose soil.
        wheel_walk_speed_m_s: ground speed in that mode.
        power_source: MMRTG or solar.
        generator_bol_w: MMRTG electric power at landing.
        solar_area_m2, solar_efficiency: solar array (0 for an MMRTG rover).
        battery_j: battery capacity.
        safe_mode_fraction: share of the capacity that must stay in the battery.
        avionics_w: electric load while awake and driving, without the drive.
        sleep_w: electric load while parked and recharging.
        drive_idle_w: drive electronics load while moving, on top of the traction power.
        drivetrain_efficiency: wheel mechanical power over drive electric power.
        thermal_conductance_w_k: heat leak of the warm body per kelvin to the ground.
        internal_setpoint_k: temperature the warm body is kept at.
        passive_heat_w: heat available without electricity (MMRTG loop or radioisotope heaters).
        heater_max_w: electric heater power available.
        actuator_min_temp_k: coldest ambient at which the drive works unheated.
        actuator_heater_w_per_k: electric power to warm the actuators, per kelvin below that.
    """

    name: str
    agency: str
    landing_year: int
    planned: bool
    mass_kg: float
    ground_clearance_m: float
    max_obstacle_height_m: float
    wheel_count: int
    wheel_diameter_m: float
    wheel_width_m: float
    contact_length_factor: float
    traction_factor: float
    max_speed_m_s: float
    autonav_speed_factor: float
    max_tilt_rad: float
    max_climb_rad: float
    wheel_walking: bool
    wheel_walk_speed_m_s: float
    power_source: PowerSource
    generator_bol_w: float
    solar_area_m2: float
    solar_efficiency: float
    battery_j: float
    safe_mode_fraction: float
    avionics_w: float
    sleep_w: float
    drive_idle_w: float
    drivetrain_efficiency: float
    thermal_conductance_w_k: float
    internal_setpoint_k: float
    passive_heat_w: float
    heater_max_w: float
    actuator_min_temp_k: float
    actuator_heater_w_per_k: float

    def __post_init__(self) -> None:
        def need(ok: bool, message: str) -> None:
            if not ok:
                raise ProfileValidationError(f"{self.name}: {message}")

        need(self.mass_kg > 0.0, "mass_kg must be positive")
        need(self.wheel_count >= 3, "wheel_count must be at least 3")
        need(self.wheel_diameter_m > 0.0 and self.wheel_width_m > 0.0, "wheel size must be positive")
        need(0.0 < self.max_obstacle_height_m <= self.ground_clearance_m,
             "need 0 < max_obstacle_height_m <= ground_clearance_m")
        need(self.max_speed_m_s > 0.0, "max_speed_m_s must be positive")
        need(0.0 < self.autonav_speed_factor <= 1.0, "autonav_speed_factor must be in (0, 1]")
        need(0.0 < self.max_climb_rad <= self.max_tilt_rad < math.pi / 2,
             "need 0 < max_climb_rad <= max_tilt_rad < 90 deg")
        need(not self.wheel_walking or self.wheel_walk_speed_m_s > 0.0,
             "a wheel-walking rover needs wheel_walk_speed_m_s > 0")
        need(self.battery_j > 0.0, "battery_j must be positive")
        need(0.0 <= self.safe_mode_fraction < 1.0, "safe_mode_fraction must be in [0, 1)")
        need(0.0 < self.drivetrain_efficiency <= 1.0, "drivetrain_efficiency must be in (0, 1]")
        if self.power_source is PowerSource.MMRTG:
            need(self.generator_bol_w > 0.0, "an MMRTG rover needs generator_bol_w > 0")
        else:
            need(self.solar_area_m2 > 0.0 and 0.0 < self.solar_efficiency < 1.0,
                 "a solar rover needs a solar array")

    @property
    def weight_n(self) -> float:
        """Weight on Mars, N."""
        from ..core.constants import G_MARS_M_S2
        return self.mass_kg * G_MARS_M_S2

    @property
    def battery_reserve_j(self) -> float:
        """Energy that must stay in the battery (safe-mode threshold), J."""
        return self.safe_mode_fraction * self.battery_j

    def mission_age_years(self, mission_year: float) -> float:
        """Years on Mars at ``mission_year``.

        Raises:
            ValidationError: the rover has not landed by that year.
        """
        if mission_year < self.landing_year:
            raise ValidationError(f"{self.name} lands in {self.landing_year}; it cannot be "
                                  f"simulated in {mission_year:g}")
        return float(mission_year) - self.landing_year


_WH: Final[float] = HOUR_S

CURIOSITY: Final[RoverProfile] = RoverProfile(
    name="Curiosity", agency="NASA", landing_year=2012, planned=False,
    mass_kg=899.0,                                  # [JPL] press kit
    ground_clearance_m=0.60,                        # [NASA] "about 60 centimeters"
    max_obstacle_height_m=0.55,                     # [verify] rock height used for MSL site safety
    wheel_count=6, wheel_diameter_m=0.50,           # [JPL]
    wheel_width_m=0.40,                             # [JPL]
    contact_length_factor=1.0, traction_factor=1.0,
    max_speed_m_s=0.038,                            # [JPL] 1.5 inches per second
    autonav_speed_factor=0.5,                       # [NASA] hazard avoidance: under half of top speed
    max_tilt_rad=math.radians(30.0),                # [NASA] fault protection limit (45 deg static)
    max_climb_rad=math.radians(30.0),
    wheel_walking=False, wheel_walk_speed_m_s=0.0,
    power_source=PowerSource.MMRTG,
    generator_bol_w=110.0,                          # [NASA] MMRTG at the start of the mission
    solar_area_m2=0.0, solar_efficiency=0.0,
    battery_j=2.0 * 42.0 * 29.6 * _WH,              # [derived] two 42 A h batteries [NASA] at 29.6 V
    safe_mode_fraction=0.47,                        # [NASA] depth of discharge up to about 53 %
    avionics_w=150.0, sleep_w=45.0, drive_idle_w=40.0,   # [verify]
    drivetrain_efficiency=0.5,                      # [verify]
    thermal_conductance_w_k=2.0, internal_setpoint_k=253.15,
    passive_heat_w=2_000.0,                         # [NASA] MMRTG waste heat through the fluid loop
    heater_max_w=200.0,
    actuator_min_temp_k=218.15, actuator_heater_w_per_k=3.0,   # [verify]
)

PERSEVERANCE: Final[RoverProfile] = RoverProfile(
    name="Perseverance", agency="NASA", landing_year=2021, planned=False,
    mass_kg=1_025.0,                                # [NASA]
    ground_clearance_m=0.60,                        # [derived] 40 cm rocks + 20 cm belly margin
    max_obstacle_height_m=0.40,                     # [NASA] "knee-high rocks up to 40 centimeters"
    wheel_count=6, wheel_diameter_m=0.525,          # [NASA]
    wheel_width_m=0.35,                             # [verify] "narrower than Curiosity's"
    contact_length_factor=1.0,
    traction_factor=1.15,                           # [verify] redesigned tread, better traction
    max_speed_m_s=0.042,                            # [NASA] 152 m/h
    autonav_speed_factor=0.75,                      # [JPL] about 110-120 m/h autonomous
    max_tilt_rad=math.radians(24.0),                # [NASA] limit raised to 24 deg in 2026
    max_climb_rad=math.radians(24.0),
    wheel_walking=False, wheel_walk_speed_m_s=0.0,
    power_source=PowerSource.MMRTG,
    generator_bol_w=110.0,                          # [NASA]
    solar_area_m2=0.0, solar_efficiency=0.0,
    battery_j=2.0 * 43.0 * 29.6 * _WH,              # [verify] two lithium-ion batteries [NASA]
    safe_mode_fraction=0.47,                        # [verify] taken from Curiosity
    avionics_w=150.0, sleep_w=45.0, drive_idle_w=40.0,   # [verify]
    drivetrain_efficiency=0.5,                      # [verify]
    thermal_conductance_w_k=2.0, internal_setpoint_k=253.15,
    passive_heat_w=2_000.0,                         # [NASA]
    heater_max_w=200.0,
    actuator_min_temp_k=218.15, actuator_heater_w_per_k=3.0,   # [verify]
)

ROSALIND_FRANKLIN: Final[RoverProfile] = RoverProfile(
    name="Rosalind Franklin", agency="ESA/NASA", landing_year=2030, planned=True,
    mass_kg=310.0,                                  # [ESA]
    ground_clearance_m=0.30,                        # [verify]
    max_obstacle_height_m=0.25,                     # [verify] one wheel diameter
    wheel_count=6, wheel_diameter_m=0.25,           # [ESA] locomotion subsystem paper
    wheel_width_m=0.11,                             # [ESA]
    contact_length_factor=1.5,                      # [verify] flexible metal wheels flatten on the soil
    traction_factor=1.0,
    max_speed_m_s=40.0 / HOUR_S,                    # [ESA] 32-40 m/h average while driving
    autonav_speed_factor=1.0,                       # that figure already is the autonomous pace
    max_tilt_rad=math.radians(25.0),                # [verify]
    max_climb_rad=math.radians(25.0),               # [verify]
    wheel_walking=True,                             # [ESA]
    wheel_walk_speed_m_s=2.0 / 1_200.0,             # [ESA] test: 2 m in 20 minutes
    power_source=PowerSource.SOLAR,
    generator_bol_w=0.0,
    solar_area_m2=2.39,                             # [ESA]
    solar_efficiency=0.145,                         # [derived] effective value giving about
                                                    # 1200 W h per sol [ESA] on a clear equatorial sol
    battery_j=1_142.0 * _WH,                        # [ESA]
    safe_mode_fraction=0.30,                        # [verify]
    avionics_w=70.0, sleep_w=12.0, drive_idle_w=15.0,    # [verify]
    drivetrain_efficiency=0.5,                      # [verify]
    thermal_conductance_w_k=0.5, internal_setpoint_k=253.15,
    passive_heat_w=30.0,                            # [verify] radioisotope heater units from NASA
    heater_max_w=60.0,
    actuator_min_temp_k=218.15, actuator_heater_w_per_k=1.0,   # [verify]
)

ROVERS: Final[dict[str, RoverProfile]] = {
    "curiosity": CURIOSITY,
    "perseverance": PERSEVERANCE,
    "rosalind_franklin": ROSALIND_FRANKLIN,
}
"""Every rover the simulator knows, by key."""


def get_rover(name: str) -> RoverProfile:
    """Profile of a rover by key or by name, case-insensitive.

    Raises:
        ValidationError: no such rover is modelled.
    """
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in ROVERS:
        raise ValidationError(f"unknown rover '{name}'; modelled rovers: {sorted(ROVERS)}")
    return ROVERS[key]


def rovers_on_mars(mission_year: float) -> list[RoverProfile]:
    """Rovers that have landed by ``mission_year``."""
    return [rover for rover in ROVERS.values() if rover.landing_year <= mission_year]
