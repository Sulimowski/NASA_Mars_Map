"""Rover physics R1-R3 and the rover profiles."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.core.config import RoverConfig, SimConfig
from src.core.constants import HOUR_S, SOL_S
from src.core.exceptions import (ConfigurationError, ProfileValidationError,
                                 TerrainValidationError, ValidationError)
from src.core.schemas import EnvSample
from src.providers.grid import GridTerrain
from src.providers.obstacles import (default_obstacle_height_m, relief_step_m,
                                     rock_obstacle_height_m)
from src.providers.synthetic import (DiurnalEnvironment, flat_terrain, patch_terrain,
                                     reference_day_sample, reference_night_sample,
                                     slope_terrain)
from src.rovers import rover_mobility, rover_power_thermal, rover_radiation
from src.rovers.profiles import (CURIOSITY, PERSEVERANCE, ROSALIND_FRANKLIN, ROVERS, get_rover,
                                 rovers_on_mars)
from src.rovers.types import DriveMode, PowerSource, RoverBlock


def step(profile, cfg: SimConfig, *, slope_deg: float = 0.0, eta: float = 1.2,
         obstacle: float = 0.0, cell_slope_deg: float | None = None):
    return rover_mobility.assess_step(
        profile, cfg.rover, slope_rad=math.radians(slope_deg),
        cell_slope_rad=math.radians(abs(slope_deg) if cell_slope_deg is None else cell_slope_deg),
        eta=eta, obstacle_height_m=obstacle, wheel_speed_m_s=profile.max_speed_m_s)


# ---- profiles -------------------------------------------------------------
def test_only_active_and_planned_rovers_are_modelled() -> None:
    assert set(ROVERS) == {"curiosity", "perseverance", "rosalind_franklin"}
    assert [r.landing_year for r in ROVERS.values()] == [2012, 2021, 2030]
    assert [r.planned for r in ROVERS.values()] == [False, False, True]
    for retired in ("Sojourner", "Spirit", "Opportunity", "Zhurong", "Sample Fetch Rover"):
        with pytest.raises(ValidationError):
            get_rover(retired)
    assert get_rover("Rosalind Franklin") is ROSALIND_FRANKLIN
    assert get_rover("CURIOSITY") is CURIOSITY


def test_profiles_carry_the_published_numbers() -> None:
    assert (CURIOSITY.mass_kg, PERSEVERANCE.mass_kg, ROSALIND_FRANKLIN.mass_kg) == (899, 1025, 310)
    assert CURIOSITY.max_speed_m_s == pytest.approx(0.038)
    assert PERSEVERANCE.max_speed_m_s * HOUR_S == pytest.approx(151.2, abs=1.0)      # 152 m/h
    assert ROSALIND_FRANKLIN.max_speed_m_s * HOUR_S == pytest.approx(40.0)
    assert (CURIOSITY.wheel_diameter_m, PERSEVERANCE.wheel_diameter_m) == (0.50, 0.525)
    assert PERSEVERANCE.max_obstacle_height_m == 0.40
    assert math.degrees(PERSEVERANCE.max_tilt_rad) == pytest.approx(24.0)
    assert math.degrees(CURIOSITY.max_tilt_rad) == pytest.approx(30.0)
    assert CURIOSITY.power_source is PERSEVERANCE.power_source is PowerSource.MMRTG
    assert ROSALIND_FRANKLIN.power_source is PowerSource.SOLAR
    assert ROSALIND_FRANKLIN.solar_area_m2 == 2.39
    assert ROSALIND_FRANKLIN.battery_j / HOUR_S == pytest.approx(1142.0)
    assert ROSALIND_FRANKLIN.wheel_walking and not CURIOSITY.wheel_walking
    assert ROSALIND_FRANKLIN.ground_clearance_m < CURIOSITY.ground_clearance_m
    assert CURIOSITY.battery_reserve_j == pytest.approx(0.47 * CURIOSITY.battery_j)


def test_a_rover_cannot_be_simulated_before_it_lands() -> None:
    assert CURIOSITY.mission_age_years(2026) == 14.0
    assert ROSALIND_FRANKLIN.mission_age_years(2030) == 0.0
    with pytest.raises(ValidationError):
        ROSALIND_FRANKLIN.mission_age_years(2026)
    with pytest.raises(ValidationError):
        PERSEVERANCE.mission_age_years(2020)
    assert [r.name for r in rovers_on_mars(2026)] == ["Curiosity", "Perseverance"]
    assert len(rovers_on_mars(2031)) == 3 and rovers_on_mars(2011) == []


def test_profile_validation() -> None:
    with pytest.raises(ProfileValidationError):
        replace(CURIOSITY, mass_kg=0.0)
    with pytest.raises(ProfileValidationError):
        replace(CURIOSITY, max_obstacle_height_m=0.9)            # above the clearance
    with pytest.raises(ProfileValidationError):
        replace(ROSALIND_FRANKLIN, solar_area_m2=0.0)
    with pytest.raises(ProfileValidationError):
        replace(CURIOSITY, generator_bol_w=0.0)
    with pytest.raises(ProfileValidationError):
        replace(ROSALIND_FRANKLIN, wheel_walk_speed_m_s=0.0)
    with pytest.raises(ConfigurationError):
        SimConfig(rover=RoverConfig(slip_critical=1.2))


# ---- R1: mobility ---------------------------------------------------------
def test_clearance_curiosity_passes_what_stops_rosalind_franklin(cfg: SimConfig) -> None:
    assert step(CURIOSITY, cfg, obstacle=0.35).passable
    assert step(PERSEVERANCE, cfg, obstacle=0.35).passable
    assert step(ROSALIND_FRANKLIN, cfg, obstacle=0.35).block is RoverBlock.CLEARANCE
    assert step(PERSEVERANCE, cfg, obstacle=0.45).block is RoverBlock.CLEARANCE
    assert step(CURIOSITY, cfg, obstacle=0.45).passable
    assert step(CURIOSITY, cfg, obstacle=0.60).block is RoverBlock.CLEARANCE
    assert step(ROSALIND_FRANKLIN, cfg, obstacle=0.20).passable


def test_tilt_and_climb_limits(cfg: SimConfig) -> None:
    assert step(CURIOSITY, cfg, slope_deg=0.0, cell_slope_deg=31.0).block is RoverBlock.TILT
    assert step(PERSEVERANCE, cfg, slope_deg=0.0, cell_slope_deg=25.0).block is RoverBlock.TILT
    assert step(CURIOSITY, cfg, slope_deg=0.0, cell_slope_deg=25.0).passable
    assert step(PERSEVERANCE, cfg, slope_deg=-26.0, cell_slope_deg=10.0).block is RoverBlock.CLIMB
    assert step(CURIOSITY, cfg, slope_deg=-26.0, cell_slope_deg=10.0, eta=1.0).passable


def test_slip_grows_with_slope_and_loose_ground(cfg: SimConfig) -> None:
    for profile in ROVERS.values():
        by_slope = [step(profile, cfg, slope_deg=d, eta=1.0).slip for d in (0, 5, 10, 15)]
        assert by_slope == sorted(by_slope) and by_slope[0] < by_slope[-1]
        by_soil = [step(profile, cfg, slope_deg=5, eta=e).slip for e in (1.0, 1.4, 1.8)]
        assert by_soil == sorted(by_soil) and by_soil[0] < by_soil[-1]
        level = step(profile, cfg, eta=1.0)
        assert level.passable and level.slip < 0.05 and level.mode is DriveMode.ROLLING
        assert level.ground_speed_m_s == pytest.approx(profile.max_speed_m_s * (1 - level.slip))
        downhill = step(profile, cfg, slope_deg=-10, eta=1.0)
        assert downhill.slip == 0.0 and downhill.thrust_n == 0.0


def test_perseverance_slips_less_than_curiosity_on_loose_ground(cfg: SimConfig) -> None:
    for slope in (5, 10, 15):
        for eta in (1.5, 2.1):
            curiosity = step(CURIOSITY, cfg, slope_deg=slope, eta=eta)
            perseverance = step(PERSEVERANCE, cfg, slope_deg=slope, eta=eta)
            if curiosity.passable:
                assert perseverance.passable and perseverance.slip < curiosity.slip
    assert step(CURIOSITY, cfg, slope_deg=15, eta=2.1).block is RoverBlock.SLIP
    assert step(PERSEVERANCE, cfg, slope_deg=15, eta=2.1).passable


def test_rosalind_franklin_wheel_walks_through_loose_slopes(cfg: SimConfig) -> None:
    rolling = step(ROSALIND_FRANKLIN, cfg, slope_deg=5, eta=2.1)
    walking = step(ROSALIND_FRANKLIN, cfg, slope_deg=12, eta=2.1)
    assert rolling.mode is DriveMode.ROLLING and rolling.passable
    assert walking.passable and walking.mode is DriveMode.WHEEL_WALKING
    assert walking.ground_speed_m_s == pytest.approx(2.0 / 1200.0)
    assert walking.ground_speed_m_s < 0.5 * rolling.ground_speed_m_s
    assert walking.slip < cfg.rover.slip_critical_wheel_walking
    no_walking = replace(ROSALIND_FRANKLIN, wheel_walking=False, wheel_walk_speed_m_s=0.0)
    assert step(no_walking, cfg, slope_deg=12, eta=2.1).block is RoverBlock.SLIP


def test_soil_and_sinkage(cfg: SimConfig) -> None:
    firm = rover_mobility.soil_from_terrain_factor(1.0, cfg.rover)
    loose = rover_mobility.soil_from_terrain_factor(2.1, cfg.rover)
    middle = rover_mobility.soil_from_terrain_factor(1.55, cfg.rover)
    assert firm.k_phi == pytest.approx(cfg.rover.k_phi_firm)
    assert loose.k_phi == pytest.approx(cfg.rover.k_phi_loose)
    assert loose.k_phi < middle.k_phi < firm.k_phi
    assert rover_mobility.soil_from_terrain_factor(5.0, cfg.rover) == loose
    load = CURIOSITY.weight_n / 6
    z_firm = rover_mobility.wheel_sinkage_m(load, 0.5, 0.4, firm)
    z_loose = rover_mobility.wheel_sinkage_m(load, 0.5, 0.4, loose)
    assert 0.0 < z_firm < 0.01 < z_loose < 0.10
    assert (rover_mobility.compaction_resistance_n(z_loose, 0.4, loose)
            > rover_mobility.compaction_resistance_n(z_firm, 0.4, firm))
    assert rover_mobility.wheel_thrust_n(0.0, load, 0.1, 0.4, firm, 1.0) == 0.0
    thrusts = [rover_mobility.wheel_thrust_n(s, load, 0.1, 0.4, firm, 1.0) for s in (0.1, 0.3, 0.6)]
    assert thrusts == sorted(thrusts)


# ---- R2: power and heat ---------------------------------------------------
def test_mmrtg_follows_the_nasa_decline(cfg: SimConfig) -> None:
    assert rover_power_thermal.mmrtg_power_w(110.0, 0.0, cfg.rover) == 110.0
    assert rover_power_thermal.mmrtg_power_w(110.0, 17.0, cfg.rover) == pytest.approx(72.0)
    curiosity_2026 = rover_power_thermal.mmrtg_power_w(110.0, CURIOSITY.mission_age_years(2026),
                                                       cfg.rover)
    perseverance_2026 = rover_power_thermal.mmrtg_power_w(
        110.0, PERSEVERANCE.mission_age_years(2026), cfg.rover)
    assert 70.0 < curiosity_2026 < 85.0 < perseverance_2026 < 110.0
    with pytest.raises(ValidationError):
        rover_power_thermal.mmrtg_power_w(110.0, -1.0, cfg.rover)


def test_night_solar_rover_generates_nothing_mmrtg_rover_keeps_going(cfg: SimConfig) -> None:
    day, night = reference_day_sample(), reference_night_sample()

    def gen(profile, env: EnvSample, year: float) -> float:
        return rover_power_thermal.generation_w(profile, env, profile.mission_age_years(year),
                                                1.0, cfg.rover)

    assert gen(ROSALIND_FRANKLIN, night, 2031) == 0.0
    assert gen(ROSALIND_FRANKLIN, day, 2031) > 100.0
    assert gen(CURIOSITY, night, 2026) == gen(CURIOSITY, day, 2026) > 70.0

    def parked_balance(profile, year: float) -> float:
        heat = rover_power_thermal.heater_demand(profile, night, profile.sleep_w, moving=False)
        return gen(profile, night, year) - profile.sleep_w - heat.power_w

    assert parked_balance(CURIOSITY, 2026) > 0.0            # recharges through the night
    assert parked_balance(PERSEVERANCE, 2026) > 0.0
    assert parked_balance(ROSALIND_FRANKLIN, 2031) < 0.0    # lives on its battery


def test_solar_array_energy_per_sol_matches_esa(cfg: SimConfig) -> None:
    env = DiurnalEnvironment()
    joules = sum(rover_power_thermal.solar_power_w(ROSALIND_FRANKLIN, env.get_env((0, 0), t), 1.0)
                 * 60.0 for t in range(0, int(SOL_S), 60))
    assert joules / HOUR_S == pytest.approx(1200.0, rel=0.05)
    dusty = rover_power_thermal.panel_dust_factor(100.0, cfg.rover)
    assert dusty == pytest.approx(math.exp(-0.2)) and rover_power_thermal.panel_dust_factor(
        0.0, cfg.rover) == 1.0


def test_drive_power_and_no_energy_recovery(cfg: SimConfig) -> None:
    level = step(CURIOSITY, cfg, eta=1.0)
    climb = step(CURIOSITY, cfg, slope_deg=10, eta=1.0)
    descent = step(CURIOSITY, cfg, slope_deg=-10, eta=1.0)
    p_level, p_climb, p_descent = (rover_power_thermal.drive_power_w(CURIOSITY, r)
                                   for r in (level, climb, descent))
    assert p_descent == pytest.approx(CURIOSITY.drive_idle_w)       # nothing comes back
    assert p_descent < p_level < p_climb
    expected = climb.thrust_n * climb.wheel_speed_m_s / CURIOSITY.drivetrain_efficiency
    assert p_climb == pytest.approx(expected + CURIOSITY.drive_idle_w)
    blocked = step(CURIOSITY, cfg, obstacle=1.0)
    assert rover_power_thermal.drive_power_w(CURIOSITY, blocked) == 0.0
    sand = step(CURIOSITY, cfg, slope_deg=5, eta=2.1)
    firm = step(CURIOSITY, cfg, slope_deg=5, eta=1.0)
    assert rover_power_thermal.drive_power_w(CURIOSITY, sand) > rover_power_thermal.drive_power_w(
        CURIOSITY, firm)


def test_heaters_and_thermal_window() -> None:
    night = reference_night_sample()
    warm = rover_power_thermal.heater_demand(CURIOSITY, reference_day_sample(),
                                             CURIOSITY.avionics_w, moving=True)
    assert warm.power_w == 0.0 and warm.deficit_w == 0.0
    cold_drive = rover_power_thermal.heater_demand(CURIOSITY, night, CURIOSITY.avionics_w,
                                                   moving=True)
    cold_parked = rover_power_thermal.heater_demand(CURIOSITY, night, CURIOSITY.sleep_w,
                                                    moving=False)
    assert cold_drive.power_w > 0.0 == cold_parked.power_w      # MMRTG heat covers the body
    arctic = replace(night, t_ground_k=140.0)
    frozen = rover_power_thermal.heater_demand(ROSALIND_FRANKLIN, arctic,
                                               ROSALIND_FRANKLIN.avionics_w, moving=True)
    assert frozen.power_w == ROSALIND_FRANKLIN.heater_max_w and frozen.deficit_w > 0.0


def test_battery_charges_discharges_and_saturates(cfg: SimConfig) -> None:
    cap = CURIOSITY.battery_j
    after = rover_power_thermal.battery_after(0.5 * cap, 100.0, 40.0, HOUR_S, CURIOSITY, cfg.rover)
    assert after == pytest.approx(0.5 * cap + 0.95 * 60.0 * HOUR_S)
    after = rover_power_thermal.battery_after(0.5 * cap, 40.0, 100.0, HOUR_S, CURIOSITY, cfg.rover)
    assert after == pytest.approx(0.5 * cap - 60.0 * HOUR_S / 0.95)
    assert rover_power_thermal.battery_after(cap, 500.0, 0.0, HOUR_S, CURIOSITY, cfg.rover) == cap
    assert rover_power_thermal.battery_after(0.0, 0.0, 100.0, HOUR_S, CURIOSITY, cfg.rover) < 0.0


# ---- R3: radiation --------------------------------------------------------
def test_particle_event_slows_the_rover_and_adds_upset_risk(cfg: SimConfig) -> None:
    quiet = reference_day_sample()
    event = replace(quiet, spe_active=True, spe_dose_rate_sv_s=cfg.rover.seu_reference_dose_rate_sv_s)
    assert rover_radiation.speed_factor(quiet, cfg.rover) == 1.0
    assert rover_radiation.speed_factor(event, cfg.rover) == cfg.rover.spe_speed_factor
    assert rover_radiation.upset_risk(quiet, HOUR_S, cfg.rover) == 0.0
    assert rover_radiation.upset_risk(event, HOUR_S, cfg.rover) == pytest.approx(
        cfg.rover.seu_rate_per_s * HOUR_S)
    stronger = replace(event, spe_dose_rate_sv_s=3 * event.spe_dose_rate_sv_s)
    assert rover_radiation.upset_risk(stronger, HOUR_S, cfg.rover) == pytest.approx(
        3 * cfg.rover.seu_rate_per_s * HOUR_S)
    assert rover_radiation.upset_risk(stronger, 1e12, cfg.rover) == 1.0
    hold = replace(cfg.rover, spe_hold=True)
    assert rover_radiation.speed_factor(event, hold) == 0.0
    assert rover_radiation.speed_factor(quiet, hold) == 1.0
    assert rover_radiation.absorbed_dose_gy(event, SOL_S, cfg.rover) > rover_radiation.absorbed_dose_gy(
        quiet, SOL_S, cfg.rover) == pytest.approx(0.64e-3 / 3.05)


# ---- obstacle height layer ------------------------------------------------
def test_rock_model_of_golombek_and_rapp() -> None:
    heights = rock_obstacle_height_m(np.array([0.0, 0.005, 0.05, 0.10, 0.20, 0.40]))
    assert heights[0] == heights[1] == 0.0                  # no obstacle rocks under 1 % cover
    assert heights.tolist() == sorted(heights.tolist())
    assert heights[3] == pytest.approx(0.5 * math.log(10.0) / (1.79 + 1.52), rel=1e-6)
    assert 0.30 < heights[3] < 0.40 < heights[4] < 0.60     # 10 % stops Rosalind Franklin only


def test_relief_steps_ignore_slopes_and_catch_ledges() -> None:
    cols = np.arange(30, dtype=float)
    ramp = np.tile(cols * 0.5, (10, 1))                     # an even 25 % slope on a 2 m grid
    assert relief_step_m(ramp).max() == pytest.approx(0.0, abs=1e-9)
    ledge = ramp.copy()
    ledge[5, 15] += 0.6
    steps = relief_step_m(ledge)
    assert steps[5, 15] == pytest.approx(2.0 * 0.6 * 8 / 9) and steps[0, 0] == 0.0
    fine = default_obstacle_height_m(ledge, 2.0, np.zeros_like(ledge))
    coarse = default_obstacle_height_m(ledge, 20.0, np.zeros_like(ledge))
    assert fine[5, 15] > 1.0 and coarse.max() == 0.0        # hills are not obstacles


def test_terrain_carries_the_obstacle_layer() -> None:
    plain = flat_terrain(5, 5, 10.0)
    assert plain.obstacle_height_m.shape == (5, 5) and plain.obstacle_height_m.max() == 0.0
    rocky = flat_terrain(5, 5, 10.0, rock_hazard=0.5)
    assert rocky.obstacle_height_m.min() == pytest.approx(
        float(rock_obstacle_height_m(np.array([0.2]))[0]))
    patch = patch_terrain(6, 10, 10.0, zone=(1, 4, 3, 6), obstacle_height_m=0.35, zone_eta=2.0)
    assert patch.obstacle_height_m[2, 4] == 0.35 and patch.obstacle_height_m[0, 0] == 0.0
    assert patch.walk_eta[2, 4] == 2.0 and patch.walk_eta[5, 9] == 1.2
    assert slope_terrain(4, 20, 2.0, grade=0.3).obstacle_height_m.max() == pytest.approx(0, abs=1e-9)
    with pytest.raises(TerrainValidationError):
        GridTerrain(elevation=np.zeros((3, 3)), cell_size_m=10.0,
                    obstacle_height_m=np.full((3, 3), -1.0))
    with pytest.raises(TerrainValidationError):
        GridTerrain(elevation=np.zeros((3, 3)), cell_size_m=10.0,
                    obstacle_height_m=np.zeros((2, 2)))
