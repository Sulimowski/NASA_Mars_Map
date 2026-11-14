"""Unit tests of algorithms A1-A12 and of the input validation."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.core.config import RoutingConfig, SimConfig
from src.core.constants import (EARTH_DAY_S, EARTH_YEAR_S, HOUR_S, O2_DENSITY_STP_KG_M3, SOL_S,
                                STEFAN_BOLTZMANN_W_M2_K4)
from src.core.exceptions import (ConfigurationError, EnvironmentValidationError,
                                 ProfileValidationError, TerrainValidationError,
                                 ValidationError)
from src.core.schemas import (AstronautProfile, CoolingMode, EnvSample, SafetyFlag, SuitProfile,
                              SuitResources, TaskType)
from src.physiology import (adaptation, capacity, dust, emergency, fatigue, injury, life_support,
                            metabolism, nutrition, radiation, thermal)
from src.physiology.fatigue import DailyEvaRecord
from src.physiology.radiation import DoseLedger
from src.providers.grid import GridTerrain
from src.providers.synthetic import reference_day_sample, reference_night_sample

TRANSIT_S = 180.0 * EARTH_DAY_S


# ---- A1 -------------------------------------------------------------------
def test_a1_fitness_at_landing(cfg: SimConfig) -> None:
    fit = adaptation.compute_fitness(0.0, TRANSIT_S, cfg.adaptation)
    assert fit.aerobic == pytest.approx(1.0 - 0.0012 * 180.0)
    assert fit.sensorimotor == pytest.approx(0.5)
    assert fit.orthostatic == pytest.approx(0.6)
    assert fit.bone_fracture_mult > 1.0


def test_a1_fitness_recovers_monotonically_to_a_plateau(cfg: SimConfig) -> None:
    values = [adaptation.compute_fitness(d * EARTH_DAY_S, TRANSIT_S, cfg.adaptation).aerobic
              for d in (0, 1, 7, 30, 60, 500)]
    assert values == sorted(values)
    assert values[-1] == pytest.approx(1.0 - cfg.adaptation.aerobic_plateau_loss, abs=1e-3)


def test_a1_loss_is_capped_and_zero_transit_loses_nothing(cfg: SimConfig) -> None:
    long_transit = adaptation.compute_fitness(0.0, 1000.0 * EARTH_DAY_S, cfg.adaptation)
    assert long_transit.aerobic == pytest.approx(1.0 - cfg.adaptation.aerobic_loss_max)
    none = adaptation.compute_fitness(0.0, 0.0, cfg.adaptation)
    assert none.aerobic == pytest.approx(1.0) and none.strength == pytest.approx(1.0)


def test_a1_rejects_negative_durations(cfg: SimConfig) -> None:
    with pytest.raises(ValidationError):
        adaptation.compute_fitness(-1.0, TRANSIT_S, cfg.adaptation)


def test_a1_first_sol_rule_and_reduced_slope_limit(cfg: SimConfig) -> None:
    assert not adaptation.eva_permitted(0.5 * SOL_S, cfg.adaptation)
    assert adaptation.eva_permitted(1.5 * SOL_S, cfg.adaptation)
    early = adaptation.compute_fitness(1.0 * EARTH_DAY_S, TRANSIT_S, cfg.adaptation)
    late = adaptation.compute_fitness(30.0 * EARTH_DAY_S, TRANSIT_S, cfg.adaptation)
    limits = (math.radians(25), math.radians(20), math.radians(10))
    assert adaptation.slope_limits_rad(early, cfg.adaptation, *limits) == (limits[2], limits[2])
    assert adaptation.slope_limits_rad(late, cfg.adaptation, *limits) == (limits[0], limits[1])


# ---- A2 -------------------------------------------------------------------
def test_a2_power_never_drops_below_standing(cfg: SimConfig) -> None:
    standing = 1.25 * 1.5 * 80.0
    for grade in (-0.35, -0.20, -0.10, 0.0):
        for speed in (0.0, 0.2, 0.8, 1.2):
            assert metabolism.walking_power_w(80, 60, speed, grade, 1.2, 1.25,
                                              cfg.metabolism) >= standing - 1e-9


def test_a2_softer_ground_and_steeper_climb_cost_more(cfg: SimConfig) -> None:
    def p(grade: float, eta: float) -> float:
        return metabolism.walking_power_w(80, 60, 0.8, grade, eta, 1.25, cfg.metabolism)

    assert p(0.0, 1.0) < p(0.0, 1.5) < p(0.0, 2.1)
    assert p(0.05, 1.2) < p(0.10, 1.2) < p(0.20, 1.2)


def test_a2_effective_grade_and_efficiency(cfg: SimConfig) -> None:
    assert metabolism.effective_grade_percent(0.10, cfg.metabolism) == pytest.approx(10.0)
    assert metabolism.effective_grade_percent(-0.10, cfg.metabolism) == pytest.approx(-5.0)
    assert metabolism.effective_grade_percent(-0.20, cfg.metabolism) == pytest.approx(-4.0)
    assert metabolism.mechanical_efficiency(-0.1, cfg.metabolism) == 0.0
    assert metabolism.mechanical_efficiency(0.05, cfg.metabolism) == pytest.approx(0.1)
    assert metabolism.mechanical_efficiency(0.50, cfg.metabolism) == pytest.approx(0.2)


def test_a2_tasks_and_invalid_inputs(cfg: SimConfig) -> None:
    fresh = metabolism.task_power_w(TaskType.SAMPLING, 0.0, cfg.metabolism)
    tired = metabolism.task_power_w(TaskType.SAMPLING, 1.0, cfg.metabolism)
    assert fresh == pytest.approx(300.0) and tired == pytest.approx(390.0)
    assert metabolism.task_power_w(TaskType.REST, 1.0, cfg.metabolism) == pytest.approx(110.0)
    with pytest.raises(ValidationError):
        metabolism.walking_power_w(80, 60, -0.1, 0.0, 1.2, 1.25, cfg.metabolism)
    with pytest.raises(ValidationError):
        metabolism.walking_power_w(0, 60, 0.5, 0.0, 1.2, 1.25, cfg.metabolism)


# ---- A3 -------------------------------------------------------------------
def _coeffs(cfg: SimConfig, grade: float) -> tuple[float, float, float, float]:
    return metabolism.pandolf_coefficients(80, 60, 1.2, grade, 1.25, cfg.metabolism)


def test_a3_speed_is_preferred_when_power_allows(cfg: SimConfig) -> None:
    assert capacity.select_speed(*_coeffs(cfg, 0.0), 0.0, 0.8333, 500.0, cfg.capacity) == 0.8333


def test_a3_uphill_binary_search_meets_the_cap(cfg: SimConfig) -> None:
    coeffs = _coeffs(cfg, 0.15)
    speed = capacity.select_speed(*coeffs, 0.15, 0.8333, 450.0, cfg.capacity)
    assert speed is not None and cfg.capacity.min_speed_m_s <= speed < 0.8333
    assert metabolism.power_from_coefficients(*coeffs, speed) <= 450.0
    faster = speed + 2.0 * cfg.capacity.speed_tolerance_m_s
    assert metabolism.power_from_coefficients(*coeffs, faster) > 450.0


def test_a3_impassable_when_minimum_speed_exceeds_the_cap(cfg: SimConfig) -> None:
    assert capacity.select_speed(*_coeffs(cfg, 0.30), 0.30, 0.8333, 200.0, cfg.capacity) is None
    assert capacity.select_speed(*_coeffs(cfg, 0.0), 0.0, 0.1, 500.0, cfg.capacity) is None


def test_a3_downhill_grid_search_returns_the_fastest_feasible_grid_speed(cfg: SimConfig) -> None:
    coeffs = _coeffs(cfg, -0.30)
    cap = 0.5 * (metabolism.power_from_coefficients(*coeffs, 0.8333)
                 + metabolism.power_from_coefficients(*coeffs, 0.3))
    speed = capacity.select_speed(*coeffs, -0.30, 0.8333, cap, cfg.capacity)
    assert speed is not None
    assert metabolism.power_from_coefficients(*coeffs, speed) <= cap
    steps = (0.8333 - speed) / cfg.capacity.downhill_speed_step_m_s
    assert steps == pytest.approx(round(steps), abs=1e-6)
    next_up = speed + cfg.capacity.downhill_speed_step_m_s
    assert metabolism.power_from_coefficients(*coeffs, next_up) > cap


def test_a3_power_cap_modifiers(cfg: SimConfig) -> None:
    base = capacity.power_cap_w(450.0, math.inf, 0.0, 1.0, 1.0, 1.0, cfg.capacity)
    assert base == pytest.approx(450.0)
    assert capacity.power_cap_w(450.0, 300.0, 0.0, 1.0, 1.0, 1.0, cfg.capacity) == pytest.approx(300.0)
    assert capacity.power_cap_w(450.0, math.inf, 1.0, 1.0, 1.0, 1.0, cfg.capacity) == pytest.approx(315.0)
    assert capacity.readiness_factor(2.0 * HOUR_S, cfg.capacity) == pytest.approx(0.9)
    assert capacity.readiness_factor(1.0e9, cfg.capacity) == 0.0


def test_a3_heart_rate(cfg: SimConfig, astronaut: AstronautProfile) -> None:
    fit = adaptation.compute_fitness(30 * EARTH_DAY_S, TRANSIT_S, cfg.adaptation)
    rest = capacity.heart_rate_hz(0.0, astronaut, fit, cfg.capacity)
    work = capacity.heart_rate_hz(400.0, astronaut, fit, cfg.capacity)
    top = capacity.heart_rate_hz(5000.0, astronaut, fit, cfg.capacity)
    assert astronaut.hr_rest_hz <= rest < work < top == astronaut.hr_max_hz
    assert capacity.heart_rate_alarm(top, astronaut, cfg.capacity)
    assert not capacity.heart_rate_alarm(rest, astronaut, cfg.capacity)


# ---- A4 -------------------------------------------------------------------
def test_a4_consumption_of_one_hour_at_300_w(cfg: SimConfig, suit: SuitProfile) -> None:
    env = reference_day_sample()
    spent = life_support.consume(300.0, 200.0, 20.0, HOUR_S, suit, env, cfg.life_support)
    volume = 300.0 * HOUR_S / 2.02e7
    assert spent.o2_kg == pytest.approx(volume * O2_DENSITY_STP_KG_M3 + 0.005)
    assert spent.co2_kg == pytest.approx(volume * 0.85 * 1.977)
    assert spent.cooling_water_kg == pytest.approx(200.0 * HOUR_S / 2.4e6)
    assert spent.battery_j == pytest.approx((60.0 + 40.0 + 20.0) * HOUR_S)
    assert spent.eva_s == HOUR_S


def test_a4_radiator_uses_no_water_and_night_adds_lights(cfg: SimConfig, suit: SuitProfile) -> None:
    radiator = replace(suit, cooling_mode=CoolingMode.RADIATOR_STORAGE)
    day, night = reference_day_sample(), reference_night_sample()
    assert life_support.consume(300, 200, 0, 600, radiator, day,
                                cfg.life_support).cooling_water_kg == 0.0
    extra = (life_support.electric_power_w(300, 0, suit, night, cfg.life_support)
             - life_support.electric_power_w(300, 0, suit, day, cfg.life_support))
    assert extra == pytest.approx(cfg.life_support.lights_power_w)


def test_a4_limiting_resource(cfg: SimConfig, suit: SuitProfile) -> None:
    left = SuitResources.full(suit)
    spent = life_support.ResourceDelta(o2_kg=0.05, co2_kg=0.01, cooling_water_kg=0.1,
                                       battery_j=5.0e5, eva_s=HOUR_S)
    name, time_left = life_support.limiting_resource(left, spent, HOUR_S, suit)
    assert name == "battery"                       # 5.0 h, ahead of the 8 h EVA limit
    assert time_left == pytest.approx(suit.battery_j / 5.0e5 * HOUR_S)
    slow_battery = replace(spent, battery_j=1.0e5)
    assert life_support.limiting_resource(left, slow_battery, HOUR_S, suit)[0] == "eva_time"
    assert life_support.limiting_resource(left, spent, 0.0, suit) == ("none", math.inf)


# ---- A5 -------------------------------------------------------------------
def test_a5_surface_temperature_solves_the_flux_balance(cfg: SimConfig, suit: SuitProfile) -> None:
    for env in (reference_day_sample(), reference_night_sample()):
        t_s = thermal.suit_surface_temperature_k(suit, env, cfg.thermal)
        el = env.sun_elevation_rad
        proj = (0.25 * math.cos(el) + 0.08 * math.sin(el)) if el > 0 else 0.0
        sigma_eps = STEFAN_BOLTZMANN_W_M2_K4 * suit.epsilon_ir
        absorbed = (suit.alpha_solar * (env.solar_direct_w_m2 * proj + 0.5 * env.solar_diffuse_w_m2)
                    + 0.5 * sigma_eps * (env.t_sky_k ** 4 + env.t_ground_k ** 4))
        h_c = thermal.convection_coefficient_w_m2_k(env, cfg.thermal)
        residual = (absorbed + h_c * (env.t_air_k - t_s)
                    + (cfg.thermal.suit_internal_temp_k - t_s) / cfg.thermal.shell_resistance_m2_k_w
                    - sigma_eps * t_s ** 4)
        assert abs(residual) < 1e-6
        assert env.t_sky_k < t_s < cfg.thermal.suit_internal_temp_k


def test_a5_night_loses_more_heat_and_runs_extremity_heaters(
        cfg: SimConfig, suit: SuitProfile, astronaut: AstronautProfile) -> None:
    day = thermal.environmental_heat(astronaut, suit, reference_day_sample(), cfg.thermal,
                                     cfg.life_support)
    night = thermal.environmental_heat(astronaut, suit, reference_night_sample(), cfg.thermal,
                                       cfg.life_support)
    assert night.q_fixed_w < day.q_fixed_w < 0.0
    assert day.extremity_heater_w == 0.0 and night.extremity_heater_w > 0.0


def test_a5_membrane_capacity_falls_as_pressure_nears_saturation(
        cfg: SimConfig, suit: SuitProfile) -> None:
    def cap(pressure: float) -> float:
        env = replace(reference_day_sample(), pressure_pa=pressure)
        return thermal.cooling_capacity_w(suit, env, cfg.thermal, cfg.life_support)

    assert cap(610.0) == pytest.approx(suit.cooling_max_w)
    assert cap(300.0) == pytest.approx(suit.cooling_max_w)
    assert 0.0 < cap(1200.0) < cap(800.0) < suit.cooling_max_w
    assert cap(2000.0) == 0.0


def test_a5_radiator_capacity_depends_on_the_sky(cfg: SimConfig, suit: SuitProfile) -> None:
    radiator = replace(suit, cooling_mode=CoolingMode.RADIATOR_STORAGE)
    day = thermal.cooling_capacity_w(radiator, reference_day_sample(), cfg.thermal, cfg.life_support)
    night = thermal.cooling_capacity_w(radiator, reference_night_sample(), cfg.thermal,
                                       cfg.life_support)
    assert 0.0 < day < night <= radiator.cooling_max_w
    assert thermal.heat_storage_limit_j(radiator, cfg.thermal) == pytest.approx(
        cfg.thermal.heat_storage_limit_j + radiator.thermal_storage_capacity_j)


def test_a5_balance_cooling_heating_and_storage(cfg: SimConfig) -> None:
    normal = thermal.thermal_balance(400.0, 0.0, -100.0, 550.0, 0.0, 60.0, 150.0, cfg.thermal)
    assert normal.cooling_w == pytest.approx(300.0) and normal.heater_w == 0.0
    assert normal.heat_storage_j == pytest.approx(0.0)
    overload = thermal.thermal_balance(700.0, 0.0, 0.0, 550.0, 0.0, 60.0, 150.0, cfg.thermal)
    assert overload.cooling_w == pytest.approx(550.0)
    assert overload.heat_storage_j == pytest.approx(150.0 * 60.0)
    cold = thermal.thermal_balance(110.0, 0.0, -300.0, 550.0, 0.0, 60.0, 150.0, cfg.thermal)
    assert cold.heater_w == pytest.approx(150.0) and cold.cooling_w == 0.0
    assert cold.heat_storage_j == pytest.approx(-40.0 * 60.0)


def test_a5_stored_heat_relaxes_exactly_for_any_step(cfg: SimConfig) -> None:
    for dt in (10.0, 600.0, 5000.0):
        result = thermal.thermal_balance(300.0, 0.0, -300.0, 550.0, 100.0e3, dt, 150.0, cfg.thermal)
        expected = 100.0e3 * math.exp(-dt / cfg.thermal.storage_relax_tau_s)
        assert result.heat_storage_j == pytest.approx(max(expected, 100.0e3 - 550.0 * dt))


def test_a5_thermal_limit_and_touch_limits(cfg: SimConfig) -> None:
    assert thermal.thermal_limit_power_w(550.0, -150.0, 0.0) == pytest.approx(700.0)
    assert thermal.thermal_limit_power_w(550.0, -150.0, 0.2) == pytest.approx(875.0)
    assert thermal.thermal_limit_power_w(100.0, 300.0, 0.0) == 0.0
    assert thermal.touch_contact_limit_s(260.0, cfg.thermal) == math.inf
    assert thermal.touch_contact_limit_s(230.0, cfg.thermal) == cfg.thermal.touch_time_cold_s
    assert thermal.touch_contact_limit_s(180.0, cfg.thermal) == cfg.thermal.touch_time_very_cold_s
    assert thermal.touch_contact_limit_s(120.0, cfg.thermal) == cfg.thermal.touch_time_extreme_s
    assert thermal.touch_contact_limit_s(350.0, cfg.thermal) == cfg.thermal.touch_time_hot_s


# ---- A6 -------------------------------------------------------------------
def test_a6_dose_rate_with_and_without_particle_event(suit: SuitProfile) -> None:
    quiet = reference_day_sample()
    event = replace(quiet, spe_active=True, spe_dose_rate_sv_s=3.0 * quiet.dose_rate_gcr_sv_s)
    assert radiation.dose_increment_sv(quiet, SOL_S, 1.0) == pytest.approx(0.64e-3)
    assert radiation.dose_increment_sv(event, SOL_S, 1.0) == pytest.approx(4 * 0.64e-3)
    assert radiation.dose_increment_sv(quiet, SOL_S, 0.5) == pytest.approx(0.32e-3)


def test_a6_ledger_rolling_windows(cfg: SimConfig) -> None:
    ledger = DoseLedger(cfg.radiation)
    ledger.add(0.0, 1.0e-3)
    ledger.add(10 * EARTH_DAY_S, 1.0e-3)
    assert ledger.dose_30d_sv(20 * EARTH_DAY_S) == pytest.approx(2.0e-3)
    assert ledger.dose_30d_sv(31 * EARTH_DAY_S) == pytest.approx(1.0e-3)
    assert ledger.dose_30d_sv(41 * EARTH_DAY_S) == pytest.approx(0.0)
    assert ledger.dose_1y_sv(41 * EARTH_DAY_S) == pytest.approx(2.0e-3)
    assert ledger.dose_1y_sv(EARTH_YEAR_S + EARTH_DAY_S) == pytest.approx(1.0e-3)
    assert ledger.dose_1y_sv(EARTH_YEAR_S + 11 * EARTH_DAY_S) == pytest.approx(0.0)
    assert ledger.career_sv == pytest.approx(2.0e-3)


def test_a6_ledger_budget_and_limits(cfg: SimConfig) -> None:
    ledger = DoseLedger(cfg.radiation, career_dose_sv=0.55)
    assert ledger.remaining_budget_sv(0.0) == pytest.approx(0.05)
    assert ledger.violations(0.0, 0.01) == []
    assert ledger.violations(0.0, 0.06) == ["career"]
    fresh = DoseLedger(cfg.radiation)
    assert fresh.remaining_budget_sv(0.0) == pytest.approx(0.5)      # 30-day BFO limit binds
    assert fresh.violations(0.0, 0.55) == ["bfo_30d"]
    assert radiation.eva_dose_budget_sv(cfg.radiation, fresh, 0.0) == cfg.radiation.per_eva_limit_sv
    assert radiation.eva_dose_budget_sv(cfg.radiation, ledger, 0.0) == cfg.radiation.per_eva_limit_sv


def test_a6_ledger_rejects_bad_input(cfg: SimConfig) -> None:
    ledger = DoseLedger(cfg.radiation)
    ledger.add(100.0, 1e-6)
    with pytest.raises(ValidationError):
        ledger.add(50.0, 1e-6)
    with pytest.raises(ValidationError):
        ledger.add(200.0, -1e-6)
    with pytest.raises(ValidationError):
        DoseLedger(cfg.radiation, career_dose_sv=-1.0)


def test_a6_shelter_rule_and_uv(cfg: SimConfig) -> None:
    assert radiation.spe_shelter_reachable(600.0, cfg.radiation)
    assert not radiation.spe_shelter_reachable(1200.0, cfg.radiation)
    assert radiation.uv_exposure_increment(reference_day_sample(), 100.0, cfg.radiation) == 50.0
    assert radiation.uv_exposure_increment(reference_night_sample(), 100.0, cfg.radiation) == 0.0


# ---- A7 -------------------------------------------------------------------
def test_a7_fatigue_growth_dead_zone_and_recovery(cfg: SimConfig) -> None:
    grown = fatigue.update_fatigue(0.0, 450.0, 450.0, HOUR_S, cfg.fatigue)
    assert grown == pytest.approx(1.9 * 0.25)
    assert fatigue.update_fatigue(0.3, 200.0, 450.0, HOUR_S, cfg.fatigue) == 0.3
    assert fatigue.update_fatigue(0.6, 110.0, 450.0, HOUR_S, cfg.fatigue) == pytest.approx(
        0.6 * math.exp(-1.0))
    assert fatigue.update_fatigue(0.95, 900.0, 450.0, 10 * HOUR_S, cfg.fatigue) == 1.0
    with pytest.raises(ValidationError):
        fatigue.update_fatigue(0.0, 300.0, 0.0, 60.0, cfg.fatigue)


def test_a7_time_to_fatigue(cfg: SimConfig) -> None:
    assert fatigue.time_to_fatigue_s(0.0, 0.7, 450.0, 450.0, cfg.fatigue) == pytest.approx(
        0.7 / (1.9 * 0.25) * HOUR_S)
    assert fatigue.time_to_fatigue_s(0.0, 0.7, 200.0, 450.0, cfg.fatigue) == math.inf
    assert fatigue.time_to_fatigue_s(0.8, 0.7, 450.0, 450.0, cfg.fatigue) == 0.0


def test_a7_hand_fatigue(cfg: SimConfig, suit: SuitProfile) -> None:
    worked = fatigue.update_hand_fatigue(0.0, 1.0, suit.pressure_pa, 3 * HOUR_S, cfg.fatigue)
    assert worked == pytest.approx(0.8)
    assert fatigue.tool_work_allowed(worked, cfg.fatigue)
    assert not fatigue.tool_work_allowed(0.81, cfg.fatigue)
    stiffer = fatigue.update_hand_fatigue(0.0, 1.0, 2 * suit.pressure_pa, HOUR_S, cfg.fatigue)
    assert stiffer == pytest.approx(2 * 0.8 / 3)
    assert fatigue.update_hand_fatigue(0.5, 0.0, suit.pressure_pa, 2 * HOUR_S,
                                       cfg.fatigue) == pytest.approx(0.5 * math.exp(-1.0))


def test_a7_daily_limits(cfg: SimConfig) -> None:
    def check(record: DailyEvaRecord, planned: float = HOUR_S) -> list[str]:
        return fatigue.daily_eva_violations(record, planned, cfg.fatigue, cfg.capacity)

    assert check(DailyEvaRecord()) == []
    assert len(check(DailyEvaRecord(evas_this_sol=1))) == 1
    assert len(check(DailyEvaRecord(suit_time_this_sol_s=7.5 * HOUR_S))) == 1
    assert len(check(DailyEvaRecord(consecutive_eva_sols=2))) == 1
    assert len(check(DailyEvaRecord(sleep_debt_s=7 * HOUR_S))) == 1


# ---- A8 -------------------------------------------------------------------
def test_a8_sweat_and_hydration(cfg: SimConfig, astronaut: AstronautProfile) -> None:
    assert nutrition.sweat_rate_kg_s(150.0, astronaut, cfg.nutrition) * HOUR_S == pytest.approx(0.1)
    assert nutrition.sweat_rate_kg_s(550.0, astronaut, cfg.nutrition) * HOUR_S == pytest.approx(0.7)
    assert nutrition.sweat_rate_kg_s(2000.0, astronaut, cfg.nutrition) * HOUR_S == pytest.approx(0.7)
    deficit, drunk = nutrition.hydration_step(0.0, 0.95, 350.0, 600.0, astronaut, cfg.nutrition)
    assert drunk == 0.0 and 0.0 < deficit < cfg.nutrition.drink_trigger_kg
    deficit, drunk = nutrition.hydration_step(0.24, 0.95, 350.0, 600.0, astronaut, cfg.nutrition)
    assert drunk > 0.24 and deficit == pytest.approx(0.0)
    deficit, drunk = nutrition.hydration_step(2.0, 0.1, 350.0, 600.0, astronaut, cfg.nutrition)
    assert drunk == pytest.approx(0.1) and deficit > 1.9


def test_a8_power_factors(cfg: SimConfig) -> None:
    assert nutrition.dehydration_factor(2.0, 0.0, 80.0, cfg.nutrition) == 0.85
    assert nutrition.dehydration_factor(2.0, 0.5, 80.0, cfg.nutrition) == 1.0
    assert nutrition.dehydration_factor(1.0, 0.0, 80.0, cfg.nutrition) == 1.0
    assert nutrition.fasting_factor(5 * HOUR_S, cfg.nutrition) == 1.0
    assert nutrition.fasting_factor(7 * HOUR_S, cfg.nutrition) == 0.9


# ---- A9 -------------------------------------------------------------------
def test_a9_risk_grows_with_every_hazard(cfg: SimConfig) -> None:
    def risk(**kw: float) -> float:
        args = dict(length_m=10.0, slope_rad=0.0, grade=0.0, rock_hazard=0.0, sensorimotor=1.0,
                    fatigue=0.0, visibility=1.0, speed_ratio=1.0, bone_fracture_mult=1.0)
        args.update(kw)
        return injury.fall_risk(cfg=cfg.injury, **args).p_fall

    base = risk()
    assert 0.0 < base < 1.0
    for change in (dict(slope_rad=0.3), dict(rock_hazard=0.8), dict(sensorimotor=0.5),
                   dict(fatigue=0.8), dict(visibility=0.4), dict(grade=-0.3),
                   dict(speed_ratio=1.5), dict(length_m=100.0)):
        assert risk(**change) > base
    assert risk(length_m=20.0) == pytest.approx(1 - (1 - base) ** 2)


def test_a9_consequences(cfg: SimConfig) -> None:
    soft = injury.fall_risk(10, 0, 0, 0.0, 1, 0, 1, 1, 1.0, cfg.injury)
    rocky = injury.fall_risk(10, 0, 0, 1.0, 1, 0, 1, 1, 1.5, cfg.injury)
    assert soft.p_breach_given_fall == 0.0
    assert rocky.p_breach_given_fall == pytest.approx(cfg.injury.breach_given_fall)
    assert rocky.p_injury_given_fall == pytest.approx(1.5 * cfg.injury.injury_given_fall)
    assert rocky.expected_harm > soft.expected_harm > 0.0


# ---- A10 ------------------------------------------------------------------
def test_a10_classification(cfg: SimConfig, suit: SuitProfile) -> None:
    assert emergency.emergency_time_limit_s(suit, cfg.emergency) == pytest.approx(1500.0)

    def flag(t: float, cooling: float = math.inf) -> SafetyFlag:
        return emergency.classify_cell(t, 1.25 * t, cooling, suit, cfg.emergency)

    assert flag(1000.0) is SafetyFlag.SAFE_EMERGENCY
    assert flag(1600.0) is SafetyFlag.MARGINAL
    assert flag(2000.0) is SafetyFlag.UNSAFE
    assert flag(math.inf) is SafetyFlag.UNSAFE
    assert flag(1000.0, cooling=600.0) is SafetyFlag.MARGINAL
    times = np.array([[1000.0, 1600.0], [2000.0, np.inf]])
    flags = emergency.emergency_flag_map(times, 1.25 * times, math.inf, suit, cfg.emergency)
    assert flags.tolist() == [[2, 1], [0, 0]]


def test_a10_cooling_failure_tolerance() -> None:
    assert emergency.cooling_failure_tolerance_s(380.0, 110.0, -150.0, 300e3) == math.inf
    assert emergency.cooling_failure_tolerance_s(380.0, 110.0, -50.0, 300e3) == pytest.approx(
        300e3 / 330.0)
    assert emergency.cooling_failure_tolerance_s(380.0, 110.0, -50.0, 300e3, 300e3) == 0.0


# ---- A12 ------------------------------------------------------------------
def test_a12_dust(cfg: SimConfig) -> None:
    assert dust.soil_fine_fraction(1.0) == 0.0 and dust.soil_fine_fraction(2.1) == 1.0
    load = dust.dust_increment_kg(1000.0, 0.0, cfg.dust)
    assert load == pytest.approx(1.0e-3)
    assert dust.dust_increment_kg(1000.0, 1.0, cfg.dust) == pytest.approx(2 * load)
    assert dust.cleaning_time_s(0.0, cfg.dust) == cfg.dust.base_cleaning_time_s
    assert dust.cleaning_time_s(load, cfg.dust) == pytest.approx(300.0 + 600.0)
    assert dust.perchlorate_flag(load, cfg.dust)
    assert not dust.perchlorate_flag(1.0e-4, cfg.dust)


# ---- validation -----------------------------------------------------------
def test_configuration_is_validated() -> None:
    with pytest.raises(ConfigurationError):
        SimConfig(routing=RoutingConfig(reserve_fraction=1.5))
    with pytest.raises(ConfigurationError):
        SimConfig(routing=RoutingConfig(max_labels_per_cell=0))


def test_profiles_are_validated() -> None:
    with pytest.raises(ProfileValidationError):
        AstronautProfile(body_mass_kg=-5.0)
    with pytest.raises(ProfileValidationError):
        AstronautProfile(hr_rest_hz=4.0)
    with pytest.raises(ProfileValidationError):
        SuitProfile(stiffness_factor=0.5)
    with pytest.raises(ProfileValidationError):
        SuitProfile(o2_primary_kg=0.0)


def test_environment_sample_is_validated() -> None:
    reference_day_sample().validate()
    with pytest.raises(EnvironmentValidationError):
        EnvSample(t_ground_k=-10.0).validate()
    with pytest.raises(EnvironmentValidationError):
        EnvSample(solar_direct_w_m2=float("nan")).validate()
    with pytest.raises(EnvironmentValidationError):
        EnvSample(sun_elevation_rad=3.0).validate()


def test_terrain_is_validated() -> None:
    elevation = np.zeros((4, 5))
    terrain = GridTerrain(elevation=elevation, cell_size_m=10.0)
    assert terrain.shape == (4, 5)
    assert len(list(terrain.neighbors((0, 0)))) == 3
    assert len(list(terrain.neighbors((2, 2)))) == 8
    distances = sorted(d for _, d in terrain.neighbors((2, 2)))
    assert distances[0] == 10.0 and distances[-1] == pytest.approx(10.0 * math.sqrt(2))
    with pytest.raises(TerrainValidationError):
        GridTerrain(elevation=elevation, cell_size_m=0.0)
    with pytest.raises(TerrainValidationError):
        GridTerrain(elevation=elevation, cell_size_m=10.0, walk_eta=np.full((4, 5), 0.5))
    with pytest.raises(TerrainValidationError):
        GridTerrain(elevation=elevation, cell_size_m=10.0, rock_hazard=np.zeros((2, 2)))
    with pytest.raises(TerrainValidationError):
        GridTerrain(elevation=np.full((4, 5), np.nan), cell_size_m=10.0)
    with pytest.raises(TerrainValidationError):
        list(terrain.neighbors((9, 9)))
