"""Calibration against the benchmark numbers of the specification (section 3.1).

Reference case: body 80 kg, suit 60 kg, terrain factor 1.2, stiffness 1.25, Mars gravity.

    3 km/h level        ~380 W    ~32 g O2 per km
    3 km/h +10 %        ~610 W    (above the sustainable power: must slow down)
    2 km/h +10 %        ~430 W    ~55 g O2 per km
    3 km/h -10 %        ~260 W    ~22 g O2 per km
"""

from __future__ import annotations

import pytest

from conftest import REFERENCE_ETA, REFERENCE_SPEED_M_S, walk_straight
from src.core.config import SimConfig
from src.core.constants import (G_EARTH_M_S2, O2_DENSITY_STP_KG_M3, SOL_S, kmh_to_ms,
                                ml_kg_min_to_m3_kg_s)
from src.core.schemas import AstronautProfile, Fitness, HabitatAtmosphere, SuitProfile
from src.physiology import capacity, decompression, life_support, metabolism
from src.providers.synthetic import flat_terrain

W, L, STIFFNESS = 80.0, 60.0, 1.25


def power(speed_kmh: float, grade: float, cfg: SimConfig, eta: float = REFERENCE_ETA) -> float:
    return metabolism.walking_power_w(W, L, kmh_to_ms(speed_kmh), grade, eta, STIFFNESS,
                                      cfg.metabolism)


def o2_g_per_km(speed_kmh: float, grade: float, cfg: SimConfig) -> float:
    seconds = 1000.0 / kmh_to_ms(speed_kmh)
    volume = life_support.o2_volume_m3(power(speed_kmh, grade, cfg), seconds)
    return 1000.0 * volume * O2_DENSITY_STP_KG_M3


def test_level_walk_power(cfg: SimConfig) -> None:
    assert power(3.0, 0.0, cfg) == pytest.approx(380.0, rel=0.03)


def test_uphill_power_at_full_speed(cfg: SimConfig) -> None:
    assert power(3.0, 0.10, cfg) == pytest.approx(610.0, rel=0.03)


def test_uphill_power_at_reduced_speed(cfg: SimConfig) -> None:
    assert power(2.0, 0.10, cfg) == pytest.approx(430.0, rel=0.10)


def test_downhill_power(cfg: SimConfig) -> None:
    assert power(3.0, -0.10, cfg) == pytest.approx(260.0, rel=0.03)


def test_level_walk_oxygen_per_km(cfg: SimConfig) -> None:
    assert o2_g_per_km(3.0, 0.0, cfg) == pytest.approx(32.0, rel=0.05)


def test_uphill_oxygen_per_km(cfg: SimConfig) -> None:
    assert o2_g_per_km(2.0, 0.10, cfg) == pytest.approx(55.0, rel=0.10)


def test_downhill_oxygen_per_km(cfg: SimConfig) -> None:
    assert 20.0 <= o2_g_per_km(3.0, -0.10, cfg) <= 25.0


def test_downhill_is_cheapest_near_ten_percent(cfg: SimConfig) -> None:
    flat, gentle, steep = power(3.0, 0.0, cfg), power(3.0, -0.10, cfg), power(3.0, -0.25, cfg)
    assert gentle < flat
    assert steep > gentle


def test_climbing_term_scales_with_gravity(cfg: SimConfig) -> None:
    """The extra power of a climb is proportional to g (potential energy is m g h)."""
    def climb_extra(g: float) -> float:
        args = (W, L, REFERENCE_SPEED_M_S)
        up = metabolism.walking_power_w(*args, 0.10, REFERENCE_ETA, STIFFNESS, cfg.metabolism, g)
        level = metabolism.walking_power_w(*args, 0.0, REFERENCE_ETA, STIFFNESS, cfg.metabolism, g)
        return up - level

    assert climb_extra(G_EARTH_M_S2) / climb_extra(3.711) == pytest.approx(G_EARTH_M_S2 / 3.711)


def test_aerobic_power_of_reference_astronaut(astronaut: AstronautProfile) -> None:
    fit = Fitness(1.0, 1.0, 1.0, 1.0, 1.0)
    assert astronaut.vo2max_earth_m3_kg_s == pytest.approx(ml_kg_min_to_m3_kg_s(40.0))
    assert capacity.aerobic_power_w(astronaut, fit) == pytest.approx(1077.0, rel=0.01)


def test_speed_on_ten_percent_climb_drops_to_about_two_kmh(cfg: SimConfig) -> None:
    c0, c1, c2, floor = metabolism.pandolf_coefficients(W, L, REFERENCE_ETA, 0.10, STIFFNESS,
                                                        cfg.metabolism)
    speed = capacity.select_speed(c0, c1, c2, floor, 0.10, REFERENCE_SPEED_M_S, 451.0,
                                  cfg.capacity)
    assert speed is not None
    assert speed * 3.6 == pytest.approx(1.9, abs=0.15)
    assert metabolism.power_from_coefficients(c0, c1, c2, floor, speed) <= 451.0


def test_one_km_level_walk_through_the_full_model(make_evaluator, suit: SuitProfile) -> None:
    """Scenario 1 through every algorithm: 20 minutes, 30-35 g of oxygen, ~380 W."""
    evaluator = make_evaluator(flat_terrain(3, 102, 10.0, eta=REFERENCE_ETA))
    last = walk_straight(evaluator, (1, 0), 100)
    elapsed = suit.max_eva_s - last.eva_remaining_s
    assert last.distance_m == pytest.approx(1000.0)
    assert elapsed == pytest.approx(1200.0, abs=60.0)
    assert 30.0 <= 1000.0 * (suit.o2_primary_kg - last.o2_kg) <= 35.0
    assert last.energy_j / elapsed == pytest.approx(380.0, rel=0.03)


def test_prebreathe_iss_versus_exploration_atmosphere(cfg: SimConfig, suit: SuitProfile) -> None:
    iss = HabitatAtmosphere(pressure_pa=101_325.0, o2_fraction=0.21)
    exploration = HabitatAtmosphere()
    assert decompression.tissue_ratio(iss, suit.pressure_pa) == pytest.approx(2.70, abs=0.02)
    assert decompression.tissue_ratio(exploration, suit.pressure_pa) == pytest.approx(1.26, abs=0.01)
    hours = decompression.prebreathe_time_s(iss, suit.pressure_pa, cfg.decompression) / 3600.0
    assert 4.0 <= hours <= 7.0
    assert decompression.prebreathe_time_s(exploration, suit.pressure_pa, cfg.decompression) == 0.0


def test_reference_fitness_after_thirty_sols(make_evaluator) -> None:
    evaluator = make_evaluator(flat_terrain(3, 3))
    assert evaluator.time_on_surface_s == pytest.approx(30.0 * SOL_S)
    assert 0.90 <= evaluator.fitness.aerobic <= 0.95
    assert evaluator.p_sustain_w > power(3.0, 0.0, evaluator.config)
