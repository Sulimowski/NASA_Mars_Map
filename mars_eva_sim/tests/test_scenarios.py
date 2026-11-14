"""Integration tests: the ten scenarios of the specification (section 16), the mission
planner, the route replay, the map layers and the adapter to the project's site data."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from conftest import START_TIME_S, make_search, walk_straight
from src.core.constants import HOUR_S, SOL_S
from src.core.exceptions import InfeasibleStateError, TerrainValidationError, ValidationError
from src.core.schemas import CoolingMode, PruneRule, SafetyFlag, SuitProfile, TaskType
from src.physiology.fatigue import DailyEvaRecord
from src.physiology.radiation import DoseLedger
from src.providers.synthetic import (DiurnalEnvironment, SpeEnvironment, flat_terrain,
                                     forbidden_zone_terrain, hill_terrain, slope_terrain,
                                     wall_terrain)
from src.reporting.map_layers import build_map_layers, plan_summary, route_profile
from src.routing.mission_planner import (MissionStatus, plan_round_trip, record_dose,
                                         simulate_path)
from src.routing.reachability import compute_reachability


def o2_per_m(evaluator, start=(1, 0), steps=20, step=(0, 1)) -> float:
    last = walk_straight(evaluator, start, steps, step)
    return (evaluator.suit.o2_primary_kg - last.o2_kg) / last.distance_m


# ---- scenario 1 is in test_calibration (1 km, 20 min, 30-35 g, 380 W) -------

# ---- scenario 2 -----------------------------------------------------------
def test_s2_ten_percent_climb_slows_to_about_two_kmh(make_evaluator) -> None:
    evaluator = make_evaluator(slope_terrain(3, 30, 10.0, grade=0.10))
    state = evaluator.initial_label((1, 0), START_TIME_S)
    tr = evaluator.walk_edge((1, 0), (1, 1), 10.0, state)
    assert tr is not None
    assert 1.6 <= tr.speed_m_s * 3.6 <= 2.2
    assert tr.metabolic_w <= evaluator.p_sustain_w + 1e-6
    assert tr.grade == pytest.approx(0.10)


# ---- scenario 3 -----------------------------------------------------------
def test_s3_gentle_descent_is_cheapest_and_steep_descent_costs_more(make_evaluator) -> None:
    flat = o2_per_m(make_evaluator(flat_terrain(3, 30, 10.0)))
    gentle = o2_per_m(make_evaluator(slope_terrain(3, 30, 10.0, grade=-0.10)))
    steep = o2_per_m(make_evaluator(slope_terrain(3, 30, 10.0, grade=-0.25)))
    assert gentle < flat
    assert steep > gentle


# ---- scenario 4 -----------------------------------------------------------
def test_s4_route_goes_around_a_thirty_degree_wall(make_evaluator) -> None:
    gap = range(14, 21)
    terrain = wall_terrain(21, 31, 10.0, wall_col=15, gap_rows=gap, shelters=[(3, 2)])
    evaluator = make_evaluator(terrain)
    astar, _, _ = make_search(evaluator, (3, 2))
    result = astar.search(evaluator.initial_label((3, 2), START_TIME_S), (3, 28))
    assert result.found
    assert result.path[0] == (3, 2) and result.path[-1] == (3, 28)
    for row, col in result.path:
        assert evaluator.is_walkable((row, col))
        if 13 <= col <= 17:
            assert row in gap, f"route crossed the wall at {(row, col)}"
    straight = 26 * 10.0                               # the detour through the pass is ~359 m
    assert result.goal_label is not None
    assert result.goal_label.distance_m == pytest.approx(359.4, abs=15.0)
    assert result.goal_label.distance_m > 1.3 * straight
    assert not evaluator.is_walkable((3, 14))          # the 30 degree face itself


def test_s4_closed_wall_reports_terrain_as_the_cause(make_evaluator) -> None:
    evaluator = make_evaluator(wall_terrain(15, 31, 10.0, wall_col=15, shelters=[(3, 2)]))
    astar, _, _ = make_search(evaluator, (3, 2))
    result = astar.search(evaluator.initial_label((3, 2), START_TIME_S), (3, 28))
    assert not result.found
    assert result.bottleneck is None
    assert "terrain" in result.message


# ---- scenario 5 -----------------------------------------------------------
def test_s5_route_never_enters_a_forbidden_zone(make_evaluator) -> None:
    terrain = forbidden_zone_terrain(15, 31, 10.0, zone=(0, 11, 12, 18), shelters=[(5, 2)])
    evaluator = make_evaluator(terrain)
    astar, _, _ = make_search(evaluator, (5, 2))
    result = astar.search(evaluator.initial_label((5, 2), START_TIME_S), (5, 28))
    assert result.found
    assert not any(terrain.forbidden[cell] for cell in result.path)
    assert max(row for row, _ in result.path) >= 11
    blocked = astar.search(evaluator.initial_label((5, 2), START_TIME_S), (5, 15))
    assert not blocked.found


# ---- scenario 6 -----------------------------------------------------------
def test_s6_goal_beyond_the_return_radius_names_oxygen(make_evaluator) -> None:
    evaluator = make_evaluator(flat_terrain(3, 90, 100.0, shelters=[(1, 0)]))
    astar, _, _ = make_search(evaluator, (1, 0))
    result = astar.search(evaluator.initial_label((1, 0), START_TIME_S), (1, 85))
    assert not result.found
    assert result.bottleneck is PruneRule.RETURN_O2
    assert "O₂" in result.message and "return" in result.message and "km" in result.message
    reached_km = result.diagnostics.prune_max_distance_m[PruneRule.RETURN_O2] / 1000.0
    assert 4.5 <= reached_km <= 7.5
    assert result.diagnostics.prune_counts[PruneRule.RETURN_O2] > 0
    near = astar.search(evaluator.initial_label((1, 0), START_TIME_S), (1, 30))
    assert near.found


# ---- scenario 7 -----------------------------------------------------------
def test_s7_first_sols_have_a_smaller_radius(make_evaluator) -> None:
    terrain = flat_terrain(3, 90, 100.0, shelters=[(1, 0)])
    early = compute_reachability(make_evaluator(terrain, sols=1.5), (1, 0),
                                 start_time_s=START_TIME_S)
    late = compute_reachability(make_evaluator(terrain, sols=60.0), (1, 0),
                                start_time_s=START_TIME_S)
    assert 0.0 < early.max_round_trip_distance_m < late.max_round_trip_distance_m
    assert early.round_trip_ok.sum() < late.round_trip_ok.sum()
    assert late.max_round_trip_distance_m == pytest.approx(5_900.0, rel=0.15)


def test_s7_no_eva_on_the_first_sol(make_evaluator) -> None:
    evaluator = make_evaluator(flat_terrain(5, 20, 25.0, shelters=[(2, 1)]), sols=0.5)
    plan = plan_round_trip(evaluator, (2, 1), (2, 15), start_time_s=START_TIME_S)
    assert plan.status is MissionStatus.FAIL
    assert any("first sol" in reason for reason in plan.reasons)


# ---- scenario 8 -----------------------------------------------------------
def test_s8_night_is_slower_and_drains_the_battery(make_evaluator, night_env) -> None:
    terrain = flat_terrain(3, 30, 10.0)
    day = walk_straight(make_evaluator(terrain), (1, 0), 20)
    night = walk_straight(make_evaluator(terrain, env=night_env), (1, 0), 20)
    assert day.transition is not None and night.transition is not None
    assert night.transition.speed_m_s < day.transition.speed_m_s
    assert night.t_s > day.t_s
    battery = make_evaluator(terrain).suit.battery_j
    assert (battery - night.battery_j) > 2.0 * (battery - day.battery_j)
    assert night.transition.heater_w > day.transition.heater_w == 0.0
    assert night.transition.p_fall > day.transition.p_fall


def test_s8_diurnal_environment_cycles(make_evaluator) -> None:
    env = DiurnalEnvironment()
    noon, midnight = env.get_env((0, 0), 0.5 * SOL_S), env.get_env((0, 0), 0.0)
    assert noon.sun_elevation_rad == pytest.approx(math.pi / 2)
    assert midnight.sun_elevation_rad == pytest.approx(-math.pi / 2)
    assert noon.t_ground_k > midnight.t_ground_k
    assert noon.solar_direct_w_m2 > 0.0 == midnight.solar_direct_w_m2
    for t in np.linspace(0.0, SOL_S, 49):
        env.get_env((0, 0), float(t)).validate()
    evaluator = make_evaluator(flat_terrain(3, 30, 10.0), env=env)
    assert (walk_straight(evaluator, (1, 0), 10, t_s=0.02 * SOL_S).t_s - 0.02 * SOL_S
            > walk_straight(evaluator, (1, 0), 10, t_s=0.5 * SOL_S).t_s - 0.5 * SOL_S)


# ---- scenario 9 -----------------------------------------------------------
def test_s9_particle_event_keeps_the_crew_near_a_shelter(make_evaluator, day_env) -> None:
    terrain = flat_terrain(3, 40, 100.0, shelters=[(1, 0)])
    quiet = make_evaluator(terrain)
    stormy = make_evaluator(terrain, env=SpeEnvironment(day_env))
    warning = quiet.config.radiation.spe_warning_time_s

    astar_q, _, _ = make_search(quiet, (1, 0))
    assert astar_q.search(quiet.initial_label((1, 0), START_TIME_S), (1, 30)).found

    astar_s, _, maps = make_search(stormy, (1, 0))
    result = astar_s.search(stormy.initial_label((1, 0), START_TIME_S), (1, 30))
    assert not result.found and result.bottleneck is PruneRule.SPE_SHELTER
    best, _ = astar_s.explore(stormy.initial_label((1, 0), START_TIME_S))
    assert best and all(maps.shelter_time_s[cell] <= warning for cell in best)
    assert max(col for _, col in best) < 10

    forecast, _, _ = make_search(quiet, (1, 0), spe_watch=True)
    assert not forecast.search(quiet.initial_label((1, 0), START_TIME_S), (1, 30)).found


def test_s9_second_shelter_opens_its_own_island(make_evaluator, day_env) -> None:
    terrain = flat_terrain(3, 40, 100.0, shelters=[(1, 0), (1, 20)])
    evaluator = make_evaluator(terrain, env=SpeEnvironment(day_env))
    _, _, maps = make_search(evaluator, (1, 0))
    warning = evaluator.config.radiation.spe_warning_time_s
    assert maps.shelter_time_s[1, 20] == 0.0
    assert maps.shelter_time_s[1, 10] > warning
    assert maps.shelter_time_s[1, 22] <= warning < maps.return_time_s[1, 22]


def test_s9_event_dose_is_four_times_the_quiet_dose(make_evaluator, day_env) -> None:
    terrain = flat_terrain(3, 30, 10.0)
    quiet = walk_straight(make_evaluator(terrain), (1, 0), 20)
    event = walk_straight(make_evaluator(terrain, env=SpeEnvironment(day_env)), (1, 0), 20)
    assert event.dose_sv == pytest.approx(4.0 * quiet.dose_sv)


# ---- scenario 10 ----------------------------------------------------------
def test_s10_monotonic_in_terrain_factor_and_grade(make_evaluator) -> None:
    by_eta = [o2_per_m(make_evaluator(flat_terrain(3, 30, 10.0, eta=eta)))
              for eta in (1.0, 1.4, 1.8, 2.1)]
    assert by_eta == sorted(by_eta) and by_eta[0] < by_eta[-1]
    by_grade = [o2_per_m(make_evaluator(slope_terrain(3, 30, 10.0, grade=g)))
                for g in (0.0, 0.05, 0.10, 0.15)]
    assert by_grade == sorted(by_grade) and by_grade[0] < by_grade[-1]


def test_s10_rocks_slow_the_walk_and_raise_the_risk(make_evaluator) -> None:
    clear = walk_straight(make_evaluator(flat_terrain(3, 30, 10.0)), (1, 0), 20)
    rocky = walk_straight(make_evaluator(flat_terrain(3, 30, 10.0, rock_hazard=0.8)), (1, 0), 20)
    assert rocky.t_s > clear.t_s and rocky.risk > clear.risk


# ---- B4: mission plan -----------------------------------------------------
@pytest.fixture
def mission(make_evaluator):
    evaluator = make_evaluator(flat_terrain(21, 41, 25.0, shelters=[(10, 2)]))
    plan = plan_round_trip(evaluator, (10, 2), (10, 38), start_time_s=START_TIME_S,
                           task=TaskType.SAMPLING, task_duration_s=1_200.0)
    return evaluator, plan


def test_b4_round_trip_on_flat_ground(mission) -> None:
    evaluator, plan = mission
    suit = evaluator.suit
    assert plan.status is MissionStatus.OK, (plan.reasons, plan.warnings)
    assert plan.final_label is not None and plan.final_label.cell == (10, 2)
    assert [leg.name for leg in plan.legs] == ["outbound", "task", "return"]
    assert plan.legs[0].distance_m == pytest.approx(900.0)
    assert plan.total_distance_m == pytest.approx(1_800.0)
    assert plan.task_completed_s == pytest.approx(1_200.0)
    assert plan.eva_duration_s == pytest.approx(2 * 1_080.0 + 1_200.0, rel=0.05)
    used_g = 1000.0 * (suit.o2_primary_kg - plan.final_label.o2_kg)
    assert 75.0 <= used_g <= 105.0
    assert plan.prebreathe_s == 0.0
    assert plan.cleaning_s > evaluator.config.dust.base_cleaning_time_s
    assert plan.safety_counts[SafetyFlag.UNSAFE.name] == 0
    assert plan.limiting_resource in {"o2", "battery", "eva_time", "co2_scrubber", "cooling_water"}
    assert math.isfinite(plan.limiting_time_left_s)


def test_b4_report_and_profile(mission) -> None:
    _, plan = mission
    summary = plan_summary(plan)
    assert summary["status"] == "OK"
    assert summary["distance_km"] == pytest.approx(1.8)
    assert 0.0 < summary["remaining"]["o2_pct"] < 100.0
    assert len(summary["legs"]) == 3 and summary["legs"][1]["length_km"] == 0.0
    assert summary["legs"][0]["time_min"] == pytest.approx(18.0, rel=0.05)
    rows = route_profile(plan.final_label)
    assert rows[0].action == "start" and rows[-1].distance_km == pytest.approx(1.8)
    times = [row.elapsed_min for row in rows]
    assert times == sorted(times)
    walking = [row for row in rows if row.action == "walk"]
    assert all(row.speed_kmh == pytest.approx(3.0) for row in walking)
    assert all(60.0 < row.heart_rate_bpm < 190.0 for row in walking)
    assert {"task"} <= {row.action for row in rows}
    assert isinstance(rows[5].to_dict()["o2_g"], float)


def test_b4_dose_ledger_and_daily_limits(make_evaluator, mission) -> None:
    evaluator, plan = mission
    ledger = DoseLedger(evaluator.config.radiation)
    record_dose(plan, ledger)
    assert ledger.career_sv == pytest.approx(plan.final_label.dose_sv) and ledger.career_sv > 0.0

    spent = DoseLedger(evaluator.config.radiation, career_dose_sv=0.6)
    no_dose = plan_round_trip(evaluator, (10, 2), (10, 38), start_time_s=START_TIME_S, ledger=spent)
    assert no_dose.status is MissionStatus.FAIL and "dose" in no_dose.reasons[0]

    second = plan_round_trip(evaluator, (10, 2), (10, 38), start_time_s=START_TIME_S,
                             daily=DailyEvaRecord(evas_this_sol=1))
    assert second.status is MissionStatus.FAIL


def test_b4_unreachable_target_fails_with_a_reason(make_evaluator) -> None:
    evaluator = make_evaluator(flat_terrain(3, 90, 100.0, shelters=[(1, 0)]))
    plan = plan_round_trip(evaluator, (1, 0), (1, 85), start_time_s=START_TIME_S)
    assert plan.status is MissionStatus.FAIL and not plan.feasible
    assert "outbound leg" in plan.reasons[0] and "O₂" in plan.reasons[0]


def test_b4_long_task_is_cut_short_and_flagged(make_evaluator) -> None:
    evaluator = make_evaluator(flat_terrain(9, 30, 25.0, shelters=[(4, 1)]))
    plan = plan_round_trip(evaluator, (4, 1), (4, 20), start_time_s=START_TIME_S,
                           task=TaskType.HAMMERING, task_duration_s=4.0 * HOUR_S)
    assert plan.status is MissionStatus.MARGINAL
    assert 0.0 < plan.task_completed_s < 4.0 * HOUR_S
    assert any("cut short" in warning for warning in plan.warnings)
    assert plan.final_label is not None and plan.final_label.cell == (4, 1)


def test_b4_invalid_cells_raise(make_evaluator) -> None:
    terrain = forbidden_zone_terrain(9, 9, 10.0, zone=(0, 2, 0, 2), shelters=[(5, 5)])
    evaluator = make_evaluator(terrain)
    with pytest.raises(InfeasibleStateError):
        plan_round_trip(evaluator, (0, 0), (5, 5), start_time_s=START_TIME_S)
    with pytest.raises(TerrainValidationError):
        plan_round_trip(evaluator, (5, 5), (50, 50), start_time_s=START_TIME_S)


def test_b4_radiator_suit_spends_no_water(make_evaluator, suit: SuitProfile) -> None:
    radiator = replace(suit, cooling_mode=CoolingMode.RADIATOR_STORAGE)
    evaluator = make_evaluator(flat_terrain(9, 30, 25.0, shelters=[(4, 1)]),
                               suit_override=radiator)
    plan = plan_round_trip(evaluator, (4, 1), (4, 20), start_time_s=START_TIME_S)
    assert plan.feasible and plan.final_label is not None
    assert plan.final_label.cooling_water_kg == radiator.cooling_water_kg
    membrane = plan_round_trip(make_evaluator(flat_terrain(9, 30, 25.0, shelters=[(4, 1)])),
                               (4, 1), (4, 20), start_time_s=START_TIME_S)
    assert membrane.final_label is not None
    assert membrane.final_label.cooling_water_kg < suit.cooling_water_kg


def test_b4_hill_route_reports_elevation(make_evaluator) -> None:
    terrain = hill_terrain(21, 41, 25.0, height_m=30.0, sigma_m=120.0, shelters=[(10, 2)])
    evaluator = make_evaluator(terrain)
    plan = plan_round_trip(evaluator, (10, 2), (10, 20), start_time_s=START_TIME_S,
                           task_duration_s=600.0)
    assert plan.feasible
    outbound, back = plan.legs[0], plan.legs[2]
    assert outbound.elevation_gain_m > 20.0 and back.elevation_loss_m > 20.0
    assert outbound.o2_kg > back.o2_kg                    # climbing costs more than descending
    assert outbound.max_slope_rad <= evaluator.slope_up_limit_rad


# ---- route replay ---------------------------------------------------------
def test_replay_reproduces_the_search(make_evaluator) -> None:
    evaluator = make_evaluator(flat_terrain(11, 30, 25.0, shelters=[(5, 1)]))
    astar, checker, _ = make_search(evaluator, (5, 1))
    result = astar.search(evaluator.initial_label((5, 1), START_TIME_S), (8, 25))
    assert result.found and result.goal_label is not None
    replay = simulate_path(evaluator, checker, result.path,
                           evaluator.initial_label((5, 1), START_TIME_S),
                           stops={len(result.path) - 1: (TaskType.OBSERVATION, 300.0)})
    assert replay.completed and replay.message == "ok"
    assert replay.final_label.distance_m == pytest.approx(result.goal_label.distance_m)
    assert replay.final_label.t_s == pytest.approx(result.goal_label.t_s + 300.0)


def test_replay_stops_at_forbidden_ground_and_rejects_bad_paths(make_evaluator) -> None:
    terrain = forbidden_zone_terrain(5, 20, 25.0, zone=(0, 5, 10, 12), shelters=[(2, 0)])
    evaluator = make_evaluator(terrain)
    _, checker, _ = make_search(evaluator, (2, 0))
    start = evaluator.initial_label((2, 0), START_TIME_S)
    replay = simulate_path(evaluator, checker, [(2, c) for c in range(15)], start)
    assert not replay.completed and replay.failed_index == 10 and replay.rule is None
    assert replay.final_label.cell == (2, 9)
    with pytest.raises(ValidationError):
        simulate_path(evaluator, checker, [(2, 0), (2, 2)], start)
    with pytest.raises(ValidationError):
        simulate_path(evaluator, checker, [(2, 1), (2, 2)], start)
    with pytest.raises(ValidationError):
        simulate_path(evaluator, checker, [], start)


def test_replay_reports_the_broken_constraint(make_evaluator) -> None:
    evaluator = make_evaluator(flat_terrain(3, 90, 100.0, shelters=[(1, 0)]))
    _, checker, _ = make_search(evaluator, (1, 0))
    replay = simulate_path(evaluator, checker, [(1, c) for c in range(86)],
                           evaluator.initial_label((1, 0), START_TIME_S))
    assert not replay.completed and replay.rule is PruneRule.RETURN_O2
    assert "return_o2" in replay.message


# ---- B5 and map layers ----------------------------------------------------
def test_b5_reachability_and_layers(make_evaluator) -> None:
    evaluator = make_evaluator(flat_terrain(41, 41, 100.0, shelters=[(20, 20)]))
    _, _, maps = make_search(evaluator, (20, 20))
    reach = compute_reachability(evaluator, (20, 20), start_time_s=START_TIME_S, maps=maps)
    assert reach.reachable.all() and reach.round_trip_ok.all()     # 2.8 km corner < 5.9 km radius
    assert reach.arrival_time_s[20, 20] == 0.0
    assert reach.arrival_time_s[20, 30] == pytest.approx(1_200.0, rel=0.03)
    quarter, half, hour, two = (reach.isochrones[t] for t in (900.0, 1800.0, 3600.0, 7200.0))
    assert 0 < quarter.sum() < half.sum() < hour.sum() <= two.sum()
    assert not (quarter & ~half).any()

    with_work = compute_reachability(evaluator, (20, 20), start_time_s=START_TIME_S, maps=maps,
                                     task=TaskType.SAMPLING, task_duration_s=3.0 * HOUR_S)
    assert 0 < with_work.round_trip_ok.sum() < reach.round_trip_ok.sum()
    assert with_work.round_trip_ok[20, 20] and not with_work.round_trip_ok[0, 0]
    assert with_work.point_of_no_return.any()
    assert not (with_work.point_of_no_return & ~with_work.round_trip_ok).any()

    layers = build_map_layers(evaluator, maps, START_TIME_S, reach)
    assert all(layer.shape == (41, 41) for layer in layers.values())
    assert layers["o2_g_per_km"][20, 20] == pytest.approx(33.7, abs=1.5)
    assert layers["time_min_per_km"][20, 20] == pytest.approx(20.0, rel=0.03)
    assert layers["return_time_min"][20, 20] == 0.0
    assert set(np.unique(layers["emergency_safety"])) <= {0, 1, 2}
    assert layers["emergency_safety"][20, 20] == int(SafetyFlag.SAFE_EMERGENCY)
    assert layers["emergency_safety"][0, 0] == int(SafetyFlag.UNSAFE)
    assert layers["isochrone_min"][20, 21] == 15.0


def test_b2_return_maps_respect_slope_direction(make_evaluator) -> None:
    """Walking home downhill is cheaper than walking home uphill."""
    evaluator = make_evaluator(slope_terrain(3, 41, 25.0, grade=0.08, shelters=[(1, 20)]))
    _, _, maps = make_search(evaluator, (1, 20))
    from_above, from_below = maps.return_o2_kg[1, 30], maps.return_o2_kg[1, 10]
    assert from_above < from_below
    assert maps.return_time_s[1, 30] < maps.return_time_s[1, 10]
    assert maps.emergency_time_s[1, 30] < maps.shelter_time_s[1, 30]


# ---- adapter to the project's site data -----------------------------------
def test_site_adapter_plans_on_the_stub_site(astronaut, suit) -> None:
    """End to end on the data team's stub: Site + Environment -> providers -> mission plan."""
    mars_map = pytest.importorskip("mars_map")
    from src.core.config import DEFAULT_CONFIG
    from src.providers.site_adapter import SiteEnvironment, terrain_from_site
    from src.routing.transition import EdgeEvaluator

    site = mars_map.load_stub()
    factor = 10
    terrain = terrain_from_site(site, downsample=factor)
    assert terrain.shape == (100, 100) and terrain.cell_size_m == pytest.approx(20.0)
    assert terrain.forbidden.any() and not terrain.passable_walk.all()
    assert 1.0 <= terrain.walk_eta.min() and terrain.walk_eta.max() <= 2.5

    env = SiteEnvironment(site, mars_map.Environment(site), sol=0, start_ltst_h=10.0,
                          downsample=factor)
    base = terrain.to_cell(*site.start)
    sample = env.get_env(base, 0.0)
    sample.validate()
    assert sample.sun_elevation_rad > 0.0 and 600.0 < sample.pressure_pa < 1200.0
    assert sample.dose_rate_gcr_sv_s == pytest.approx(0.67e-3 / SOL_S, rel=0.25)

    evaluator = EdgeEvaluator(terrain, env, astronaut, suit, DEFAULT_CONFIG,
                              time_on_surface_s=30.0 * SOL_S)
    target_m = next((t.x_m, t.y_m) for t in site.targets if t.name == "Boxwork ridges")
    plan = plan_round_trip(evaluator, base, terrain.to_cell(*target_m), start_time_s=0.0,
                           task=TaskType.SAMPLING, task_duration_s=45.0 * 60.0)
    assert plan.feasible, plan.reasons
    assert plan.final_label is not None and plan.final_label.cell == base
    assert 1.5 <= plan.total_distance_m / 1000.0 <= 3.5
    assert not any(terrain.forbidden[label.cell] for label in plan.labels())
