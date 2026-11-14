"""Rovers in the generic graph search: traverses, reach, map layers, the three rovers compared."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.core.config import DEFAULT_CONFIG
from src.core.constants import HOUR_S, SOL_S
from src.core.exceptions import InfeasibleStateError, TerrainValidationError, ValidationError
from src.providers.synthetic import (DiurnalEnvironment, SpeEnvironment, flat_terrain,
                                     patch_terrain)
from src.reporting.rover_layers import (BLOCK_CODES, build_rover_layers, rover_comparison,
                                        rover_plan_summary, rover_route_profile)
from src.routing.agent import TraversalModel
from src.routing.rover_planner import (compare_rovers, compute_rover_reachability,
                                       plan_rover_traverse)
from src.routing.rover_transition import RoverEdgeEvaluator
from src.rovers.profiles import CURIOSITY, PERSEVERANCE, ROSALIND_FRANKLIN, ROVERS
from src.rovers.types import DriveMode, RoverBlock, RoverRule

MORNING_S = 0.35 * SOL_S
MIDNIGHT_S = 0.0
START, GOAL = (4, 2), (4, 27)


def rover(profile, terrain, env=None, year: float = 2031, config=DEFAULT_CONFIG, **kwargs):
    return RoverEdgeEvaluator(terrain, env or DiurnalEnvironment(), profile, config,
                              mission_year=year, **kwargs)


def rock_field(full_width: bool = False):
    rows = (0, 9) if full_width else (0, 7)
    return patch_terrain(9, 30, 10.0, zone=(*rows, 12, 16), obstacle_height_m=0.35)


# ---- construction ---------------------------------------------------------
def test_rover_evaluator_checks_the_mission_year_and_cells() -> None:
    terrain = rock_field()
    with pytest.raises(ValidationError):
        rover(ROSALIND_FRANKLIN, terrain, year=2026)
    curiosity = rover(CURIOSITY, terrain, year=2026)
    assert curiosity.age_years == 14.0
    assert isinstance(curiosity, TraversalModel)            # usable by the generic algorithms
    franklin = rover(ROSALIND_FRANKLIN, terrain)
    assert franklin.cell_block((2, 13)) is RoverBlock.CLEARANCE
    assert curiosity.cell_block((2, 13)) is RoverBlock.NONE
    assert franklin.cell_block((40, 0)) is RoverBlock.OUT_OF_BOUNDS
    with pytest.raises(InfeasibleStateError):
        franklin.initial_label((2, 13), MORNING_S)
    with pytest.raises(TerrainValidationError):
        plan_rover_traverse(franklin, START, (99, 99), start_time_s=MORNING_S)
    with pytest.raises(ValidationError):
        franklin.initial_label(START, MORNING_S, battery_fraction=1.5)


# ---- clearance ------------------------------------------------------------
def test_curiosity_crosses_the_rocks_rosalind_franklin_drives_around() -> None:
    terrain = rock_field()
    straight = plan_rover_traverse(rover(CURIOSITY, terrain), START, GOAL, start_time_s=MORNING_S)
    around = plan_rover_traverse(rover(ROSALIND_FRANKLIN, terrain), START, GOAL,
                                 start_time_s=MORNING_S)
    assert straight.found and around.found
    assert straight.distance_m == pytest.approx(250.0)
    assert around.distance_m > straight.distance_m + 10.0
    assert any(12 <= col < 16 and row < 7 for row, col in straight.path)
    assert not any(12 <= col < 16 and row < 7 for row, col in around.path)
    assert around.blocked_steps.get("clearance", 0) > 0
    assert "clearance" not in straight.blocked_steps


def test_a_rock_belt_stops_rosalind_franklin_with_the_reason() -> None:
    terrain = rock_field(full_width=True)
    blocked = plan_rover_traverse(rover(ROSALIND_FRANKLIN, terrain), START, GOAL,
                                  start_time_s=MORNING_S)
    assert not blocked.found and blocked.bottleneck is None and blocked.final_label is None
    assert "obstacles taller than 0.25 m" in blocked.message
    assert plan_rover_traverse(rover(CURIOSITY, terrain), START, GOAL,
                               start_time_s=MORNING_S).found
    assert plan_rover_traverse(rover(PERSEVERANCE, terrain), START, GOAL,
                               start_time_s=MORNING_S).found
    inside = plan_rover_traverse(rover(ROSALIND_FRANKLIN, terrain), START, (4, 13),
                                 start_time_s=MORNING_S)
    assert not inside.found and "cannot stand in the goal cell: clearance" in inside.message


# ---- loose ground ---------------------------------------------------------
def test_loose_slope_curiosity_bogs_perseverance_grips_franklin_walks() -> None:
    terrain = patch_terrain(9, 30, 10.0, zone=(0, 9, 12, 16), zone_eta=2.1, grade=0.38)
    plans = {key: plan_rover_traverse(rover(profile, terrain), START, GOAL,
                                      start_time_s=MORNING_S, max_duration_s=8 * SOL_S)
             for key, profile in ROVERS.items()}
    curiosity, perseverance, franklin = (plans[k] for k in ROVERS)
    assert not curiosity.found and "wheel slip" in curiosity.message
    assert perseverance.found and perseverance.final_label.wheel_walk_m == 0.0
    assert perseverance.final_label.max_slip > 0.3
    assert franklin.found and franklin.final_label.wheel_walk_m > 20.0
    walking = [l.transition for l in franklin.final_label.chain()
               if l.transition and l.transition.mode is DriveMode.WHEEL_WALKING]
    assert walking and all(tr.speed_m_s == pytest.approx(2.0 / 1200.0) for tr in walking)
    assert franklin.duration_s > perseverance.duration_s

    firm = patch_terrain(9, 30, 10.0, zone=(0, 9, 12, 16), zone_eta=1.0, grade=0.38)
    assert plan_rover_traverse(rover(CURIOSITY, firm), START, GOAL, start_time_s=MORNING_S).found


def test_loose_ground_costs_time_energy_and_slip() -> None:
    sand = patch_terrain(9, 30, 10.0, zone=(0, 9, 12, 16), zone_eta=2.1)
    for profile in ROVERS.values():
        firm_plan = plan_rover_traverse(rover(profile, flat_terrain(9, 30, 10.0, eta=1.0)), START,
                                        GOAL, start_time_s=MORNING_S)
        sand_plan = plan_rover_traverse(rover(profile, sand), START, GOAL, start_time_s=MORNING_S)
        assert firm_plan.found and sand_plan.found
        assert sand_plan.drive_time_s > firm_plan.drive_time_s
        assert sand_plan.final_label.energy_used_j > firm_plan.final_label.energy_used_j
        assert sand_plan.final_label.max_slip > 2.0 * firm_plan.final_label.max_slip
        assert sand_plan.final_label.cost > firm_plan.final_label.cost


# ---- day and night --------------------------------------------------------
def test_rovers_wait_for_daylight_and_only_the_solar_one_pays_for_the_night() -> None:
    terrain = flat_terrain(9, 30, 10.0)
    night = {key: plan_rover_traverse(rover(profile, terrain), START, (4, 8),
                                      start_time_s=MIDNIGHT_S)
             for key, profile in ROVERS.items()}
    for plan in night.values():
        chain = plan.final_label.chain()
        assert plan.found and chain[1].action == "wait"
        assert chain[1].t_s >= 0.25 * SOL_S                     # parked until after sunrise
        assert chain[1].transition.speed_m_s == 0.0
    mmrtg, solar = night["curiosity"].final_label.chain()[1], night[
        "rosalind_franklin"].final_label.chain()[1]
    assert mmrtg.min_battery_j == pytest.approx(CURIOSITY.battery_j)      # generator covers the night
    assert solar.min_battery_j < 0.97 * ROSALIND_FRANKLIN.battery_j       # the battery carries it
    assert solar.transition.battery_low_j == solar.min_battery_j

    day = plan_rover_traverse(rover(CURIOSITY, terrain), START, (4, 8), start_time_s=MORNING_S)
    assert day.final_label.wait_s == 0.0 and day.duration_s < night["curiosity"].duration_s


def test_with_night_driving_allowed_the_mmrtg_rover_goes_the_solar_one_runs_dry() -> None:
    terrain = flat_terrain(9, 30, 10.0)
    curiosity = plan_rover_traverse(
        rover(CURIOSITY, terrain, year=2026, allow_night_driving=True), START, GOAL,
        start_time_s=MIDNIGHT_S)
    franklin = plan_rover_traverse(
        rover(ROSALIND_FRANKLIN, terrain, allow_night_driving=True), START, GOAL,
        start_time_s=MIDNIGHT_S, battery_fraction=0.8)
    assert curiosity.found and curiosity.final_label.wait_s == 0.0
    assert curiosity.duration_s < 0.25 * SOL_S                  # arrives before sunrise
    assert all(l.transition.generation_w > 70.0 for l in curiosity.final_label.chain()[1:])
    assert franklin.found and franklin.final_label.wait_s > 0.0   # had to stop and wait for the Sun
    assert franklin.duration_s > curiosity.duration_s
    first_drive = franklin.final_label.chain()[1].transition
    assert first_drive.generation_w == 0.0 and first_drive.d_battery_j < 0.0
    assert first_drive.heater_w > 0.0                            # cold actuators at night

    tight = plan_rover_traverse(
        rover(ROSALIND_FRANKLIN, terrain, allow_night_driving=True), START, GOAL,
        start_time_s=MIDNIGHT_S, battery_fraction=0.8, max_duration_s=0.2 * SOL_S)
    assert not tight.found and tight.bottleneck in {RoverRule.BATTERY_RESERVE,
                                                    RoverRule.TIME_BUDGET}


def test_mmrtg_age_shows_in_the_energy_balance() -> None:
    terrain = flat_terrain(9, 30, 10.0)
    young = plan_rover_traverse(rover(CURIOSITY, terrain, year=2012), START, GOAL,
                                start_time_s=MORNING_S)
    old = plan_rover_traverse(rover(CURIOSITY, terrain, year=2031), START, GOAL,
                              start_time_s=MORNING_S)
    assert young.final_label.battery_j > old.final_label.battery_j
    assert young.final_label.chain()[1].transition.generation_w == pytest.approx(110.0)


# ---- battery --------------------------------------------------------------
def test_low_battery_forces_a_recharge_and_an_empty_one_stops_the_rover() -> None:
    terrain = flat_terrain(9, 30, 10.0)
    evaluator = rover(CURIOSITY, terrain, year=2026)
    low = plan_rover_traverse(evaluator, START, GOAL, start_time_s=MORNING_S,
                              battery_fraction=0.50, max_duration_s=6 * SOL_S)
    assert low.found and low.final_label.wait_s > HOUR_S
    assert low.final_label.min_battery_j >= CURIOSITY.battery_reserve_j
    assert "wait" in {label.action for label in low.final_label.chain()}
    empty = plan_rover_traverse(evaluator, START, GOAL, start_time_s=MORNING_S,
                                battery_fraction=0.40)
    assert not empty.found and empty.bottleneck is RoverRule.BATTERY_RESERVE
    assert "safe-mode" in empty.message


# ---- radiation ------------------------------------------------------------
def test_particle_event_makes_the_traverse_slower_and_riskier() -> None:
    terrain = flat_terrain(9, 30, 10.0)
    quiet = plan_rover_traverse(rover(PERSEVERANCE, terrain), START, (4, 12),
                                start_time_s=MORNING_S)
    stormy_env = SpeEnvironment(DiurnalEnvironment())
    stormy = plan_rover_traverse(rover(PERSEVERANCE, terrain, env=stormy_env), START, (4, 12),
                                 start_time_s=MORNING_S)
    assert quiet.found and stormy.found
    assert quiet.final_label.risk == 0.0 < stormy.final_label.risk
    assert stormy.drive_time_s == pytest.approx(2.0 * quiet.drive_time_s, rel=0.05)
    assert stormy.final_label.dose_gy > quiet.final_label.dose_gy

    hold = replace(DEFAULT_CONFIG, rover=replace(DEFAULT_CONFIG.rover, spe_hold=True))
    held = rover(PERSEVERANCE, terrain, env=stormy_env, config=hold)
    assert held.operations_block(START, MORNING_S) is RoverBlock.SPE_HOLD
    stuck = plan_rover_traverse(held, START, (4, 12), start_time_s=MORNING_S,
                                max_duration_s=1.5 * SOL_S)
    assert not stuck.found and stuck.bottleneck is RoverRule.TIME_BUDGET


# ---- reach, layers, reports -----------------------------------------------
def test_reach_of_curiosity_is_wider_than_of_rosalind_franklin() -> None:
    terrain = rock_field(full_width=True)
    reach = {key: compute_rover_reachability(rover(profile, terrain), START,
                                             start_time_s=MORNING_S, max_duration_s=3 * SOL_S,
                                             isochrones_s=(HOUR_S, 2 * HOUR_S, 4 * HOUR_S))
             for key, profile in (("curiosity", CURIOSITY), ("franklin", ROSALIND_FRANKLIN))}
    curiosity, franklin = reach["curiosity"], reach["franklin"]
    assert curiosity.reachable.all()
    assert not franklin.reachable[:, 12:].any() and franklin.reachable[:, :12].all()
    assert franklin.reachable.sum() < curiosity.reachable.sum()
    assert (curiosity.round_trip_ok == curiosity.reachable).all()
    masks = [curiosity.isochrones[t] for t in sorted(curiosity.isochrones)]
    assert all(not (a & ~b).any() for a, b in zip(masks, masks[1:]))
    assert masks[0].sum() < masks[-1].sum()
    assert curiosity.arrival_time_s[START] == 0.0


def test_rover_map_layers() -> None:
    terrain = rock_field()
    curiosity, franklin = rover(CURIOSITY, terrain), rover(ROSALIND_FRANKLIN, terrain)
    reach = compute_rover_reachability(franklin, START, start_time_s=MORNING_S)
    layers = build_rover_layers(franklin, MORNING_S, reach)
    assert all(layer.shape == (9, 30) for layer in layers.values())
    assert layers["cell_block"][2, 13] == BLOCK_CODES[RoverBlock.CLEARANCE]
    assert layers["cell_block"][4, 2] == 0 and layers["passable_directions"][4, 5] == 8
    assert layers["obstacle_margin_m"][2, 13] < 0.0 < layers["obstacle_margin_m"][4, 2]
    assert layers["time_min_per_km"][4, 5] == pytest.approx(1000 / 40 * 60, rel=0.08)   # 40 m/h
    assert not layers["reachable"][2, 13] and layers["reachable"][8, 13]
    assert layers["wheel_walking_share"][4, 5] == 0.0
    theirs = build_rover_layers(curiosity, MORNING_S)
    assert theirs["cell_block"].max() == 0 and "reachable" not in theirs
    assert theirs["time_min_per_km"][4, 5] < layers["time_min_per_km"][4, 5]
    assert theirs["energy_wh_per_km"][4, 5] > layers["energy_wh_per_km"][4, 5]
    night = build_rover_layers(curiosity, MIDNIGHT_S)
    assert night["passable_directions"].max() == 0                 # nobody drives at night


def test_profile_summary_and_comparison() -> None:
    terrain = rock_field()
    plans = compare_rovers(terrain, DiurnalEnvironment(), START, GOAL, start_time_s=MORNING_S,
                           mission_year=2031)
    assert set(plans) == set(ROVERS) and all(p is not None and p.found for p in plans.values())
    capacities = {key: profile.battery_j for key, profile in ROVERS.items()}
    rows = rover_comparison(plans, capacities)
    assert [r["rover"] for r in rows] == ["Curiosity", "Perseverance", "Rosalind Franklin"]
    assert rows[1]["drive_time_h"] < rows[0]["drive_time_h"] < rows[2]["drive_time_h"]

    plan = plans["perseverance"]
    summary = rover_plan_summary(plan, PERSEVERANCE.battery_j)
    assert summary["found"] and summary["distance_m"] == pytest.approx(250.0)
    assert summary["mean_drive_speed_m_h"] == pytest.approx(0.042 * 0.75 * HOUR_S, rel=0.06)
    assert 47.0 < summary["battery_min_pct"] <= summary["battery_end_pct"] + 1e-9 <= 100.0
    profile = rover_route_profile(plan.final_label, PERSEVERANCE.battery_j)
    assert profile[0].action == "start" and profile[-1].distance_m == pytest.approx(250.0)
    assert [r.elapsed_h for r in profile] == sorted(r.elapsed_h for r in profile)
    assert all(r.mode == "rolling" and r.speed_m_h > 100.0 for r in profile[1:])
    assert isinstance(profile[3].to_dict()["battery_pct"], float)

    now = compare_rovers(terrain, DiurnalEnvironment(), START, GOAL, start_time_s=MORNING_S,
                         mission_year=2026)
    assert now["rosalind_franklin"] is None and now["curiosity"].found
    assert rover_comparison(now, capacities)[2]["message"] == "not on Mars in this year"
    failed = plan_rover_traverse(rover(ROSALIND_FRANKLIN, rock_field(full_width=True)), START,
                                 GOAL, start_time_s=MORNING_S)
    assert rover_plan_summary(failed, ROSALIND_FRANKLIN.battery_j)["found"] is False


# ---- the data teams' stub site --------------------------------------------
def test_rovers_on_the_stub_site() -> None:
    """Site data and environment model of the other teams drive all three rovers."""
    mars_map = pytest.importorskip("mars_map")
    from src.providers.site_adapter import SiteEnvironment, terrain_from_site

    site = mars_map.load_stub()
    terrain = terrain_from_site(site, downsample=10)
    obstacles = terrain.obstacle_height_m
    assert obstacles.shape == (100, 100) and obstacles.min() >= 0.0
    shares = [float((obstacles > profile.max_obstacle_height_m).mean())
              for profile in (CURIOSITY, PERSEVERANCE, ROSALIND_FRANKLIN)]
    assert shares[0] <= shares[1] < shares[2] < 0.5               # the small rover is stopped most

    env = SiteEnvironment(site, mars_map.Environment(site), sol=0, start_ltst_h=8.0, downsample=10)
    start = terrain.to_cell(*site.start)
    goal = terrain.to_cell(1000.0, 500.0)
    plans = compare_rovers(terrain, env, start, goal, start_time_s=0.0, mission_year=2031)
    for key, plan in plans.items():
        assert plan is not None and plan.message
        if plan.found:
            assert plan.path[0] == start and plan.path[-1] == goal
            assert plan.final_label.min_battery_j >= ROVERS[key].battery_reserve_j
            assert not any(terrain.forbidden[cell] for cell in plan.path)
    assert plans["curiosity"].found and plans["perseverance"].found
    assert plans["perseverance"].drive_time_s < plans["curiosity"].drive_time_s
