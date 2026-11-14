"""H10, H2, H11: verdict on an external route, router modifiers, reference tables,
and the end-to-end run on the teams' real code (the stub site)."""

from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np
import pytest

from conftest import START_TIME_S, make_search
from src.comms.coverage import GridCoverage, coverage_from_radio_map, coverage_from_terrain
from src.core.config import DEFAULT_CONFIG, SimConfig
from src.core.constants import SOL_S
from src.core.exceptions import InfeasibleStateError, ValidationError
from src.core.schemas import (AstronautProfile, CoolingMode, PruneRule, SuitProfile, TaskType)
from src.planning.sol_selection import assess_sols, best_sol, sol_series_from_site
from src.providers.route_adapter import (RouteStop, rasterize_route, route_from_eva_plans,
                                         route_from_mission_plan, route_from_points)
from src.providers.site_adapter import (SiteEnvironment, series_value, site_sols,
                                        terrain_from_site)
from src.providers.synthetic import flat_terrain, forbidden_zone_terrain, slope_terrain
from src.reporting import reference_tables
from src.reporting.map_layers import build_map_layers, route_report_summary
from src.routing import modifiers
from src.routing.mission_planner import MissionStatus
from src.routing.route_evaluation import chain_scale, evaluate_route
from src.routing.transition import EdgeEvaluator

OUT_AND_BACK = [(12.5, 112.5), (712.5, 112.5), (12.5, 112.5)]


# ---- H10: verdict on an external route ------------------------------------
def test_out_and_back_route_with_a_stop_is_accepted(make_evaluator) -> None:
    evaluator = make_evaluator(flat_terrain(9, 40, 25.0, shelters=[(4, 0)]))
    route = route_from_points(
        OUT_AND_BACK, stops=[RouteStop("rock", 712.5, 112.5, 900.0, TaskType.SAMPLING)])
    report = evaluate_route(evaluator, route, start_time_s=START_TIME_S)
    assert report.status is MissionStatus.OK and report.completed and report.rule is None
    assert report.cells[0] == report.cells[-1] == (4, 0) and len(report.cells) == 57
    assert report.distance_m == pytest.approx(1400.0)
    assert report.duration_s == pytest.approx(2 * 840.0 + 900.0, rel=0.03)
    assert [leg.name for leg in report.legs] == ["leg 1", "task 1", "leg 2"]
    assert report.legs[1].distance_m == 0.0 and report.legs[1].duration_s == pytest.approx(900.0)
    assert report.contact is None and report.reference_delta == {}
    assert report.limiting_resource != "none"

    summary = route_report_summary(report)
    assert summary["status"] == "OK" and summary["broken_constraint"] is None
    assert summary["distance_km"] == pytest.approx(1.4)
    assert len(summary["legs"]) == 3 and "radio" not in summary
    assert 0.0 < summary["remaining"]["o2_pct"] < 100.0


def test_diagonal_route_is_walked_at_its_true_length(make_evaluator) -> None:
    """A route at an angle to the grid is not charged for the staircase of cells."""
    evaluator = make_evaluator(flat_terrain(30, 60, 10.0, shelters=[(2, 2)]))
    points = [(25.0, 25.0), (525.0, 235.0)]
    cells = rasterize_route(evaluator.terrain, points)
    scale = chain_scale(10.0, cells, tuple(points))
    assert 0.9 < scale < 1.0
    report = evaluate_route(evaluator, route_from_points(points), start_time_s=START_TIME_S)
    assert report.distance_m == pytest.approx(math.hypot(500.0, 210.0), rel=0.01)


def test_author_totals_are_compared(make_evaluator) -> None:
    evaluator = make_evaluator(flat_terrain(9, 40, 25.0, shelters=[(4, 0)]))
    route = dataclasses.replace(route_from_points(OUT_AND_BACK),
                                reference={"time_s": 1000.0, "distance_m": 1400.0, "o2_kg": 0.01})
    report = evaluate_route(evaluator, route, start_time_s=START_TIME_S)
    assert set(report.reference_delta) == {"time_s", "distance_m", "o2_kg"}
    assert report.reference_delta["time_s"] == pytest.approx(680.0, rel=0.05)
    assert report.reference_delta["distance_m"] == pytest.approx(0.0, abs=1.0)
    difference = route_report_summary(report)["difference_from_author"]
    assert set(difference) == {"time_min", "distance_km", "o2_g"} and difference["o2_g"] > 30.0


def test_radio_gap_makes_the_route_marginal(make_evaluator) -> None:
    evaluator = make_evaluator(flat_terrain(9, 80, 25.0, shelters=[(4, 0)]))
    far = [(12.5, 112.5), (1912.5, 112.5), (12.5, 112.5)]
    mask = np.ones((9, 80), dtype=bool)
    mask[:, 40:] = False
    report = evaluate_route(evaluator, route_from_points(far), start_time_s=START_TIME_S,
                            coverage=GridCoverage(mask))
    assert report.status is MissionStatus.MARGINAL and report.completed
    assert report.contact is not None and report.contact.violates
    assert report.contact.longest_gap_s > DEFAULT_CONFIG.comms.max_no_contact_s
    assert any("radio contact" in warning for warning in report.warnings)
    radio = route_report_summary(report)["radio"]
    assert radio["over_limit"] and radio["gaps"] == 1 and radio["longest_gap_min"] > 15.0

    covered = evaluate_route(evaluator, route_from_points(far), start_time_s=START_TIME_S,
                             coverage=coverage_from_terrain(evaluator.terrain, (4, 0),
                                                            evaluator.config.comms))
    assert covered.contact is not None and not covered.contact.violates


def test_blocked_route_fails_with_the_exact_reason(make_evaluator) -> None:
    forbidden = make_evaluator(forbidden_zone_terrain(5, 30, 25.0, zone=(0, 5, 14, 16),
                                                      shelters=[(2, 0)]))
    report = evaluate_route(forbidden, route_from_points([(12.5, 62.5), (700.0, 62.5)]),
                            start_time_s=START_TIME_S)
    assert report.status is MissionStatus.FAIL and not report.completed and report.rule is None
    assert "forbidden zone" in report.message and "(2, 14)" in report.message
    assert report.final_label.cell == (2, 13)

    steep = make_evaluator(slope_terrain(5, 30, 25.0, grade=0.40, shelters=[(2, 0)]))
    report = evaluate_route(steep, route_from_points([(12.5, 62.5), (700.0, 62.5)]),
                            start_time_s=START_TIME_S)
    assert report.status is MissionStatus.FAIL and "needs" in report.message
    assert "W" in report.message and "sustainable power" in report.message

    cliff = make_evaluator(slope_terrain(5, 30, 25.0, grade=-0.40, shelters=[(2, 0)]))
    report = evaluate_route(cliff, route_from_points([(12.5, 62.5), (700.0, 62.5)]),
                            start_time_s=START_TIME_S)
    assert "descent of 22 deg exceeds the limit of 20 deg" in report.message


def test_route_too_long_breaks_the_return_oxygen_rule(make_evaluator) -> None:
    evaluator = make_evaluator(flat_terrain(3, 90, 100.0, shelters=[(1, 0)]))
    report = evaluate_route(evaluator, route_from_points([(50.0, 150.0), (8550.0, 150.0)]),
                            start_time_s=START_TIME_S)
    assert report.status is MissionStatus.FAIL and report.rule is PruneRule.RETURN_O2
    assert route_report_summary(report)["broken_constraint"] == "return_o2"
    assert 4.5 <= report.distance_m / 1000.0 <= 7.5


def test_evaluate_route_rejects_unusable_input(make_evaluator) -> None:
    terrain = forbidden_zone_terrain(5, 30, 25.0, zone=(0, 5, 0, 2), shelters=[(2, 10)])
    evaluator = make_evaluator(terrain)
    with pytest.raises(InfeasibleStateError):
        evaluate_route(evaluator, route_from_points([(12.5, 62.5), (700.0, 62.5)]),
                       start_time_s=START_TIME_S)
    with pytest.raises(ValidationError):
        evaluate_route(evaluator, route_from_points([(300.0, 62.5), (9000.0, 62.5)]),
                       start_time_s=START_TIME_S)
    with pytest.raises(ValidationError):
        evaluate_route(evaluator, route_from_points(
            [(300.0, 62.5), (600.0, 62.5)], stops=[RouteStop("off", 400.0, 5000.0, 60.0)]),
            start_time_s=START_TIME_S)


# ---- H2: modifiers and cube export ----------------------------------------
def test_router_modifiers(cfg: SimConfig) -> None:
    day, night = math.radians(40.0), math.radians(-10.0)
    reference = cfg.routing.standard_fatigue
    assert modifiers.speed_multiplier(day, reference, cfg) == pytest.approx(1.0)
    assert modifiers.speed_multiplier(day, 0.0, cfg) == 1.0
    assert modifiers.speed_multiplier(night, reference, cfg) == pytest.approx(0.5)
    tired = modifiers.speed_multiplier(day, 1.0, cfg)
    assert tired == pytest.approx(0.7 / (1.0 - 0.3 * reference))
    assert modifiers.o2_multiplier(day, 1.0, cfg) == pytest.approx(1.0 / tired)
    assert modifiers.o2_multiplier(night, reference, cfg) == pytest.approx(2.0)
    with pytest.raises(ValidationError):
        modifiers.speed_multiplier(day, 1.5, cfg)


def test_cost_cubes_are_exported_for_another_router(make_evaluator, tmp_path) -> None:
    evaluator = make_evaluator(flat_terrain(6, 8, 10.0, shelters=[(0, 0)]))
    _, _, maps = make_search(evaluator, (0, 0))
    arrays = modifiers.export_cost_cubes(maps.nominal, evaluator.cell_size_m)
    assert arrays["time_s"].shape == (8, 6, 8) and arrays["directions"].shape == (8, 2)
    assert arrays["time_s"][3, 2, 2] == pytest.approx(12.0)             # one 10 m step east
    assert arrays["directions"][3].tolist() == [0.0, 1.0]
    assert np.isinf(arrays["time_s"][3, 2, 7])                          # off the east edge
    path = modifiers.save_cost_cubes(tmp_path / "cubes", maps.nominal, evaluator.cell_size_m)
    assert path.suffix == ".npz"
    with np.load(path) as loaded:
        assert str(loaded["schema"]) == modifiers.CUBE_SCHEMA
        assert float(loaded["cell_size_m"]) == 10.0
        np.testing.assert_array_equal(loaded["o2_kg"], maps.nominal.o2_kg)


# ---- H11 / H0: reference tables -------------------------------------------
def test_terrain_slope_table(astronaut: AstronautProfile, suit: SuitProfile) -> None:
    rows = reference_tables.terrain_slope_table(astronaut, suit)
    assert len(rows) == 4 * 10

    def row(ground: str, slope: float) -> dict:
        return next(r for r in rows if r["ground"] == ground and r["slope_deg"] == slope)

    level = row("firm regolith", 0.0)
    assert level["speed_kmh"] == pytest.approx(3.0) and level["power_w"] == pytest.approx(377, abs=3)
    assert level["o2_g_per_km"] == pytest.approx(33.7, abs=0.5)
    assert level["time_min_per_km"] == pytest.approx(20.0)
    assert row("bedrock", 25.0)["speed_kmh"] is None            # beyond sustainable power
    assert row("loose sand", 0.0)["speed_kmh"] < 3.0
    assert row("bedrock", 10.0)["speed_kmh"] < row("bedrock", 5.0)["speed_kmh"] < 3.0
    assert row("bedrock", -10.0)["o2_g_per_km"] < row("bedrock", 0.0)["o2_g_per_km"]
    for ground in reference_tables.GROUND_TYPES:
        costs = [row(ground, s)["o2_g_per_km"] for s in (0.0, 5.0, 10.0)]
        assert costs == sorted(costs)


def test_walking_and_reach_tables(astronaut: AstronautProfile, suit: SuitProfile) -> None:
    walking = reference_tables.continuous_walking_table(astronaut, suit)
    level = next(r for r in walking if r["ground"] == "firm regolith" and r["slope_deg"] == 0.0)
    assert 2.0 < level["until_rest_h"] < level["until_return_only_h"] < 4.0
    assert level["distance_until_rest_km"] == pytest.approx(3.0 * level["until_rest_h"])
    climb = next(r for r in walking if r["ground"] == "firm regolith" and r["slope_deg"] == 10.0)
    assert climb["until_rest_h"] < level["until_rest_h"]
    assert climb["share_of_sustainable"] == pytest.approx(1.0, abs=0.01)

    reach = reference_tables.max_distance_table(astronaut, suit)
    assert reach[0]["radius_km"] == pytest.approx(5.9, abs=0.2)        # same as the search (B5)
    assert reach[0]["limited_by"] == "oxygen"
    radii = [r["radius_km"] for r in reach]
    assert radii == sorted(radii, reverse=True) and radii[-1] > 0.0
    assert all(r["round_trip_km"] == pytest.approx(2 * r["radius_km"]) for r in reach)


def test_limiting_resource_and_mission_params(astronaut: AstronautProfile, suit: SuitProfile) -> None:
    table = reference_tables.limiting_resource_table(suit)
    assert [r["power_w"] for r in table] == [150.0, 250.0, 350.0, 450.0]
    assert table[0]["oxygen_h"] > table[-1]["oxygen_h"]
    assert all(r["eva_time_h"] == pytest.approx(6.4) for r in table)
    assert "cooling_water_h" not in table[0] and "cooling_water_h" in table[-1]
    radiator = dataclasses.replace(suit, cooling_mode=CoolingMode.RADIATOR_STORAGE)
    assert "cooling_water_h" not in reference_tables.limiting_resource_table(radiator)[-1]
    cold = reference_tables.limiting_resource_table(suit, heater_w=100.0)
    assert cold[0]["battery_h"] < table[0]["battery_h"]

    params = reference_tables.recommended_mission_params(astronaut, suit)
    assert set(params) == {"max_slope_deg", "hard_max_slope_deg", "walk_speed_flat_kmh",
                           "eva_max_hours", "return_reserve_frac", "max_dose_msv",
                           "max_distance_km"}
    assert params["walk_speed_flat_kmh"] == pytest.approx(3.0)
    assert params["hard_max_slope_deg"] == pytest.approx(25.0) and params["eva_max_hours"] == 8.0
    assert 8.0 <= params["max_distance_km"] <= 11.0

    text = reference_tables.to_markdown(table)
    assert text.splitlines()[0].startswith("| power_w |") and text.count("\n") == len(table) + 1
    assert "—" in reference_tables.to_markdown([{"a": None, "b": 1.0}])
    assert reference_tables.to_markdown([]) == ""


def test_radio_layer_in_the_map_layers(make_evaluator, cfg: SimConfig) -> None:
    evaluator = make_evaluator(flat_terrain(6, 8, 10.0, shelters=[(0, 0)]))
    _, _, maps = make_search(evaluator, (0, 0))
    coverage = coverage_from_terrain(evaluator.terrain, (0, 0), cfg.comms)
    layers = build_map_layers(evaluator, maps, START_TIME_S, coverage=coverage)
    assert layers["radio_contact"].shape == (6, 8) and layers["radio_contact"].all()
    assert "radio_contact" not in build_map_layers(evaluator, maps, START_TIME_S)


# ---- the teams' real code on the stub site --------------------------------
@pytest.fixture(scope="module")
def stub():
    """Stub site without the dust and wind series, with the teams' models built on it."""
    mars_map = pytest.importorskip("mars_map")
    pipeline = pytest.importorskip("mars_map.pipeline")
    site = mars_map.load_stub()
    series = {k: v for k, v in site.series.items()
              if k not in ("dust_opacity_tau", "wind_max_ms")}
    site = dataclasses.replace(site, series=series)
    model = mars_map.Environment(site)
    plans = pipeline.plan_site(site, sol=0, env=model, check_radio=False)
    ground = mars_map.build_astronaut_terrain(site)
    factor = 10
    terrain = terrain_from_site(site, downsample=factor, ground_model=ground)
    environment = SiteEnvironment(site, model, sol=0, start_ltst_h=10.0, downsample=factor)
    evaluator = EdgeEvaluator(terrain, environment, AstronautProfile(), SuitProfile(),
                              DEFAULT_CONFIG, time_on_surface_s=30.0 * SOL_S)
    base = terrain.to_cell(*site.start)
    _, _, maps = make_search(evaluator, base, t_s=0.0)
    return dict(mars_map=mars_map, site=site, model=model, plans=plans, ground=ground,
                factor=factor, terrain=terrain, evaluator=evaluator, base=base, maps=maps)


def test_stub_site_works_without_dust_and_wind_series(stub) -> None:
    site, terrain = stub["site"], stub["terrain"]
    assert "dust_opacity_tau" not in site.series and "wind_max_ms" not in site.series
    assert series_value(site, "dust_opacity_tau", 0) is None
    assert site_sols(site) == list(range(30))
    sample = stub["evaluator"].env_at(stub["base"], 0.0)
    sample.validate()
    assert 600.0 < sample.pressure_pa < 1200.0 and sample.sun_elevation_rad > 0.0
    assert not hasattr(sample, "tau") and not hasattr(sample, "wind_m_s")
    assert terrain.shape == (100, 100) and terrain.forbidden.any()


def test_stub_ground_model_of_the_terrain_team_is_used(stub) -> None:
    terrain, ground, factor = stub["terrain"], stub["ground"], stub["factor"]
    block = np.asarray(ground.terramech_factor)[:factor, :factor].mean()
    assert terrain.walk_eta[0, 0] == pytest.approx(float(block), rel=1e-6)
    own = terrain_from_site(stub["site"], downsample=factor)
    assert own.shape == terrain.shape and not np.allclose(own.walk_eta, terrain.walk_eta)
    assert (own.passable_walk == terrain.passable_walk).mean() > 0.9
    with pytest.raises(Exception):
        terrain_from_site(stub["site"], downsample=factor,
                          ground_model=dataclasses.replace(
                              ground, impassable=np.zeros((3, 3), dtype=bool)))


def test_stub_real_site_series_are_used_when_present(stub) -> None:
    """The real site has pressure and separate air and ground temperatures."""
    site, model = stub["site"], stub["model"]
    series = dict(site.series)
    series["pressure_pa"] = [850.0] * 30
    series["ground_temp_min_c"] = [-85.0] * 30
    series["ground_temp_max_c"] = [0.0] * 30
    real_like = dataclasses.replace(site, series=series)
    real_model = copy.copy(model)
    real_model.site = real_like
    real_model._insolation = {}
    environment = SiteEnvironment(real_like, real_model, sol=0, start_ltst_h=13.0,
                                  downsample=stub["factor"])
    sample = environment.get_env(stub["base"], 0.0)
    sample.validate()
    assert sample.pressure_pa == pytest.approx(850.0, rel=0.02)
    assert sample.t_air_k < sample.t_ground_k               # air series is colder at midday
    assert sample.t_ground_k - sample.t_air_k > 5.0


def test_stub_sol_selection_rejects_the_particle_event(stub) -> None:
    site = stub["site"]
    cfg = DEFAULT_CONFIG.sol_selection
    result = assess_sols(sol_series_from_site(site, cfg), cfg,
                         latitude_rad=math.radians(site.center_lat))
    rejected = {a.sol: a.reasons for a in result if not a.allowed}
    events = [a.sol for a in result if a.particle_event]
    assert len(events) == 2 and events[1] == events[0] + 1
    assert set(rejected) == {events[0] - 1, events[0], events[1], events[1] + 1}
    assert stub["model"].is_particle_event(events[0])
    best = best_sol(result)
    assert best is not None and best.sol not in rejected and best.window_s is not None


def test_stub_router_plans_pass_through_the_simulator(stub) -> None:
    """Every plan of the routing team gets a verdict; the totals are comparable."""
    mars_map, evaluator, terrain = stub["mars_map"], stub["evaluator"], stub["terrain"]
    site = stub["site"]
    radio_map = mars_map.radio.coverage_map(site.layers["elevation"], site.resolution_m,
                                            site.start)
    coverage = coverage_from_radio_map(radio_map, terrain)
    own = coverage_from_terrain(terrain, stub["base"], DEFAULT_CONFIG.comms)
    assert (own.mask == coverage.mask).mean() > 0.9         # two independent models agree

    verdicts = {}
    for name, plan in stub["plans"].plans.items():
        route = route_from_mission_plan(plan)
        assert len(route.stops) == len(plan.visited_targets)
        report = evaluate_route(evaluator, route, start_time_s=0.0, maps=stub["maps"],
                                coverage=coverage)
        summary = route_report_summary(report)
        verdicts[name] = summary["status"]
        assert report.cells[0] == stub["base"] and report.contact is not None
        assert set(summary["difference_from_author"]) == {
            "time_min", "distance_km", "o2_g", "dose_msv", "battery_wh"}
        if report.completed:
            assert abs(summary["difference_from_author"]["distance_km"]) < 0.1
            assert report.final_label.cell == stub["base"]
            assert not any(terrain.forbidden[label.cell] for label in report.final_label.chain())
        else:
            assert "blocked" in report.message or report.rule is not None
    assert "eco" in verdicts and verdicts["eco"] in {"OK", "MARGINAL"}


def test_stub_eva_plans_document_gives_the_same_route(stub) -> None:
    document = stub["plans"].to_dict()
    plan = stub["plans"].plans["eco"]
    from_json = route_from_eva_plans(document, "eco")
    from_object = route_from_mission_plan(plan)
    assert [s.name for s in from_json.stops] == [s.name for s in from_object.stops]
    assert from_json.reference["time_s"] == pytest.approx(from_object.reference["time_s"], rel=0.01)
    cells_json = rasterize_route(stub["terrain"], from_json.points_m)
    cells_object = rasterize_route(stub["terrain"], from_object.points_m)
    assert cells_json[0] == cells_object[0] and cells_json[-1] == cells_object[-1]
    shared = len(set(cells_json) & set(cells_object)) / len(set(cells_object))
    assert shared > 0.8
