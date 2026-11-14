"""The routing team's planner running on the astronaut model.

Part 1: the array form of the walking model equals the scalar one.
Part 2: ``mars_map.pipeline`` on the stub site prices routes with that model,
writes the ``eva-plan/1`` JSON with the agreed waypoint keys, and carries the
verdict of the sequential simulator.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.core.config import SimConfig
from src.core.constants import HOUR_S, O2_DENSITY_STP_KG_M3, SOL_S
from src.core.exceptions import ValidationError
from src.core.schemas import AstronautProfile, SuitProfile, TaskType
from src.physiology import capacity, life_support, metabolism, vectorized

WAYPOINT_KEYS = {"leg", "x_m", "y_m", "elev_m", "dist_m", "t_min", "slope_deg", "looseness",
                 "sinkage_cm", "terrain_class", "dose_rate_usv_h", "dose_msv", "temp_c", "o2_pct",
                 "battery_pct", "lat", "lon", "radio"}


# ---- part 1: array model == scalar model ----------------------------------
def scalar_step(astronaut: AstronautProfile, suit: SuitProfile, cfg: SimConfig, dist: float,
                rise: float, eta: float, factor: float, cap: float) -> tuple[float, float] | None:
    grade = rise / dist
    coeffs = metabolism.pandolf_coefficients(astronaut.body_mass_kg, suit.mass_kg, eta, grade,
                                             suit.stiffness_factor, cfg.metabolism)
    upper = max(cfg.capacity.min_speed_m_s, astronaut.preferred_speed_m_s * factor)
    speed = capacity.select_speed(*coeffs, grade, upper, cap, cfg.capacity)
    if speed is None:
        return None
    return speed, metabolism.power_from_coefficients(*coeffs, speed)


def test_array_model_matches_the_scalar_model(astronaut: AstronautProfile, suit: SuitProfile,
                                              cfg: SimConfig) -> None:
    cap = vectorized.planning_power_cap_w(astronaut, cfg, 30.0 * SOL_S)
    grades = np.array([-0.35, -0.25, -0.15, -0.10, -0.05, 0.0, 0.03, 0.08, 0.15, 0.25, 0.40])
    etas = np.array([1.0, 1.2, 1.5, 2.1, 2.6])
    factors = np.array([1.0, 0.7, 0.45])
    g, e, f = (a.ravel() for a in np.meshgrid(grades, etas, factors, indexing="ij"))
    dist = np.full(g.shape, 10.0)
    steps = vectorized.walking_steps(dist, dist * g, e, f, 0.0, astronaut=astronaut, suit=suit,
                                     cfg=cfg, power_cap_w=cap)
    feasible_count = 0
    for i in range(g.size):
        expected = scalar_step(astronaut, suit, cfg, 10.0, 10.0 * g[i], e[i], f[i], cap)
        if expected is None:
            # The scalar descent scan is 0.02 m/s coarse: it may miss a feasible sliver.
            if steps.feasible[i]:
                assert g[i] < 0.0 and steps.speed_m_s[i] < cfg.capacity.min_speed_m_s + 0.021
            else:
                assert np.isinf(steps.time_s[i]) and np.isinf(steps.o2_kg[i])
            continue
        feasible_count += 1
        assert steps.feasible[i], f"grade {g[i]}, eta {e[i]}, factor {f[i]}"
        speed, power = expected
        tolerance = 0.021 if g[i] < 0.0 else 2.0 * cfg.capacity.speed_tolerance_m_s
        assert steps.speed_m_s[i] == pytest.approx(speed, abs=tolerance)
        assert steps.speed_m_s[i] >= speed - 1e-9          # the closed form is never slower
        assert steps.metabolic_w[i] <= cap + 1e-6
        if abs(steps.speed_m_s[i] - speed) < 1e-9:
            time_s = float(np.hypot(10.0, 10.0 * g[i])) / speed
            assert steps.metabolic_w[i] == pytest.approx(power)
            assert steps.time_s[i] == pytest.approx(time_s)
            assert steps.o2_kg[i] == pytest.approx(life_support.o2_consumption_kg(power, time_s, suit))
    assert feasible_count > 60 and (~steps.feasible).sum() > 10


def test_array_model_reference_numbers(astronaut: AstronautProfile, suit: SuitProfile,
                                       cfg: SimConfig) -> None:
    steps = vectorized.walking_steps(1000.0, 0.0, 1.2, 1.0, 0.0, astronaut=astronaut, suit=suit,
                                     cfg=cfg, power_cap_w=450.0)
    assert float(steps.speed_m_s) == pytest.approx(3.0 / 3.6)
    assert float(steps.metabolic_w) == pytest.approx(377.0, abs=1.0)
    assert float(steps.time_s) == pytest.approx(1200.0)
    assert 1000.0 * float(steps.o2_kg) == pytest.approx(33.7, abs=0.3)
    assert float(steps.battery_j) == pytest.approx((60.0 + 40.0 * 377.0 / 300.0) * 1200.0, rel=0.01)

    heated = vectorized.walking_steps(1000.0, 0.0, 1.2, 1.0, 50.0, astronaut=astronaut, suit=suit,
                                      cfg=cfg, power_cap_w=450.0)
    assert float(heated.battery_j - steps.battery_j) == pytest.approx(50.0 * 1200.0)

    shaped = vectorized.walking_steps(np.full((2, 3), 10.0), np.zeros((2, 3)), 1.2, 1.0, 0.0,
                                      astronaut=astronaut, suit=suit, cfg=cfg, power_cap_w=450.0)
    assert shaped.time_s.shape == (2, 3) and shaped.feasible.all()
    with pytest.raises(ValidationError):
        vectorized.walking_steps(10.0, 0.0, 1.2, 1.0, 0.0, astronaut=astronaut, suit=suit,
                                 cfg=cfg, power_cap_w=0.0)


def test_planning_cap_heaters_and_stationary_work(astronaut: AstronautProfile, suit: SuitProfile,
                                                  cfg: SimConfig) -> None:
    rested = vectorized.planning_power_cap_w(astronaut, cfg, 30.0 * SOL_S, fatigue=0.0)
    planning = vectorized.planning_power_cap_w(astronaut, cfg, 30.0 * SOL_S)
    assert planning == pytest.approx(rested * (1.0 - 0.3 * cfg.fatigue.rest_trigger))
    assert vectorized.planning_power_cap_w(astronaut, cfg, 1.5 * SOL_S) < planning

    heaters = vectorized.extremity_heater_w(vectorized.celsius_to_kelvin(
        np.array([-20.0, -40.0, -60.0, -200.0])), suit, cfg)
    assert heaters.tolist() == pytest.approx([0.0, 0.0, 30.0, suit.heater_max_w])

    power, o2, battery = vectorized.stationary_work(45 * 60.0, TaskType.SAMPLING, 0.0,
                                                    suit=suit, cfg=cfg)
    assert power == pytest.approx(300.0 * (1.0 + 0.3 * cfg.fatigue.rest_trigger))
    assert o2 == pytest.approx(life_support.o2_consumption_kg(power, 2700.0, suit))
    assert battery == pytest.approx(
        float(vectorized.electric_power_w(power, 0.0, suit, cfg)) * 2700.0)
    assert vectorized.o2_kg_to_litres(O2_DENSITY_STP_KG_M3) == pytest.approx(1000.0)
    with pytest.raises(ValidationError):
        vectorized.stationary_work(-1.0, TaskType.REST, 0.0, suit=suit, cfg=cfg)


# ---- part 2: the routing team's planner on this model ---------------------
@pytest.fixture(scope="module")
def planned():
    mars_map = pytest.importorskip("mars_map")
    pipeline = pytest.importorskip("mars_map.pipeline")
    site = mars_map.load_stub()
    result = pipeline.plan_site(site, sol=0)
    return dict(mars_map=mars_map, pipeline=pipeline, site=site, result=result,
                document=result.to_dict())


def test_router_uses_the_astronaut_model(planned) -> None:
    from mars_map import biomechanics

    params = biomechanics.SuitParams()
    assert params.body_mass_kg == 80.0 and params.suit_mass_kg == 60.0
    assert params.o2_tank_l == pytest.approx(0.55 / O2_DENSITY_STP_KG_M3 * 1000.0)
    assert params.battery_wh == pytest.approx(700.0)
    mission = planned["site"].mission

    time_s, power, o2_l, energy_wh = biomechanics.locomotion(
        np.array([1000.0]), np.array([0.0]), 1.2, 1.0, mission)
    assert float(time_s[0]) == pytest.approx(1200.0)                 # 3 km/h, not a mission constant
    assert float(power[0]) == pytest.approx(377.0, abs=1.0)
    assert float(o2_l[0]) * O2_DENSITY_STP_KG_M3 == pytest.approx(33.7, abs=0.3)   # grams
    assert float(energy_wh[0]) == pytest.approx((60 + 40 * 377 / 300) * 1200 / HOUR_S, rel=0.01)

    blocked = biomechanics.locomotion(np.array([10.0]), np.array([4.0]), 2.1, 1.0, mission)
    assert np.isinf(blocked[0][0]) and np.isinf(blocked[2][0])       # 22 deg on loose sand

    seconds, o2_work, wh_work = biomechanics.dwell(45.0)
    assert seconds == 2700.0 and o2_work > 0.0 and wh_work > 0.0
    lighter = biomechanics.locomotion(np.array([1000.0]), np.array([0.0]), 1.2, 1.0, mission,
                                      biomechanics.SuitParams(suit_mass_kg=30.0))
    assert float(lighter[1][0]) < float(power[0])                    # parameters reach the model


def test_document_keeps_the_agreed_waypoint_format(planned) -> None:
    document = planned["document"]
    assert document["schema"] == "eva-plan/1"
    assert set(document["strategies"]) == {"shielded", "eco", "science"}
    json.loads(json.dumps(document))                                 # serialisable as is
    for name, plan in document["strategies"].items():
        waypoints = plan["waypoints"]
        assert waypoints, name
        for waypoint in (waypoints[0], waypoints[len(waypoints) // 2], waypoints[-1]):
            assert set(waypoint) == WAYPOINT_KEYS
            assert waypoint["radio"] in {"clear", "degraded", "shadow"}
        times = [w["t_min"] for w in waypoints]
        oxygen = [w["o2_pct"] for w in waypoints]
        assert times == sorted(times) and oxygen == sorted(oxygen, reverse=True)
        assert waypoints[0]["o2_pct"] == 100.0 and waypoints[0]["dist_m"] == 0.0
        assert plan["limits"]["o2_tank_l"] == pytest.approx(384.9, abs=0.1)
        assert plan["limits"]["battery_wh"] == 700.0
    assert document["suit"]["body_mass_kg"] == 80.0


def test_walking_pace_in_the_plans_comes_from_the_model(planned) -> None:
    """Level firm stretches are walked at 3 km/h; no step exceeds it or the power cap."""
    from mars_map import biomechanics

    cap = biomechanics.SuitParams().power_cap_w()
    plan = planned["result"].plans["eco"]
    assert plan.segments
    fastest = 0.0
    for segment in plan.segments:
        profile = segment.profile
        for before, after in zip(profile, profile[1:]):
            minutes = after["t_min"] - before["t_min"]
            if minutes > 0.0:
                fastest = max(fastest, (after["dist_m"] - before["dist_m"]) / (minutes * 60.0))
    assert fastest == pytest.approx(3.0 / 3.6, rel=0.05)
    mean_speed_kmh = plan.total_distance_km / (sum(s.duration_min for s in plan.segments) / 60.0)
    assert 1.5 < mean_speed_kmh <= 3.0
    assert cap == pytest.approx(384.0, abs=2.0)


def test_plans_carry_the_simulator_verdict(planned) -> None:
    result, document = planned["result"], planned["document"]
    assert "physiology" in result.timings_s
    for name, plan in result.plans.items():
        verdict = plan.physiology
        assert verdict == document["strategies"][name]["physiology"]
        assert verdict["status"] in {"OK", "MARGINAL", "FAIL"}, verdict
        if not plan.visited_targets:
            continue
        for key in ("eva_time_h", "o2_remaining_pct", "battery_remaining_pct", "peak_fatigue",
                    "peak_heart_rate_bpm", "rest_min", "limiting_resource", "emergency_safety",
                    "radio", "difference_from_plan", "warnings", "message"):
            assert key in verdict, key
        assert 0.0 <= verdict["peak_fatigue"] <= 1.0
        assert 60 <= verdict["peak_heart_rate_bpm"] <= 190
        if verdict["status"] != "FAIL":
            # The planner and the simulator price the same walk with the same model.
            assert verdict["eva_time_h"] == pytest.approx(plan.total_time_hours, rel=0.12)
            assert verdict["o2_remaining_pct"] == pytest.approx(plan.o2_remaining_pct, abs=8.0)
            assert abs(verdict["difference_from_plan"]["distance_km"]) < 0.1


def test_document_reports_the_sol(planned) -> None:
    assessment = planned["document"]["conditions"]["sol_assessment"]
    assert assessment["sol"] == 0 and assessment["allowed"] is True and assessment["reasons"] == []
    assert assessment["best_sol"] is not None and 20 <= assessment["allowed_sols"] <= 30
    start, end = assessment["window_ltst_h"]
    assert 6.0 < start < 8.0 and 16.0 < end < 18.0


def test_particle_event_sol_is_flagged(planned) -> None:
    site, pipeline = planned["site"], planned["pipeline"]
    doses = site.series["radiation_msv_per_sol"]
    event_sol = site.series["sol"][int(np.argmax(doses))]
    result = pipeline.plan_site(site, sol=event_sol, use_environment=False,
                                strategies=["eco"], check_radio=False)
    assert result.conditions.particle_event
    assert result.sol_assessment is not None and result.sol_assessment["allowed"] is False
    assert "particle event" in result.sol_assessment["reasons"][0]
    verdict = result.plans["eco"].physiology
    assert verdict["status"] in {"FAIL", "OK"}          # FAIL when a walk was planned at all
    if result.plans["eco"].visited_targets:
        assert verdict["status"] == "FAIL" and verdict["broken_constraint"] == "spe_shelter"


def test_planning_without_the_check_and_without_an_environment(planned) -> None:
    site, pipeline = planned["site"], planned["pipeline"]
    result = pipeline.plan_site(site, sol=0, use_environment=False, strategies=["science"],
                                check_radio=False, check_physiology=False)
    plan = result.plans["science"]
    assert plan.physiology == {} and "physiology" not in result.timings_s
    assert result.to_dict()["strategies"]["science"]["physiology"] == {}
