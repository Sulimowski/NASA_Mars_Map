"""Input boundary for routes from other teams: loaders, rasterization, stops."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from src.core.config import SimConfig
from src.core.constants import O2_DENSITY_STP_KG_M3
from src.core.exceptions import ValidationError
from src.core.schemas import TaskType
from src.providers.route_adapter import (RouteInput, RouteStop, cell_of, polyline_length_m,
                                         rasterize_route, route_from_eva_plans,
                                         route_from_mission_plan, route_from_points,
                                         route_from_track, simplify_polyline, stop_indices)
from src.providers.synthetic import flat_terrain


def assert_chain(cells: list[tuple[int, int]]) -> None:
    for a, b in zip(cells, cells[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1, f"{a} -> {b} is not one step"


# ---- rasterization --------------------------------------------------------
def test_straight_and_diagonal_lines() -> None:
    terrain = flat_terrain(10, 10, 10.0)
    east = rasterize_route(terrain, [(5.0, 5.0), (95.0, 5.0)])
    assert east == [(0, c) for c in range(10)]
    diagonal = rasterize_route(terrain, [(5.0, 5.0), (95.0, 95.0)])
    assert diagonal == [(i, i) for i in range(10)]
    back = rasterize_route(terrain, [(95.0, 95.0), (5.0, 5.0)])
    assert back == diagonal[::-1]


def test_any_polyline_becomes_a_connected_chain() -> None:
    terrain = flat_terrain(40, 40, 5.0)
    points = [(12.0, 7.0), (150.0, 33.0), (171.0, 180.0), (20.0, 140.0), (12.0, 7.0)]
    cells = rasterize_route(terrain, points)
    assert_chain(cells)
    assert cells[0] == cells[-1] == cell_of(terrain, 12.0, 7.0)
    for x_m, y_m in points:
        assert cell_of(terrain, x_m, y_m) in cells


def test_dense_route_along_a_cell_boundary_does_not_zigzag() -> None:
    """A fine-grid route wobbling across a coarse cell boundary stays one straight chain."""
    terrain = flat_terrain(10, 40, 20.0)
    wobble = [(float(x), 100.0 + (0.8 if (x // 2) % 2 else -0.8)) for x in range(10, 790, 2)]
    cells = rasterize_route(terrain, wobble)
    assert_chain(cells)
    chain_m = 20.0 * sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(cells, cells[1:]))
    assert chain_m <= 1.03 * (780.0 - 10.0)
    assert len(simplify_polyline(wobble, 10.0)) == 2
    assert polyline_length_m([(0.0, 0.0), (3.0, 4.0), (3.0, 10.0)]) == pytest.approx(11.0)


def test_simplify_keeps_real_corners() -> None:
    corner = [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (100.0, 50.0), (100.0, 100.0)]
    assert simplify_polyline(corner, 5.0) == [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]
    assert simplify_polyline(corner[:2], 5.0) == corner[:2]
    assert simplify_polyline([(1.0, 1.0)] * 4, 5.0) == [(1.0, 1.0), (1.0, 1.0)]


def test_rasterize_rejects_bad_routes() -> None:
    terrain = flat_terrain(10, 10, 10.0)
    with pytest.raises(ValidationError):
        rasterize_route(terrain, [])
    with pytest.raises(ValidationError):
        rasterize_route(terrain, [(5.0, 5.0), (500.0, 5.0)])
    with pytest.raises(ValidationError):
        cell_of(terrain, -1.0, 5.0)


# ---- stops ----------------------------------------------------------------
def test_stops_snap_to_the_route_in_travel_order(cfg: SimConfig) -> None:
    terrain = flat_terrain(3, 20, 10.0)
    cells = rasterize_route(terrain, [(5.0, 15.0), (195.0, 15.0), (5.0, 15.0)])   # out and back
    first = RouteStop("A", 55.0, 15.0, 600.0, TaskType.SAMPLING)
    far = RouteStop("B", 195.0, 12.0, 300.0, TaskType.HAMMERING)
    late = RouteStop("C", 55.0, 15.0, 120.0, TaskType.OBSERVATION)        # same place, way back
    found = stop_indices(terrain, cells, [first, far, late], cfg.route_adapter)
    assert found == {5: (TaskType.SAMPLING, 600.0), 19: (TaskType.HAMMERING, 300.0),
                     33: (TaskType.OBSERVATION, 120.0)}
    merged = stop_indices(terrain, cells, [first, RouteStop("A2", 56.0, 14.0, 60.0)],
                          cfg.route_adapter)
    assert merged == {5: (TaskType.SAMPLING, 660.0)}
    with pytest.raises(ValidationError):
        stop_indices(terrain, cells, [RouteStop("lost", 100.0, 900.0, 60.0)], cfg.route_adapter)


# ---- loaders --------------------------------------------------------------
def test_points_accept_pairs_mappings_and_objects() -> None:
    route = route_from_points([(1.0, 2.0), (1.0, 2.0), {"x_m": 3, "y_m": 4},
                               SimpleNamespace(x_m=5.0, y_m=6.0)])
    assert route.points_m == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    assert route.stops == () and route.source == "points" and not route.reference
    track = route_from_track([SimpleNamespace(x_m=10.0, y_m=20.0, sol=3, sols_here=12),
                              SimpleNamespace(x_m=30.0, y_m=20.0, sol=15, sols_here=0)])
    assert track.points_m == ((10.0, 20.0), (30.0, 20.0)) and track.source == "track"


def test_route_input_is_validated() -> None:
    with pytest.raises(ValidationError):
        RouteInput(points_m=())
    with pytest.raises(ValidationError):
        RouteInput(points_m=((0.0, float("nan")),))
    with pytest.raises(ValidationError):
        RouteInput(points_m=((0.0, 0.0),), stops=(RouteStop("x", 0.0, 0.0, -5.0),))


def _fake_plan() -> SimpleNamespace:
    stops = [
        {"name": "Base", "kind": "start", "x_m": 0.0, "y_m": 0.0, "dwell_min": 0.0},
        {"name": "Outcrop", "kind": "target", "x_m": 100.0, "y_m": 0.0, "dwell_min": 30.0},
        {"name": "Base", "kind": "return", "x_m": 0.0, "y_m": 0.0, "dwell_min": 0.0},
    ]
    legs = [SimpleNamespace(points_m=[(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)]),
            SimpleNamespace(points_m=[(100.0, 0.0), (50.0, 0.0), (0.0, 0.0)])]
    return SimpleNamespace(segments=legs, stops=stops, strategy="eco", total_time_hours=1.5,
                           total_distance_km=0.2, total_dose_msv=0.04, total_o2_liters=100.0,
                           total_energy_wh=250.0)


def test_route_from_a_mission_plan_object() -> None:
    route = route_from_mission_plan(_fake_plan(), task_for=lambda name: TaskType.HAMMERING)
    assert route.points_m == ((0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (50.0, 0.0), (0.0, 0.0))
    assert route.stops == (RouteStop("Outcrop", 100.0, 0.0, 1800.0, TaskType.HAMMERING),)
    assert route.source == "mission_plan:eco"
    assert route.reference["time_s"] == pytest.approx(5400.0)
    assert route.reference["distance_m"] == pytest.approx(200.0)
    assert route.reference["dose_sv"] == pytest.approx(4.0e-5)
    assert route.reference["o2_kg"] == pytest.approx(0.1 * O2_DENSITY_STP_KG_M3)
    assert route.reference["energy_j"] == pytest.approx(250.0 * 3600.0)
    with pytest.raises(ValidationError):
        route_from_mission_plan(SimpleNamespace(segments=[]))


def test_route_from_an_eva_plans_document(tmp_path) -> None:
    document = {
        "schema": "eva-plan/1",
        "strategies": {"science": {
            "waypoints": [{"x_m": 0.0, "y_m": 0.0, "t_min": 0.0}, {"x_m": 60.0, "y_m": 0.0},
                          {"x_m": 0.0, "y_m": 0.0}],
            "stops": [{"name": "S", "kind": "target", "x_m": 60.0, "y_m": 0.0, "dwell_min": 10}],
            "summary": {"total_time_hours": 0.5, "total_distance_km": 0.12,
                        "total_dose_msv": 0.01, "o2_used_liters": 20.0, "energy_used_wh": 50.0},
        }, "empty": {"waypoints": []}},
    }
    route = route_from_eva_plans(document, "science")
    assert len(route.points_m) == 3 and route.stops[0].dwell_s == 600.0
    assert route.stops[0].task is TaskType.SAMPLING
    assert route.reference["time_s"] == pytest.approx(1800.0)
    path = tmp_path / "eva_plans.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert route_from_eva_plans(path, "science").points_m == route.points_m
    with pytest.raises(ValidationError):
        route_from_eva_plans(document, "shielded")
    with pytest.raises(ValidationError):
        route_from_eva_plans(document, "empty")
    with pytest.raises(ValidationError):
        route_from_eva_plans({"schema": "other/2", "strategies": {}}, "science")
    with pytest.raises(ValidationError):
        route_from_eva_plans(tmp_path / "missing.json", "science")
