"""Route profile and raster layers in the units a map or a report shows.

This is the output boundary of the simulator. Inside, everything is SI; here
values become km, km/h, minutes, degrees, percent, grams, Wh, bpm, mSv.

    route_profile      one row per step of a route: where, when, how fast, what is left
    plan_summary       totals and verdict of a mission plan
    route_report_summary   verdict on an externally planned route
    build_map_layers   rasters: cost per km, safety times, safety class, reachability, contact
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..comms.coverage import GridCoverage
from ..core.constants import HOUR_S, MINUTE_S, hz_to_bpm, ms_to_kmh
from ..core.schemas import Label
from ..physiology import emergency
from ..routing.mission_planner import LegSummary, MissionPlan, cooling_tolerance_s
from ..routing.reachability import ReachabilityResult
from ..routing.route_evaluation import RouteReport
from ..routing.shelter_maps import ShelterMaps
from ..routing.transition import EdgeEvaluator


@dataclass(frozen=True, slots=True)
class ProfileRow:
    """State after one step of a route, in report units."""

    index: int
    row: int
    col: int
    action: str
    elapsed_min: float
    distance_km: float
    speed_kmh: float
    grade_pct: float
    slope_deg: float
    metabolic_w: float
    heart_rate_bpm: float
    o2_g: float
    co2_capacity_g: float
    cooling_water_kg: float
    battery_wh: float
    eva_remaining_min: float
    fatigue: float
    hand_fatigue: float
    hydration_deficit_l: float
    heat_storage_kj: float
    dose_msv: float
    risk: float

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary, ready for JSON."""
        return asdict(self)


def route_profile(final_label: Label) -> list[ProfileRow]:
    """Step-by-step profile of the route that ends at ``final_label``."""
    chain = final_label.chain()
    t0 = chain[0].t_s
    rows: list[ProfileRow] = []
    for index, label in enumerate(chain):
        tr = label.transition
        rows.append(ProfileRow(
            index=index, row=label.row, col=label.col, action=label.action,
            elapsed_min=(label.t_s - t0) / MINUTE_S,
            distance_km=label.distance_m / 1000.0,
            speed_kmh=ms_to_kmh(tr.speed_m_s) if tr else 0.0,
            grade_pct=100.0 * tr.grade if tr else 0.0,
            slope_deg=math.degrees(tr.slope_rad) if tr else 0.0,
            metabolic_w=tr.metabolic_w if tr else 0.0,
            heart_rate_bpm=hz_to_bpm(tr.heart_rate_hz) if tr else 0.0,
            o2_g=1000.0 * label.o2_kg,
            co2_capacity_g=1000.0 * label.co2_capacity_kg,
            cooling_water_kg=label.cooling_water_kg,
            battery_wh=label.battery_j / HOUR_S,
            eva_remaining_min=label.eva_remaining_s / MINUTE_S,
            fatigue=label.fatigue, hand_fatigue=label.hand_fatigue,
            hydration_deficit_l=label.hydration_deficit_kg,
            heat_storage_kj=label.heat_storage_j / 1000.0,
            dose_msv=1000.0 * label.dose_sv, risk=label.risk))
    return rows


def _leg_dict(leg: LegSummary) -> dict[str, Any]:
    return {
        "name": leg.name,
        "length_km": leg.distance_m / 1000.0,
        "time_min": leg.duration_s / MINUTE_S,
        "elevation_gain_m": leg.elevation_gain_m,
        "elevation_loss_m": leg.elevation_loss_m,
        "max_slope_deg": math.degrees(leg.max_slope_rad),
        "o2_g": 1000.0 * leg.o2_kg,
        "energy_kj": leg.energy_j / 1000.0,
        "dose_msv": 1000.0 * leg.dose_sv,
        "rest_min": leg.rest_s / MINUTE_S,
    }


def route_report_summary(report: RouteReport) -> dict[str, Any]:
    """Verdict on an externally planned route, in report units.

    ``difference_from_author`` is the simulator total minus the total the author
    of the route claimed (positive = the simulator expects more).
    """
    start, final = report.start_label, report.final_label
    summary: dict[str, Any] = {
        "status": report.status.value,
        "completed": report.completed,
        "message": report.message,
        "broken_constraint": report.rule.value if report.rule else None,
        "source": report.source,
        "warnings": list(report.warnings),
        "eva_time_h": report.duration_s / HOUR_S,
        "distance_km": report.distance_m / 1000.0,
        "dose_msv": 1000.0 * (final.dose_sv - start.dose_sv),
        "o2_used_g": 1000.0 * (start.o2_kg - final.o2_kg),
        "battery_used_wh": (start.battery_j - final.battery_j) / HOUR_S,
        "final_fatigue": final.fatigue,
        "remaining": {
            "o2_pct": 100.0 * final.o2_kg / start.o2_kg if start.o2_kg else 0.0,
            "battery_pct": 100.0 * final.battery_j / start.battery_j if start.battery_j else 0.0,
            "eva_time_min": final.eva_remaining_s / MINUTE_S,
        },
        "limiting_resource": report.limiting_resource,
        "limiting_time_left_h": report.limiting_time_left_s / HOUR_S,
        "safety_counts": dict(report.safety_counts),
        "legs": [_leg_dict(leg) for leg in report.legs],
    }
    if report.contact is not None:
        summary["radio"] = {
            "no_contact_min": report.contact.no_contact_s / MINUTE_S,
            "longest_gap_min": report.contact.longest_gap_s / MINUTE_S,
            "contact_share_pct": 100.0 * report.contact.contact_share,
            "gaps": len(report.contact.gaps),
            "over_limit": report.contact.violates,
        }
    delta = report.reference_delta
    if delta:
        scale = {"time_s": ("time_min", 1.0 / MINUTE_S), "distance_m": ("distance_km", 1.0e-3),
                 "o2_kg": ("o2_g", 1.0e3), "dose_sv": ("dose_msv", 1.0e3),
                 "energy_j": ("battery_wh", 1.0 / HOUR_S)}
        summary["difference_from_author"] = {
            scale[key][0]: value * scale[key][1] for key, value in delta.items() if key in scale}
    return summary


def plan_summary(plan: MissionPlan) -> dict[str, Any]:
    """Totals and verdict of a mission plan, in report units."""
    summary: dict[str, Any] = {
        "status": plan.status.value,
        "reasons": list(plan.reasons),
        "warnings": list(plan.warnings),
        "prebreathe_min": plan.prebreathe_s / MINUTE_S,
        "eva_time_h": plan.eva_duration_s / HOUR_S,
        "cleaning_min": plan.cleaning_s / MINUTE_S,
        "distance_km": plan.total_distance_m / 1000.0,
        "task_completed_min": plan.task_completed_s / MINUTE_S,
        "limiting_resource": plan.limiting_resource,
        "limiting_time_left_h": plan.limiting_time_left_s / HOUR_S,
        "perchlorate_flag": plan.perchlorate_flag,
        "safety_counts": dict(plan.safety_counts),
        "legs": [_leg_dict(leg) for leg in plan.legs],
    }
    final, start = plan.final_label, plan.start_label
    if final is not None and start is not None:
        summary["remaining"] = {
            "o2_pct": 100.0 * final.o2_kg / start.o2_kg if start.o2_kg else 0.0,
            "battery_pct": 100.0 * final.battery_j / start.battery_j if start.battery_j else 0.0,
            "co2_scrubber_pct": (100.0 * final.co2_capacity_kg / start.co2_capacity_kg
                                 if start.co2_capacity_kg else 0.0),
            "cooling_water_pct": (100.0 * final.cooling_water_kg / start.cooling_water_kg
                                  if start.cooling_water_kg else 0.0),
            "eva_time_min": final.eva_remaining_s / MINUTE_S,
        }
        summary["dose_msv"] = 1000.0 * (final.dose_sv - start.dose_sv)
        summary["final_fatigue"] = final.fatigue
        summary["peak_heat_storage_kj"] = max(
            abs(label.heat_storage_j) for label in final.chain()) / 1000.0
    return summary


def _mean_per_km(cube: NDArray[np.float64], length_m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mean over the passable directions of ``cube / length`` per kilometre; NaN where none."""
    finite = np.isfinite(cube) & np.isfinite(length_m) & (length_m > 0.0)
    ratio = np.zeros_like(cube)
    np.divide(cube, length_m, out=ratio, where=finite)
    count = finite.sum(axis=0)
    total = np.where(finite, ratio, 0.0).sum(axis=0)
    out = np.full(count.shape, np.nan)
    np.divide(total * 1000.0, count, out=out, where=count > 0)
    return out


def build_map_layers(evaluator: EdgeEvaluator, maps: ShelterMaps, t_s: float,
                     reach: ReachabilityResult | None = None,
                     coverage: GridCoverage | None = None) -> dict[str, NDArray[Any]]:
    """Raster layers for the map, each ``[rows, cols]``.

    Layers:
        time_min_per_km, o2_g_per_km, energy_kj_per_km, risk_per_km
            mean cost of leaving the cell, over the passable directions (NaN if none)
        return_time_min, return_o2_g, shelter_time_min, emergency_time_min
            cost of reaching safety (inf where unreachable)
        emergency_safety
            ``SafetyFlag`` value per cell (0 unsafe, 1 marginal, 2 safe)
        arrival_time_min, round_trip_ok, isochrone_min, point_of_no_return
            only when ``reach`` is given; ``isochrone_min`` holds the smallest
            isochrone containing the cell, NaN outside all of them
        radio_contact
            only when ``coverage`` is given; True where the base can be reached

    Args:
        t_s: mission time used for the cooling-failure tolerance.
    """
    cubes = maps.nominal
    tolerance = cooling_tolerance_s(evaluator, maps.base, t_s)
    layers: dict[str, NDArray[Any]] = {
        "time_min_per_km": _mean_per_km(cubes.time_s, cubes.length_m) / MINUTE_S,
        "o2_g_per_km": _mean_per_km(cubes.o2_kg, cubes.length_m) * 1000.0,
        "energy_kj_per_km": _mean_per_km(cubes.energy_j, cubes.length_m) / 1000.0,
        "risk_per_km": _mean_per_km(cubes.risk, cubes.length_m),
        "return_time_min": maps.return_time_s / MINUTE_S,
        "return_o2_g": maps.return_o2_kg * 1000.0,
        "shelter_time_min": maps.shelter_time_s / MINUTE_S,
        "emergency_time_min": maps.emergency_time_s / MINUTE_S,
        "emergency_safety": emergency.emergency_flag_map(
            maps.emergency_time_s, maps.shelter_time_s, tolerance, evaluator.suit,
            evaluator.config.emergency),
    }
    if reach is not None:
        isochrone = np.full(reach.arrival_time_s.shape, np.nan)
        for limit in sorted(reach.isochrones, reverse=True):
            isochrone[reach.isochrones[limit]] = limit / MINUTE_S
        layers["arrival_time_min"] = reach.arrival_time_s / MINUTE_S
        layers["round_trip_ok"] = reach.round_trip_ok
        layers["isochrone_min"] = isochrone
        layers["point_of_no_return"] = reach.point_of_no_return
    if coverage is not None:
        layers["radio_contact"] = coverage.mask
    return layers
