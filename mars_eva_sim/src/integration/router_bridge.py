"""Sequential check of a finished plan of the routing team's planner.

The planner prices every step with the walking model (A2-A4) at a fixed fatigue
level. What builds up along a route cannot be priced per step: fatigue and the
rests it forces, stored heat, water, the oxygen needed to walk back from every
point, the reach of the emergency oxygen. ``PlanChecker`` walks the finished
plan through the sequential simulator and returns that verdict as a plain
dictionary in report units, which the planner stores in the plan under
``physiology`` and writes to its JSON.

``sol_assessment`` adds the verdict on the sol itself (H3) to the plan set.

The site, the ground model and the environment model are the other teams'
objects, duck-typed; this module does not import their package.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..comms.coverage import GridCoverage, coverage_from_radio_map, coverage_from_terrain
from ..core.config import DEFAULT_CONFIG, SimConfig
from ..core.constants import SOL_S, hz_to_bpm
from ..core.exceptions import MarsEvaError
from ..core.schemas import AstronautProfile, SuitProfile
from ..planning.diurnal import SeriesEnvironment
from ..planning.sol_selection import assess_sols, best_sol, sol_series_from_site
from ..providers.interfaces import EnvironmentProvider
from ..providers.route_adapter import route_from_mission_plan
from ..providers.site_adapter import SiteEnvironment, site_sols, terrain_from_site
from ..providers.synthetic import ConstantEnvironment
from ..reporting.map_layers import route_report_summary
from ..routing.route_evaluation import evaluate_route
from ..routing.shelter_maps import ShelterMaps, compute_shelter_maps
from ..routing.transition import EdgeEvaluator

MARS_HOUR_S = SOL_S / 24.0


class PlanChecker:
    """Prepared simulator for one site and one sol; checks any number of plans."""

    def __init__(self, site: Any, *, ground_model: Any | None = None,
                 environment_model: Any | None = None, sol: int | None = None,
                 start_ltst_h: float = 10.0, astronaut: AstronautProfile | None = None,
                 suit: SuitProfile | None = None, time_on_surface_s: float = 30.0 * SOL_S,
                 config: SimConfig = DEFAULT_CONFIG, radio_map: Any | None = None,
                 max_cells_per_side: int = 100) -> None:
        """
        Args:
            site: the ``Site`` the plans were made on.
            ground_model: the terrain team's ground model for the site.
            environment_model: the environment team's model; without it the weather
                comes from the site series, and without series from typical values.
            sol: sol of the EVA.
            start_ltst_h: local true solar time at which the crew leaves, Mars hours.
            astronaut, suit: profiles the plans were priced with.
            time_on_surface_s: time since landing.
            radio_map: the radio team's coverage map (0 shadow, 1 degraded, 2 clear);
                without it line of sight is computed from the terrain.
            max_cells_per_side: the site is merged into cells until it is no larger
                than this, which keeps the sequential check fast.
        """
        shape = np.asarray(site.layers["elevation"]).shape
        self.factor = max(1, math.ceil(max(shape) / max_cells_per_side))
        self.config = config
        self.terrain = terrain_from_site(site, downsample=self.factor, ground_model=ground_model)
        self.environment: EnvironmentProvider = self._environment(
            site, environment_model, sol, start_ltst_h)
        self.evaluator = EdgeEvaluator(self.terrain, self.environment,
                                       astronaut or AstronautProfile(), suit or SuitProfile(),
                                       config, time_on_surface_s=time_on_surface_s)
        self.base = self.terrain.to_cell(*site.start)
        self.coverage: GridCoverage = (
            coverage_from_radio_map(np.asarray(radio_map), self.terrain) if radio_map is not None
            else coverage_from_terrain(self.terrain, self.base, config.comms))
        self._maps: ShelterMaps | None = None

    def _environment(self, site: Any, model: Any | None, sol: int | None,
                     start_ltst_h: float) -> EnvironmentProvider:
        sols = site_sols(site)
        if sol is None and sols:
            sol = sols[0]
        if model is not None and sol is not None and getattr(model, "has_temperature", True):
            return SiteEnvironment(site, model, sol=sol, start_ltst_h=start_ltst_h,
                                   downsample=self.factor)
        if sol is not None and sols:
            return SeriesEnvironment.from_site(site, sol, start_ltst_h=start_ltst_h,
                                               cfg=self.config.sol_selection)
        return ConstantEnvironment()

    @property
    def maps(self) -> ShelterMaps:
        """Safety maps of the site, built on first use."""
        if self._maps is None:
            self._maps = compute_shelter_maps(self.evaluator, self.base, 0.0)
        return self._maps

    def check(self, plan: Any) -> dict[str, Any]:
        """Verdict of the simulator on a plan, in report units.

        Never raises for a plan the simulator cannot walk: the reason is returned
        in ``message`` with ``status`` "FAIL" (a constraint or an impassable step)
        or "UNKNOWN" (the plan could not be placed on the simulator grid).
        """
        if not getattr(plan, "segments", None) or not getattr(plan, "visited_targets", None):
            return {"status": "OK", "message": "no walk planned", "warnings": []}
        try:
            route = route_from_mission_plan(plan)
            report = evaluate_route(self.evaluator, route, start_time_s=0.0, maps=self.maps,
                                    coverage=self.coverage)
        except MarsEvaError as error:
            return {"status": "UNKNOWN", "message": str(error), "warnings": []}
        summary = route_report_summary(report)
        labels = report.final_label.chain()
        steps = [label.transition for label in labels if label.transition is not None]
        walking = [tr for tr in steps if tr.distance_m > 0.0]
        out: dict[str, Any] = {
            "status": summary["status"],
            "message": summary["message"],
            "broken_constraint": summary["broken_constraint"],
            "warnings": summary["warnings"],
            "eva_time_h": round(summary["eva_time_h"], 2),
            "distance_km": round(summary["distance_km"], 3),
            "o2_remaining_pct": round(summary["remaining"]["o2_pct"], 1),
            "battery_remaining_pct": round(summary["remaining"]["battery_pct"], 1),
            "limiting_resource": summary["limiting_resource"],
            "limiting_time_left_h": _rounded(summary["limiting_time_left_h"], 2),
            "final_fatigue": round(summary["final_fatigue"], 3),
            "peak_fatigue": round(max(label.fatigue for label in labels), 3),
            "peak_hand_fatigue": round(max(label.hand_fatigue for label in labels), 3),
            "peak_heart_rate_bpm": round(max((hz_to_bpm(tr.heart_rate_hz) for tr in steps),
                                             default=0.0)),
            "mean_walking_power_w": round(
                sum(tr.energy_j for tr in walking) / max(sum(tr.dt_s for tr in walking), 1e-9)),
            "rest_min": round(sum(leg["rest_min"] for leg in summary["legs"]), 1),
            "peak_heat_storage_kj": round(max(abs(label.heat_storage_j) for label in labels)
                                          / 1000.0, 1),
            "water_deficit_l": round(report.final_label.hydration_deficit_kg, 2),
            "emergency_safety": summary["safety_counts"],
            "radio": {k: (round(v, 1) if isinstance(v, float) else v)
                      for k, v in summary.get("radio", {}).items()},
            "difference_from_plan": {k: round(v, 3)
                                     for k, v in summary.get("difference_from_author", {}).items()},
            "grid_cell_m": self.terrain.cell_size_m,
        }
        return out


def _rounded(value: float, digits: int) -> float | None:
    return round(value, digits) if math.isfinite(value) else None


def sol_assessment(site: Any, sol: int | None,
                   config: SimConfig = DEFAULT_CONFIG) -> dict[str, Any] | None:
    """Verdict on the sol of the EVA (H3), or ``None`` when the site has no series.

    Returns:
        ``sol``, ``allowed``, ``reasons``, the ``best_sol`` of the series, the number
        of ``allowed_sols`` and the daylight ``window_ltst_h`` as [start, end] hours.
    """
    if not site_sols(site):
        return None
    cfg = config.sol_selection
    series = sol_series_from_site(site, cfg)
    results = assess_sols(series, cfg, latitude_rad=math.radians(float(site.center_lat)))
    if sol is None:
        sol = series.sols[0]
    chosen = next((a for a in results if a.sol == sol), None)
    if chosen is None:
        return None
    best = best_sol(results)
    window = chosen.window_s
    return {
        "sol": chosen.sol,
        "allowed": chosen.allowed,
        "reasons": list(chosen.reasons),
        "dose_msv": round(chosen.dose_sv * 1000.0, 3),
        "best_sol": None if best is None else best.sol,
        "allowed_sols": sum(1 for a in results if a.allowed),
        "window_ltst_h": None if window is None else [round(window[0] / MARS_HOUR_S, 2),
                                                      round(window[1] / MARS_HOUR_S, 2)],
    }
