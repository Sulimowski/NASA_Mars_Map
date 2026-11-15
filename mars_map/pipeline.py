"""One call from a Site to three EVA plans and the JSON the web frontend reads.

    from mars_map import load_stub
    from mars_map.pipeline import plan_site

    result = plan_site(load_stub(), sol=0)       # or Site.load("data/<real site>")
    result.plans["science"].visited_targets
    result.to_json("data/gale_stub/eva_plans.json")
    result.shelter((1200.0, 900.0))              # escape route during a particle event

From the command line:

    python -m mars_map.pipeline                  # stub site, sol 0
    python -m mars_map.pipeline --site data/gale --sol 3650 --ltst 10

Stages: Environment (radiation and temperature maps) -> ground model -> router
(coarse graph) -> three plans -> physiology check. The first three are done once
per site and sol; replanning with other targets or strategies only repeats the rest.

The physiology check walks every finished plan through the sequential astronaut
simulator (mars_eva_sim): fatigue and rests, stored heat, water, the oxygen to walk
back from every point, emergency reach, radio silence. Its verdict is stored in
plan.physiology and written to the JSON.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import mars_map  # noqa: F401  (run directly, e.g. PyCharm Run: make relative imports work)
    __package__ = "mars_map"

from mars_eva_sim.src.core.constants import SOL_S
from mars_eva_sim.src.integration.router_bridge import PlanChecker, sol_assessment

from .astronaut_router import STRATEGIES, EvaRouter, MissionPlan, ShelterRoute
from .astronaut_terrain import build_astronaut_terrain, uniform_dose_rate_msv_h
from .biomechanics import SuitParams
from .dust_devils import DevilHazard
from .environment import Environment
from .site import TERRAIN_CLASSES, Site, resolve_path

SCHEMA = "eva-plan/1"


@dataclass
class EvaConditions:
    """Weather the plans were made for."""
    sol: int | None                        # None: the site has no series
    ltst_h: float                          # local true solar time of the temperature map
    ls_deg: float | None
    particle_event: bool
    dust_tau: float | None
    dose_rate_msv_h: np.ndarray            # mSv per Earth hour, site.shape
    ground_temp_c: np.ndarray | None       # degrees C, site.shape
    radiation_source: str                  # "environment" (map) or "series" (uniform)
    start_h: float | None = None           # local solar hour the EVA starts
    start_reason: str = ""                 # why that hour

    def summary(self) -> dict:
        def stats(a, digits):
            return {"min": round(float(a.min()), digits), "median": round(float(np.median(a)), digits),
                    "max": round(float(a.max()), digits)}

        return {
            "sol": self.sol, "ltst_h": self.ltst_h,
            "ls_deg": None if self.ls_deg is None else round(self.ls_deg, 1),
            "particle_event": self.particle_event, "dust_tau": self.dust_tau,
            "radiation_source": self.radiation_source,
            "start_h": self.start_h, "start_reason": self.start_reason,
            "dose_rate_usv_h": stats(self.dose_rate_msv_h * 1000.0, 2),
            "ground_temp_c": None if self.ground_temp_c is None else stats(self.ground_temp_c, 1),
        }


def build_conditions(site: Site, env: Environment | None = None, sol: int | None = None,
                     ltst_h: float = 11.0) -> EvaConditions:
    """Radiation and temperature maps for a sol.

    With an Environment the maps follow the terrain; without one the radiation is
    the uniform series value and there is no temperature.
    """
    if env is None:
        doses = site.series.get("radiation_msv_per_sol") or []
        sols = site.series.get("sol") or list(range(len(doses)))
        index = sols.index(sol) if sol in sols else 0
        rate = np.full(site.shape, uniform_dose_rate_msv_h(site, index), np.float32)
        event = bool(doses) and doses[index] > 2.0 * float(np.median(doses))
        return EvaConditions(sols[index] if sols else None, ltst_h, None, event, None, rate, None,
                             "series")
    sols = env.sols
    if sol is None:
        sol = sols[0] if sols else 0
    return EvaConditions(
        sol=sol if sols else None, ltst_h=ltst_h, ls_deg=env.ls(sol),
        particle_event=env.is_particle_event(sol),
        dust_tau=env.dust_tau(sol) if "dust_opacity_tau" in site.series else None,
        dose_rate_msv_h=env.dose_rate(sol) / np.float32(1000.0),      # the map is in uSv/h
        ground_temp_c=env.temperature(sol, ltst_h) if env.has_temperature else None,
        radiation_source="environment")


@dataclass
class EvaPlanSet:
    """Everything one planning run produced."""
    site: Site
    conditions: EvaConditions
    router: EvaRouter
    plans: dict[str, MissionPlan]
    timings_s: dict[str, float] = field(default_factory=dict)
    emergency: dict[str, list[dict]] = field(default_factory=dict)
    sol_assessment: dict | None = None     # is this sol fit for an EVA, and which is best

    def shelter(self, current_pos_m: tuple[float, float], **kwargs) -> ShelterRoute:
        """Escape route to the nearest natural radiation shelter from any point."""
        return self.router.shelter(current_pos_m, **kwargs)

    def to_dict(self, waypoint_step_m: float = 6.0) -> dict:
        site, m = self.site, self.site.mission
        w, h = site.extent_m
        elev, sci = site.layers["elevation"], site.layers["science_score"]

        def point(x, y, **extra):
            lat, lon = site.to_latlon(x, y)
            return {"x_m": round(x, 1), "y_m": round(y, 1),
                    "elev_m": round(float(elev[site.to_pixel(x, y)]), 2),
                    "lat": round(lat, 6), "lon": round(lon, 6), **extra}

        plans = {}
        for name, plan in self.plans.items():
            plans[name] = plan.to_dict(waypoint_step_m)
            if name in self.emergency:
                plans[name]["emergency_shelters"] = self.emergency[name]
        return {
            "schema": SCHEMA,
            "units": {
                "x_m, y_m": "metres east / south of the north-west corner of the site",
                "elev_m": "metres, same datum as the elevation layer",
                "dist_m": "metres along the route from the start",
                "t_min": "minutes since leaving the airlock",
                "dose_rate_usv_h": "microsieverts per Earth hour",
                "dose_msv": "millisieverts accumulated since the start",
                "o2_pct, battery_pct": "percent left in the tank / battery",
                "looseness": "0 firm ground .. 1 loose sand or boulder field",
                "sinkage_cm": "boot sinkage, centimetres",
                "terrain_class": {str(k): v for k, v in TERRAIN_CLASSES.items()},
                "radio": "clear | degraded (Fresnel zone obstructed) | shadow (no line of sight)",
            },
            "site": {
                "name": site.name, "source": site.source,
                "resolution_m": site.resolution_m, "shape": list(site.shape),
                "extent_m": [w, h],
                "center_lat": site.center_lat, "center_lon": site.center_lon,
                "start": point(*site.start),
                "targets": [point(t.x_m, t.y_m, name=t.name, priority=t.priority,
                                  dwell_min=t.dwell_min, note=t.note,
                                  science_score=round(float(sci[site.to_pixel(t.x_m, t.y_m)]), 3))
                            for t in site.targets],
                "no_go": [[[round(x, 1), round(y, 1)] for x, y in poly] for poly in site.no_go],
            },
            "conditions": {**self.conditions.summary(), "sol_assessment": self.sol_assessment},
            "mission": asdict(m),
            "suit": asdict(self.router.suit),
            "strategies": plans,
            "comparison": [
                {"strategy": name, "title": p.title, "feasible": p.feasible,
                 "targets": p.visited_targets, "time_h": round(p.total_time_hours, 2),
                 "distance_km": round(p.total_distance_km, 2),
                 "dose_msv": round(p.total_dose_msv, 3),
                 "o2_used_l": round(p.total_o2_liters, 1),
                 "energy_used_wh": round(p.total_energy_wh, 1),
                 "science_value": round(p.science_value, 2)}
                for name, p in self.plans.items()],
            "timings_s": {k: round(v, 3) for k, v in self.timings_s.items()},
        }

    def to_json(self, path: str | Path, waypoint_step_m: float = 6.0) -> Path:
        out = Path(path)
        out.write_text(json.dumps(self.to_dict(waypoint_step_m), ensure_ascii=False, indent=1),
                       encoding="utf-8")
        return out


_STATUS_RANK = {"OK": 0, "MARGINAL": 1, "UNKNOWN": 2, "FAIL": 3}
_BLOCKED_STEP = re.compile(r"step into \((\d+), (\d+)\)")


def _repair(plan: MissionPlan, name: str, site: Site, suit: SuitParams, conditions: EvaConditions,
            devils, start_h: float, checker, check_radio: bool, rounds: int = 4) -> MissionPlan:
    """Close the cells where the astronaut simulator got stuck and plan again.

    Each round closes the cell the latest attempt got stuck in (closures add up), and the
    best attempt by the simulator's verdict is kept.
    """
    best, current, closed, seen = plan, plan, [], set()
    cell = float(checker.terrain.cell_size_m)
    for _ in range(rounds):
        status = current.physiology.get("status")
        msg = current.physiology.get("message", "")
        m = _BLOCKED_STEP.search(msg)
        if status != "FAIL" or not m or (int(m[1]), int(m[2])) in seen:
            break
        late = re.search(r"after ([\d.]+) km", msg)
        if " W " in msg and late and float(late[1]) > 0.6 * current.total_distance_km:
            break                       # a tired crew member late in the walk: shorten it instead (below)
        r, c = int(m[1]), int(m[2])
        seen.add((r, c))
        closed.append([(c * cell, r * cell), ((c + 1) * cell, r * cell),
                       ((c + 1) * cell, (r + 1) * cell), (c * cell, (r + 1) * cell)])
        trial_site = copy.copy(site)
        trial_site.no_go = list(site.no_go) + closed
        terrain = build_astronaut_terrain(trial_site, suit=suit, dose_rate_msv_h=conditions.dose_rate_msv_h,
                                          ground_temp_c=conditions.ground_temp_c)
        router = EvaRouter(trial_site, terrain, suit, devils=devils, start_h=start_h)
        try:
            trial = router.plan(name, check_radio=check_radio)
        except RuntimeError:
            break
        trial.physiology = checker.check(trial)
        if _STATUS_RANK.get(trial.physiology.get("status"), 3) < _STATUS_RANK.get(best.physiology.get("status"), 3):
            trial.warnings.append(f"Rerouted around {len(closed)} spot(s) the astronaut simulator "
                                  "could not walk (steep for a tired crew member, or too narrow).")
            best = trial
        current = trial

    # Still failing on power: the walk is too long for the crew member, not one bad climb.
    # Drop the least valuable stop (lowest priority, then lowest science) and plan again.
    targets = list(site.targets)
    for _ in range(2):
        msg = best.physiology.get("message", "")
        if best.physiology.get("status") != "FAIL" or " W " not in msg or len(best.visited_targets) < 2:
            break
        visited = [t for t in targets if t.name in best.visited_targets]
        sci = site.layers["science_score"]
        drop = max(visited, key=lambda t: (t.priority, -float(sci[site.to_pixel(t.x_m, t.y_m)])))
        if drop.priority == 1:
            break                                       # never drop a must-visit stop
        targets = [t for t in targets if t is not drop]
        terrain = build_astronaut_terrain(site, suit=suit, dose_rate_msv_h=conditions.dose_rate_msv_h,
                                          ground_temp_c=conditions.ground_temp_c)
        router = EvaRouter(site, terrain, suit, devils=devils, start_h=start_h)
        try:
            trial = router.plan(name, targets=targets, check_radio=check_radio)
        except RuntimeError:
            break
        trial.physiology = checker.check(trial)
        if _STATUS_RANK.get(trial.physiology.get("status"), 3) < _STATUS_RANK.get(best.physiology.get("status"), 3):
            trial.warnings.append(f"Left out {drop.name}: with it the walk is too long for the crew "
                                  "member (the astronaut simulator fails a climb late in the EVA).")
            best = trial
    return best


def plan_site(site: Site, sol: int | None = None, ltst_h: float = 11.0,
              env: Environment | None = None, use_environment: bool = True,
              suit: SuitParams | None = None, strategies: list[str] | None = None,
              check_radio: bool = True, shelters: bool = False,
              check_physiology: bool = True, start_h: float | None = None,
              dust_devils: bool = True) -> EvaPlanSet:
    """Plan the EVA of a site for one sol.

    sol       -- a sol of site.series (default: the first one).
    ltst_h    -- local true solar time of the ground-temperature map, about mid-EVA.
    start_h   -- local solar hour the EVA starts. None: chosen to keep dust devil encounters
                 low (a first plan is timed at every start; among the starts within 10 % of
                 the lowest risk, the one nearest 09:30 wins). Without a forecast: ltst_h - 1.
    dust_devils -- route around dust devil risk when the site has a forecast.
    env       -- a prepared Environment, to reuse it between runs (it takes seconds to build).
    use_environment=False -- skip the terrain-following maps: uniform radiation, no temperature.
    shelters  -- also precompute an escape route from every stop of every plan.
    check_physiology -- walk every plan through the astronaut simulator (a few seconds).
    """
    timings: dict[str, float] = {}
    clock = time.perf_counter

    t = clock()
    if env is None and use_environment:
        env = Environment(site)
    timings["environment_build"] = clock() - t

    t = clock()
    conditions = build_conditions(site, env, sol, ltst_h)
    timings["environment_maps"] = clock() - t

    t = clock()
    suit = suit or SuitParams()
    terrain = build_astronaut_terrain(site, suit=suit, dose_rate_msv_h=conditions.dose_rate_msv_h,
                                      ground_temp_c=conditions.ground_temp_c)
    timings["terrain"] = clock() - t

    t = clock()
    devils = DevilHazard.for_site(site, env, conditions.sol) if (dust_devils and env is not None) else None
    router = EvaRouter(site, terrain, suit, devils=devils,
                       start_h=start_h if start_h is not None else max(0.0, ltst_h - 1.0))
    timings["router"] = clock() - t

    def checker_at(hour: float) -> PlanChecker:
        return PlanChecker(
            site, ground_model=terrain, environment_model=env, sol=conditions.sol,
            start_ltst_h=hour, astronaut=suit.astronaut(), suit=suit.suit(),
            time_on_surface_s=suit.sols_on_surface * SOL_S, config=suit.config)

    conditions.start_h, conditions.start_reason = router.start_h, "given" if start_h is not None else "default"
    if start_h is None and devils is not None:
        # When to go out. A first plan is timed at every start hour; starts are ranked by
        # dust devil risk (within 10 % counts as equal, then nearer 09:30 is better), and the
        # first one the astronaut simulator does not fail is taken (the heat of the day
        # changes how hard the same climb is).
        t = clock()
        default_h = router.start_h
        probe = router.plan("science", check_radio=False)
        by = [(float(h), float(e)) for h, e in probe.dust_devils.get("by_start") or []]
        if by:
            low = max(min(e for _, e in by), 1e-6)
            ranked = sorted(by, key=lambda he: (round(he[1] / low, 1), abs(he[0] - 9.5)))
            chosen = None
            for h, e in ranked[:5]:
                # a single climb the simulator cannot walk is repaired later (see _repair);
                # any other failure (heat, oxygen, water) rules the start hour out
                verdict = checker_at(h).check(probe) if check_physiology else {}
                if verdict.get("status") != "FAIL" or _BLOCKED_STEP.search(verdict.get("message", "")):
                    chosen = (h, e)
                    break
            at_default = probe.dust_devils["expected"]
            if chosen is None:
                conditions.start_reason = "default: every low-risk start fails the physiology check"
            else:
                router.start_h = chosen[0]
                conditions.start_reason = (f"dust devils: {chosen[1]:.3f} expected encounters "
                                           f"against {at_default:.3f} at {default_h:.1f} h")
            conditions.start_h = router.start_h
        timings["start_time"] = clock() - t

    plans: dict[str, MissionPlan] = {}
    for name in strategies or list(STRATEGIES):
        t = clock()
        plans[name] = router.plan(name, check_radio=check_radio)
        timings[f"plan_{name}"] = clock() - t
        if conditions.particle_event:
            plans[name].warnings.insert(
                0, "Солнечная вспышка в этот сол: мощность дозы многократно выше обычной, "
                   "выход лучше перенести.")

    result = EvaPlanSet(site, conditions, router, plans, timings,
                        sol_assessment=sol_assessment(site, conditions.sol, suit.config))
    if check_physiology:
        t = clock()
        checker = checker_at(router.start_h)
        for plan in plans.values():
            plan.physiology = checker.check(plan)
        timings["physiology"] = clock() - t
        # Plan, check, repair. The simulator walks the plan on its own coarser grid with
        # fatigue and heat building up; where it finds a climb the crew member cannot hold,
        # that cell is closed and the plan is made again (twice at most). A repaired plan
        # replaces the first one only if the simulator likes it better.
        t = clock()
        for name in list(plans):
            plans[name] = _repair(plans[name], name, site, suit, conditions, devils, router.start_h,
                                  checker, check_radio)
        timings["physiology_repair"] = clock() - t
    if shelters:
        t = clock()
        for name, plan in plans.items():
            result.emergency[name] = [
                {"from": stop["name"], **router.shelter((stop["x_m"], stop["y_m"])).to_dict()}
                for stop in plan.stops if stop["kind"] == "target"]
        timings["shelters"] = clock() - t
    timings["planning_total"] = sum(v for k, v in timings.items()
                                    if not k.startswith("environment"))
    return result


# ----------------------------------------------------------------------------
# Preview picture and command line
# ----------------------------------------------------------------------------
COLORS = {"shielded": "#22d3ee", "eco": "#84cc16", "science": "#e11d48"}


def render(result: EvaPlanSet, out_path: str | Path, shelter_from: tuple[float, float] | None = None
           ) -> Path:
    """Routes of all plans over the looseness, dose-rate and elevation maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    site, router = result.site, result.router
    w, h = site.extent_m
    extent = (0, w, h, 0)
    fig, axes = plt.subplots(2, 2, figsize=(17, 15))

    def base(ax, data, title, cmap, **kw):
        im = ax.imshow(data, cmap=cmap, extent=extent, **kw)
        fig.colorbar(im, ax=ax, shrink=0.8)
        for poly in site.no_go:
            xs, ys = zip(*(poly + poly[:1]))
            ax.plot(xs, ys, color="red", lw=1.2, ls="--")
        ax.set_title(title)
        ax.set_xlabel("x, m (east)")
        ax.set_ylabel("y, m (south)")

    def routes(ax, lw=2.0):
        for k, (name, plan) in enumerate(result.plans.items()):
            for i, seg in enumerate(plan.segments):
                xs, ys = zip(*seg.points_m)
                ax.plot(xs, ys, color=COLORS.get(name, "white"), lw=lw + 1.6 - 0.8 * k,
                        alpha=0.9, label=name if i == 0 else None)
        ax.plot(*site.start, marker="^", color="white", mec="k", ms=11)
        for t in site.targets:
            ax.plot(t.x_m, t.y_m, marker="*", color="yellow", mec="k", ms=13)
            ax.annotate(f"{t.name} (P{t.priority})", (t.x_m, t.y_m), xytext=(6, 6),
                        textcoords="offset points", fontsize=7, color="k",
                        bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))

    base(axes[0, 0], router.terrain.looseness, "Soil looseness and the three routes", "YlOrBr",
         vmin=0, vmax=1)
    routes(axes[0, 0])
    axes[0, 0].legend(loc="lower left", fontsize=9)

    rate = result.conditions.dose_rate_msv_h * 1000.0
    base(axes[0, 1], rate, "Dose rate, µSv/h, and an escape route to shelter", "inferno")
    routes(axes[0, 1], lw=1.2)
    if shelter_from is not None:
        esc = result.shelter(shelter_from)
        xs, ys = zip(*esc.segment.points_m)
        axes[0, 1].plot(xs, ys, color="white", lw=2.5, ls=":")
        axes[0, 1].plot(*shelter_from, marker="o", color="white", mec="k", ms=8)
        axes[0, 1].plot(*esc.shelter_m, marker="P", color="#22c55e", mec="k", ms=13)

    impassable = np.ma.masked_where(~router.terrain.impassable, router.terrain.impassable)
    base(axes[1, 0], site.layers["elevation"], "Elevation, m; black = impassable; "
         "x = radio shadow, v = relay drop point", "terrain")
    axes[1, 0].imshow(impassable, cmap="gray_r", extent=extent, alpha=0.9, vmin=0, vmax=1)
    routes(axes[1, 0], lw=1.2)
    for name, plan in result.plans.items():
        for zone in plan.radio.get("shadow_zones", []):
            axes[1, 0].plot(*zip(zone["from"], zone["to"]), color="k", lw=0, marker="x", ms=6)
        for relay in plan.radio.get("relay_points", []):
            axes[1, 0].plot(relay["x_m"], relay["y_m"], marker="v", color=COLORS.get(name, "w"),
                            mec="k", ms=9)

    ax = axes[1, 1]
    for name, plan in result.plans.items():
        pts = plan.waypoints(step_m=4.0)
        ax.plot([p["t_min"] / 60 for p in pts], [p["elev_m"] for p in pts],
                color=COLORS.get(name, "gray"), lw=2,
                label=f"{name}: {len(plan.visited_targets)} targets, {plan.total_time_hours:.2f} h, "
                      f"{plan.total_distance_km:.2f} km, {plan.total_dose_msv * 1000:.0f} µSv, "
                      f"O2 {plan.o2_remaining_pct:.0f}% left")
    usable = site.mission.eva_max_hours * (1 - site.mission.return_reserve_frac)
    ax.axvline(usable, color="red", ls="--", lw=1)
    ax.text(usable, ax.get_ylim()[0], " usable EVA time", color="red", fontsize=8, va="bottom")
    ax.set_xlabel("Time since start, h (flat pieces = work at a target)")
    ax.set_ylabel("Elevation, m")
    ax.set_title("Elevation along each plan")
    ax.legend(loc="upper left", fontsize=8)

    c = result.conditions
    fig.suptitle(f"{site.name} — EVA plans for sol {c.sol}"
                 f"{'  (SOLAR PARTICLE EVENT)' if c.particle_event else ''}  |  "
                 f"radiation: {c.radiation_source}  |  planned in "
                 f"{result.timings_s.get('planning_total', 0):.2f} s", fontsize=13)
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=95)
    plt.close(fig)
    return out


def print_report(result: EvaPlanSet) -> None:
    c = result.conditions.summary()
    print(f"site {result.site.name}: sol {c['sol']}, Ls {c['ls_deg']}, "
          f"particle event: {c['particle_event']}, dose rate {c['dose_rate_usv_h']} µSv/h")
    for name, p in result.plans.items():
        print(f"\n[{name}] {p.title} | {'OK' if p.feasible else 'НЕ УКЛАДЫВАЕТСЯ'}")
        print(f"  цели: {p.visited_targets or '—'}")
        print(f"  {p.total_distance_km:.2f} км, {p.total_time_hours:.2f} ч, набор +{p.total_elevation_gain_m:.0f} м, "
              f"макс. уклон {p.max_slope_deg:.1f}°")
        print(f"  O2 осталось {p.o2_remaining_pct:.0f}%, батарея {p.battery_remaining_pct:.0f}%, "
              f"доза {p.total_dose_msv * 1000:.1f} мкЗв, научная ценность {p.science_value:.1f}")
        for s in p.skipped_targets:
            print(f"  пропущена: {s['name']} (P{s['priority']}) — {s['reason']}")
        for wmsg in p.warnings:
            print(f"  ! {wmsg}")
        ph = p.physiology
        if ph.get("eva_time_h") is not None:
            print(f"  физиология: {ph['status']} — {ph['message']}; усталость до "
                  f"{ph['peak_fatigue']:.2f}, пульс до {ph['peak_heart_rate_bpm']}, отдых "
                  f"{ph['rest_min']:.0f} мин, первым кончится {ph['limiting_resource']}")
            for wmsg in ph["warnings"]:
                print(f"  ! {wmsg}")
        elif ph:
            print(f"  физиология: {ph['status']} — {ph['message']}")
    print("\nвремя, с: " + ", ".join(f"{k} {v:.2f}" for k, v in result.timings_s.items()))


def main() -> None:
    # The report is printed in Russian; a Windows console in a legacy code page turns
    # it into question marks unless the stream is told to write UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Plan the EVA of a site: three alternative routes.")
    ap.add_argument("--site", default=None, help="site folder (default: the stub site)")
    ap.add_argument("--sol", type=int, default=None)
    ap.add_argument("--ltst", type=float, default=11.0, help="local solar time of the temperature map")
    ap.add_argument("--no-environment", action="store_true",
                    help="uniform radiation, no temperature (faster start)")
    ap.add_argument("--shelters", action="store_true", help="precompute escape routes from every stop")
    ap.add_argument("--out", default=None, help="output folder (default: the site folder)")
    ap.add_argument("--start", type=float, default=None,
                    help="local solar hour the EVA starts (default: chosen for low dust devil risk)")
    ap.add_argument("--no-dust-devils", action="store_true", help="ignore the dust devil forecast")
    args = ap.parse_args()

    if args.site is None:
        from . import STUB_DIR, load_stub
        site, folder = load_stub(), STUB_DIR
    else:
        folder = resolve_path(args.site)
        site = Site.load(folder)
    out = resolve_path(args.out) if args.out else folder
    out.mkdir(parents=True, exist_ok=True)

    result = plan_site(site, sol=args.sol, ltst_h=args.ltst,
                       use_environment=not args.no_environment, shelters=args.shelters,
                       start_h=args.start, dust_devils=not args.no_dust_devils)
    print_report(result)

    far = max(site.targets, key=lambda t: np.hypot(t.x_m - site.start[0], t.y_m - site.start[1]),
              default=None)
    probe = (far.x_m, far.y_m) if far else tuple(site.start)
    try:
        esc = result.shelter(probe)
        print(f"\nукрытие из точки {probe}: {esc.segment.distance_m:.0f} м, {esc.eta_min:.1f} мин, "
              f"мощность дозы -{esc.reduction_pct:.0f}% ({esc.compute_s:.2f} с)")
    except RuntimeError as e:                 # the demo probe must not cost us the plans
        print(f"\nукрытие из точки {probe}: {e}")
        probe = None

    print(f"json    {result.to_json(out / 'eva_plans.json')}")
    print(f"preview {render(result, out / 'eva_plans.png', shelter_from=probe)}")


if __name__ == "__main__":
    main()
