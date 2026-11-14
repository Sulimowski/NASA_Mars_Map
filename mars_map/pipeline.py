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
(coarse graph) -> three plans. The first three are done once per site and sol;
replanning with other targets or strategies only repeats the last one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import mars_map  # noqa: F401  (run directly, e.g. PyCharm Run: make relative imports work)
    __package__ = "mars_map"

from .astronaut_router import STRATEGIES, EvaRouter, MissionPlan, ShelterRoute
from .astronaut_terrain import build_astronaut_terrain, uniform_dose_rate_msv_h
from .biomechanics import SuitParams
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

    def summary(self) -> dict:
        def stats(a, digits):
            return {"min": round(float(a.min()), digits), "median": round(float(np.median(a)), digits),
                    "max": round(float(a.max()), digits)}

        return {
            "sol": self.sol, "ltst_h": self.ltst_h,
            "ls_deg": None if self.ls_deg is None else round(self.ls_deg, 1),
            "particle_event": self.particle_event, "dust_tau": self.dust_tau,
            "radiation_source": self.radiation_source,
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
            "conditions": self.conditions.summary(),
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


def plan_site(site: Site, sol: int | None = None, ltst_h: float = 11.0,
              env: Environment | None = None, use_environment: bool = True,
              suit: SuitParams | None = None, strategies: list[str] | None = None,
              check_radio: bool = True, shelters: bool = False) -> EvaPlanSet:
    """Plan the EVA of a site for one sol.

    sol       -- a sol of site.series (default: the first one).
    ltst_h    -- local true solar time of the ground-temperature map, about mid-EVA.
    env       -- a prepared Environment, to reuse it between runs (it takes seconds to build).
    use_environment=False -- skip the terrain-following maps: uniform radiation, no temperature.
    shelters  -- also precompute an escape route from every stop of every plan.
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
    router = EvaRouter(site, terrain, suit)
    timings["router"] = clock() - t

    plans: dict[str, MissionPlan] = {}
    for name in strategies or list(STRATEGIES):
        t = clock()
        plans[name] = router.plan(name, check_radio=check_radio)
        timings[f"plan_{name}"] = clock() - t
        if conditions.particle_event:
            plans[name].warnings.insert(
                0, "Солнечная вспышка в этот сол: мощность дозы многократно выше обычной, "
                   "выход лучше перенести.")

    result = EvaPlanSet(site, conditions, router, plans, timings)
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
    print("\nвремя, с: " + ", ".join(f"{k} {v:.2f}" for k, v in result.timings_s.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan the EVA of a site: three alternative routes.")
    ap.add_argument("--site", default=None, help="site folder (default: the stub site)")
    ap.add_argument("--sol", type=int, default=None)
    ap.add_argument("--ltst", type=float, default=11.0, help="local solar time of the temperature map")
    ap.add_argument("--no-environment", action="store_true",
                    help="uniform radiation, no temperature (faster start)")
    ap.add_argument("--shelters", action="store_true", help="precompute escape routes from every stop")
    ap.add_argument("--out", default=None, help="output folder (default: the site folder)")
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
                       use_environment=not args.no_environment, shelters=args.shelters)
    print_report(result)

    far = max(site.targets, key=lambda t: np.hypot(t.x_m - site.start[0], t.y_m - site.start[1]),
              default=None)
    probe = (far.x_m, far.y_m) if far else tuple(site.start)
    esc = result.shelter(probe)
    print(f"\nукрытие из точки {probe}: {esc.segment.distance_m:.0f} м, {esc.eta_min:.1f} мин, "
          f"мощность дозы -{esc.reduction_pct:.0f}% ({esc.compute_s:.2f} с)")

    print(f"json    {result.to_json(out / 'eva_plans.json')}")
    print(f"preview {render(result, out / 'eva_plans.png', shelter_from=probe)}")


if __name__ == "__main__":
    main()
