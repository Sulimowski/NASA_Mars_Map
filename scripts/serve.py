"""The app server: the pages, plus live EVA planning for points chosen on the map.

    .venv\\Scripts\\python scripts\\serve.py              # http://127.0.0.1:8770/gui/
    .venv\\Scripts\\python scripts\\serve.py --port 8771

It serves the project folder like `python -m http.server` (so `gui/` and
`data/<site>/view3d.html` open as before) and adds two addresses:

    GET  /api/health                  {"ok": true, "sites": [...]}
    POST /api/plan                    plan a walk for a base and points chosen by the user
    POST /api/reroute                 during a walk: a new route from where the astronaut is,
                                      around an area to avoid (a dust devil), to the rest of
                                      the stops and back to the base

`/api/plan` takes JSON:

    {"site": "gale_boxwork",
     "start": [x_m, y_m],
     "targets": [{"name": "Point 1", "x": x_m, "y": y_m, "priority": 1, "dwell": 20}, ...],
     "sol": 4650,                      # optional, default: the first sol the site has
     "vehicles": ["astronaut", "rover"]}   # optional, default: both

Every vehicle asked for is planned for the same base and points: `astronaut` is a walk
(three plans, with the physiology check), `rover` is a drive (two plans, wheel and battery
model, no physiology check). The answer has one set of plans per vehicle under `modes`.

and answers with the three plans in the shape the page draws (`mars_map/viewer/plans.py`),
the points as the planner used them (moved to walkable ground if they were on blocked
cells) and notes about what was moved. Planning takes 5-20 s; the first request for a
site is slower because the site and its radiation / temperature model are loaded once
and kept in memory.

`/api/reroute` takes JSON:

    {"site": "gale_boxwork", "sol": 4650, "strategy": "science",
     "from": [x_m, y_m],                              # where the astronaut is now
     "via": [{"name": "Point 2", "x": x_m, "y": y_m}, ...],   # stops still to visit, in order
     "base": [x_m, y_m],
     "avoid": [{"x1": .., "y1": .., "x2": .., "y2": .., "r": 80}, ...]}   # corridors to keep out of

Each `avoid` entry is a corridor: everything within `r` metres of the segment (x1, y1)-(x2, y2),
which is the track a dust devil is expected to sweep. It answers with the route as points,
its length and time, and the names of stops it had to drop because the corridor cuts them
off. It keeps the order of the stops and does not run the physiology check, so it takes
about a second: it is meant to be called while the walk is under way.

Only binds to this computer unless `--bind 0.0.0.0` is given.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import threading
import time
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mars_map import DATA_DIR, Site, Target  # noqa: E402
from mars_map.environment import Environment  # noqa: E402
from mars_map.astronaut_router import STRATEGIES, EvaRouter  # noqa: E402
from mars_map.astronaut_terrain import build_astronaut_terrain  # noqa: E402
from mars_map.biomechanics import SuitParams  # noqa: E402
from mars_map.pipeline import build_conditions, plan_site  # noqa: E402
from mars_map.rover import RoverParams, build_rover_terrain  # noqa: E402
from mars_map.viewer.plans import plans_view  # noqa: E402

SITE_NAME = re.compile(r"^[a-z0-9_]{1,64}$")
MAX_TARGETS = 12
MAX_BODY = 64_000
SNAP_RADIUS_M = 60.0
MAX_AVOID = 6
VEHICLES = ("astronaut", "rover")
AVOID_RADIUS_M = (10.0, 400.0)                 # accepted range of a corridor's half-width


def corridor_polygon(x1: float, y1: float, x2: float, y2: float, r: float, n: int = 10
                     ) -> list[tuple[float, float]]:
    """Outline of everything within `r` of the segment: two half-circles joined by straight sides."""
    a = np.arctan2(y2 - y1, x2 - x1) if (x1, y1) != (x2, y2) else 0.0
    out = []
    for cx, cy, a0 in ((x2, y2, a - np.pi / 2), (x1, y1, a + np.pi / 2)):
        for k in range(n + 1):
            t = a0 + np.pi * k / n
            out.append((float(cx + r * np.cos(t)), float(cy + r * np.sin(t))))
    return out


class PlanError(ValueError):
    """A request the planner cannot serve; the message is shown to the user."""


class Planner:
    """Sites and their environment models, loaded once. One plan at a time."""

    def __init__(self):
        self._sites: dict[str, tuple[Site, Environment]] = {}
        self._conditions: dict[tuple[str, int | None], object] = {}
        self._lock = threading.Lock()

    @staticmethod
    def available() -> list[str]:
        return sorted(p.parent.name for p in DATA_DIR.glob("*/site.json")
                      if (p.parent / "layers.npz").exists())

    def _load(self, name: str) -> tuple[Site, Environment]:
        if name not in self._sites:
            if not SITE_NAME.match(name) or name not in self.available():
                raise PlanError(f"unknown site {name!r}; built sites: {', '.join(self.available())}")
            site = Site.load(DATA_DIR / name)
            self._sites[name] = (site, Environment(site))
        return self._sites[name]

    @staticmethod
    def _snap(site: Site, x: float, y: float, what: str, notes: list[str]) -> tuple[float, float]:
        """A point on blocked ground moves to the nearest open cell within SNAP_RADIUS_M."""
        w, h = site.extent_m
        if not (0 <= x < w and 0 <= y < h):
            raise PlanError(f"{what} is outside the map")
        blocked = site.layers["blocked"] if "blocked" in site.layers else site.layers["terrain_class"] == 3
        r, c = site.to_pixel(x, y)
        if not blocked[r, c]:
            return x, y
        k = int(SNAP_RADIUS_M / site.resolution_m)
        r0, r1, c0, c1 = max(0, r - k), min(blocked.shape[0], r + k + 1), max(0, c - k), min(blocked.shape[1], c + k + 1)
        free = np.argwhere(~blocked[r0:r1, c0:c1])
        if free.size == 0:
            raise PlanError(f"{what} is on blocked ground and there is no open ground within "
                            f"{SNAP_RADIUS_M:.0f} m of it")
        d = np.hypot(free[:, 0] + r0 - r, free[:, 1] + c0 - c)
        rr, cc = free[int(np.argmin(d))] + (r0, c0)
        nx, ny = site.to_metres(int(rr), int(cc))
        notes.append(f"{what} was on blocked ground and was moved {float(d.min()) * site.resolution_m:.0f} m "
                     f"to the nearest open ground.")
        return float(nx), float(ny)

    def plan(self, req: dict) -> dict:
        name = str(req.get("site", ""))
        start, targets = req.get("start"), req.get("targets")
        if not (isinstance(start, list) and len(start) == 2):
            raise PlanError("the base is not set")
        if not isinstance(targets, list) or not targets:
            raise PlanError("add at least one point to visit")
        if len(targets) > MAX_TARGETS:
            raise PlanError(f"at most {MAX_TARGETS} points")

        with self._lock:                                   # the planner is heavy: one at a time
            t0 = time.perf_counter()
            base, env = self._load(name)
            notes: list[str] = []
            try:
                sx, sy = self._snap(base, float(start[0]), float(start[1]), "The base", notes)
                chosen = []
                for i, t in enumerate(targets):
                    label = str(t.get("name") or f"Point {i + 1}")[:60]
                    x, y = self._snap(base, float(t["x"]), float(t["y"]), label, notes)
                    chosen.append(Target(label, round(x, 1), round(y, 1),
                                         priority=min(3, max(1, int(t.get("priority", 1)))),
                                         dwell_min=min(240.0, max(0.0, float(t.get("dwell", 20)))),
                                         note=str(t.get("note", ""))[:300]))
            except (KeyError, TypeError, ValueError) as e:
                if isinstance(e, PlanError):
                    raise
                raise PlanError(f"bad point in the request: {e}") from e

            site = copy.copy(base)                         # layers and series are shared, not copied
            site.start = (round(sx, 1), round(sy, 1))
            site.targets = chosen
            sol = req.get("sol")
            sol = int(sol) if sol is not None else None
            wanted = [v for v in (req.get("vehicles") or VEHICLES) if v in VEHICLES] or list(VEHICLES)
            measured = bool(site.series.get("radiation_msv_per_sol"))
            modes: dict[str, dict] = {}
            failed: dict[str, str] = {}
            for vehicle in wanted:                         # the same base and points for each
                try:
                    result = plan_site(site, sol=sol, env=env, vehicle=vehicle)
                    view = plans_view(result.to_dict(), radiation_measured=measured)
                    if view is None:
                        raise RuntimeError("no plan returned")
                    modes[vehicle] = view
                except (RuntimeError, ValueError, KeyError) as e:
                    failed[vehicle] = str(e)
            if not modes:
                raise PlanError("the planner could not plan this trip: " + "; ".join(failed.values()))
            for vehicle, why in failed.items():
                notes.append(f"No plan for the {'rover' if vehicle == 'rover' else 'walk'}: {why}")
            if (name, sol) not in self._conditions:         # ready for the first reroute of the trip
                self._conditions[(name, sol)] = build_conditions(base, env, sol, 11.0)
            plans = modes.get("astronaut") or next(iter(modes.values()))
            return {
                "plans": plans, "modes": modes, "notes": notes,
                "start": list(site.start),
                "targets": [{"name": t.name, "x": t.x_m, "y": t.y_m, "priority": t.priority,
                             "dwell": t.dwell_min, "note": t.note} for t in chosen],
                "seconds": round(time.perf_counter() - t0, 1),
            }


    def reroute(self, req: dict) -> dict:
        """A route from where the astronaut is, around the corridors, through the stops left."""
        name = str(req.get("site", ""))
        strategy = str(req.get("strategy", "science"))
        if strategy not in STRATEGIES:
            strategy = "science"
        try:
            here = (float(req["from"][0]), float(req["from"][1]))
            home = (float(req["base"][0]), float(req["base"][1]))
            via = [(str(v.get("name", f"Stop {i + 1}"))[:60], float(v["x"]), float(v["y"]))
                   for i, v in enumerate(req.get("via") or [])][:MAX_TARGETS]
            zones = []
            for z in (req.get("avoid") or [])[:MAX_AVOID]:
                r = min(AVOID_RADIUS_M[1], max(AVOID_RADIUS_M[0], float(z["r"])))
                zones.append(corridor_polygon(float(z["x1"]), float(z["y1"]),
                                              float(z["x2"]), float(z["y2"]), r))
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise PlanError(f"bad reroute request: {e}") from e
        if not zones:
            raise PlanError("nothing to avoid")

        with self._lock:
            t0 = time.perf_counter()
            base, env = self._load(name)
            sol = req.get("sol")
            sol = int(sol) if sol is not None else None
            if (name, sol) not in self._conditions:         # radiation and temperature maps of the sol
                self._conditions[(name, sol)] = build_conditions(base, env, sol, 11.0)
            cond = self._conditions[(name, sol)]

            site = copy.copy(base)
            site.no_go = list(base.no_go) + zones           # the corridors are closed ground
            if req.get("vehicle") == "rover":               # wheels: the rover's ground model
                mover = RoverParams()
                terrain = build_rover_terrain(site, mover, dose_rate_msv_h=cond.dose_rate_msv_h,
                                              ground_temp_c=cond.ground_temp_c,
                                              sun_w_m2=cond.sun_w_m2)
                if strategy not in mover.strategies:
                    strategy = "science"
            else:
                mover = SuitParams()
                terrain = build_astronaut_terrain(site, suit=mover, dose_rate_msv_h=cond.dose_rate_msv_h,
                                                  ground_temp_c=cond.ground_temp_c)
            router = EvaRouter(site, terrain, mover)

            points: list[tuple[float, float]] = []
            dist = minutes = 0.0
            dropped: list[str] = []
            at, at_name = here, "Current position"
            for label, x, y in via + [("Base", *home)]:
                try:
                    seg = router.leg(at, (x, y), strategy, at_name, label)
                except RuntimeError:
                    if label == "Base":
                        raise PlanError("no way back to the base that keeps out of the corridor") from None
                    dropped.append(label)                   # cut off by the corridor: skip this stop
                    continue
                points += [(float(px), float(py)) for px, py in seg.points_m]
                dist += seg.distance_m
                minutes += seg.duration_min
                at, at_name = (x, y), label
            if len(points) < 2:
                raise PlanError("no route found")
            keep = max(1, len(points) // 600)
            thin = points[::keep] + ([points[-1]] if (len(points) - 1) % keep else [])
            return {"points": [[round(x, 1), round(y, 1)] for x, y in thin],
                    "distance_m": round(dist, 1), "duration_min": round(minutes, 1),
                    "dropped": dropped, "strategy": strategy,
                    "seconds": round(time.perf_counter() - t0, 2)}


class Handler(SimpleHTTPRequestHandler):
    planner: Planner

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/health":
            return self._json(200, {"ok": True, "sites": self.planner.available()})
        return super().do_GET()

    def do_POST(self):
        route = self.path.split("?")[0]
        if route not in ("/api/plan", "/api/reroute"):
            return self._json(404, {"error": "unknown address"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                raise PlanError("empty or oversized request")
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(req, dict):
                raise PlanError("the request must be a JSON object")
            answer = self.planner.plan(req) if route == "/api/plan" else self.planner.reroute(req)
            return self._json(200, answer)
        except PlanError as e:
            return self._json(400, {"error": str(e)})
        except json.JSONDecodeError:
            return self._json(400, {"error": "the request is not valid JSON"})
        except Exception as e:                              # a bug: say so, keep serving
            traceback.print_exc()
            return self._json(500, {"error": f"planner error: {type(e).__name__}: {e}"})

    def end_headers(self):
        # Pages are rebuilt by scripts; never let the browser show a stale one.
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Log API calls and errors; stay quiet about every page and image that is served.
        # For an error (a missing file, say) the first argument is the status code, not
        # the request line, so it is turned into text before looking inside it.
        first = str(args[0]) if args else ""
        if "/api/" in first or fmt.startswith("code "):
            super().log_message(fmt, *args)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):                # the planner prints Russian text
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--bind", default="127.0.0.1", help="0.0.0.0 to reach it from other devices")
    args = ap.parse_args()

    Handler.planner = Planner()
    server = ThreadingHTTPServer((args.bind, args.port), partial(Handler, directory=str(ROOT)))
    host = "127.0.0.1" if args.bind in ("0.0.0.0", "") else args.bind
    print(f"Serving on http://{host}:{args.port}/gui/")
    for name in Handler.planner.available():
        if (DATA_DIR / name / "view3d.html").exists():
            print(f"  map: http://{host}:{args.port}/data/{name}/view3d.html")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
