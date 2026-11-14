# GUI

The screen an astronaut or a mission planner uses: the map of the site, the planned EVA
routes on it, and the numbers that say whether a walk is safe.

Status: iPad mode, step 1 — a 3D Mars globe with our locations (one so far). Tap a location
to fly to it and open its card; "Open location" opens the existing 3D map of the site.
Helmet mode: not started.

## Run

Serve the project root (the GUI reads `gui/` and `data/`), then open http://localhost:8770/gui/

```bash
.venv\Scripts\python scripts\serve.py
```

`scripts/serve.py` serves the same files as `python -m http.server 8770` and adds live
planning: in a site's 3D map the user can set a base and points and get routes for them
(`POST /api/plan`, see the main [README](../README.md)). A plain `http.server` still works
for looking at the globe and the precomputed plans.

On an iPad in the same Wi-Fi: start it with `--bind 0.0.0.0` and open `http://<computer IP>:8770/gui/`.

Stack: plain HTML + ES modules + three.js 0.160 (in `vendor/`), no build step.

| File | What it is |
|---|---|
| `index.html` | the page |
| `css/app.css` | all styles; landscape = side panel, portrait = bottom sheet |
| `js/globe.js` | the globe: planet shader, atmosphere, stars, pins, touch input, fly-to |
| `js/app.js` | locations list, cards, selection |
| `locations.json` | our locations: name, lat/lon, facts, links to the site map and plans |
| `assets/mars_4k.jpg` | planet texture, made by `scripts/make_globe.py` (Viking MDIM 2.1 colour mosaic, NASA / USGS) |

To add a location: one more entry in `locations.json`.

## What the GUI reads

Two inputs, both produced by the other branches. The GUI does not compute anything itself.

| Input | Made by | Where | What it holds |
|---|---|---|---|
| Site | Data (`scripts/build_gale_site.py`) | `data/<site>/site.json` + `layers.npz` | the multi-layer map: elevation, slope, terrain class, minerals, photo, 31 layers; start, targets, series per sol. See [DATA_CONTRACT.md](../DATA_CONTRACT.md) |
| EVA plans | Alg (`python -m mars_map.pipeline`) | `data/<site>/eva_plans.json` | three alternative routes with time, oxygen, battery, dose, radio and shelters |

`data/` is not in git. To get a site, build it (see the main [README](../README.md)) or unzip
`gale_boxwork_site.zip` into `data/`.

The 3D map viewer already exists on this branch: `scripts/view3d.py` turns a site folder into
one HTML page (`data/<site>/view3d.html`) with three.js. The GUI can grow from it or reuse its
mesh and layer textures.

## Sample Alg output

`gui/samples/` holds real `eva_plans.json` files so the GUI can be built before the branches
are merged:

| File | Site | Notes |
|---|---|---|
| `samples/gale_boxwork/eva_plans.json` | real Gale boxwork site | sol 4640, 7 targets, 3 plans visit 2-3 of them, shelters included. `dust_tau` is null (no dust series yet) |
| `samples/gale_stub/eva_plans.json` | synthetic stub | sol 0, 5 targets, `dust_tau` present |
| `samples/gale_stub/eva_plans.png` | stub | Alg's own preview of the three plans, for reference |

Made from Alg commit `a7e6929` with `plan_site(site, shelters=True).to_json(...)`.

Note: `astronaut_plan.json` (keys `feasible`, `warnings`, `visited_targets`, `summary`,
`waypoints`) is an older Alg format with one plan only. Build against `eva_plans.json`.

## `eva_plans.json` in short

Full description: `EVA_PLANNER.md` on the Alg branch. Schema tag `"eva-plan/1"`.

```
schema, units                 units of every field
site                          name, resolution_m, shape, extent_m, center_lat/lon, start, targets[], no_go[]
conditions                    sol, ltst_h, ls_deg, particle_event, dust_tau, dose_rate_usv_h{min,median,max}, ground_temp_c{...}
mission, suit                 the limits and suit parameters used
comparison[]                  one row per plan: targets, time_h, distance_km, dose_msv, o2_used_l, energy_used_wh, science_value
strategies.shielded|eco|science
    title, feasible, warnings[]
    visited_targets[], skipped_targets[{name, priority, reason}]
    summary, limits, usage_pct          percent of each limit used
    stops[]                   start, targets, return: x_m, y_m, arrive/depart minute, o2_pct, battery_pct, dose_msv, note
    legs[]                    from, to, distance, duration, gain, max slope, O2, energy, dose, vertices_m[[x, y], ...]
    radio                     shadow_m, degraded_m, shadow_zones[], relay_points[]
    waypoints[]               one point about every 6 m (values for colouring the route and for a timeline)
    emergency_shelters[]      escape route from every target stop
```

A waypoint:

```json
{"leg": 0, "x_m": 342.0, "y_m": 1044.1, "elev_m": -3581.59, "lat": -4.812745, "lon": 137.371838,
 "dist_m": 30.0, "t_min": 1.09, "slope_deg": 5.2, "looseness": 0.121, "sinkage_cm": 0.46,
 "terrain_class": 0, "dose_rate_usv_h": 18.42, "dose_msv": 0.0003, "temp_c": -6.6,
 "o2_pct": 99.8, "battery_pct": 99.8, "radio": "clear"}
```

Coordinates are the same as in the site: `x_m` east and `y_m` south from the north-west corner
of the site, in metres. Pixel = `(row = y_m / 2, col = x_m / 2)` at 2 m per pixel.

## What the GUI should show (first draft, to agree on)

1. **Map** with layer switcher (photo, relief, slope, terrain, minerals, cost, reachable).
2. **Routes** of the three plans on the map, one colour each; start, targets, skipped targets.
3. **Plan comparison**: the `comparison[]` table, with the chosen plan highlighted.
4. **Plan detail**: stops with arrival time and O2 / battery / dose; limit usage bars from `usage_pct`.
5. **Timeline / profile** along the route from `waypoints`: elevation, O2, battery, dose, radio state.
6. **Radio**: shadow stretches on the route and relay drop points.
7. **Emergency**: shelter routes from each stop (`emergency_shelters`).
8. **Conditions**: sol, season, dust, particle event, dose rate range, temperatures.

## Open decisions

- Stack: extend the existing static three.js viewer (no build step, opens from one file), or a
  separate web app (for example Vite + React). To decide before writing code.
- Live replanning (GUI calls the Python planner) or only showing precomputed `eva_plans.json`.

## Known issues in the inputs

- Alg CLI on the real site: `python -m mars_map.pipeline --site data/gale_boxwork --shelters`
  plans fine, then crashes in its demo shelter query from the fixed point (1000, 1820), which is
  isolated on that site. `plan_site(...).to_json(...)` from Python works.
- `scripts/make_env_maps.py` on the real site: KeyError `dust_opacity_tau` (the site has no dust series).
- On the real site, targets at Curiosity stops have `science_score` 0.0 in the plans.
