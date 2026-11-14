# EVA planner

Plans a walk on Mars: which targets to visit, in what order, by which route, and what it
costs in time, oxygen, battery and radiation dose. Works on any `Site` from
[DATA_CONTRACT.md](DATA_CONTRACT.md) and uses the radiation and temperature maps from
[ENV_MAPS.md](ENV_MAPS.md) when the site has the data for them.

## Run it

```bash
.venv\Scripts\python -m mars_map.pipeline                  # stub site, sol 0
.venv\Scripts\python -m mars_map.pipeline --sol 17         # a solar particle event sol
.venv\Scripts\python -m mars_map.pipeline --site data/<real site> --sol <sol> --shelters
```

It prints the three plans and writes `eva_plans.json` (for the frontend) and `eva_plans.png`
(to look at) into the site folder. In PyCharm: open `mars_map/pipeline.py` and press Run.

```python
from mars_map import load_stub
from mars_map.pipeline import plan_site

result = plan_site(load_stub(), sol=0)             # or Site.load("data/<real site>")
plan = result.plans["science"]                     # also "shielded", "eco"
plan.visited_targets, plan.total_time_hours, plan.total_dose_msv
result.to_json("eva_plans.json")

escape = result.shelter((1200.0, 900.0))           # nearest radiation shelter from any point
```

Without the pipeline, on a site alone (uniform radiation, no temperature):

```python
from mars_map import EvaRouter, plan_pareto, plan_eva_mission, route_segment, find_nearest_shelter

plans = plan_pareto(site)                          # {"shielded": ..., "eco": ..., "science": ...}
plan  = plan_eva_mission(site, strategy="eco")
leg   = route_segment(site, (100, 200), (900, 1500))
```

## Files

| File | What it does |
|---|---|
| `mars_map/biomechanics.py` | Walking speed, metabolic power, oxygen and battery per step, taken from the astronaut model in `mars_eva_sim`. `SuitParams` |
| `mars_map/astronaut_terrain.py` | Ground model: boot sinkage, speed and energy factors, impassable mask, weather rasters |
| `mars_map/astronaut_router.py` | Route search, smoothing, target choice, strategies, shelter. `EvaRouter` |
| `mars_map/radio.py` | Line of sight to the base with the first Fresnel zone |
| `mars_map/pipeline.py` | Environment -> ground model -> router -> three plans -> physiology check -> JSON |
| `mars_eva_sim/` | The astronaut model: walking physics used by the router, and the sequential simulator that checks every finished plan |

## The three plans

| Strategy | Route prefers | Targets |
|---|---|---|
| `shielded` | ground with a lower dose rate (under cliffs, in hollows) | must-visit ones; optional ones only if they add little exposure; uses at most 85 % of every limit |
| `eco` | firm, level ground; lowest oxygen use | must-visit ones; optional ones only if cheap in oxygen; at most 85 % of every limit |
| `science` | the fastest way | as much science as fits, up to 100 % of the limits |

Priority always wins: a priority-1 target outweighs any number of lower ones. Within a
priority the `science_score` under the target decides.

Limits checked for every plan, return leg included:

- time: `eva_max_hours * (1 - return_reserve_frac)`
- distance: `max_distance_km`
- dose: `max_dose_msv`
- oxygen and battery: the tank and the battery minus a 20 % reserve (`SuitParams`)

If not everything fits, lower-priority targets are dropped and listed in `skipped_targets`
with the reason. A plan that visits nothing is still valid: it says to stay at the base.

## How a route is found

1. **Coarse graph.** The site is reduced to about 200 x 200 cells. Two coarse cells are linked
   only if the straight line between them crosses no impassable fine cell, so thin cliffs stay
   closed. One Dijkstra per waypoint gives the cost of every leg between all waypoints.
2. **Target choice.** Every order of every subset that fits the limits is scored.
3. **Fine legs.** Each chosen leg is searched again at full resolution with A*, inside a
   corridor around its coarse path.
4. **String pulling.** Grid corners are replaced by a straight line wherever the line stays one
   cell away from anything impassable, never exceeds the hard slope, and costs no more than
   the piece it replaces. A line through sand or through a hotter spot is rejected.
5. **Report.** The final route is walked sample by sample with the same physics the search used.

Measured on the stub (1000 x 1000 cells, 2 m): the three plans take about 1.3 s in total. The
radiation and temperature maps are prepared once per site and sol and take about 3.5 s more;
pass a ready `Environment` to `plan_site(..., env=env)` to skip that when replanning.

## Physics

- **Soil.** Bekker-Wong sinkage `p = (k_c / b + k_phi) * z^n` under one boot. The soil moduli
  come from thermal inertia (low = loose). Loose sand gives about 5 cm, firm ground under 1 cm.
- **Energy, speed, oxygen, battery** come from the astronaut model in `mars_eva_sim`
  (`src/physiology/vectorized.py`, the array form of its algorithms A2-A4), so the router,
  the report and the simulator use one set of formulas and parameters:
  - *Energy.* Pandolf load-carriage equation with the terrain coefficient from the sinkage
    and the rock cover; the climbing term scales with Mars gravity, level walking less than
    that; descents recover half of the grade term and pay for braking beyond 10 %. About
    380 W walking at 3 km/h on firm level ground.
  - *Speed.* The preferred pace (3 km/h) times the terrain speed factor, lowered until the
    metabolic power fits what the crew member can hold (about 384 W for a crew member 30
    sols after landing). `walk_speed_flat_kmh` is no longer used.
  - *Impossible steps.* Where even the slowest walk needs more than that power (a steep
    climb over loose ground) the step is blocked, like a cliff.
  - *Oxygen* follows the metabolic power plus suit leakage; *battery* = avionics + fan
    (scaled by the workload) + glove and boot heaters that follow the ground temperature.
- **Dose** = dose rate under the feet x time, also while working at a target.

## Physiology check

A step price cannot hold what builds up along a route. After a plan is assembled the
pipeline walks it through the sequential simulator of `mars_eva_sim` (on the site merged
to about 100 x 100 cells) and stores the verdict in `plan.physiology`:

- `status`: `OK`, `MARGINAL` (walkable, with warnings) or `FAIL` (a constraint breaks or a
  step is impassable; `message` and `broken_constraint` say where and why);
- fatigue and the rests it forces, peak heart rate, mean walking power, stored heat, water;
- the oxygen and time needed to walk back from every point, emergency reach of the
  secondary oxygen, the longest stretch without radio contact;
- `difference_from_plan`: simulator total minus the plan total, per quantity.

`plan_site(..., check_physiology=False)` skips it (it takes a few seconds per site).
`conditions.sol_assessment` says whether the sol is fit for an EVA (no particle event
within one sol, daylight window) and which sol of the series is best.

## Emergency: solar particle event

`result.shelter((x_m, y_m))` or `find_nearest_shelter(site, (x_m, y_m), dose_rate_map)` returns
the route to the spot that gives the lowest total dose: walking there plus waiting there for the
rest of `shelter_hours` (default 2 h), among spots within `max_walk_min` (default 30 min).
`worthwhile` is False when nothing in reach is at least 3 % better than staying put.
With a prepared router the answer takes about 0.1 s.

## Radio

Every 10 m of a route is checked for line of sight to a 10 m mast at the base (UHF 400 MHz):
`clear`, `degraded` (line of sight, but the ground intrudes into the first Fresnel zone) or
`shadow` (terrain blocks the ray). For every shadow stretch of 20 m or more the plan names the
last point still in contact as a relay drop point, and how much of the stretch a relay there covers.

## `eva_plans.json`

```
schema        "eva-plan/1"
units         meaning of every field
site          name, resolution_m, shape, extent_m, center_lat/lon, start, targets[], no_go[]
conditions    sol, ltst_h, ls_deg, particle_event, dose_rate_usv_h{min,median,max}, ground_temp_c{...},
              sol_assessment{sol, allowed, reasons[], best_sol, allowed_sols, window_ltst_h}
mission       the MissionParams used
suit          the SuitParams used
comparison[]  one line per strategy: targets, time_h, distance_km, dose_msv, o2_used_l, science_value
strategies    { shielded | eco | science : plan }
timings_s
```

A plan:

```
strategy, title, feasible, warnings[]
visited_targets[], skipped_targets[{name, priority, reason}]
summary       distance, time, elevation gain, max slope, O2 and battery left, dose, science value
limits, usage_pct{time_s, distance_m, o2_l, energy_wh, dose_msv}     percent of each limit used
stops[]       start, targets, return: position, arrive/depart minute, O2 %, battery %, dose
legs[]        from, to, distance, duration, gain/drop, max slope, O2, energy, dose, vertices_m[[x, y], ...]
radio         shadow_m, degraded_m, shadow_zones[], relay_points[]
physiology    status, message, broken_constraint, warnings[], eva_time_h, o2/battery remaining,
              limiting_resource, peak_fatigue, peak_heart_rate_bpm, rest_min, emergency_safety,
              radio{...}, difference_from_plan{...}
waypoints[]   the route for drawing, one point about every 6 m
emergency_shelters[]   only with --shelters: escape route from every target stop
```

A waypoint:

```json
{"leg": 0, "x_m": 1001.0, "y_m": 151.0, "elev_m": -4483.97, "lat": -4.735649, "lon": 137.380017,
 "dist_m": 30.0, "t_min": 1.15, "slope_deg": 3.0, "looseness": 0.116, "sinkage_cm": 0.53,
 "terrain_class": 0, "dose_rate_usv_h": 27.82, "dose_msv": 0.0005, "temp_c": -22.3,
 "o2_pct": 99.8, "battery_pct": 99.8, "radio": "clear"}
```

`legs[].vertices_m` are the corners of the smoothed route (a few per leg): enough to draw the
path. `waypoints` carry the values along it for colouring and for the timeline.

## Assumptions to replace with real numbers

- The boot size in `SuitParams` and the Bekker-Wong moduli in `astronaut_terrain.py`.
- Every parameter of the astronaut model (`mars_eva_sim/src/core/config.py`, the values
  marked `[verify]`), including masses, oxygen tank, battery and the Pandolf scaling.
- Strategy weights in `STRATEGIES`.

## Known gaps

- One weather snapshot per plan: the temperature map is for one local time, not for each hour
  of the walk.
- Target choice uses leg costs of the fastest route for all strategies; the plan is then
  checked against the real legs and re-chosen if it does not fit.
- On the stub, radiation varies by a few percent away from cliffs, so `shielded` differs from
  the others mainly by taking fewer targets. It will separate more on data with real gradients.
- `degraded` radio is the normal state for antennas this low; only `shadow` is treated as a problem.
