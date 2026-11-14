# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hackathon project "Interplanetary Survival Guide: Martian Map": plan an astronaut's walk (EVA) across a Mars site, taking terrain, radiation, temperature and suit consumables into account. Pure Python (numpy + matplotlib), no build step, no test suite and no linter configured.

Branches are split by team role: `Alg` (algorithms, the working branch for this code), `Data`, `GUI`, `main`.

## Commands

Windows, venv in `.venv`:

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\make_stub_site.py        # (re)generate data/gale_stub/ + preview.png
.venv\Scripts\python scripts\check_reachability.py    # flood-fill sanity check, exit 1 if a target is unreachable
.venv\Scripts\python scripts\make_env_maps.py --sol 23   # radiation / temperature maps -> env_preview.png
.venv\Scripts\python -m mars_map.astronaut_router     # plan the default EVA -> astronaut_plan.json + astronaut_route_preview.png
```

`scripts/make_stub_site.py` accepts `--size`, `--res`, `--seed`, `--out`. Generation is deterministic (`seed=42`), so everyone works on the same site.

## Architecture

Three layers, each built on the one before. `DATA_CONTRACT.md` and `ENV_MAPS.md` are the authoritative descriptions of the first two; read them before changing anything they cover.

1. **Site data contract** (`mars_map/site.py`, `synthetic.py`). A `Site` holds same-shaped 2-D rasters in `site.layers`, per-sol series in `site.series`, `start`, `targets`, `no_go` polygons and `MissionParams`. `load_stub()` returns the synthetic `gale_stub` site, generating it if missing. The contract is meant to stay fixed when real HiRISE/CTX/THEMIS data replaces the stub, so algorithms must read data only through `Site`.
2. **Environment maps** (`environment.py` + `terrain.py`, `sun.py`, `radiation.py`, `temperature.py`). `Environment(site)` precomputes horizon angles and sky view once (a few seconds), then gives per-cell maps for a sol and local solar time: `dose_rate`, `temperature`, `sunlit`, `direct_sunlight`. The series give one site-wide number per sol; the model only spreads it over the terrain. Tunable assumptions live in `RadiationParams` and `TemperatureParams`.
3. **Astronaut routing** (`astronaut_terrain.py`, `astronaut_router.py`). `build_astronaut_terrain(site)` turns the layers into walking rasters (`looseness`, `terramech_factor`, `speed_multiplier`, `impassable`, `nodata_penalty`). `route_segment` runs A* on the 8-connected grid; `plan_eva_mission` chains start → targets → start, adds dwell costs and checks the result against `site.mission`, returning a `MissionPlan`.

### Conventions that are easy to get wrong

- Grid: `layer[row, col]`, row 0 = north, col 0 = west. Points are `(x_m, y_m)` = metres east / metres **south** of the NW corner. Convert with `site.to_pixel()` / `site.to_metres()`, never by hand.
- Units differ between layers 2 and 3: `Environment.dose_rate()` returns **µSv per Earth hour**, while `route_segment(radiation_rate_msv_h=...)` expects **mSv per hour**. Divide by 1000 when wiring them together.
- The router is not yet connected to `Environment`: without a radiation raster it uses the series value of sol 0 spread evenly over the sol, and it ignores temperature, time of day and the choice of sol.
- Time: `ltst_h` is local true solar time in Mars hours (one Mars hour = 1.0275 Earth hours); dose rates and `eva_max_hours` are in Earth hours. The stub has no season, so sol 0 is pinned to Ls 150.
- The `slope` layer is the non-directional maximum slope and is used only for masks. Routing computes the directional slope of each step from `elevation`.
- `MissionParams` values and the module constants at the top of `astronaut_router.py` (masses, O2 tank, battery) are placeholders. The questions in `README.md` (radiation limits, thermal tolerance, walking endurance, terrain speed) are the research meant to replace them.
- The stub contains deliberate traps the algorithms must handle: a mesa reachable only by its NW ramp, a fresh crater with a boulder field, a dune field with a no-go core, a `nodata` wedge in the NE corner, a solar particle event (dose ×4 for 2 sols) and a dust storm in the series.
- Modules meant to be run directly use the `if __package__ in (None, ""):` block to fix `sys.path`; keep it when adding a runnable module. Paths like `data/...` resolve from the repo root via `resolve_path`, not the working directory.
- `.gitignore` lists `data/`, yet `data/gale_stub/` is currently committed. Regenerating the site rewrites tracked files.
