# Map data contract (for the algorithms team)

Until we have real Gale crater data, we work on a **synthetic site** called `gale_stub`, shaped
like the Mount Sharp foothills. The format below will not change: when real HiRISE/CTX/THEMIS
data is plugged in, algorithm code should keep working without edits.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\make_stub_site.py         # creates data/gale_stub/
.venv\Scripts\python scripts\check_reachability.py     # sanity check: all targets reachable
```

Generation is deterministic: `seed=42` gives everyone the same site. The `data/` folder is not
in git; everyone generates it locally (~3 seconds).

```python
from mars_map import load_stub

site = load_stub()                      # generates the site if it is missing
# or Site.load("data/gale_stub"): the path is resolved from the repo root, not the working dir

elev  = site.layers["elevation"]        # np.ndarray (1000, 1000) float32
slope = site.layers["slope"]
r, c  = site.to_pixel(*site.start)      # metres -> pixel
for t in site.targets:
    print(t.name, t.priority, site.to_pixel(t.x_m, t.y_m))
```

Picture of all layers: `data/gale_stub/preview.png`.

## Grid

- 2000 × 2000 m, **2 m/pixel**, arrays are `1000 × 1000`. Change the scale with `--size 4000 --res 1`.
- `layer[row, col]`: `row 0` = **north**, `col 0` = **west** (like an image).
- Points are in metres `(x_m, y_m)`: `x` east, `y` **south** from the north-west corner.
  `col = x_m / res`, `row = y_m / res`. Converters: `site.to_pixel()`, `site.to_metres()`, `site.to_latlon()`.

## Layers (`site.layers`), all the same shape

| Key | Type | Units | Meaning |
|---|---|---|---|
| `elevation` | float32 | m | height |
| `slope` | float32 | degrees | slope, derived from elevation |
| `roughness` | float32 | m | local relief (std of elevation in a ~7 m window) |
| `rock_abundance` | float32 | 0..1 | fraction of area covered by rocks |
| `thermal_inertia` | float32 | SI | low = loose sand, high = bedrock |
| `terrain_class` | uint8 | class | 0 bedrock, 1 sand, 2 boulders, 3 impassable |
| `science_score` | float32 | 0..1 | scientific value of the spot |
| `nodata` | bool | — | `True` = no high-res data, values here are coarser |

## Everything else

- `site.series`: per-sol series for the whole site (30 sols): `radiation_msv_per_sol`,
  `dust_opacity_tau`, `temp_min_c`, `temp_max_c`, `wind_max_ms`. The series contains a **solar
  particle event** (dose ×4 for 2 sols) and a **dust storm**: an EVA-day picker should avoid them.
- `site.start`: start and finish point (the base).
- `site.targets`: science stops: `name`, `x_m`, `y_m`, `priority` (1 must visit … 3 optional),
  `dwell_min` (minutes of work at the stop), `note`.
- `site.no_go`: polygons in metres the crew must not enter.
- `site.mission`: crew limits (`MissionParams`): slopes 15°/25°, speed 2 km/h, EVA 6 h,
  25 % reserve, dose 1 mSv, distance 8 km. **These are placeholders.** The real numbers are the
  answers to the questions in `README.md` (radiation, temperature, speed per terrain type). Once
  you find them, update the values in `mars_map/site.py`.

## Traps built in on purpose

- **Mesa** with cliffs: the "Mesa top outcrop" target is reachable only via the ramp on the north-west side.
- **Fresh crater** with steep walls and a boulder field around it (the "Fresh crater ejecta" target is on its edge).
- **Dunes** in the west: walkable but slow; the dune core is no-go.
- **Data gap** (north-east corner): decide how the algorithm treats `nodata`.

## Expected algorithm output (proposal)

- route: a list of points `(x_m, y_m)`;
- per leg: length, time, elevation gain, max slope, terrain type;
- totals: EVA time, dose, distance, targets visited, remaining reserve.
