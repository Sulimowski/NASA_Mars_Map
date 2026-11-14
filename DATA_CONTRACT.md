# Map data contract (for the algorithms and GUI teams)

Every map area is a `Site`: layers on one grid, per-sol series, a base, targets and crew limits.
There are two sites in the same format:

| Site | What it is | How to get it |
|---|---|---|
| `gale_stub` | **synthetic**, shaped like the Mount Sharp foothills, with traps built in on purpose | `scripts\make_stub_site.py`, 3 seconds, no internet |
| `gale_boxwork` | **real NASA data**, the boxwork area of Gale crater | `scripts\ingest_region.py` then `scripts\build_site.py`, see `README.md` |

Code written against the stub works on the real site if it follows the rules in
"Writing code that works on both" below.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\make_stub_site.py         # creates data/gale_stub/
.venv\Scripts\python scripts\check_reachability.py     # sanity check: all targets reachable
```

Generation is deterministic: `seed=42` gives everyone the same stub. The `data/` folder is not
in git; everyone generates it locally.

```python
from mars_map import Site, load_stub

site = load_stub()                      # generates the stub if it is missing
# site = Site.load("data/gale_boxwork") # the real site; paths are resolved from the repo root

elev  = site.layers["elevation"]        # np.ndarray (1000, 1000) float32
slope = site.layers["slope"]
r, c  = site.to_pixel(*site.start)      # metres -> pixel
for t in site.targets:
    print(t.name, t.priority, site.to_pixel(t.x_m, t.y_m))
```

Pictures of the layers: `data/<site>/preview.png`.

## Grid

- 2000 × 2000 m, **2 m/pixel**, arrays are `1000 × 1000`.
- `layer[row, col]`: `row 0` = **north**, `col 0` = **west** (like an image).
- Points are in metres `(x_m, y_m)`: `x` east, `y` **south** from the north-west corner.
  `col = x_m / res`, `row = y_m / res`. Converters: `site.to_pixel()`, `site.to_metres()`, `site.to_latlon()`.

## Required layers

Present in every site, all the same shape, never `NaN`.

| Key | Type | Units | Meaning |
|---|---|---|---|
| `elevation` | float32 | m | height above the Mars datum |
| `slope` | float32 | degrees | slope, derived from elevation |
| `roughness` | float32 | m | local relief (std of elevation in a ~7 m window) |
| `rock_abundance` | float32 | 0..1 | stub: fraction of area under rocks. Real site: a **proxy** from image texture, see below |
| `thermal_inertia` | float32 | J m⁻² K⁻¹ s⁻¹ᐟ² | low = loose sand, high = bedrock |
| `terrain_class` | uint8 | class | 0 bedrock, 1 sand, 2 boulders, 3 impassable |
| `science_score` | float32 | 0..1 | scientific value of the spot |
| `nodata` | bool | — | `True` = no high-res data here, values are coarser |

## Optional layers

May be absent, so check `if "cost" in site.layers`. Float layers marked "may be NaN" have `NaN`
where their source has no data.

For routing (added by fusion; present in `gale_boxwork` and `gale_fixture`, not in the stub):

| Key | Type | Meaning |
|---|---|---|
| `blocked` | bool | hard limit: impassable class, slope over `hard_max_slope_deg`, or inside a no-go polygon. Never route through |
| `cost` | float32 | how many times slower than flat firm ground (≥ 1), `inf` where blocked |
| `value` | float32 | science value of standing on the cell, 0..1 |
| `reachable` | bool | connected to the base through cells within `max_slope_deg` (real site only) |
| `source_res` | float32 | native resolution in metres of the elevation source used for the cell |
| `ridge_index` | float32 | height in metres above the surrounding 30 m (positive = ridge, e.g. boxwork) |

Real data only (`gale_boxwork`):

| Key | Units | Meaning |
|---|---|---|
| `optical` | 0..1 | HiRISE grayscale image, 1 m source |
| `optical_ir`, `optical_red`, `optical_bg` | 0..1 | HiRISE colour bands (near-IR, red, blue-green). **May be NaN**: covers 43 % |
| `optical_texture` | 0..1 | local contrast of the image at 1 m |
| `mineral_sulfate`, `mineral_sulfate_mono`, `mineral_clay`, `mineral_hydration`, `mineral_silica`, `mineral_hematite`, `mineral_olivine`, `mineral_pyroxene` | index | CRISM spectral index: higher = stronger signature, ≤ 0 = not detected. Unitless, not percent |
| `mineral_dominant` | uint8 | which mineral wins per cell: 0 none, 1 hydrated sulfate, 2 monohydrated sulfate, 3 clay, 4 hydrated minerals, 5 hydrated silica, 6 hematite, 7 olivine, 8 pyroxene |
| `themis_day_ir`, `themis_night_ir` | 8-bit brightness | THEMIS infrared pictures, not temperature |
| `albedo` | 0..1 | TES albedo. The source pixel is 7.4 km, so it is nearly constant over the site |

The names for `mineral_dominant` are in `mars_map.fusion.science.MINERAL_NAMES`.

## Series, base, targets, limits

- `site.series`: per-sol values for the whole site, all the same length, plus `sol`.

  | Key | Stub | Real site |
  |---|---|---|
  | `radiation_msv_per_sol` | yes, with a solar particle event (dose ×4 for 2 sols) | yes, 0.45–0.50, quiet period |
  | `temp_min_c`, `temp_max_c` | yes | yes (air) |
  | `dust_opacity_tau`, `wind_max_ms` | yes, with a dust storm | **no**: the public weather feed does not carry them |
  | `ground_temp_min_c`, `ground_temp_max_c`, `pressure_pa`, `radiation_dose_ugy_per_sol` | no | yes |

- `site.start`: start and finish point (the base).
- `site.targets`: science stops: `name`, `x_m`, `y_m`, `priority` (1 must visit … 3 optional),
  `dwell_min` (minutes of work at the stop), `note`. On the real site the note of a Curiosity
  stop ends with a link to a photo taken there.
- `site.no_go`: polygons in metres the crew must not enter. Empty on the real site.
- `site.mission`: crew limits (`MissionParams`): slopes 15°/25°, speed 2 km/h, EVA 6 h,
  25 % reserve, dose 1 mSv, distance 8 km. **These are placeholders.** Once the real numbers
  are found, update the values in `mars_map/site.py`.

## Writing code that works on both

1. Rely only on the **required layers**. Treat everything else as optional.
2. If `cost` and `blocked` are there, use them; otherwise build your own from `slope` and
   `terrain_class`.
3. Read series with `site.series.get("dust_opacity_tau")` and handle `None`.
4. Do not assume `rock_abundance` counts rocks. On the real site it is the local contrast of
   the image, scaled so the roughest 1 % of the region reads 0.5. It ranks ground from smooth
   to rugged and also lights up on cliff edges.
5. The real site has no sand class (thermal inertia at 100 m cannot see it) and no no-go zones.
6. Never route through `inf` cost. On the real site 17.5 % of cells are blocked and only a
   third of the region is reachable from the base within the comfortable slope.

## Traps built into the stub on purpose

- **Mesa** with cliffs: the "Mesa top outcrop" target is reachable only via the ramp on the north-west side.
- **Fresh crater** with steep walls and a boulder field around it (the "Fresh crater ejecta" target is on its edge).
- **Dunes** in the west: walkable but slow; the dune core is no-go.
- **Data gap** (north-east corner): decide how the algorithm treats `nodata`.

## Expected algorithm output (proposal)

- route: a list of points `(x_m, y_m)`;
- per leg: length, time, elevation gain, max slope, terrain type;
- totals: EVA time, dose, distance, targets visited, remaining reserve.
