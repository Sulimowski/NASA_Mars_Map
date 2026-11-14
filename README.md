# Data branch — data extraction and the multi-layer map

This branch turns raw NASA data into one `Site` object: a stack of map layers on one grid,
plus weather and radiation series, a base and science targets. It is the format the `Alg`
branch already works with (see [`DATA_CONTRACT.md`](DATA_CONTRACT.md)).

**Status (2026-11-14):** the first target from `main` is met on real data. There is a real
`Site` for the boxwork area of Gale crater with elevation, radiation and minerals, ready for
`Alg` (routing) and `GUI` (3D view).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\ingest_gale.py --rad-sols 4640-4669
.venv\Scripts\python scripts\build_gale_site.py
```

- `ingest_gale.py` downloads real data for the region: about 650 MB of traffic (540 MB of it
  is radiation, 18 MB per sol, deleted after parsing), 100–200 MB stays in `data/raw/`.
  It took about 15 minutes on our connection. Everything is cached: a second run downloads
  nothing. Without `--rad-sols` it loads only 5 early sols, which fall into a weather gap.
- `build_gale_site.py` fuses it into `data/gale_boxwork/` (`layers.npz`, `site.json`,
  `preview.png`). Takes about a minute.

```python
from mars_map import Site

site = Site.load("data/gale_boxwork")
site.layers["elevation"]          # 1000 x 1000, metres
site.layers["cost"]               # how many times slower than flat firm ground, inf = blocked
site.series["radiation_msv_per_sol"]
site.start, site.targets          # base and science stops, in site metres
```

No internet? `scripts\make_stub_site.py` makes the synthetic site `data/gale_stub/` in 3 seconds.

## 3D view

```bash
.venv\Scripts\python scripts\view3d.py --open
```

Builds `data/gale_boxwork/view3d.html`: one self-contained page (7.6 MB) with the terrain in
3D and every layer draped over it. Open it in a browser, or send the file to someone.
`scripts\view3d.py data/gale_stub` does the same for any other site folder.

- Left sheet: pick a layer (photo, false colour, elevation, slope, terrain class, walking cost,
  reachable area, minerals, thermal inertia), its legend, overlay strength, relief
  exaggeration, sun direction, contours, blocked ground.
- Right sheet: values under the cursor, the stops (click one to fly there; Curiosity stops
  show the rover's photo from the spot), the EVA plan, radiation and temperature per sol.
- Drag to orbit, right-drag to pan, wheel to zoom, `H` hides the sheets.

Shown automatically when present:
- `data/<site>/astronaut_plan.json`, the route written by the `Alg` router (distance, time,
  oxygen, battery, dose, GO / NO-GO);
- Curiosity's track, for regions the rover drove through.

Checked through a local server (`--serve`) in a desktop browser. If opening the file by
double-click shows a blank page in your browser, use `--serve`. The page template is
[`mars_map/viewer/template.html`](mars_map/viewer/template.html).

## Giving the real site to `Alg` and `GUI`

The site is two files, `layers.npz` and `site.json` (68 MB together). `data/` is not in git,
so there are two ways to get it:

1. **Build it yourself**: the two commands in the quick start. Needs this branch's code.
2. **Take the ready folder**: unzip `gale_boxwork_site.zip` into `data/` so that
   `data/gale_boxwork/site.json` exists. Needs only `mars_map/site.py`, which `Alg` already
   has, so no merge is required. The zip is made with:

   ```bash
   powershell Compress-Archive data\gale_boxwork\layers.npz,data\gale_boxwork\site.json data\gale_boxwork_site.zip
   ```

Then `Site.load("data/gale_boxwork")` works exactly like the stub.

Tested against `Alg` as of commit `272afd3`, without changing its code:

- **Router** (`mars_map/astronaut_router.py`): works. It planned a 3.39 km, 3.9 h walk to both
  priority-1 Curiosity stops and flagged it NO-GO on battery (13 % left).
- **Environment maps** (`scripts/make_env_maps.py`): **crash** with
  `KeyError: 'dust_opacity_tau'`. Real data has no dust opacity (see the contract). `Alg`
  needs a default there, for example `site.series.get("dust_opacity_tau")` with a fallback.

## The real site: `gale_boxwork`

Centre -4.812 N, 137.383 E, 2 × 2 km at 2 m per pixel, on the flank of Mount Sharp.
The centre comes from Curiosity's own track: the rover worked on the boxwork ridges here in
sols 4500–5070. Terrain is hard: 458 m of relief, median slope 12.7°, 17.5 % of cells blocked.

### Layers by data type

| Data type | Layers | Source | Native resolution |
|---|---|---|---|
| Topographic | `elevation`, `slope`, `roughness`, `ridge_index` | HiRISE DTM (CTX, HRSC, MOLA as fallback) | 1 m, covers 100 % |
| Optical | `optical`, `optical_texture` | HiRISE orthoimage, grayscale | 1 m, covers 100 % |
| Optical, colour | `optical_ir`, `optical_red`, `optical_bg` | HiRISE orthoimage, colour strip | 1 m, covers 43 % |
| Spectral / chemical | `mineral_sulfate`, `mineral_sulfate_mono`, `mineral_clay`, `mineral_hydration`, `mineral_silica`, `mineral_hematite`, `mineral_olivine`, `mineral_pyroxene`, `mineral_dominant`, `science_score` | CRISM MTRDR, 3 scenes | 18 m |
| Thermal | `thermal_inertia` (physical units) | THEMIS quantitative mosaic | 100 m |
| Thermal, pictures | `themis_day_ir`, `themis_night_ir` (8-bit brightness), `albedo` | THEMIS, TES | 100 m, 7.4 km |
| Radiation | series `radiation_msv_per_sol`, `radiation_dose_ugy_per_sol` | Curiosity RAD | one value per sol |
| Atmospheric | series `temp_min_c`, `temp_max_c`, `ground_temp_min_c`, `ground_temp_max_c`, `pressure_pa` | Curiosity REMS | one value per sol |
| For routing | `terrain_class`, `rock_abundance`, `blocked`, `cost`, `value`, `reachable`, `nodata`, `source_res` | fusion | grid |

Series cover sols 4640–4669 (30 sols, while Curiosity stood on the boxwork). Dose is
0.45–0.50 mSv per sol.

### Base and targets

The base is the flattest 40 m patch inside the area a crew can walk from the rover's proven
ground without exceeding the comfortable slope (15°). That area is 33.5 % of the region.
All 8 targets are inside it.

| Priority | Target | Position, m (x east, y south) | Why |
|---|---|---|---|
| — | Base | 315, 1031 | flattest reachable patch |
| 1 | Curiosity stop, sol 4752 | 883, 765 | rover worked here 37 sols |
| 1 | Curiosity stop, sol 5045 | 1273, 1751 | rover worked here 24 sols |
| 2 | Curiosity stop, sol 4868 | 923, 1135 | rover worked here 21 sols |
| 2 | Curiosity stop, sol 5021 | 1345, 1779 | rover worked here 20 sols |
| 2 | Hydrated sulfate exposure | 1385, 1441 | strongest CRISM detection (score 0.97) |
| 2 | Hydrated minerals exposure | 1543, 101 | CRISM, bound water (score 0.68) |
| 2 | Hydrated minerals exposure | 1583, 531 | CRISM, bound water, weak (score 0.25) |
| 2 | Hydrated silica / Al-OH exposure | 1341, 1181 | CRISM, weak (score 0.18) |

Each Curiosity stop carries, in `target.note`, a link to a photo the rover took on the spot:
what an astronaut would see there.

### Limits you must know

- **`rock_abundance` is a proxy, not a measurement.** It is the local contrast of the 1 m
  image, scaled so the roughest 1 % of the region reads 0.5. It also lights up on cliff edges
  and rock layering. It ranks ground from smooth to rugged; it does not count rocks.
- **No sand class.** Thermal inertia is 300–462 over the whole region (rocky ground), and at
  100 m per pixel it cannot see the sand ripples that are visible in the image.
- **Mineral indices are unitless and noisy.** A cell counts as a detection above median + 3
  sigma after smoothing over one CRISM pixel; 3.7 % of the region passes. Sulfates, hydrated
  minerals and hydrated silica are detected. Hematite, olivine, pyroxene and clay are not.
- **Optional layers can contain `NaN`** where their source has no data (the colour bands on
  57 % of the region). Core layers are always filled, and `nodata` says where they are coarse.
- **Radiation is one number per sol** for the whole region. There is no terrain shielding.
- **No dust opacity and no wind.** The public REMS feed does not carry them.
- `MissionParams` (slope limits, speed, EVA time) and `CostWeights` are placeholders until
  `Alg` has real numbers.

## Pipeline

```
catalog (metadata only: what exists, footprints, URLs)
        │
        ▼
fetch only the region ──▶ normalize to the region grid ──▶ processed layers + series
        │
        ▼
fuse (finest source per pixel) ──▶ interpret (terrain, minerals, cost) ──▶ base + targets ──▶ Site
```

Rules:
- We do **not** download all of Mars. The catalog holds metadata for everything; rasters are
  read only over chosen regions (often just the pixels we need) and cached.
- Rasters live as files, the database only stores where they are:
  `data/raw/` (as downloaded), `data/processed/<region>/<layer>__<source>.tif`, `data/catalog.sqlite`.
- `data/` is never committed.

Choosing the region automatically (analyzer + rating, tasks B2 and B3) is not built yet.
`gale_boxwork` was chosen by hand from Curiosity's track.

## Shared agreements

- `Site` contract: [`mars_map/site.py`](mars_map/site.py), described in
  [`DATA_CONTRACT.md`](DATA_CONTRACT.md). Shared with `Alg`, changed only by agreement.
- Catalog schema: [`mars_map/catalog/schema.sql`](mars_map/catalog/schema.sql).
- Region grid and processed layer format: [`mars_map/grid.py`](mars_map/grid.py).

Check them on your machine: `.venv\Scripts\python scripts\check_step0.py`

### Catalog tables

| Table | What it holds |
|---|---|
| `sources` | instrument + provider (HiRISE DTM from PDS ODE, CRISM, THEMIS, …) |
| `products` | everything that exists remotely: kind, resolution, footprint, bbox, download URL |
| `regions` | areas we work on: centre, size, resolution (this defines the region grid) |
| `layers` | processed rasters of a region: layer name, unit, source, file path, coverage |
| `series` | per-sol values of a region (radiation, weather) |
| `ratings` | region analyzer output, one row per metric (empty until B2/B3) |

Longitudes are east-positive in -180..180, latitudes planetocentric.

### Processed layer format

- One layer = one single-band GeoTIFF: `data/processed/<region>/<layer>__<source>.tif`.
- All layers of a region share one grid (`RegionGrid`): square, local equirectangular projection
  centred on the region, Mars sphere radius 3 396 190 m, so 1 pixel = `resolution_m` true metres.
- Row 0 = north, col 0 = west, same as `Site`.
- SI units. Missing data = `NaN` for float layers, `255` for `uint8` layers.
- Always use `write_layer()` / `read_layer()` from `mars_map/grid.py`, never open the files directly.
- The same layer may exist from several sources (`elevation__hirise_dtm.tif`,
  `elevation__mola.tif`); fusion picks the finest one per pixel.

### Regions

Defined in `mars_map/grid.py`, both scripts take `--region`:

- `gale_boxwork` (default): the real site described above.
- `gale_mount_sharp`: -4.75 N, 137.38 E, 3.5 km north of the boxwork (Curiosity sols
  2595–3700). The first test region; the synthetic stub and the fixture use its grid.

## Part A — ingestion (`mars_map/catalog/`, `mars_map/ingest/`)

Offline check (fake products, a local web server, a generated DTM):

```bash
.venv\Scripts\python scripts\check_ingest.py
```

| Module | What it does |
|---|---|
| [`catalog/db.py`](mars_map/catalog/db.py) | `connect()`, `add_source`, `add_products`, `find_products(bbox, kind)` (finest first, with `coverage` of the bbox by the exact footprint), `add_layer`, `add_region`, `add_series` |
| [`ingest/loaders.py`](mars_map/ingest/loaders.py) | `load_catalog(db, bbox)`: HiRISE DTMs from PDS ODE; CTX DTMs, HiRISE images, THEMIS mosaics from USGS STAC |
| [`ingest/fetch.py`](mars_map/ingest/fetch.py) | `fetch_product()` whole file with resume; `fetch_window()` reads only the pixels over a region from a remote raster (this is how 2–23 GB mosaics are used) |
| [`ingest/normalize.py`](mars_map/ingest/normalize.py) | `normalize()`: reproject to the region grid, nodata → NaN, units, mosaic of several products with the height offset between DTMs removed, register in `layers` |
| [`ingest/global_layers.py`](mars_map/ingest/global_layers.py) | MOLA 463 m and HRSC+MOLA 200 m elevation, THEMIS day/night IR, TES albedo |
| [`ingest/series.py`](mars_map/ingest/series.py) | REMS weather for every mission sol; RAD dose per requested sol (`--rad-sols`) |
| [`ingest/minerals.py`](mars_map/ingest/minerals.py) | CRISM summary-parameter cubes: downloads only the 8 bands we use (2 MB each instead of a 130 MB cube) |
| [`ingest/optical.py`](mars_map/ingest/optical.py) | HiRISE orthoimages that ship with the DTMs: grayscale, colour bands, and `optical_texture` computed at the native 1 m |
| [`ingest/thermal.py`](mars_map/ingest/thermal.py) | THEMIS quantitative thermal inertia: picks the 30° × 60° tile, reads only the region |
| [`ingest/rover.py`](mars_map/ingest/rover.py) | Curiosity waypoints (track, long stops) and a link to a photo taken at a stop |

### Sources (all verified 2026-11)

| Source | Provider | Where | File used |
|---|---|---|---|
| `hirise_dtm` | PDS ODE | `ihid=MRO&iid=HIRISE&pt=DTM` | `DTEEC_*.IMG`, 1 m |
| `hirise_ortho`, `hirise_color` | PDS ODE | file list of the same DTM products | `*_RED_C_01_ORTHO.JP2`, `*_IRB_C_01_ORTHO.JP2`, 1 m |
| `crism` | PDS ODE | `ihid=MRO&iid=CRISM&pt=MTRDR` | `*_su*j_mtr3.img` + `.hdr`, 18 m |
| `ctx_dtm`, `ctx_optical` | USGS STAC | `mro_ctx_controlled_usgs_dtms` | assets `geoid_adjusted_dem` (20 m), `orthoimage` |
| `hirise_optical` | USGS STAC | `mro_hirise_uncontrolled_observations` | asset `image`, 0.25 m (catalogued, not used) |
| `themis` | USGS STAC | `mo_themis_controlled_mosaics` | asset `image`, 100 m, 8-bit |
| `themis_ti` | USGS Astropedia (S3) | `THEMIS_TI_Mosaic_Quant_<tile>_100mpp.cub` | ISIS cube, 100 m, latitudes -60..60 |
| `mola`, `hrsc_mola`, `tes_albedo` | USGS planetarymaps | global mosaics | GeoTIFF |
| `rems` | mars.nasa.gov | weather feed `?feed=weather&category=msl` | JSON |
| `rad` | PDS PPI | `MSL-M-RAD-3-RDR-V1.0` | one text file per sol, 18 MB |
| rover track | mars.nasa.gov | `mmgis-maps/MSL/Layers/json/MSL_waypoints.json` | GeoJSON |
| rover photos | mars.nasa.gov | `api/v1/raw_image_items/?condition_2=<sol>:sol:in` | JSON with image links |

Working query examples:

```
https://oderest.rsl.wustl.edu/live2/?query=products&target=mars&results=fmp&output=JSON&ihid=MRO&iid=HIRISE&pt=DTM&westlon=137.366&eastlon=137.400&minlat=-4.829&maxlat=-4.795&limit=100
https://stac.astrogeology.usgs.gov/api/collections/mro_ctx_controlled_usgs_dtms/items?bbox=137.366,-4.829,137.400,-4.795&limit=100
```

In ODE longitudes are east 0..360; `results=fmp` adds the file list (`Product_files`) and the
footprint (`Footprint_C0_geometry`, WKT). In STAC longitudes are -180..180 and pages are chained
by the `next` link.

Things to know:
- **TLS on PDS ODE.** Its server does not send the intermediate certificate, so plain Python
  fails with `CERTIFICATE_VERIFY_FAILED` on a clean machine. `mars_map/ingest/__init__.py`
  switches verification to the operating system's trust store (`truststore`). Verification
  stays on.
- Gale has thousands of CTX stereo-pair DTMs, so the STAC loader stops at `max_items` (200).
- STAC also lists HiRISE DTMs (`mro_hirise_socet_dtms`), but some of its links are dead (404).
  We take HiRISE DTMs from ODE.
- All elevations are metres above the MOLA areoid, but the sources disagree by a constant. On
  the first test region the median difference to HiRISE was -5 m for CTX, -11 m for HRSC+MOLA
  and -51 m for MOLA. Fusion blends it out at the seams.
- REMS has gaps. Sols 4616–4753 and 4779–4995 are complete runs; the old default RAD sols
  453–457 fall into a gap, which is why the quick start asks for 4640–4669.
- Reading a JP2 window over HTTP takes minutes, so orthoimages under 150 MB are downloaded
  whole and cropped locally.
- Not found: TES thermal inertia and TES mineral maps as direct downloads. THEMIS thermal
  inertia and CRISM replace them at far better resolution.

## Part B — fusion and interpretation (`mars_map/fusion/`, `mars_map/analysis/`)

Part B on fake data only (stub → fake catalog → fusion → `Site`), with the error against the
stub it was cut from:

```bash
.venv\Scripts\python scripts\build_fixture_site.py
```

| Module | What it does |
|---|---|
| [`analysis/fixture.py`](mars_map/analysis/fixture.py) | fake catalog with 4 regions and Gale layers degraded like real sources; lets B run without A |
| [`fusion/stack.py`](mars_map/fusion/stack.py) | `fuse_region()`: finest source per pixel, coarse sources blended in at the seam (no fake cliffs), `nodata` and `source_res`, `slope`, `roughness`, `terrain_class`, series cut to their common sols |
| [`fusion/science.py`](mars_map/fusion/science.py) | CRISM indices → detection scores, `science_score`, `mineral_dominant`; `rock_abundance` proxy from image texture; `ridge_index`; `pick_base()`, `mineral_targets()` |
| [`fusion/derived.py`](mars_map/fusion/derived.py) | `blocked` (hard mask), `cost`, `value`, `walkable_mask()`, `reachable_mask()` |

How well fusion keeps the truth, measured on the fixture: elevation is exact where the fine
source exists and off by 0.19 m on average where only coarse sources do; terrain class agrees
on 97.6 % of cells; no false cliffs at source seams.

## What is left

- [ ] **B2. Region analyzer** (`analysis/regions.py`): data completeness, safety and science
      metrics per region. *Done when* it prints a metrics table for several regions.
- [ ] **B3. Rating** (`analysis/rating.py`): weighted sum of the metrics into `ratings`.
      *Done when* changing a weight changes the ranking.
- [ ] **B6. New layer types in the stub**: `optical`, `mineral_*`, `reachable`, so `Alg` and
      `GUI` can develop against them offline.
- [ ] **Sand detection** from the image (dark, smooth, flat ground); thermal inertia cannot do it.
- [ ] **Radiation shielding by terrain** (sky view from the DTM) and **shadows by time of day**.
- [ ] **Geologic map** (USGS units) as a layer.
- [ ] **Real limits** in `MissionParams` and `CostWeights`, from the `Alg` research questions.
- [ ] **A second region** (Jezero) to prove the pipeline is not Gale-only. The rover track
      and series modules are Curiosity-specific.

## Git

- Work on `Data`, or branch from it and merge back through a pull request.
- Shared files (`mars_map/site.py`, `catalog/schema.sql`, `mars_map/grid.py`, `requirements.txt`)
  change only by agreement.
