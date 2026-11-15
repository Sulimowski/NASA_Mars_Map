# Data — from NASA archives to a `Site`

How the map data is downloaded, fused and interpreted: the regions, the layers and where each
comes from, the processed-file format, and what to know about every source. The project
overview is in the [README](README.md); the `Site` contract in [DATA_CONTRACT.md](DATA_CONTRACT.md).

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
All 6 targets are inside it.

| Priority | Target | Position, m (x east, y south) | Why |
|---|---|---|---|
| — | Base | 315, 1031 | flattest reachable patch |
| 1 | Curiosity stop, sol 4752 | 883, 765 | rover worked here 37 sols |
| 1 | Curiosity stop, sol 5045 | 1273, 1751 | rover worked here 24 sols |
| 2 | Curiosity stop, sol 4868 | 923, 1135 | rover worked here 21 sols |
| 2 | Curiosity stop, sol 5021 | 1345, 1779 | rover worked here 20 sols |
| 2 | Hydrated sulfate exposure | 1385, 1441 | strongest CRISM detection (score 0.76) |
| 2 | Hydrated minerals exposure | 1543, 101 | CRISM, bound water (score 0.33) |

Each Curiosity stop carries, in `target.note`, a link to a photo the rover took on the spot:
what an astronaut would see there.

### Limits you must know

- **`rock_abundance` is a proxy, not a measurement.** It is the local contrast of the 1 m
  image, scaled so the roughest 1 % of the region reads 0.5. It also lights up on cliff edges
  and rock layering. It ranks ground from smooth to rugged; it does not count rocks.
- **Sand is read from the image, not measured.** Thermal inertia alone found no sand (300–462
  here, and at 100 m per pixel it cannot see strips a few metres wide). `sand_fraction`
  ([`fusion/sand.py`](mars_map/fusion/sand.py)) reads it from HiRISE and the DTM instead:
  dark, smooth, flat ground in lows (sand sheets), or strong image contrast on ground the DTM
  sees as flat and smooth (ripple fields). THEMIS and CRISM olivine only nudge it by ±20 %.
  Cells with `sand_fraction` ≥ 0.5 are terrain class "sand", and the boot sinkage softens
  gradually with the fraction. Result: Gale boxwork 0.9 % sand, Jezero 17 %. Check against
  THEMIS on 100 m blocks, with the map built without THEMIS: at Jezero more mapped sand goes
  with lower thermal inertia (r = −0.31) and a colder night (r = −0.21), as it should; at Gale
  there is too little sand to show anything (r ≈ 0). The USGS dune database cannot be used to
  check it: it maps dune fields larger than about 1 km², and our 2 km sites have none.
- **Mineral indices are unitless and noisy.** A cell counts as a detection when its index,
  smoothed over one CRISM pixel, is above a fixed threshold; the score runs from 0 at the
  threshold to 1 at a "strong signature" level. The thresholds are the same in every region
  (`DETECTION` in `mars_map/fusion/science.py`), so scores can be compared between regions.
  **They are our estimates** from the three regions we have, not a published calibration, and
  a CRISM scene that reads low as a whole will under-detect. In Gale 3.6 % of the region
  passes: sulfates, hydrated minerals and hydrated silica are detected; hematite, olivine,
  pyroxene and clay are not.
- **The science score is not a resource score.** It ranks how clearly a mineral shows and how
  interesting it is (sulfates and clays weigh 1.0, olivine and pyroxene 0.4).
- **Optional layers can contain `NaN`** where their source has no data (the colour bands on
  57 % of the region). Core layers are always filled, and `nodata` says where they are coarse.
- **Radiation is measured as one number per sol** for the whole region. The environment model
  spreads it over the terrain (sky hidden by cliffs, air column), which is a few percent.
- **No dust opacity and no wind.** The public REMS feed does not carry them.
- `MissionParams` (slope limits, speed, EVA time) and `CostWeights` are placeholders until
  `Alg` has real numbers.

## Regions

Defined in `mars_map/grid.py` (`REGIONS`); `ingest_region.py` and `build_site.py` take
`--region`. Each is 2 × 2 km at 2 m per pixel. A new region is one `RegionGrid(...)` line.

| Region | Centre | What it is | Built on real data |
|---|---|---|---|
| `gale_boxwork` (default) | -4.812 N, 137.383 E | boxwork ridges on Mount Sharp, Curiosity sols 4500–5070 | yes, described above |
| `gale_mount_sharp` | -4.75 N, 137.38 E | 3.5 km north of the boxwork, Curiosity sols 2595–3700. The first test region; the synthetic stub and the fixture use its grid | elevation and coarse layers only |
| `jezero_delta` | 18.458 N, 77.402 E | front of the river delta in Jezero crater, Perseverance sols 400–800 | yes |
| `nili_planum` | 18.432 N, 77.238 E | plains outside the western rim of Jezero, Perseverance sols 1600+ | yes, coarse |

How the three built regions compare:

| | `gale_boxwork` | `jezero_delta` | `nili_planum` |
|---|---|---|---|
| Elevation source | HiRISE 1 m | HiRISE 1 m | CTX 20 m (no HiRISE DTM exists there) |
| Photo | HiRISE 1 m + colour strip | HiRISE 1 m | CTX 20 m |
| Layers in the site | 31 | 28 | 27 |
| Relief, median slope | 458 m, 12.7° | 121 m, 5.2° | 238 m, 7.7° |
| Blocked | 17.5 % | 1.6 % | 0.0 % (slopes are smoothed away at 20 m) |
| Reachable from the base | 33.5 % | 90.9 % | 87.9 % |
| Minerals detected | 3.6 %: hydrated sulfate, hydrated minerals, silica | 88 %: olivine 82 %, clay 6.3 % | 0.3 % |
| Targets | 4 rover stops, 2 mineral | 3 rover stops, 4 clay | 4 rover stops |
| Weather | REMS, any sol | MEDA, 4 sols | MEDA, 4 sols |
| Radiation | RAD, per sol | none | none |

What differs outside Gale:

- **Rover.** `ingest/rover.py` picks the rover by where the region is: Curiosity within 150 km
  of its landing site, Perseverance within 150 km of its own, none elsewhere. Track, stops
  and photos follow.
- **No radiation.** Perseverance has no radiation detector. Jezero and Nili Planum sites have
  no `radiation_msv_per_sol`; code that needs dose must handle its absence.
- **Weather is four sols** (1130–1133): that is all the public Perseverance feed holds.
- **Olivine at Jezero is everywhere but weak.** It covers 82 % of the square and never scores
  above 0.36. The clay patches (25 ha, up to 0.92) are the strong detections.
- **`nili_planum` is a coarse map.** With 20 m elevation there is no rock proxy, nothing is
  classed impassable and the single CRISM scene covers a quarter of the square.

## Giving a real site to someone else

A site is two files, `layers.npz` and `site.json` (about 70 MB together). `site.json` is in git,
`layers.npz` is not: either build it with the commands in the [README](README.md#build-a-site-from-nasa-data),
or zip `layers.npz` and unzip it into `data/<site>/`. `Site.load("data/gale_boxwork")` then works
exactly like the stub.

A `layers.npz` built by older code may lack layers that newer code derives. `Site.load` fills
them in (today: `sand_fraction`, and sandy cells become terrain class "sand"), so pages and plans
built from an old file keep the sand. Before this, a page rebuilt from an old file silently lost it.

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
- Raw downloads, processed rasters and `layers.npz` are never committed. The finished pages and
  plans are (`data/<site>/view3d.html`, `site.json`, `eva_plans.json`), so a fresh clone can show
  the maps without building anything.

Choosing the region automatically (analyzer + rating, tasks B2 and B3) is not built yet.
The regions were chosen by hand from the rovers' tracks.

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

## Part A — ingestion (`mars_map/catalog/`, `mars_map/ingest/`)

Offline check (fake products, a local web server, a generated DTM):

```bash
.venv\Scripts\python scripts\check_ingest.py
```

| Module | What it does |
|---|---|
| [`catalog/db.py`](mars_map/catalog/db.py) | `connect()`, `add_source`, `add_products`, `find_products(bbox, kind)` (finest first, with `coverage` of the bbox by the exact footprint), `add_layer`, `add_region`, `add_series` |
| [`ingest/loaders.py`](mars_map/ingest/loaders.py) | `load_catalog(db, bbox)`: HiRISE DTMs from PDS ODE; CTX DTMs with their orthoimages, HiRISE images, THEMIS mosaics from USGS STAC |
| [`ingest/fetch.py`](mars_map/ingest/fetch.py) | `fetch_product()` whole file with resume; `fetch_window()` reads only the pixels over a region from a remote raster (this is how 2–23 GB mosaics are used); `source_crs()` repairs the projection of HiRISE ortho JP2 files |
| [`ingest/normalize.py`](mars_map/ingest/normalize.py) | `normalize()`: reproject to the region grid, nodata → NaN, units, mosaic of several products with the height offset between DTMs removed, register in `layers` |
| [`ingest/global_layers.py`](mars_map/ingest/global_layers.py) | MOLA 463 m and HRSC+MOLA 200 m elevation, THEMIS day/night IR, TES albedo |
| [`ingest/series.py`](mars_map/ingest/series.py) | Curiosity: REMS weather for every mission sol, RAD dose per requested sol (`--rad-sols`). Perseverance: MEDA weather from the public feed |
| [`ingest/minerals.py`](mars_map/ingest/minerals.py) | CRISM summary-parameter cubes: downloads only the 8 bands we use (2 MB each instead of a 130 MB cube) |
| [`ingest/optical.py`](mars_map/ingest/optical.py) | HiRISE orthoimages that ship with the DTMs: grayscale, colour bands, and `optical_texture` computed at the native 1 m. Where there is no HiRISE orthoimage, the 20 m CTX orthoimage becomes `optical` |
| [`ingest/thermal.py`](mars_map/ingest/thermal.py) | THEMIS quantitative thermal inertia: picks the 30° × 60° tile, reads only the region |
| [`ingest/rover.py`](mars_map/ingest/rover.py) | Curiosity or Perseverance, picked by region: waypoints (track, long stops) and a link to a photo taken at a stop |

### Sources (all verified 2026-11)

| Source | Provider | Where | File used |
|---|---|---|---|
| `hirise_dtm` | PDS ODE | `ihid=MRO&iid=HIRISE&pt=DTM` | `DTEEC_*.IMG`, 1 m |
| `hirise_ortho`, `hirise_color` | PDS ODE | file list of the same DTM products | `*_RED_C_01_ORTHO.JP2`, `*_IRB_C_01_ORTHO.JP2`, 1 m |
| `crism` | PDS ODE | `ihid=MRO&iid=CRISM&pt=MTRDR` | `*_su*j_mtr3.img` + `.hdr`, 18 m |
| `ctx_dtm`, `ctx_optical` | USGS STAC | `mro_ctx_controlled_usgs_dtms` | assets `geoid_adjusted_dem` or `dtm` (20 m), `orthoimage` (20 m) |
| `hirise_optical` | USGS STAC | `mro_hirise_uncontrolled_observations` | asset `image`, 0.25 m (catalogued, not used) |
| `themis` | USGS STAC | `mo_themis_controlled_mosaics` | asset `image`, 100 m, 8-bit |
| `themis_ti` | USGS Astropedia (S3) | `THEMIS_TI_Mosaic_Quant_<tile>_100mpp.cub` | ISIS cube, 100 m, latitudes -60..60 |
| `mola`, `hrsc_mola`, `tes_albedo` | USGS planetarymaps | global mosaics | GeoTIFF |
| `rems` | mars.nasa.gov | weather feed `?feed=weather&category=msl` | JSON |
| `rad` | PDS PPI | `MSL-M-RAD-3-RDR-V1.0` | one text file per sol, 18 MB |
| rover waypoints (for the rover-stop targets) | mars.nasa.gov | `mmgis-maps/MSL/Layers/json/MSL_waypoints.json` | GeoJSON |
| rover photos | mars.nasa.gov | `api/v1/raw_image_items/?condition_2=<sol>:sol:in` | JSON with image links |
| `meda` (Perseverance weather) | mars.nasa.gov | weather feed `?feed=weather&category=mars2020` | JSON, a handful of sols |
| Perseverance track | mars.nasa.gov | `mmgis-maps/M20/Layers/json/M20_waypoints.json` | GeoJSON |
| Perseverance photos | mars.nasa.gov | `rss/api/?feed=raw_images&category=mars2020&sol=<sol>` | JSON with image links |

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
- **HiRISE ortho JP2 files carry a wrong projection field** outside the equatorial band: the
  standard parallel is stored as the latitude of origin, so read literally the image lands
  15° of latitude away. `source_crs()` in `ingest/fetch.py` repairs it. Gale never showed it
  because its standard parallel is 0.
- **The CTX DTM asset has two names** in the same STAC collection: `geoid_adjusted_dem` around
  Gale, `dtm` around Jezero. Both are metres above the areoid (checked against MOLA).
- **No HiRISE DTM west of the Jezero rim**, so `nili_planum` runs on CTX.
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
| [`fusion/science.py`](mars_map/fusion/science.py) | CRISM indices → detection scores with fixed thresholds per mineral (`DETECTION`, same in every region), `science_score`, `mineral_dominant`; `rock_abundance` proxy from image texture; `ridge_index`; `pick_base()`, `mineral_targets()` |
| [`fusion/derived.py`](mars_map/fusion/derived.py) | `blocked` (hard mask), `cost`, `value`, `walkable_mask()`, `reachable_mask()` |

How well fusion keeps the truth, measured on the fixture: elevation is exact where the fine
source exists and off by 0.19 m on average where only coarse sources do; terrain class agrees
on 97.6 % of cells; no false cliffs at source seams.
