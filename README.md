# Data branch — data extraction and the multi-layer map

Goal of this branch: turn raw NASA data into one `Site` object (the format the `Alg` branch
already works with, see `DATA_CONTRACT.md` on `Alg`).

**First target (from `main`):** a real `Site` for Gale crater with geometry (elevation), radiation
and useful minerals, so `Alg` can run point-to-point routing with stops and `GUI` can show it in 3D.

## Pipeline

```
catalog of everything (metadata only)
        │
        ▼
region analyzer ── rating ──▶ chosen region
        │
        ▼
download only that region ──▶ normalize to one grid ──▶ fuse ──▶ Site
```

Rules:
- We do **not** download all of Mars. The catalog holds metadata (footprint, resolution, URL)
  for everything; rasters are downloaded only for chosen regions and cached.
- Rasters live as files, the database only stores where they are:
  `data/raw/` (as downloaded), `data/processed/<region>/<layer>.tif`, `data/catalog.sqlite`.
- `data/` is never committed.

## Step 0 — shared agreements (done)

- [x] Shared contract (`mars_map/site.py`, `DATA_CONTRACT.md`, stub generator) copied from `Alg`.
      `site.py` is changed only by agreement with `Alg`.
- [x] Catalog schema: [`mars_map/catalog/schema.sql`](mars_map/catalog/schema.sql).
- [x] Processed layer format and region grid: [`mars_map/grid.py`](mars_map/grid.py).
- [x] `rasterio` and `geopandas` added to `requirements.txt`.

Check that it all works on your machine:

```bash
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\check_step0.py
```

### Catalog tables

| Table | What it holds |
|---|---|
| `sources` | instrument + provider (HiRISE DTM from USGS, TES from Trek, …) |
| `products` | everything that exists remotely: kind, resolution, footprint, bbox, download URL |
| `regions` | candidate areas: centre, size, resolution (this defines the region grid) |
| `layers` | processed rasters of a region: layer name, unit, source, file path, coverage |
| `series` | per-sol values of a region (radiation, weather) |
| `ratings` | region analyzer output, one row per metric |

Longitudes are east-positive in -180..180, latitudes planetocentric.

### Processed layer format

- One layer = one single-band GeoTIFF: `data/processed/<region>/<layer>__<source>.tif`.
- All layers of a region share one grid (`RegionGrid`): square, local equirectangular projection
  centred on the region, Mars sphere radius 3 396 190 m, so 1 pixel = `resolution_m` true metres.
- Row 0 = north, col 0 = west, same as `Site`.
- SI units. Missing data = `NaN` for float layers, `255` for `uint8` layers.
- Always use `write_layer()` / `read_layer()` from `mars_map/grid.py`, never open the files directly.
- The same layer may exist from several sources (`elevation__hirise_dtm.tif`,
  `elevation__mola.tif`); fusion picks the best one per pixel.

## A — ingestion (parsers and the database)

Works in `mars_map/catalog/` and `mars_map/ingest/`. Branch: `data/ingest`.

Check part A without internet (fake products, a local web server, a generated DTM):

```bash
.venv\Scripts\python scripts\check_ingest.py
```

Run part A on real data for Gale (catalog → fetch → normalize → global layers → series).
About 100 MB of traffic (mostly RAD, 18 MB per sol), 1–2 minutes. Everything is cached, a
second run downloads nothing:

```bash
.venv\Scripts\python scripts\ingest_gale.py
```

It writes `data/catalog.sqlite`, `data/raw/<source>/` and `data/processed/gale_mount_sharp/`.

- [x] **A1. Catalog module.** [`mars_map/catalog/db.py`](mars_map/catalog/db.py): `connect()`
      creates the SQLite file from `schema.sql`; `add_source`, `add_products`,
      `find_products(bbox, kind)`, `add_layer`, plus `add_region`, `add_series`, `set_local_path`.
      `find_products` returns products finest first, each with `coverage` = share of the bbox
      inside its exact footprint. Re-adding a product keeps its `local_path`.
- [x] **A2. Catalog loader.** [`mars_map/ingest/loaders.py`](mars_map/ingest/loaders.py):
      `load_catalog(db, bbox)`. Both APIs were verified to return footprints and download URLs
      (working queries below). `find_products(gale_bbox, "dtm")` returns real HiRISE and CTX DTMs.
- [x] **A3. Fetcher with cache.** [`mars_map/ingest/fetch.py`](mars_map/ingest/fetch.py):
      `fetch_product()` downloads a whole product into `data/raw/<source>/` through a `.part`
      file, resumes a broken download with an HTTP Range request, raises `FetchError` otherwise.
      `fetch_window()` reads only the pixels over a region from a remote raster and caches the
      crop, which is how the global mosaics (2–23 GB) and 500 MB DTM strips are used.
- [x] **A4. Normalizer.** [`mars_map/ingest/normalize.py`](mars_map/ingest/normalize.py):
      `normalize()` reprojects to the region grid, sets nodata to NaN, converts units, mosaics
      several products of one source (removing the constant height offset between DTMs),
      writes `data/processed/<region>/<layer>__<source>.tif` and registers it in `layers`.
- [ ] **A5. Global coarse layers.** [`mars_map/ingest/global_layers.py`](mars_map/ingest/global_layers.py).
      Done: MOLA 463 m and HRSC+MOLA 200 m elevation, THEMIS day/night IR 100 m (8-bit
      brightness, not kelvin), TES albedo. **Missing:** TES thermal inertia and the TES mineral
      maps, no working direct URL found yet.
- [x] **A6. Time series.** [`mars_map/ingest/series.py`](mars_map/ingest/series.py):
      REMS weather for every mission sol (`temp_min_c`, `temp_max_c`, `ground_temp_min_c`,
      `ground_temp_max_c`, `pressure_pa`) and RAD dose per sol (`radiation_dose_ugy_per_sol`
      measured, `radiation_msv_per_sol` = measured × quality factor 3.05). RAD is 18 MB per
      sol, so sols are loaded on request: `ingest_gale.py --rad-sols 453-482`.
      The REMS feed has no dust opacity and no wind (`dust_opacity_tau`, `wind_max_ms`).

### Working queries (verified 2026-11)

PDS ODE REST, HiRISE DTMs over a box. Longitudes are east 0..360. `results=fmp` adds the file
list (`Product_files`, the `.IMG` with `Type = Product` is the DTM) and the footprint
(`Footprint_C0_geometry`, WKT):

```
https://oderest.rsl.wustl.edu/live2/?query=products&target=mars&results=fmp&output=JSON&ihid=MRO&iid=HIRISE&pt=DTM&westlon=137.346&eastlon=137.414&minlat=-4.767&maxlat=-4.733&limit=100
```

USGS STAC, items of a collection over a box. Longitudes are -180..180, `geometry` is the
footprint, `assets.<key>.href` is a cloud-optimized GeoTIFF, pages are chained by the `next` link:

```
https://stac.astrogeology.usgs.gov/api/collections/mro_ctx_controlled_usgs_dtms/items?bbox=137.346,-4.767,137.414,-4.733&limit=100
```

| Source | Provider | Query | File used |
|---|---|---|---|
| `hirise_dtm` | ODE | `ihid=MRO&iid=HIRISE&pt=DTM` | `DTEEC_*.IMG`, 1 m |
| `ctx_dtm` | STAC | `mro_ctx_controlled_usgs_dtms` | asset `geoid_adjusted_dem`, 20 m |
| `ctx_optical` | STAC | same collection | asset `orthoimage` |
| `hirise_optical` | STAC | `mro_hirise_uncontrolled_observations` | asset `image`, 0.25 m |
| `themis` | STAC | `mo_themis_controlled_mosaics` | asset `image`, 100 m |

Things to know:
- Gale has thousands of CTX stereo-pair DTMs, so the STAC loader stops at `max_items` (200).
- STAC also lists HiRISE DTMs (`mro_hirise_socet_dtms`), but some of its links are dead (404).
  We take HiRISE DTMs from ODE.
- Three HiRISE DTMs cover the Gale region completely; the pipeline uses one. Where one is not
  enough, the normalizer mosaics several.
- All elevations (HiRISE, CTX geoid-adjusted, MOLA) are metres above the MOLA areoid, but the
  sources still disagree by a constant: over Gale, median difference to HiRISE is -5 m for CTX,
  -11 m for HRSC+MOLA, -51 m for MOLA. Fusion has to remove it at the seams.
- Open for meeting point 2: `fuse_region` refuses series that cover different sols, and real
  series do (REMS has ~4700 sols, RAD only the sols that were requested).

Priority for the first target: A1 → A2 → A3 → A4 (elevation), then A6 (radiation), then A5 (minerals).

## B — analysis (regions, rating, fusion)

Works in `mars_map/analysis/` and `mars_map/fusion/`. Branch: `data/analysis`.
Does not wait for A: uses the synthetic site and a small fake catalog until real data arrives.

Run the whole of part B on fake data (stub → fake catalog → fusion → derived layers → `Site`):

```bash
.venv\Scripts\python scripts\build_fixture_site.py
```

It writes `data/fixture/` (catalog + degraded layers) and `data/gale_fixture/` (the fused site
with `preview.png`), and prints how far the fused site is from the stub it was cut from.

- [x] **B1. Fake catalog fixture.** `analysis/fixture.py`: 4 regions (Gale, Jezero, Coprates,
      Arcadia) with invented product footprints, plus Gale layers cut from the stub and degraded
      like real sources (HiRISE with a gap, CTX 20 m, MOLA 463 m, THEMIS 100 m, CRISM 18 m).
- [ ] **B2. Region analyzer.** `analysis/regions.py`: for a region compute data completeness
      (which kinds exist, coverage %, best resolution), safety (walkable %, mean slope) and
      science (number of mineral / geologic units).
      *Done when:* it prints a metrics table for the fixture regions.
- [ ] **B3. Rating.** `analysis/rating.py`: weighted sum of normalized metrics, weights in a
      config file, result stored in `ratings`.
      *Done when:* changing a weight changes the ranking, and the top region is printed.
- [x] **B4. Fusion to one grid.** `fusion/stack.py`: `fuse_region()` takes the processed layers
      of a region, picks the finest source per pixel (HiRISE > CTX > MOLA), blends coarse sources
      in at the seam (no fake cliffs), builds `nodata` and `source_res`, derives `slope`,
      `roughness`, `terrain_class`, returns a valid `Site`.
      Tested on the fixture only; real data from A4 is the next check.
- [x] **B5. Derived layers.** `fusion/derived.py`: `blocked` (hard mask: impassable, too steep,
      no-go), `cost` (times slower than flat firm ground, `inf` where blocked), `value`
      (science score for now). Weights in `CostWeights` are placeholders until `Alg` has real
      speeds per terrain type.
- [ ] **B6. Missing layer types in the stub.** Add `optical`, separate mineral layers and
      `geologic_unit` to the synthetic generator, so `Alg` and `GUI` can use them before real data.
      Coordinate with `Alg`: new layers must not break existing code.

Priority for the first target: B1 → B4 → B5, then B2 → B3, then B6.

## A + B together: a real `Site` for Gale

```bash
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\ingest_gale.py        # part A: real data -> catalog + processed layers
.venv\Scripts\python scripts\build_gale_site.py    # part B: fusion -> data/gale_real/ + preview.png
```

`Site.load("data/gale_real")` then gives the same object as the stub, built from real data.

What the real site has now (first joint run, 2026-11-14):

- Elevation from one HiRISE DTM, 1 m native, 100 % of the region; slope and roughness from it.
  Relief is 407 m over 2 km, median slope 9°, 4.9 % of cells blocked.
- Extra layers carried through: `themis_night_ir`, `themis_day_ir` (8-bit DN, not kelvin), `albedo`.
- Series from Curiosity: REMS weather and RAD dose, cut to the sols all of them share.

Known gaps:

- **No rock abundance, thermal inertia or minerals** in real data yet, so `terrain_class` only
  tells bedrock from impassable (no sand, no boulders), `science_score` is 0 and `cost` comes
  from slope and roughness alone.
- **No targets and no base location.** The start point is picked automatically near the centre.
- **Series are short.** The default RAD sols 453–457 fall into a gap of the REMS feed, only
  sols 455–456 are common. Load more sols with `ingest_gale.py --rad-sols`.
- The region centre (-4.75 N, 137.38 E) is an approximate guess, not a verified boxwork location.

## Meeting points

1. **A2 done** → B switches the analyzer from the fake catalog to the real one.
2. [x] **A4 done for Gale** → B runs fusion on real elevation instead of synthetic.
3. [x] **B4 produces a real `Site`** → hand it to `Alg` and `GUI`; their code should work unchanged.

## Git

- One branch per person (`data/ingest`, `data/analysis`), merge into `Data` through pull requests.
- Shared files (`mars_map/site.py`, `catalog/schema.sql`, `requirements.txt`) change only by agreement.
