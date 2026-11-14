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

- [ ] **A1. Catalog module.** `mars_map/catalog/db.py`: create the SQLite file from `schema.sql`,
      functions `add_source`, `add_products`, `find_products(bbox, kind)`, `add_layer`.
      *Done when:* a test inserts 3 fake products and finds them by bounding box.
- [ ] **A2. Catalog loader.** Fill `products` with metadata for HiRISE, CTX and DTM products from
      PDS ODE REST and/or USGS STAC. **First verify** that these APIs return footprints and
      download URLs; write down the working query in this README.
      *Done when:* `find_products(gale_bbox, "dtm")` returns real products with URLs.
- [ ] **A3. Fetcher with cache.** `ingest/fetch.py`: download a product into `data/raw/`, skip if
      already there, resume or fail cleanly on a broken download.
      *Done when:* the second call for the same product does not hit the network.
- [ ] **A4. Normalizer.** `ingest/normalize.py`: reproject to the region grid, crop to the region,
      convert units, set nodata, write `data/processed/<region>/<layer>.tif`, register in `layers`.
      *Done when:* a Gale DTM becomes `elevation.tif` on the region grid with heights in metres.
- [ ] **A5. Global coarse layers.** MOLA elevation, THEMIS day/night IR, TES thermal inertia and
      mineral maps, cropped per region through A4 (they are the fallback where high-res is missing).
- [ ] **A6. Time series.** `ingest/series.py`: Curiosity RAD dose and REMS weather per sol into
      the `series` table.

Priority for the first target: A1 → A2 → A3 → A4 (elevation), then A6 (radiation), then A5 (minerals).

## B — analysis (regions, rating, fusion)

Works in `mars_map/analysis/` and `mars_map/fusion/`. Branch: `data/analysis`.
Does not wait for A: uses the synthetic site and a small fake catalog until real data arrives.

- [ ] **B1. Fake catalog fixture.** 3–4 regions (Gale, Jezero, two more) with invented product
      footprints, written through A's schema. Lets B2/B3 start on day one.
- [ ] **B2. Region analyzer.** `analysis/regions.py`: for a region compute data completeness
      (which kinds exist, coverage %, best resolution), safety (walkable %, mean slope) and
      science (number of mineral / geologic units).
      *Done when:* it prints a metrics table for the fixture regions.
- [ ] **B3. Rating.** `analysis/rating.py`: weighted sum of normalized metrics, weights in a
      config file, result stored in `ratings`.
      *Done when:* changing a weight changes the ranking, and the top region is printed.
- [ ] **B4. Fusion to one grid.** `fusion/stack.py`: take processed layers of a region, resample
      to one grid, apply source priority (HiRISE > CTX > MOLA), build the `nodata` mask, derive
      `slope` and `roughness`, return a valid `Site`.
      *Done when:* `Site.validate()` passes and `scripts/check_reachability.py` runs on the result.
- [ ] **B5. Derived layers.** `fusion/derived.py`: `cost` (weighted slope / rocks / sand),
      `value` (from minerals and geology), hard masks (impassable, no-go).
      Hard limits are masks, never weights.
- [ ] **B6. Missing layer types in the stub.** Add `optical`, separate mineral layers and
      `geologic_unit` to the synthetic generator, so `Alg` and `GUI` can use them before real data.
      Coordinate with `Alg`: new layers must not break existing code.

Priority for the first target: B1 → B4 → B5, then B2 → B3, then B6.

## Meeting points

1. **A2 done** → B switches the analyzer from the fake catalog to the real one.
2. **A4 done for Gale** → B runs fusion on real elevation instead of synthetic.
3. **B4 produces a real `Site`** → hand it to `Alg` and `GUI`; their code should work unchanged.

## Git

- One branch per person (`data/ingest`, `data/analysis`), merge into `Data` through pull requests.
- Shared files (`mars_map/site.py`, `catalog/schema.sql`, `requirements.txt`) change only by agreement.
