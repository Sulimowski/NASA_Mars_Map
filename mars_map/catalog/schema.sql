-- Catalog of Mars data: what exists, where it is, what we already processed.
-- Rasters are NOT stored here, only metadata and file paths.
-- Longitudes are east-positive, -180..180. Latitudes are planetocentric.
-- Change this file only by agreement between A and B.

PRAGMA foreign_keys = ON;

-- Where data comes from (one row per instrument/provider pair).
CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,      -- e.g. "hirise_dtm", "ctx_mosaic", "tes_thermal_inertia"
    instrument  TEXT NOT NULL,             -- HiRISE, CTX, THEMIS, TES, CRISM, MOLA, RAD, REMS
    provider    TEXT NOT NULL,             -- PDS ODE, USGS STAC, Trek, ...
    base_url    TEXT,
    license     TEXT
);

-- Everything that EXISTS remotely (metadata only, nothing downloaded yet).
CREATE TABLE IF NOT EXISTS products (
    id            INTEGER PRIMARY KEY,
    source_id     INTEGER NOT NULL REFERENCES sources(id),
    product_id    TEXT NOT NULL,           -- provider's own id, e.g. "DTEEC_010573_1755_010639_1755_U01"
    kind          TEXT NOT NULL CHECK (kind IN
                    ('dtm', 'optical', 'thermal', 'mineral', 'geology', 'series')),
    resolution_m  REAL,                    -- metres per pixel, NULL for vector / tables
    footprint_wkt TEXT,                    -- exact outline, WKT POLYGON in lon/lat
    min_lon       REAL NOT NULL,
    min_lat       REAL NOT NULL,
    max_lon       REAL NOT NULL,
    max_lat       REAL NOT NULL,
    url           TEXT NOT NULL,           -- direct download link
    size_bytes    INTEGER,
    acquired      TEXT,                    -- ISO date of observation
    local_path    TEXT,                    -- set by the fetcher once it is in data/raw/
    UNIQUE (source_id, product_id)
);
CREATE INDEX IF NOT EXISTS idx_products_bbox ON products (kind, min_lon, max_lon, min_lat, max_lat);

-- Candidate exploration areas. size_m and resolution_m define the region grid.
CREATE TABLE IF NOT EXISTS regions (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,    -- e.g. "gale_mount_sharp"
    center_lat    REAL NOT NULL,
    center_lon    REAL NOT NULL,
    size_m        REAL NOT NULL,           -- square side
    resolution_m  REAL NOT NULL,
    note          TEXT
);

-- Processed rasters on a region grid (output of the normalizer, input of fusion).
CREATE TABLE IF NOT EXISTS layers (
    id            INTEGER PRIMARY KEY,
    region_id     INTEGER NOT NULL REFERENCES regions(id),
    name          TEXT NOT NULL,           -- layer key, e.g. "elevation", "thermal_inertia"
    unit          TEXT NOT NULL,
    source_id     INTEGER NOT NULL REFERENCES sources(id),
    native_res_m  REAL NOT NULL,           -- resolution of the source before resampling
    path          TEXT NOT NULL,           -- relative to repo root: data/processed/<region>/<file>.tif
    coverage_frac REAL NOT NULL,           -- 0..1 share of region cells with real data
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (region_id, name, source_id)    -- same layer may exist from several sources (HiRISE + CTX)
);

-- Per-sol values that apply to a whole region (radiation, weather).
CREATE TABLE IF NOT EXISTS series (
    region_id  INTEGER NOT NULL REFERENCES regions(id),
    name       TEXT NOT NULL,              -- e.g. "radiation_msv_per_sol"
    sol        INTEGER NOT NULL,
    value      REAL NOT NULL,
    source_id  INTEGER REFERENCES sources(id),
    PRIMARY KEY (region_id, name, sol)
);

-- Region analyzer output: one row per metric, plus the final "rating".
CREATE TABLE IF NOT EXISTS ratings (
    region_id   INTEGER NOT NULL REFERENCES regions(id),
    metric      TEXT NOT NULL,             -- e.g. "coverage_dtm", "walkable_frac", "rating"
    value       REAL NOT NULL,
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (region_id, metric)
);
