"""Fake catalog + fake processed layers, so part B can work before part A delivers.

It imitates what ingestion (A) will produce for real:

* `data/fixture/catalog.sqlite` built from the shared schema, with 4 regions and
  invented product footprints (rich coverage for Gale and Jezero, poor for the rest);
* processed GeoTIFF layers for Gale, cut from the synthetic stub and degraded the
  way real sources are: HiRISE has a gap, CTX is 20 m, MOLA is 463 m, THEMIS is 100 m.

Everything here is invented. Nothing in this file touches real NASA data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from ..grid import GALE_MOUNT_SHARP, RegionGrid, coverage_frac, write_layer
from ..site import DATA_DIR, PROJECT_ROOT, Site

FIXTURE_DIR = DATA_DIR / "fixture"
FIXTURE_DB = FIXTURE_DIR / "catalog.sqlite"
SCHEMA = PROJECT_ROOT / "mars_map" / "catalog" / "schema.sql"

# name, instrument, native resolution in metres
SOURCES = [
    ("hirise_dtm", "HiRISE", 1.0),
    ("ctx_dtm", "CTX", 20.0),
    ("mola", "MOLA", 463.0),
    ("hirise_optical", "HiRISE", 0.25),
    ("ctx_optical", "CTX", 6.0),
    ("themis", "THEMIS", 100.0),
    ("crism", "CRISM", 18.0),
    ("rad", "RAD", None),
    ("rems", "REMS", None),
]

# region, lat, lon, {kind: number of fake products}
REGIONS = [
    (GALE_MOUNT_SHARP, {"dtm": 6, "optical": 14, "thermal": 3, "mineral": 5, "series": 2},
     "Curiosity site, Mount Sharp foothills"),
    (RegionGrid("jezero_delta", 18.44, 77.45, 2000.0, 2.0),
     {"dtm": 5, "optical": 12, "thermal": 3, "mineral": 6, "series": 1},
     "Perseverance site, river delta"),
    (RegionGrid("coprates_chasma", -13.2, -61.0, 2000.0, 2.0),
     {"dtm": 1, "optical": 4, "thermal": 2, "mineral": 2}, "Valles Marineris wall, steep"),
    (RegionGrid("arcadia_planitia", 46.0, -171.0, 2000.0, 2.0),
     {"optical": 2, "thermal": 2}, "Northern plains, almost no high-res data"),
]

KIND_SOURCE = {"dtm": "hirise_dtm", "optical": "hirise_optical", "thermal": "themis",
               "mineral": "crism", "series": "rad"}


def coarsen(a: np.ndarray, k: int) -> np.ndarray:
    """Imitate a coarse source resampled to the fine grid: block mean, then bilinear."""
    n0, n1 = a.shape
    p0, p1 = (-n0) % k, (-n1) % k
    p = np.pad(a.astype(np.float64), ((0, p0), (0, p1)), mode="edge")
    coarse = p.reshape(p.shape[0] // k, k, p.shape[1] // k, k).mean(axis=(1, 3))
    cy = (np.arange(coarse.shape[0]) + 0.5) * k - 0.5      # block centres in fine pixels
    cx = (np.arange(coarse.shape[1]) + 0.5) * k - 0.5
    rows = np.array([np.interp(np.arange(n1), cx, r) for r in coarse])
    out = np.array([np.interp(np.arange(n0), cy, c) for c in rows.T]).T
    return out.astype(np.float32)


def _fake_products(rng, grid: RegionGrid, kind: str, count: int):
    lon0, lat0, lon1, lat1 = grid.bounds_lonlat
    w = lon1 - lon0
    for i in range(count):
        dx, dy = rng.uniform(-0.6, 0.6, 2) * w
        s = rng.uniform(0.8, 3.0) * w                       # footprint bigger than region
        a, b, c, d = grid.center_lon + dx - s, grid.center_lat + dy - s, \
            grid.center_lon + dx + s, grid.center_lat + dy + s
        wkt = f"POLYGON(({a} {b},{c} {b},{c} {d},{a} {d},{a} {b}))"
        yield (f"FAKE_{grid.name}_{kind}_{i}", kind, wkt, a, b, c, d,
               f"http://example.invalid/{grid.name}/{kind}/{i}")


def build_fixture(stub: Site, db_path: Path = FIXTURE_DB) -> Path:
    """(Re)create the fixture catalog and the Gale processed layers. Returns the db path."""
    if stub.shape != GALE_MOUNT_SHARP.shape:
        raise ValueError(f"stub {stub.shape} does not match {GALE_MOUNT_SHARP.shape}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA.read_text(encoding="utf-8"))
    rng = np.random.default_rng(7)

    res_of = {}
    for name, instrument, res in SOURCES:
        db.execute("INSERT INTO sources (name, instrument, provider) VALUES (?,?,'fixture')",
                   (name, instrument))
        res_of[name] = res
    sid = dict(db.execute("SELECT name, id FROM sources"))

    for grid, counts, note in REGIONS:
        db.execute("INSERT INTO regions (name, center_lat, center_lon, size_m, resolution_m, note) "
                   "VALUES (?,?,?,?,?,?)", (grid.name, grid.center_lat, grid.center_lon,
                                            grid.size_m, grid.resolution_m, note))
        for kind, count in counts.items():
            src = KIND_SOURCE[kind]
            for pid, k, wkt, a, b, c, d, url in _fake_products(rng, grid, kind, count):
                db.execute("INSERT INTO products (source_id, product_id, kind, resolution_m, "
                           "footprint_wkt, min_lon, min_lat, max_lon, max_lat, url) "
                           "VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (sid[src], pid, k, res_of[src], wkt, a, b, c, d, url))

    # --- Gale processed layers, degraded like real sources ---------------------
    g = GALE_MOUNT_SHARP
    region_id = db.execute("SELECT id FROM regions WHERE name=?", (g.name,)).fetchone()[0]
    L, gap = stub.layers, stub.layers["nodata"]
    k = lambda metres: max(1, int(round(metres / g.resolution_m)))  # noqa: E731

    def with_gap(a, mask):
        out = a.astype(np.float32).copy()
        out[mask] = np.nan
        return out

    rows, cols = np.indices(g.shape)
    ctx_gap = gap & (rows < 0.12 * g.shape[0])               # CTX also misses the far corner
    layers = [  # layer, source, unit, array
        ("elevation", "hirise_dtm", "m", with_gap(L["elevation"], gap)),
        ("elevation", "ctx_dtm", "m", with_gap(coarsen(L["elevation"], k(20)), ctx_gap)),
        ("elevation", "mola", "m", coarsen(L["elevation"], k(463))),
        ("rock_abundance", "hirise_optical", "0..1", with_gap(L["rock_abundance"], gap)),
        ("thermal_inertia", "themis", "J m^-2 K^-1 s^-1/2", coarsen(L["thermal_inertia"], k(100))),
        ("science_score", "crism", "0..1", coarsen(L["science_score"], k(18))),
    ]
    root = FIXTURE_DIR / "processed"
    for layer, src, unit, arr in layers:
        path = write_layer(g.layer_path(layer, src, root=root), arr, g, layer, unit, src)
        db.execute("INSERT INTO layers (region_id, name, unit, source_id, native_res_m, path, "
                   "coverage_frac) VALUES (?,?,?,?,?,?,?)",
                   (region_id, layer, unit, sid[src], res_of[src],
                    path.relative_to(PROJECT_ROOT).as_posix(), coverage_frac(arr)))

    for name, values in stub.series.items():
        if name == "sol":
            continue
        src = sid["rad" if name.startswith("radiation") else "rems"]
        db.executemany("INSERT INTO series (region_id, name, sol, value, source_id) VALUES (?,?,?,?,?)",
                       [(region_id, name, sol, v, src) for sol, v in zip(stub.series["sol"], values)])
    db.commit()
    db.close()
    return db_path
