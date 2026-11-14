"""The catalog: one SQLite file that knows what data exists and what we already have.

Only metadata lives here (footprints, URLs, file paths). The schema is `schema.sql`
next to this file; nobody else writes SQL against the catalog for inserts, so the
rules (east-positive longitudes, upserts, relative paths) are kept in one place.

    db = connect()                                   # data/catalog.sqlite, created if missing
    add_source(db, "hirise_dtm", "HiRISE", "PDS ODE")
    add_products(db, "hirise_dtm", [{...}, {...}])
    for p in find_products(db, grid.bounds_lonlat, "dtm"):
        print(p.product_id, p.resolution_m, p.coverage, p.url)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from ..grid import RegionGrid
from ..site import DATA_DIR, PROJECT_ROOT

CATALOG_DB = DATA_DIR / "catalog.sqlite"
SCHEMA = Path(__file__).with_name("schema.sql")

BBox = tuple[float, float, float, float]          # (min_lon, min_lat, max_lon, max_lat)


@dataclass
class Product:
    """One row of `products`, plus how much of the queried bbox it covers."""
    id: int
    source: str
    product_id: str
    kind: str
    resolution_m: float | None
    footprint_wkt: str | None
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    url: str
    size_bytes: int | None
    acquired: str | None
    local_path: str | None
    coverage: float | None = None    # 0..1 share of the query bbox inside the footprint

    @property
    def bbox(self) -> BBox:
        return self.min_lon, self.min_lat, self.max_lon, self.max_lat


def wrap_lon(lon: float) -> float:
    """Any longitude -> east-positive -180..180 (the catalog convention)."""
    lon = (lon + 180.0) % 360.0 - 180.0
    return 180.0 if lon == -180.0 else lon


def rel_path(path: str | Path) -> str:
    """Paths are stored relative to the repo root when they are inside it."""
    p = Path(path)
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def connect(path: str | Path = CATALOG_DB) -> sqlite3.Connection:
    """Open the catalog, creating the file and the tables if they are missing."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA.read_text(encoding="utf-8"))
    db.execute("PRAGMA foreign_keys = ON")
    return db


# --- sources ---------------------------------------------------------------------

def add_source(db: sqlite3.Connection, name: str, instrument: str, provider: str,
               base_url: str | None = None, license: str | None = None) -> int:
    """Insert or update a source, return its id."""
    db.execute(
        "INSERT INTO sources (name, instrument, provider, base_url, license) VALUES (?,?,?,?,?) "
        "ON CONFLICT(name) DO UPDATE SET instrument=excluded.instrument, "
        "provider=excluded.provider, base_url=excluded.base_url, license=excluded.license",
        (name, instrument, provider, base_url, license))
    db.commit()
    return source_id(db, name)


def source_id(db: sqlite3.Connection, name: str) -> int:
    row = db.execute("SELECT id FROM sources WHERE name=?", (name,)).fetchone()
    if row is None:
        known = [r[0] for r in db.execute("SELECT name FROM sources ORDER BY 1")]
        raise KeyError(f"source {name!r} not in catalog (add_source first), known: {known}")
    return row[0]


# --- products --------------------------------------------------------------------

def _bbox_of_wkt(wkt: str) -> BBox:
    from shapely import wkt as _wkt
    return _wkt.loads(wkt).bounds


def add_products(db: sqlite3.Connection, source: str, products: Iterable[Mapping]) -> int:
    """Insert or update products of one source. Returns how many were written.

    Each product is a mapping with `product_id`, `kind`, `url` and either a `bbox`
    (min_lon, min_lat, max_lon, max_lat) or a `footprint_wkt`; optional
    `resolution_m`, `size_bytes`, `acquired`. Re-adding a product updates its
    metadata but keeps `local_path`, so reloading the catalog never forgets downloads.
    """
    sid = source_id(db, source)
    rows = []
    for p in products:
        wkt = p.get("footprint_wkt")
        bbox = p.get("bbox") or (_bbox_of_wkt(wkt) if wkt else None)
        if bbox is None:
            raise ValueError(f"product {p.get('product_id')!r} has neither bbox nor footprint_wkt")
        lon0, lat0, lon1, lat1 = bbox
        if lon0 > lon1 or lat0 > lat1:
            raise ValueError(f"product {p['product_id']!r}: bbox {bbox} is not (min, min, max, max)")
        rows.append((sid, p["product_id"], p["kind"], p.get("resolution_m"), wkt,
                     lon0, lat0, lon1, lat1, p["url"], p.get("size_bytes"), p.get("acquired")))
    db.executemany(
        "INSERT INTO products (source_id, product_id, kind, resolution_m, footprint_wkt, "
        "min_lon, min_lat, max_lon, max_lat, url, size_bytes, acquired) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(source_id, product_id) DO UPDATE SET kind=excluded.kind, "
        "resolution_m=excluded.resolution_m, footprint_wkt=excluded.footprint_wkt, "
        "min_lon=excluded.min_lon, min_lat=excluded.min_lat, max_lon=excluded.max_lon, "
        "max_lat=excluded.max_lat, url=excluded.url, "
        "size_bytes=COALESCE(excluded.size_bytes, products.size_bytes), "
        "acquired=COALESCE(excluded.acquired, products.acquired)", rows)
    db.commit()
    return len(rows)


def _coverage(bbox: BBox, p: Product) -> float:
    """Share of `bbox` covered by the product (exact footprint if we have one)."""
    from shapely import wkt as _wkt
    from shapely.geometry import box
    query = box(*bbox)
    if query.area == 0:
        return 1.0
    try:
        shape = _wkt.loads(p.footprint_wkt) if p.footprint_wkt else box(*p.bbox)
        if not shape.is_valid:
            shape = shape.buffer(0)
        return float(shape.intersection(query).area / query.area)
    except Exception:                                   # broken WKT: fall back to the bbox
        return float(box(*p.bbox).intersection(query).area / query.area)


def find_products(db: sqlite3.Connection, bbox: BBox, kind: str | None = None,
                  source: str | None = None, min_coverage: float = 0.0) -> list[Product]:
    """Products whose bounding box intersects `bbox` = (min_lon, min_lat, max_lon, max_lat).

    Sorted finest resolution first, then by coverage of the bbox. `coverage` uses the
    exact footprint when the catalog has it, so a tilted strip that only clips the
    corner of the region shows up with a small number instead of looking like a hit.
    """
    lon0, lat0, lon1, lat1 = bbox
    sql = ("SELECT p.id, s.name, p.product_id, p.kind, p.resolution_m, p.footprint_wkt, "
           "p.min_lon, p.min_lat, p.max_lon, p.max_lat, p.url, p.size_bytes, p.acquired, "
           "p.local_path FROM products p JOIN sources s ON s.id = p.source_id "
           "WHERE p.min_lon <= ? AND p.max_lon >= ? AND p.min_lat <= ? AND p.max_lat >= ?")
    args: list = [lon1, lon0, lat1, lat0]
    if kind is not None:
        sql += " AND p.kind = ?"
        args.append(kind)
    if source is not None:
        sql += " AND s.name = ?"
        args.append(source)
    out = []
    for row in db.execute(sql, args):
        p = Product(*row)
        p.coverage = _coverage(bbox, p)
        if p.coverage >= min_coverage:
            out.append(p)
    out.sort(key=lambda p: (p.resolution_m if p.resolution_m is not None else float("inf"),
                            -p.coverage, p.product_id))
    return out


def set_local_path(db: sqlite3.Connection, product: Product | int, path: str | Path | None) -> None:
    """Remember where the fetcher put a product (None = not downloaded any more)."""
    pid = product.id if isinstance(product, Product) else product
    stored = None if path is None else rel_path(path)
    db.execute("UPDATE products SET local_path=? WHERE id=?", (stored, pid))
    db.commit()
    if isinstance(product, Product):
        product.local_path = stored


# --- regions, layers, series -----------------------------------------------------

def add_region(db: sqlite3.Connection, grid: RegionGrid, note: str | None = None) -> int:
    """Insert or update a region from its grid, return its id."""
    db.execute(
        "INSERT INTO regions (name, center_lat, center_lon, size_m, resolution_m, note) "
        "VALUES (?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET center_lat=excluded.center_lat, "
        "center_lon=excluded.center_lon, size_m=excluded.size_m, "
        "resolution_m=excluded.resolution_m, note=COALESCE(excluded.note, regions.note)",
        (grid.name, grid.center_lat, grid.center_lon, grid.size_m, grid.resolution_m, note))
    db.commit()
    return region_id(db, grid.name)


def region_id(db: sqlite3.Connection, name: str) -> int:
    row = db.execute("SELECT id FROM regions WHERE name=?", (name,)).fetchone()
    if row is None:
        known = [r[0] for r in db.execute("SELECT name FROM regions ORDER BY 1")]
        raise KeyError(f"region {name!r} not in catalog (add_region first), known: {known}")
    return row[0]


def get_region(db: sqlite3.Connection, name: str) -> RegionGrid:
    region_id(db, name)
    return RegionGrid(*db.execute(
        "SELECT name, center_lat, center_lon, size_m, resolution_m FROM regions WHERE name=?",
        (name,)).fetchone())


def add_layer(db: sqlite3.Connection, region: str, name: str, unit: str, source: str,
              native_res_m: float, path: str | Path, coverage_frac: float) -> int:
    """Register a processed raster. One row per (region, layer, source): re-running replaces it."""
    db.execute(
        "INSERT INTO layers (region_id, name, unit, source_id, native_res_m, path, coverage_frac) "
        "VALUES (?,?,?,?,?,?,?) ON CONFLICT(region_id, name, source_id) DO UPDATE SET "
        "unit=excluded.unit, native_res_m=excluded.native_res_m, path=excluded.path, "
        "coverage_frac=excluded.coverage_frac, created_at=datetime('now')",
        (region_id(db, region), name, unit, source_id(db, source), native_res_m,
         rel_path(path), coverage_frac))
    db.commit()
    return db.execute(
        "SELECT id FROM layers WHERE region_id=? AND name=? AND source_id=?",
        (region_id(db, region), name, source_id(db, source))).fetchone()[0]


def add_series(db: sqlite3.Connection, region: str, name: str, values: Mapping[int, float],
               source: str | None = None) -> int:
    """Store per-sol values {sol: value} of a region. Existing sols are overwritten."""
    rid = region_id(db, region)
    sid = source_id(db, source) if source else None
    db.executemany(
        "INSERT INTO series (region_id, name, sol, value, source_id) VALUES (?,?,?,?,?) "
        "ON CONFLICT(region_id, name, sol) DO UPDATE SET value=excluded.value, "
        "source_id=excluded.source_id",
        [(rid, name, int(sol), float(v), sid) for sol, v in values.items()])
    db.commit()
    return len(values)
