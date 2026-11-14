"""Catalog loaders: ask NASA what exists over a bounding box and write it to `products`.

Nothing is downloaded here except small JSON answers. Two providers, both verified
(2026-11) to return footprints and direct download URLs:

* PDS ODE REST (https://oderest.rsl.wustl.edu/live2/) - HiRISE DTMs as PDS3 .IMG.
  Longitudes in the API are east-positive 0..360.
* USGS Astrogeology STAC (https://stac.astrogeology.usgs.gov/api/) - CTX DTMs with
  orthoimages, HiRISE images, THEMIS IR mosaics, all as cloud-optimized GeoTIFFs.

    load_catalog(db, GALE_MOUNT_SHARP.bounds_lonlat)
"""

from __future__ import annotations

import json
import sqlite3
import urllib.parse
import urllib.request

from ..catalog.db import BBox, add_products, add_source, wrap_lon

ODE_URL = "https://oderest.rsl.wustl.edu/live2/"
STAC_URL = "https://stac.astrogeology.usgs.gov/api"
USER_AGENT = "mars-map-ingest/1.0"
TIMEOUT_S = 90

# name -> (instrument, provider, base_url)
SOURCES = {
    "hirise_dtm": ("HiRISE", "PDS ODE", ODE_URL),
    "ctx_dtm": ("CTX", "USGS STAC", STAC_URL),
    "ctx_optical": ("CTX", "USGS STAC", STAC_URL),
    "hirise_optical": ("HiRISE", "USGS STAC", STAC_URL),
    "themis": ("THEMIS", "USGS STAC", STAC_URL),
}

# STAC collection -> list of (source, kind, asset key, resolution if `gsd` is not the right one)
STAC_COLLECTIONS = {
    # heights above the areoid (same datum as MOLA and the HiRISE DTMs)
    # The asset key is not the same across the collection: "geoid_adjusted_dem" around
    # Gale, plain "dtm" around Jezero. Both are metres above the areoid (checked against MOLA).
    "mro_ctx_controlled_usgs_dtms": [("ctx_dtm", "dtm", ("geoid_adjusted_dem", "dtm"), None),
                                     ("ctx_optical", "optical", "orthoimage", None)],
    "mro_hirise_uncontrolled_observations": [("hirise_optical", "optical", "image", None)],
    "mo_themis_controlled_mosaics": [("themis", "thermal", "image", None)],
}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.load(resp)


def _register(db: sqlite3.Connection, source: str) -> None:
    instrument, provider, base = SOURCES[source]
    add_source(db, source, instrument, provider, base_url=base, license="NASA public domain")


# --- PDS ODE ---------------------------------------------------------------------

def ode_query_url(bbox: BBox, ihid: str, iid: str, pt: str, limit: int = 100, offset: int = 0) -> str:
    """The working ODE query: products of one type whose footprint touches the bbox."""
    lon0, lat0, lon1, lat1 = bbox
    params = {"query": "products", "target": "mars", "results": "fmp", "output": "JSON",
              "ihid": ihid, "iid": iid, "pt": pt,
              "westlon": f"{lon0 % 360:.5f}", "eastlon": f"{lon1 % 360:.5f}",
              "minlat": f"{lat0:.5f}", "maxlat": f"{lat1:.5f}",
              "limit": limit, "offset": offset}
    return ODE_URL + "?" + urllib.parse.urlencode(params)


def _ode_products(answer: dict) -> list[dict]:
    res = answer["ODEResults"]
    if res.get("Status") != "Success":
        raise RuntimeError(f"ODE error: {res.get('Error', res)}")
    prods = res.get("Products")
    if not isinstance(prods, dict):                 # "No Products Found"
        return []
    p = prods["Product"]
    return p if isinstance(p, list) else [p]        # a single hit is not wrapped in a list


def _ode_files(product: dict) -> list[dict]:
    f = (product.get("Product_files") or {}).get("Product_file", [])
    return f if isinstance(f, list) else [f]


def load_ode_hirise_dtm(db: sqlite3.Connection, bbox: BBox, max_items: int = 500) -> int:
    """HiRISE stereo DTMs (DTEEC/DTEPC .IMG, ~1 m) from PDS ODE."""
    _register(db, "hirise_dtm")
    rows, offset = [], 0
    while offset < max_items:
        page = _ode_products(_get_json(ode_query_url(bbox, "MRO", "HIRISE", "DTM", 100, offset)))
        for p in page:
            data = [f for f in _ode_files(p) if f.get("Type") == "Product"
                    and f["FileName"].upper().startswith("DTE") and f["FileName"].upper().endswith(".IMG")]
            if not data:
                continue
            wkt = p.get("Footprint_C0_geometry") or p.get("Footprint_geometry")
            if wkt and "EMPTY" in wkt:
                wkt = None
            rows.append({
                "product_id": p["pdsid"], "kind": "dtm", "url": data[0]["URL"],
                "resolution_m": float(p["Map_scale"]) if p.get("Map_scale") else None,
                "footprint_wkt": wkt,
                "bbox": (wrap_lon(float(p["Westernmost_longitude"])), float(p["Minimum_latitude"]),
                         wrap_lon(float(p["Easternmost_longitude"])), float(p["Maximum_latitude"])),
                "size_bytes": int(float(data[0]["KBytes"]) * 1024) if data[0].get("KBytes") else None,
                "acquired": (p.get("Observation_time") or p.get("UTC_start_time") or "")[:10] or None,
            })
        if len(page) < 100:
            break
        offset += 100
    return add_products(db, "hirise_dtm", rows)


# --- USGS STAC -------------------------------------------------------------------

def stac_query_url(collection: str, bbox: BBox, limit: int = 100) -> str:
    """The working STAC query: items of a collection intersecting the bbox (lon -180..180)."""
    return (f"{STAC_URL}/collections/{collection}/items?"
            f"bbox={bbox[0]:.5f},{bbox[1]:.5f},{bbox[2]:.5f},{bbox[3]:.5f}&limit={limit}")


def _geojson_to_wkt(geom: dict | None) -> str | None:
    if not geom:
        return None
    try:
        from shapely.geometry import shape
        return shape(geom).wkt
    except Exception:
        return None


def load_stac(db: sqlite3.Connection, collection: str, bbox: BBox, max_items: int = 200) -> int:
    """One STAC collection -> products. Follows `next` links up to `max_items` items.

    Gale alone has thousands of CTX stereo-pair DTMs, hence the cap: the first
    `max_items` are plenty to choose from.
    """
    targets = STAC_COLLECTIONS[collection]
    for source, *_ in targets:
        _register(db, source)
    rows: dict[str, list[dict]] = {source: [] for source, *_ in targets}
    url, seen = stac_query_url(collection, bbox, min(100, max_items)), 0
    while url and seen < max_items:
        page = _get_json(url)
        feats = page.get("features", [])
        for f in feats[: max_items - seen]:
            b = f.get("bbox") or []
            # some items carry a projected bbox (metres); only trust it when it looks like degrees
            if len(b) == 4 and all(abs(v) <= 360 for v in b):
                box = (wrap_lon(b[0]), min(b[1], b[3]), wrap_lon(b[2]), max(b[1], b[3]))
                if box[0] > box[2]:
                    box = None
            else:
                box = None
            wkt = _geojson_to_wkt(f.get("geometry"))
            if box is None and wkt is None:
                continue
            props = f.get("properties", {})
            for source, kind, asset, res in targets:
                keys = (asset,) if isinstance(asset, str) else asset
                a = next((f["assets"][k] for k in keys if k in f.get("assets", {})), None)
                if not a:
                    continue
                rows[source].append({
                    "product_id": f["id"], "kind": kind, "url": a["href"],
                    "resolution_m": res if res is not None else props.get("gsd"),
                    "footprint_wkt": wkt, "bbox": box,
                    "acquired": (props.get("datetime") or props.get("start_datetime") or "")[:10] or None,
                })
        seen += len(feats)
        nxt = [l["href"] for l in page.get("links", []) if l.get("rel") == "next"]
        url = nxt[0] if nxt and feats else None
    return sum(add_products(db, source, r) for source, r in rows.items())


def load_catalog(db: sqlite3.Connection, bbox: BBox, max_items: int = 200,
                 verbose: bool = True) -> dict[str, int]:
    """Everything we know how to list over `bbox`. Returns {loader: products written}."""
    counts = {"ode:hirise_dtm": load_ode_hirise_dtm(db, bbox)}
    for collection in STAC_COLLECTIONS:
        counts[f"stac:{collection}"] = load_stac(db, collection, bbox, max_items)
    if verbose:
        for k, v in counts.items():
            print(f"  catalog {k}: {v} products")
    return counts
