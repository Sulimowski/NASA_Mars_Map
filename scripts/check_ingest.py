"""Offline checks for part A (ingestion). No internet needed: a tiny local web server
plays NASA, and the "raw DTM" is generated from a known formula.

A1  three fake products are inserted and found by bounding box
A3  the second fetch of the same file does not hit the network; a broken download
    resumes; a hopeless one fails cleanly
A4  a raw DTM in another projection, in centimetres, with gaps, becomes `elevation`
    on the region grid in metres; two overlapping DTMs are mosaicked without a step
A6  the REMS and RAD parsers read their formats

    .venv\\Scripts\\python scripts\\check_ingest.py
"""

import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mars_map.catalog import (add_products, add_source, connect, find_products,  # noqa: E402
                              set_local_path)
from mars_map.grid import GALE_MOUNT_SHARP, MARS_SPHERE_RADIUS_M, read_layer  # noqa: E402
from mars_map.ingest.fetch import FetchError, fetch, fetch_product, fetch_window  # noqa: E402
from mars_map.ingest.normalize import normalize  # noqa: E402
from mars_map.ingest.series import parse_rad_dose_rate, parse_rems  # noqa: E402

G = GALE_MOUNT_SHARP


# --- A1 --------------------------------------------------------------------------

def check_catalog(tmp: Path):
    db = connect(tmp / "catalog.sqlite")
    add_source(db, "hirise_dtm", "HiRISE", "test")
    add_source(db, "ctx_optical", "CTX", "test")
    lon0, lat0, lon1, lat1 = G.bounds_lonlat
    w = lon1 - lon0
    full = (lon0 - w, lat0 - w, lon1 + w, lat1 + w)                  # covers the whole region
    west_half = f"POLYGON(({lon0 - w} {lat0 - w},{G.center_lon} {lat0 - w}," \
                f"{G.center_lon} {lat1 + w},{lon0 - w} {lat1 + w},{lon0 - w} {lat0 - w}))"
    n = add_products(db, "hirise_dtm", [
        {"product_id": "FAKE_FULL", "kind": "dtm", "resolution_m": 2.0, "bbox": full,
         "url": "http://example.invalid/full.img", "size_bytes": 10},
        {"product_id": "FAKE_HALF", "kind": "dtm", "resolution_m": 1.0, "footprint_wkt": west_half,
         "url": "http://example.invalid/half.img"},
        {"product_id": "FAKE_FAR", "kind": "dtm", "resolution_m": 1.0,
         "bbox": (10.0, 10.0, 11.0, 11.0), "url": "http://example.invalid/far.img"},
    ])
    n += add_products(db, "ctx_optical", [{"product_id": "FAKE_IMG", "kind": "optical",
                                           "resolution_m": 6.0, "bbox": full,
                                           "url": "http://example.invalid/img.tif"}])
    assert n == 4

    hits = find_products(db, G.bounds_lonlat, "dtm")
    assert [p.product_id for p in hits] == ["FAKE_HALF", "FAKE_FULL"], hits      # finest first
    assert abs(hits[0].coverage - 0.5) < 1e-6 and abs(hits[1].coverage - 1.0) < 1e-9
    assert [p.product_id for p in find_products(db, G.bounds_lonlat, "dtm", min_coverage=0.9)] \
        == ["FAKE_FULL"]
    assert len(find_products(db, G.bounds_lonlat)) == 3                          # any kind
    assert [p.product_id for p in find_products(db, (10.2, 10.2, 10.3, 10.3), "dtm")] == ["FAKE_FAR"]
    assert find_products(db, (50, 50, 51, 51)) == []

    # reloading the catalog must not forget what was downloaded
    set_local_path(db, hits[1], tmp / "full.img")
    add_products(db, "hirise_dtm", [{"product_id": "FAKE_FULL", "kind": "dtm", "resolution_m": 2.0,
                                     "bbox": full, "url": "http://example.invalid/full_v2.img"}])
    again = find_products(db, G.bounds_lonlat, "dtm", min_coverage=0.9)[0]
    assert again.url.endswith("full_v2.img") and again.local_path and again.size_bytes == 10
    db.close()
    print("A1 catalog ok: 3 fake DTMs inserted, 2 found over Gale by bbox, sorted by resolution")


# --- A3 --------------------------------------------------------------------------

PAYLOAD = bytes(range(256)) * 4000                    # ~1 MB


class FakeNasa(BaseHTTPRequestHandler):
    hits = 0                # GET requests served
    break_after = None      # cut the connection after this many bytes (once)
    always_break = False

    def log_message(self, *a):
        pass

    def do_GET(self):
        cls = FakeNasa
        cls.hits += 1
        if self.path != "/dtm.img":
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(PAYLOAD)}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if cls.break_after is not None:
            body = body[: cls.break_after]
            if not cls.always_break:
                cls.break_after = None
        self.wfile.write(body)
        self.wfile.flush()


def check_fetch(tmp: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeNasa)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/dtm.img"
    try:
        # 1. plain download, then cache
        a = fetch(url, tmp / "raw" / "a.img")
        assert a.read_bytes() == PAYLOAD and FakeNasa.hits == 1
        fetch(url, tmp / "raw" / "a.img")
        assert FakeNasa.hits == 1, "second fetch hit the network"

        # 2. same through the catalog: local_path is remembered
        db = connect(tmp / "fetch.sqlite")
        add_source(db, "hirise_dtm", "HiRISE", "test")
        add_products(db, "hirise_dtm", [{"product_id": "P", "kind": "dtm", "url": url,
                                         "bbox": G.bounds_lonlat, "size_bytes": len(PAYLOAD)}])
        p = find_products(db, G.bounds_lonlat, "dtm")[0]
        path = fetch_product(db, p, tmp / "raw", progress=False)
        assert path == tmp / "raw" / "hirise_dtm" / "dtm.img" and FakeNasa.hits == 2
        p = find_products(db, G.bounds_lonlat, "dtm")[0]              # fresh row from the db
        assert p.local_path and fetch_product(db, p, tmp / "raw") == path and FakeNasa.hits == 2
        db.close()

        # 3. connection drops at 300 kB: nothing appears under the final name, retry resumes
        FakeNasa.break_after = 300_000
        b = fetch(url, tmp / "raw" / "b.img", retries=2)
        assert b.read_bytes() == PAYLOAD and FakeNasa.hits == 4       # 1 broken + 1 resumed
        assert not (tmp / "raw" / "b.img.part").exists()

        # 4. it keeps dropping: clean failure, partial kept, a later call finishes the job
        FakeNasa.break_after, FakeNasa.always_break = 200_000, True
        c = tmp / "raw" / "c.img"
        try:
            fetch(url, c, retries=2)
            raise AssertionError("broken download was accepted")
        except FetchError:
            pass
        part = c.with_name("c.img.part")
        assert not c.exists() and part.stat().st_size == 400_000      # two attempts, both kept
        FakeNasa.break_after, FakeNasa.always_break = None, False
        assert fetch(url, c).read_bytes() == PAYLOAD and not part.exists()

        # 5. a missing file is an error, not an empty file
        try:
            fetch(url + ".nope", tmp / "raw" / "d.img")
            raise AssertionError("404 was accepted")
        except FetchError:
            assert not (tmp / "raw" / "d.img").exists()
    finally:
        server.shutdown()
    print("A3 fetch ok: cached second call made 0 requests, broken download resumed, "
          "failures leave no half file")


# --- A4 --------------------------------------------------------------------------

def truth(lon, lat):
    """Invented ground: a tilted plane plus a bump, metres. Smooth, so resampling is exact enough."""
    dx = np.radians(lon - G.center_lon) * MARS_SPHERE_RADIUS_M * np.cos(np.radians(G.center_lat))
    dy = np.radians(lat - G.center_lat) * MARS_SPHERE_RADIUS_M
    return -4400.0 + 0.05 * dx - 0.03 * dy + 40.0 * np.exp(-(dx ** 2 + dy ** 2) / 600.0 ** 2)


def write_raw_dtm(path: Path, lon_from: float, lon_to: float, shift_m: float = 0.0) -> Path:
    """A provider-style DTM: global equirectangular (lat_ts=0), 1.5 m, int32 CENTIMETRES,
    nodata -32767, covering only longitudes lon_from..lon_to of the region (+ margin)."""
    import rasterio
    from rasterio.transform import from_origin

    lon0, lat0, lon1, lat1 = G.bounds_lonlat
    pad = 0.002
    res = 1.5
    k = np.radians(1.0) * MARS_SPHERE_RADIUS_M                        # metres per degree, lat_ts=0
    x0, x1 = (lon_from - pad) * k, (lon_to + pad) * k
    y0, y1 = (lat0 - pad) * k, (lat1 + pad) * k
    w, h = int((x1 - x0) / res), int((y1 - y0) / res)
    xs = x0 + (np.arange(w) + 0.5) * res
    ys = y1 - (np.arange(h) + 0.5) * res
    lon, lat = np.meshgrid(xs / k, ys / k)
    cm = np.round((truth(lon, lat) + shift_m) * 100.0).astype(np.int32)
    cm[:200, :200] = -32767                                           # a 300 m hole in the NW corner
    crs = f"+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +R={MARS_SPHERE_RADIUS_M} +units=m +no_defs"
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="int32",
                       crs=crs, transform=from_origin(x0, y1, res, res), nodata=-32767) as dst:
        dst.write(cm, 1)
    return path


def check_normalize(tmp: Path):
    db = connect(tmp / "norm.sqlite")
    add_source(db, "hirise_dtm", "HiRISE", "test")
    lon0, lat0, lon1, lat1 = G.bounds_lonlat
    mid = lon0 + 0.6 * (lon1 - lon0)
    west = write_raw_dtm(tmp / "west.tif", lon0, mid)                             # west 60 %
    east = write_raw_dtm(tmp / "east.tif", lon0 + 0.3 * (lon1 - lon0), lon1, shift_m=7.0)  # 7 m off

    # the fetcher's crop of a local "remote" file keeps pixels untouched
    crop = fetch_window(str(west), "hirise_dtm", G, tmp / "raw")
    assert crop is not None and fetch_window(str(west), "hirise_dtm", G, tmp / "raw") == crop

    rows, cols = np.indices(G.shape)
    lat = G.center_lat + np.degrees((G.size_m / 2 - (rows + 0.5) * G.resolution_m) / MARS_SPHERE_RADIUS_M)
    lon = G.center_lon + np.degrees(((cols + 0.5) * G.resolution_m - G.size_m / 2)
                                    / (MARS_SPHERE_RADIUS_M * np.cos(np.radians(G.center_lat))))
    want = truth(lon, lat)

    # one strip: metres, on the grid, gap where the strip ends
    path = normalize(db, G, "elevation", "m", "hirise_dtm", [crop], native_res_m=1.5, scale=0.01,
                     valid_range=(-900_000, 2_200_000), processed_root=tmp / "processed")
    elev, tags = read_layer(path)
    assert elev.shape == G.shape and elev.dtype == np.float32 and tags["unit"] == "m"
    have = np.isfinite(elev)
    assert 0.6 < have.mean() < 0.7, have.mean()
    assert have[:, : G.shape[1] // 2].mean() > 0.97 and not have[:, -50:].any()
    assert not have[:50, :50].any(), "provider nodata (-32767) leaked into the layer"
    err = np.abs(elev - want)[have]
    assert err.max() < 0.05, f"elevation off by up to {err.max():.3f} m"

    # two strips with a 7 m disagreement: full coverage and no cliff at the seam
    path = normalize(db, G, "elevation", "m", "hirise_dtm", [west, east], native_res_m=1.5,
                     scale=0.01, valid_range=(-900_000, 2_200_000), align_offsets=True,
                     processed_root=tmp / "processed")
    elev, _ = read_layer(path)
    assert 0.99 < np.isfinite(elev).mean() < 1.0          # only the hole is left
    err = np.abs(elev - want)[np.isfinite(elev)]
    assert err.max() < 0.1, f"seam step left: {err.max():.2f} m"

    row = db.execute("SELECT r.name, l.name, l.unit, s.name, l.native_res_m, l.path, l.coverage_frac "
                     "FROM layers l JOIN regions r ON r.id=l.region_id JOIN sources s ON s.id=l.source_id"
                     ).fetchall()
    assert len(row) == 1 and row[0][:5] == (G.name, "elevation", "m", "hirise_dtm", 1.5), row
    assert row[0][5].endswith("elevation__hirise_dtm.tif") and row[0][6] > 0.99

    # a raster that misses the region registers nothing
    add_source(db, "ctx_dtm", "CTX", "test")
    far = write_raw_dtm(tmp / "far.tif", lon1 + 1.0, lon1 + 1.02)
    assert normalize(db, G, "elevation", "m", "ctx_dtm", [far], 20.0,
                     processed_root=tmp / "processed") is None
    assert db.execute("SELECT count(*) FROM layers").fetchone()[0] == 1
    db.close()
    print(f"A4 normalize ok: cm -> m on the {G.shape[0]}x{G.shape[1]} grid, max error "
          f"{err.max() * 100:.1f} cm, two strips merged without a seam, layer registered")


# --- A6 --------------------------------------------------------------------------

def check_series_parsers():
    feed = {"soles": [
        {"sol": "10", "min_temp": "-75", "max_temp": "-3", "pressure": "740",
         "min_gts_temp": "-90", "max_gts_temp": "6"},
        {"sol": "11", "min_temp": "--", "max_temp": "-5", "pressure": "--"},
    ]}
    s = parse_rems(feed)
    assert s["temp_min_c"] == {10: -75.0} and s["temp_max_c"] == {10: -3.0, 11: -5.0}
    assert s["pressure_pa"] == {10: 740.0} and s["ground_temp_max_c"] == {10: 6.0}

    rad = ("[DOSIMETRY_TOTAL_DOSE_B: 00]\n\n9.9\n\n[DOSIMETRY_TOTAL_DOSE_E: 00]\n\n8.5\n\n"
           "[DOSIMETRY_LET_B_A1: 00]\n\n 0.1 0.2\n\n[DOSIMETRY_TOTAL_DOSE_E: 01]\n\n9.5\n\n"
           "[DOSIMETRY_TOTAL_DOSE_E: 02]\n\n0.00000\n")
    assert abs(parse_rad_dose_rate(rad) - 9.0) < 1e-9
    assert parse_rad_dose_rate("[OBSERVATION: 00]\n") is None
    print("A6 parsers ok: REMS feed and RAD dosimetry")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        check_catalog(tmp)
        check_fetch(tmp)
        check_normalize(tmp)
        check_series_parsers()
    print("part A offline checks OK")
