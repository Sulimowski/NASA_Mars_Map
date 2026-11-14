"""Checks that the step-0 agreements actually work together.

1. catalog schema creates and accepts rows;
2. a layer written with write_layer comes back identical, on the right grid;
3. the grid lines up with the Site convention (row 0 = north) and real lon/lat.
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mars_map import load_stub  # noqa: E402
from mars_map.grid import (GALE_MOUNT_SHARP, coverage_frac, read_layer,  # noqa: E402
                           write_layer)


def check_schema():
    db = sqlite3.connect(":memory:")
    db.executescript((ROOT / "mars_map" / "catalog" / "schema.sql").read_text(encoding="utf-8"))
    g = GALE_MOUNT_SHARP
    lon0, lat0, lon1, lat1 = g.bounds_lonlat
    db.execute("INSERT INTO sources (name, instrument, provider) VALUES ('hirise_dtm','HiRISE','test')")
    db.execute("INSERT INTO products (source_id, product_id, kind, resolution_m, min_lon, min_lat, "
               "max_lon, max_lat, url) VALUES (1,'FAKE_1','dtm',1.0,?,?,?,?,'http://example.invalid/x')",
               (lon0 - 0.1, lat0 - 0.1, lon1 + 0.1, lat1 + 0.1))
    db.execute("INSERT INTO regions (name, center_lat, center_lon, size_m, resolution_m) VALUES (?,?,?,?,?)",
               (g.name, g.center_lat, g.center_lon, g.size_m, g.resolution_m))
    hit = db.execute("SELECT product_id FROM products WHERE kind='dtm' AND min_lon<=? AND max_lon>=? "
                     "AND min_lat<=? AND max_lat>=?", (lon1, lon0, lat1, lat0)).fetchall()
    assert hit == [("FAKE_1",)], hit
    tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1")]
    print("schema ok:", ", ".join(tables))


def check_layer_roundtrip():
    import rasterio
    from pyproj import Transformer

    site = load_stub()
    g = GALE_MOUNT_SHARP
    assert site.shape == g.shape, (site.shape, g.shape)

    elev = site.layers["elevation"].astype(np.float32).copy()
    elev[site.layers["nodata"]] = np.nan                 # mark the data gap as missing
    with tempfile.TemporaryDirectory() as tmp:
        path = write_layer(g.layer_path("elevation", "synthetic", root=tmp), elev, g,
                           layer="elevation", unit="m", source="synthetic")
        back, tags = read_layer(path)
        assert np.array_equal(back, elev, equal_nan=True)
        assert tags["unit"] == "m" and tags["layer"] == "elevation"
        with rasterio.open(path) as src:
            # pixel centre of the grid must be the region centre in lon/lat
            x, y = src.xy(g.shape[0] / 2 - 0.5, g.shape[1] / 2 - 0.5)
            to_lonlat = Transformer.from_crs(src.crs, f"+proj=longlat +R={3396190} +no_defs",
                                             always_xy=True)
            lon, lat = to_lonlat.transform(x, y)
            assert abs(lon - g.center_lon) < 1e-6 and abs(lat - g.center_lat) < 1e-6, (lon, lat)
            # row 0 must be the northern edge
            _, y_top = src.xy(0, 0)
            _, y_bottom = src.xy(g.shape[0] - 1, 0)
            assert y_top > y_bottom
            size_kb = path.stat().st_size / 1024
    print(f"layer roundtrip ok: {back.shape} {back.dtype}, coverage {coverage_frac(back):.1%}, "
          f"{size_kb:.0f} KB, centre lon/lat = ({lon:.4f}, {lat:.4f})")


if __name__ == "__main__":
    check_schema()
    check_layer_roundtrip()
    print("step 0 OK")
