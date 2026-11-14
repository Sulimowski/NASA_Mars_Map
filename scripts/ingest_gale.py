"""Part A end to end on real NASA data, for one region (default: Gale, Mount Sharp).

catalog (ODE + STAC) -> pick DTMs -> fetch -> normalize -> global layers -> time series

    .venv\\Scripts\\python scripts\\ingest_gale.py                # ~150 MB of traffic, crops only
    .venv\\Scripts\\python scripts\\ingest_gale.py --full         # download whole DTM files instead
    .venv\\Scripts\\python scripts\\ingest_gale.py --rad-sols 453-482

Everything is cached: a second run does not download anything again.
Output: data/catalog.sqlite, data/raw/, data/processed/<region>/.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mars_map.catalog import add_region, connect, find_products  # noqa: E402
from mars_map.grid import REGIONS, read_layer  # noqa: E402
from mars_map.ingest.fetch import FetchError, fetch_product, fetch_window  # noqa: E402
from mars_map.ingest.global_layers import ingest_global_layers  # noqa: E402
from mars_map.ingest.loaders import load_catalog  # noqa: E402
from mars_map.ingest.normalize import ELEVATION_RANGE_M, normalize  # noqa: E402
from mars_map.ingest.minerals import ingest_minerals, load_ode_crism  # noqa: E402
from mars_map.ingest.optical import ingest_optical, load_ode_hirise_ortho  # noqa: E402
from mars_map.ingest.series import load_rad, load_rems  # noqa: E402
from mars_map.ingest.thermal import ingest_thermal_inertia  # noqa: E402


def pick_cover(products, bbox, max_n):
    """Greedy: take the product that adds the most uncovered area, until nothing is gained."""
    from shapely import wkt
    from shapely.geometry import box
    query, covered, chosen = box(*bbox), None, []
    shapes = {p.id: (wkt.loads(p.footprint_wkt) if p.footprint_wkt else box(*p.bbox)).buffer(0)
              for p in products}
    left = list(products)
    while left and len(chosen) < max_n:
        def gain(p):
            s = shapes[p.id].intersection(query)
            return (s.difference(covered).area if covered is not None else s.area) / query.area
        best = max(left, key=gain)
        if gain(best) < 0.01:
            break
        s = shapes[best.id].intersection(query)
        covered = s if covered is None else covered.union(s)
        chosen.append(best)
        left.remove(best)
    return chosen, (covered.area / query.area if covered is not None else 0.0)


def get_raw(db, product, grid, full):
    """Whole file (--full) or only the pixels over the region."""
    if full:
        return fetch_product(db, product)
    return fetch_window(product.url, product.source, grid)


def ingest_dtm(db, grid, source, max_n, full):
    found = find_products(db, grid.bounds_lonlat, "dtm", source=source)
    chosen, cover = pick_cover(found, grid.bounds_lonlat, max_n)
    print(f"  {source}: {len(found)} in catalog, using {len(chosen)} "
          f"(footprints cover {cover:.0%} of the region)")
    paths = []
    for p in chosen:
        size = f"{p.size_bytes / 1e6:.0f} MB" if p.size_bytes else "size unknown"
        try:
            raw = get_raw(db, p, grid, full)
        except FetchError as e:
            print(f"    {p.product_id}: SKIPPED, {e}")
            continue
        if raw:
            paths.append(raw)
            print(f"    {p.product_id} ({p.resolution_m:.1f} m, {size}) -> {raw.relative_to(ROOT)}")
    if not paths:
        return None
    res = min(p.resolution_m for p in chosen)
    return normalize(db, grid, "elevation", "m", source, paths, res,
                     valid_range=ELEVATION_RANGE_M, align_offsets=True)


def parse_sols(text):
    if not text:
        return []
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="download whole DTM products (0.3-0.8 GB each)")
    ap.add_argument("--max-hirise", type=int, default=3, help="max HiRISE DTMs to mosaic")
    ap.add_argument("--max-items", type=int, default=200, help="max items per STAC collection")
    ap.add_argument("--rad-sols", default="453-457", help="RAD sols, e.g. 453-482 (18 MB per sol), '' = skip")
    ap.add_argument("--skip-catalog", action="store_true", help="reuse the catalog as it is")
    ap.add_argument("--region", default="gale_boxwork", choices=sorted(REGIONS))
    args = ap.parse_args()

    grid = REGIONS[args.region]
    db = connect()
    add_region(db, grid)

    print(f"[1/7] catalog over {grid.name} {tuple(round(v, 3) for v in grid.bounds_lonlat)}")
    if not args.skip_catalog:
        load_catalog(db, grid.bounds_lonlat, max_items=args.max_items)
    dtms = find_products(db, grid.bounds_lonlat, "dtm")
    print(f"  find_products(gale_bbox, 'dtm') -> {len(dtms)} products, finest:")
    for p in dtms[:4]:
        print(f"    {p.source:11s} {p.resolution_m:5.1f} m  cover {p.coverage:4.0%}  {p.url}")

    print("[2/7] elevation")
    ingest_dtm(db, grid, "hirise_dtm", args.max_hirise, args.full)
    ingest_dtm(db, grid, "ctx_dtm", 1, args.full)

    print("[3/7] global coarse layers")
    ingest_global_layers(db, grid)

    print("[4/7] time series (Curiosity)")
    for name, n in load_rems(db, grid.name).items():
        print(f"  REMS {name}: {n} sols")
    load_rad(db, grid.name, parse_sols(args.rad_sols))

    print("[5/7] minerals (CRISM)")
    if not args.skip_catalog:
        print(f"  catalog ode:crism: {load_ode_crism(db, grid.bounds_lonlat)} products")
    for layer in ingest_minerals(db, grid):
        print(f"  {layer}: ok")

    print("[6/7] optical (HiRISE orthoimages)")
    if not args.skip_catalog:
        print(f"  catalog ode:hirise_ortho: {load_ode_hirise_ortho(db, grid.bounds_lonlat)} products")
    for layer in ingest_optical(db, grid):
        print(f"  {layer}: ok")

    print("[7/7] thermal inertia (THEMIS, quantitative)")
    ingest_thermal_inertia(db, grid)

    print("\nprocessed layers:")
    for name, src, unit, res, path, cov in db.execute(
            "SELECT l.name, s.name, l.unit, l.native_res_m, l.path, l.coverage_frac FROM layers l "
            "JOIN sources s ON s.id=l.source_id JOIN regions r ON r.id=l.region_id "
            "WHERE r.name=? ORDER BY l.name, l.native_res_m", (grid.name,)):
        a, _ = read_layer(ROOT / path)
        import numpy as np
        print(f"  {name:16s} {src:11s} {res:7.1f} m  cover {cov:6.1%}  "
              f"{np.nanmin(a):9.1f} .. {np.nanmax(a):9.1f} {unit}")
    db.close()


if __name__ == "__main__":
    main()
