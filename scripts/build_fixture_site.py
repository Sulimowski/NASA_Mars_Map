"""End-to-end run of part B on fake data:

    stub -> fake catalog + degraded layers -> fusion -> derived layers -> Site

and a comparison of the fused site with the stub it was cut from. The stub is the
ground truth here, so the numbers show how much the pipeline loses.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mars_map import DATA_DIR, load_stub  # noqa: E402
from mars_map.analysis.fixture import build_fixture  # noqa: E402
from mars_map.fusion import add_derived, fuse_region  # noqa: E402
from mars_map.grid import GALE_MOUNT_SHARP  # noqa: E402
from mars_map.preview import render  # noqa: E402

OUT = DATA_DIR / "gale_fixture"


def main():
    stub = load_stub()
    db = build_fixture(stub)
    print(f"fixture catalog: {db}")

    site = fuse_region(db, GALE_MOUNT_SHARP.name, start=stub.start, targets=stub.targets,
                       no_go=stub.no_go, mission=stub.mission)
    add_derived(site)
    site.save(OUT)
    render(site, OUT / "preview.png")
    print(f"fused site: {OUT}  ({site.source})")

    S, F = stub.layers, site.layers
    gap = S["nodata"]
    res, counts = np.unique(F["source_res"], return_counts=True)
    print("\nelevation source per pixel:")
    for r, c in zip(res, counts):
        print(f"  {r:6.0f} m : {c / gap.size:6.1%}")

    d = np.abs(F["elevation"] - S["elevation"])
    print(f"\nelevation error vs stub:  outside gap max {d[~gap].max():.3f} m | "
          f"inside gap mean {d[gap].mean():.2f} m, max {d[gap].max():.2f} m")
    print(f"slope error vs stub:      outside gap mean {np.abs(F['slope'] - S['slope'])[~gap].mean():.2f} deg | "
          f"inside gap mean {np.abs(F['slope'] - S['slope'])[gap].mean():.2f} deg")
    print(f"terrain class agreement:  {(F['terrain_class'] == S['terrain_class']).mean():.1%}")
    for k, name in {0: "bedrock", 1: "sand", 2: "boulders", 3: "impassable"}.items():
        truth = S["terrain_class"] == k
        found = F["terrain_class"] == k
        hit = (truth & found).sum() / max(truth.sum(), 1)
        print(f"  {name:10s} truth {truth.mean():6.2%}  fused {found.mean():6.2%}  recall {hit:5.1%}")

    cost = F["cost"]
    ok = np.isfinite(cost)
    print(f"\nblocked: {F['blocked'].mean():.1%} of site | cost on walkable cells: "
          f"median {np.median(cost[ok]):.2f}, p95 {np.percentile(cost[ok], 95):.2f}, max {cost[ok].max():.2f}")
    for t in site.targets:
        r, c = site.to_pixel(t.x_m, t.y_m)
        print(f"  {t.name:22s} cost {cost[r, c]:5.2f}  value {F['value'][r, c]:.2f}  blocked {bool(F['blocked'][r, c])}")


if __name__ == "__main__":
    main()
