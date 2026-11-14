"""Sanity check: can every target be reached on foot from the start?

Walkable = slope <= mission.max_slope_deg, class != impassable, outside no-go.
This is NOT a route planner, just a flood fill to prove the site is solvable.
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mars_map import Site  # noqa: E402


def no_go_mask(site: Site) -> np.ndarray:
    rows, cols = site.shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    pts = np.column_stack([((cc + 0.5) * site.resolution_m).ravel(),
                           ((rr + 0.5) * site.resolution_m).ravel()])
    mask = np.zeros(rows * cols, bool)
    for poly in site.no_go:
        mask |= MplPath(poly).contains_points(pts)
    return mask.reshape(rows, cols)


def main(folder="data/gale_stub"):
    site = Site.load(folder)
    L = site.layers
    walk = (L["slope"] <= site.mission.max_slope_deg) & (L["terrain_class"] != 3)
    walk &= ~no_go_mask(site)

    rows, cols = site.shape
    seen = np.zeros_like(walk)
    q = deque([site.to_pixel(*site.start)])
    seen[q[0]] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and walk[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    print(f"walkable: {walk.mean():.1%} of site, reachable from start: {seen.mean():.1%}")
    ok = True
    for t in site.targets:
        hit = seen[site.to_pixel(t.x_m, t.y_m)]
        ok &= bool(hit)
        print(f"  {'OK  ' if hit else 'FAIL'} {t.name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:])
