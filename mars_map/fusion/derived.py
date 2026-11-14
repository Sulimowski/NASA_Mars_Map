"""Layers derived for the algorithms: where you may not go, how hard it is, how valuable.

* `blocked` (bool)  -- hard limits. A route must never enter these cells.
* `cost`    (float) -- how many times slower than firm flat ground (>= 1); inf where blocked.
* `value`   (0..1)  -- science value of standing on the cell.

Hard limits are a mask, never a weight: no amount of science makes a cliff walkable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..site import Site


@dataclass
class CostWeights:
    """Working assumptions. `Alg` owns the real numbers (speed per terrain type)."""
    slope: float = 2.0        # extra cost at max_slope_deg, grows with the square of slope
    sand: float = 1.0         # loose sand: 2x slower
    boulders: float = 2.0     # boulder field: 3x slower
    rock: float = 1.0         # per unit of rock abundance, on top of the class
    roughness: float = 0.5    # per metre of local relief
    coarse_data: float = 0.5  # penalty where only coarse data exists (hidden hazards)


def no_go_mask(site: Site) -> np.ndarray:
    """Rasterize the no-go polygons (given in metres) onto the grid."""
    if not site.no_go:
        return np.zeros(site.shape, bool)
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    shapes = [({"type": "Polygon", "coordinates": [[tuple(p) for p in poly] + [tuple(poly[0])]]}, 1)
              for poly in site.no_go]
    # pixel (col, row) -> (x_m, y_m): y grows south, same as the grid
    transform = Affine.scale(site.resolution_m, site.resolution_m)
    return rasterize(shapes, out_shape=site.shape, transform=transform,
                     fill=0, dtype="uint8", all_touched=True).astype(bool)


def blocked_mask(site: Site) -> np.ndarray:
    L, m = site.layers, site.mission
    return (L["terrain_class"] == 3) | (L["slope"] > m.hard_max_slope_deg) | no_go_mask(site)


def walkable_mask(site: Site) -> np.ndarray:
    """Cells a suited crew can cross comfortably: not blocked and within `max_slope_deg`."""
    return ~blocked_mask(site) & (site.layers["slope"] <= site.mission.max_slope_deg)


def snap_to(mask: np.ndarray, site: Site, x_m: float, y_m: float,
            max_m: float = 20.0) -> tuple[float, float] | None:
    """Nearest cell of `mask` within `max_m` of a point, as (x_m, y_m). None if there is none."""
    r0, c0 = site.to_pixel(x_m, y_m)
    k = int(np.ceil(max_m / site.resolution_m))
    ra, rb = max(r0 - k, 0), min(r0 + k + 1, mask.shape[0])
    ca, cb = max(c0 - k, 0), min(c0 + k + 1, mask.shape[1])
    rows, cols = np.nonzero(mask[ra:rb, ca:cb])
    if rows.size == 0:
        return None
    d = np.hypot(rows + ra - r0, cols + ca - c0)
    i = int(d.argmin())
    if d[i] * site.resolution_m > max_m:
        return None
    return site.to_metres(int(rows[i] + ra), int(cols[i] + ca))


def reachable_mask(site: Site, x_m: float, y_m: float) -> np.ndarray:
    """Everything connected to a point through walkable cells (4-neighbour flood fill)."""
    from collections import deque

    walk = walkable_mask(site)
    seed = snap_to(walk, site, x_m, y_m)
    seen = np.zeros(walk.shape, bool)
    if seed is None:
        return seen
    rows, cols = walk.shape
    q = deque([site.to_pixel(*seed)])
    seen[q[0]] = True
    while q:
        r, c = q.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and walk[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def cost_layer(site: Site, w: CostWeights | None = None) -> np.ndarray:
    w = w or CostWeights()
    L, m = site.layers, site.mission
    cost = np.ones(site.shape, np.float32)
    cost += w.slope * (L["slope"] / m.max_slope_deg) ** 2
    cost += w.sand * (L["terrain_class"] == 1)
    cost += w.boulders * (L["terrain_class"] == 2)
    cost += w.rock * L["rock_abundance"]
    cost += w.roughness * L["roughness"]
    cost += w.coarse_data * L["nodata"]
    cost[blocked_mask(site)] = np.inf
    return cost


def value_layer(site: Site) -> np.ndarray:
    """Science value. For now the science score; minerals and geology join later (B6)."""
    return site.layers["science_score"].astype(np.float32)


def add_derived(site: Site, weights: CostWeights | None = None) -> Site:
    """Adds `blocked`, `cost`, `value` to `site.layers` (in place) and returns the site."""
    site.layers["blocked"] = blocked_mask(site)
    site.layers["cost"] = cost_layer(site, weights)
    site.layers["value"] = value_layer(site)
    site.validate()
    return site
