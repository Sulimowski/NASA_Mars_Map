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
