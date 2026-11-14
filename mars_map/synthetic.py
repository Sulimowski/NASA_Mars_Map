"""Synthetic stand-in for a Gale crater / Mount Sharp foothills site.

The terrain is fake but shaped like the real thing, so algorithms meet the
same kinds of problems they will meet on real data:

* ground rising to the south (toward Mount Sharp)
* a mesa with cliffs and one walkable ramp (route must find the ramp)
* craters with steep rims, one fresh crater with a boulder field
* a dune field (loose sand: low thermal inertia, slow walking)
* a strip where high-res data is missing (nodata = True, coarser values)
* per-sol radiation with a solar particle event, and a dust event

Same seed -> same site, so everyone on the team works on identical data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):          # run directly (PyCharm Run / python synthetic.py)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mars_map.site import MissionParams, Site, Target, slope_deg
else:
    from .site import MissionParams, Site, Target, slope_deg

# Approximate location of the Mount Sharp boxwork area (for lat/lon labels only).
GALE_CENTER_LAT = -4.75
GALE_CENTER_LON = 137.38


def _fractal_noise(rng, n: int, beta: float = 2.2) -> np.ndarray:
    """Noise with a 1/f^beta spectrum, normalised to std 1."""
    white = rng.standard_normal((n, n))
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    k = np.hypot(fx, fy)
    k[0, 0] = 1.0
    spec = np.fft.fft2(white) / k ** (beta / 2)
    spec[0, 0] = 0
    out = np.real(np.fft.ifft2(spec))
    return out / out.std()


def _box_mean(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)x(2r+1) window, edges padded."""
    p = np.pad(a, r + 1, mode="edge").astype(np.float64)
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s / (k * k))[: a.shape[0], : a.shape[1]]


def _crater(x, y, cx, cy, radius, fresh: bool):
    d = 2 * radius
    depth = (0.2 if fresh else 0.08) * d
    rim = (0.04 if fresh else 0.015) * d
    r = np.hypot(x - cx, y - cy) / radius
    inside = rim - depth * (1 - r ** 2)
    outside = rim / np.maximum(r, 1) ** 3
    return np.where(r < 1, inside, outside), r


def make_gale_like_site(size_m: float = 2000.0, resolution_m: float = 2.0,
                        n_sols: int = 30, seed: int = 42) -> Site:
    rng = np.random.default_rng(seed)
    n = int(round(size_m / resolution_m))
    coords = (np.arange(n) + 0.5) * resolution_m
    x, y = np.meshgrid(coords, coords)           # x east, y south (metres)
    s = size_m / 2000.0                          # feature scale vs. default size

    # --- elevation --------------------------------------------------------
    elev = -4500.0 + 0.06 * y                    # ~3.4 deg rise toward Mount Sharp
    elev += 5.0 * _fractal_noise(rng, n, beta=3.6)       # rolling terrain
    elev += 0.12 * _fractal_noise(rng, n, beta=2.0)      # small-scale texture

    # Mesa with cliffs; the NW side is a gentle ramp.
    mcx, mcy, mr, mh = 1350 * s, 1100 * s, 220 * s, 30.0
    ang = np.arctan2(y - mcy, x - mcx)
    ramp_dir = np.deg2rad(-135)                  # pointing NW (up-left on the map)
    ramp_w = np.clip((np.cos(ang - ramp_dir) - np.cos(np.deg2rad(35)))
                     / (np.cos(np.deg2rad(12)) - np.cos(np.deg2rad(35))), 0, 1)
    ramp_w = ramp_w * ramp_w * (3 - 2 * ramp_w)  # smoothstep, no hard seams
    in_ramp = ramp_w > 0.5
    edge_w = 8.0 + (220.0 * s - 8.0) * ramp_w
    dist_edge = np.hypot(x - mcx, y - mcy) - mr
    mesa = mh / (1 + np.exp(dist_edge / (edge_w / 4)))
    elev += mesa

    craters = [  # cx, cy, radius, fresh
        (650, 1450, 120, True),
        (1500, 450, 60, False),
        (800, 600, 35, False),
        (300, 1750, 50, False),
        (1100, 1650, 80, False),
    ]
    fresh_r = None
    for cx, cy, cr, fresh in craters:
        dz, r = _crater(x, y, cx * s, cy * s, cr * s, fresh)
        elev += dz
        if fresh:
            fresh_r = r

    # Dune field (west): ripples on a sand sheet.
    dune = ((x - 250 * s) / (230 * s)) ** 2 + ((y - 950 * s) / (430 * s)) ** 2
    sand = dune < 1.0
    sand_soft = np.clip(1.3 - dune, 0, 1)
    ripples = 1.0 * np.sin(2 * np.pi * (0.8 * x + 0.6 * y) / 40.0)
    elev += sand_soft * (ripples + 2.0)

    elev = elev.astype(np.float32)

    # --- nodata strip: no HiRISE here, only coarse (CTX-like) data -----------
    nodata = (x + 0.8 * y) > 2050 * s            # NE corner wedge
    nodata &= y < 700 * s
    coarse = _box_mean(elev, r=max(1, int(10 / resolution_m)))
    elev = np.where(nodata, coarse, elev).astype(np.float32)

    # --- derived layers ----------------------------------------------------
    slope = slope_deg(elev, resolution_m)
    rr = max(1, int(3 / resolution_m))
    mean = _box_mean(elev, rr)
    roughness = np.sqrt(np.maximum(_box_mean(elev.astype(np.float64) ** 2, rr) - mean ** 2, 0))

    rock = 0.05 + 0.04 * _fractal_noise(rng, n, beta=1.5)
    ejecta = np.clip(1.0 - (fresh_r - 1.0) / 1.2, 0, 1) * (fresh_r > 0.9)
    rock += 0.45 * ejecta * (0.7 + 0.3 * rng.random((n, n)))
    rock += 0.25 * np.clip(-dist_edge / 15, 0, 1) * (dist_edge > -40) * (~in_ramp)  # talus
    rock[sand] *= 0.2
    rock = np.clip(rock, 0, 1).astype(np.float32)

    ti = 350 + 300 * rock - 200 * sand_soft + 25 * _fractal_noise(rng, n, beta=2.0)
    ti = np.clip(ti, 80, 900).astype(np.float32)

    mission = MissionParams()
    tclass = np.zeros((n, n), np.uint8)
    tclass[sand] = 1
    tclass[rock > 0.3] = 2
    tclass[slope > mission.hard_max_slope_deg] = 3

    # Science interest: boxwork at the mesa foot, clay layer upslope, fresh ejecta.
    boxwork = np.exp(-((x - 1170 * s) ** 2 + (y - 920 * s) ** 2) / (2 * (90 * s) ** 2))
    clay = np.exp(-((elev - np.percentile(elev, 92)) / 4.0) ** 2) * (y > 1500 * s)
    sci = 0.1 + 0.9 * boxwork + 0.6 * clay + 0.4 * ejecta + 0.05 * _fractal_noise(rng, n)
    sci = np.clip(sci, 0, 1).astype(np.float32)

    layers = {
        "elevation": elev,
        "slope": slope,
        "roughness": roughness.astype(np.float32),
        "rock_abundance": rock,
        "thermal_inertia": ti,
        "terrain_class": tclass,
        "science_score": sci,
        "nodata": nodata,
    }

    # --- per-sol series ----------------------------------------------------
    sols = np.arange(n_sols)
    dose = 0.67 + 0.03 * rng.standard_normal(n_sols)        # mSv/sol, RAD-like
    spe = (sols >= n_sols * 0.55) & (sols < n_sols * 0.55 + 2)
    dose[spe] *= 4.0                                        # solar particle event
    tau = 0.6 + 0.05 * rng.standard_normal(n_sols)
    storm = (sols >= n_sols * 0.73) & (sols < n_sols * 0.73 + 5)
    tau[storm] += np.array([0.4, 0.9, 1.2, 0.8, 0.4])[: storm.sum()]
    series = {
        "sol": sols.tolist(),
        "radiation_msv_per_sol": np.round(dose, 3).tolist(),
        "dust_opacity_tau": np.round(tau, 2).tolist(),
        "temp_min_c": np.round(-78 + 3 * rng.standard_normal(n_sols), 1).tolist(),
        "temp_max_c": np.round(-12 + 4 * rng.standard_normal(n_sols) - 6 * (tau - 0.6), 1).tolist(),
        "wind_max_ms": np.round(np.clip(9 + 3 * rng.standard_normal(n_sols) + 8 * storm, 2, None), 1).tolist(),
    }

    def p(px, py):
        return (px * s, py * s)

    targets = [
        Target("Boxwork ridges", *p(1170, 920), priority=1, dwell_min=45,
               note="Main goal: mineral veins at the foot of the mesa"),
        Target("Mesa top outcrop", *p(1380, 1080), priority=1, dwell_min=30,
               note="Only reachable via the NW ramp"),
        Target("Fresh crater ejecta", *p(860, 1380), priority=2, dwell_min=25,
               note="Deep-origin rocks; boulder field, careful footing"),
        Target("Dune margin", *p(480, 820), priority=3, dwell_min=15,
               note="Sand sample at the edge, do not walk into the dunes"),
        Target("Clay-bearing layer", *p(1000, 1820), priority=2, dwell_min=30,
               note="Upslope toward Mount Sharp"),
    ]

    def circle(cx, cy, r, k=12):
        return [(cx + r * np.cos(a), cy + r * np.sin(a))
                for a in np.linspace(0, 2 * np.pi, k, endpoint=False)]

    no_go = [
        [(round(a, 1), round(b, 1)) for a, b in circle(650 * s, 1450 * s, 105 * s)],  # crater
        [(round(a, 1), round(b, 1)) for a, b in circle(230 * s, 950 * s, 150 * s)],   # dune core
    ]

    site = Site(
        name="gale_stub",
        source=f"synthetic (seed={seed}); Gale/Mount Sharp-like, NOT real data",
        resolution_m=resolution_m,
        center_lat=GALE_CENTER_LAT,
        center_lon=GALE_CENTER_LON,
        layers=layers,
        series=series,
        start=p(1000, 120),
        targets=targets,
        no_go=no_go,
        mission=mission,
    )
    site.validate()
    return site


if __name__ == "__main__":
    from mars_map import STUB_DIR
    print(f"saved: {make_gale_like_site().save(STUB_DIR)}")
