"""Dust devil risk for route planning: where and when a walker meets one.

The hazards model (`python -m hazards.train_dust_devils`) forecasts, for each season
bin of 30 deg Ls and each local hour, the rate at which vortices with a pressure drop of
0.8 Pa or more pass over ONE point: lambda(h), encounters per Earth hour. That is a
site-wide number. This module spreads it over the terrain and over the walk:

    rate(x, h) = lambda(h) * S(x)          encounters per hour at cell x, hour h
    E          = sum over the walk of rate(x(t), h(t)) * dt     expected encounters
    P(>=1)     = 1 - exp(-E)               chance of meeting at least one

S(x) is a relative susceptibility, mean 1 over the walkable ground, so the site-wide
rate is kept. Dust devils are convective vortices: they grow where the ground heats the
air most, and they form and wander on open, level ground. The working assumptions:

* heating  -- sunlight a cell collects over the sol (slope, aspect, cast shadows: the
              environment model) over the square root of thermal inertia, which sets
              how far the surface warms above the air; low-inertia sand and dust warm most;
* level    -- steep ground breaks the convective cells: the factor falls off past ~12 deg
              of slope averaged over ~40 m;
* dust     -- loose sand and dust make a vortex carry more grit, which is what hurts a
              suit and its visor; mapped sand raises the factor by up to 40 %.

These shape WHERE within a 2 km site the risk is higher; the absolute level is the
forecast's. They are not fitted to vortex tracks, so treat S as a ranking, not a
measurement.

    hz = DevilHazard.for_site(site, env, sol)      # None without a forecast file
    hz.rate_per_s(hour)  -> map, encounters per second at that local hour
    hz.exposure(xs, ys, t_s, start_h)              # expected encounters along a walk
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .site import Site

ROOT = Path(__file__).resolve().parent.parent
FORECAST_DIR = ROOT / "hazards" / "data"
MARS_HOUR_S = 3600.0 * 1.0275          # one local (Mars) hour in Earth seconds
LEVEL_SLOPE_DEG = 12.0
LEVEL_WINDOW_M = 40.0
SAND_BOOST = 0.4
S_RANGE = (0.15, 3.0)


def read_forecast(name: str) -> dict[float, list[float]] | None:
    """{season bin start Ls: [encounters per Earth hour at one point, for hours 0..23]}."""
    path = FORECAST_DIR / f"{name}_dust_devils.csv"
    if not path.exists():
        return None
    out: dict[float, list[float]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ls, h = float(row["ls_from"]), int(row["hour"])
            out.setdefault(ls, [0.0] * 24)[h % 24] = max(0.0, float(row["vortices_per_h"]))
    return out


def hourly_rate(forecast: dict[float, list[float]], ls_deg: float) -> list[float]:
    """The 24 hourly rates of the season bin that holds ls_deg."""
    bins = sorted(forecast)
    ls = ls_deg % 360.0
    pick = max((b for b in bins if b <= ls), default=bins[-1])
    return list(forecast[pick])


def _box_mean(a: np.ndarray, r: int) -> np.ndarray:
    p = np.pad(a.astype(np.float64), r + 1, mode="edge")
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s / (k * k))[: a.shape[0], : a.shape[1]]


def susceptibility(site: Site, insolation: np.ndarray | None = None) -> np.ndarray:
    """S(x): relative likelihood of a dust devil passing a cell, mean 1 on walkable ground."""
    L = site.layers
    shape = L["elevation"].shape
    ti = np.nan_to_num(L.get("thermal_inertia", np.full(shape, 350.0)), nan=350.0)
    ins = np.ones(shape) if insolation is None else np.nan_to_num(insolation, nan=1.0)
    heating = np.clip(ins, 0.0, None) / np.sqrt(np.clip(ti, 80.0, None))
    heating /= max(float(np.median(heating)), 1e-12)

    r = max(1, int(round(LEVEL_WINDOW_M / 2 / site.resolution_m)))
    slope = _box_mean(np.nan_to_num(L["slope"]), r)
    level = np.exp(-(slope / LEVEL_SLOPE_DEG) ** 2)

    sand = np.clip(np.nan_to_num(L["sand_fraction"]), 0, 1) if "sand_fraction" in L else 0.0
    dust = 1.0 + SAND_BOOST * sand

    s = heating * level * dust
    walk = ~(L["blocked"].astype(bool) if "blocked" in L else L["terrain_class"] == 3)
    mean = float(s[walk].mean()) if walk.any() else float(s.mean())
    return np.clip(s / max(mean, 1e-12), *S_RANGE).astype(np.float32)


@dataclass
class DevilHazard:
    """Dust devil risk of one site and season: S map plus the 24 hourly rates."""
    s_map: np.ndarray                  # relative susceptibility, mean 1 on walkable ground
    hourly: list[float]                # encounters per Earth hour at one point, local hours 0..23
    ls_deg: float
    resolution_m: float
    source: str = "hazards model forecast"

    @classmethod
    def for_site(cls, site: Site, env=None, sol: int | None = None) -> "DevilHazard | None":
        fc = read_forecast(site.name)
        if fc is None:
            return None
        if env is not None and sol is None and env.sols:
            sol = env.sols[0]
        ls = env.ls(sol) if (env is not None and sol is not None) else 90.0
        ins = env.insolation(sol) if (env is not None and sol is not None) else None
        return cls(susceptibility(site, ins), hourly_rate(fc, ls), float(ls), float(site.resolution_m))

    # ---- time ---------------------------------------------------------------
    def rate_at(self, hour: float) -> float:
        """Site-wide encounters per Earth hour at a local hour (linear between whole hours)."""
        h = hour % 24.0
        i = int(math.floor(h))
        f = h - i
        return self.hourly[i] * (1 - f) + self.hourly[(i + 1) % 24] * f

    def rate_per_s(self, hour: float) -> np.ndarray:
        """Map of encounters per Earth second at a local hour."""
        return self.s_map * (self.rate_at(hour) / 3600.0)

    @staticmethod
    def hour_after(start_h: float, elapsed_s: float) -> float:
        return start_h + elapsed_s / MARS_HOUR_S

    # ---- along a walk -----------------------------------------------------------
    def exposure(self, xs: np.ndarray, ys: np.ndarray, t_s: np.ndarray, start_h: float) -> float:
        """Expected encounters along samples (x, y) reached at times t_s after the start."""
        xs, ys, t_s = (np.asarray(a, np.float64) for a in (xs, ys, t_s))
        if len(xs) < 2:
            return 0.0
        rows, cols = self.s_map.shape
        c = np.clip((xs / self.resolution_m).astype(np.intp), 0, cols - 1)
        r = np.clip((ys / self.resolution_m).astype(np.intp), 0, rows - 1)
        s = self.s_map[r, c]
        dt = np.diff(t_s)
        mid = 0.5 * (t_s[1:] + t_s[:-1])
        lam = np.array([self.rate_at(self.hour_after(start_h, m)) for m in mid]) / 3600.0
        return float((s[1:] * lam * dt).sum())

    def exposure_by_start(self, xs, ys, t_s, starts) -> list[float]:
        """The same walk started at each of `starts` (local hours): expected encounters."""
        return [self.exposure(xs, ys, t_s, h) for h in starts]

    @staticmethod
    def probability(expected: float) -> float:
        """Chance of at least one encounter."""
        return 1.0 - math.exp(-max(expected, 0.0))
