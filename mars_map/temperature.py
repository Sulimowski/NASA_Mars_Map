"""Ground temperature and sunlight maps for a given sol and time of day.

The site-wide minimum and maximum for a sol come from the data (site.series).
This module turns them into a map at a chosen local time:

* time of day  -- coldest at sunrise, warmest about an hour after noon, fast drop
                  after sunset then slow cooling through the night;
* ground type  -- low thermal inertia (sand, dust) swings more: hotter by day,
                  colder at night. Bedrock stays closer to the mean;
* sunlight     -- slopes facing the Sun and cells out of shadow warm up more.
                  Dust (tau) turns direct light into scattered light, so under a
                  dusty sky slopes and shadows matter less.

This is a parametric model, not a heat-conduction solver. Known gap: every cell
peaks at the same time (an east-facing slope really warms earlier than a west one).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .sun import day_length_h, sun_vector
from .terrain import horizon_toward


@dataclass
class TemperatureParams:
    """Working assumptions, to be tuned against real ground temperatures."""
    peak_lag_h: float = 1.0             # ground is warmest this long after noon
    night_decay: float = 3.0            # larger = faster drop right after sunset
    # Diurnal swing ~ (thermal inertia) ** exponent, relative to the site median.
    inertia_exponent: float = -0.35
    inertia_factor_min: float = 0.5
    inertia_factor_max: float = 1.8
    max_insolation_ratio: float = 1.6   # cap for steep Sun-facing slopes


def diurnal_shape(ltst_h: float, day_len_h: float,
                  params: TemperatureParams = TemperatureParams()) -> float:
    """Position within the daily swing: 0 at the sunrise minimum, 1 at the peak."""
    d = min(max(day_len_h, 0.5), 23.5)
    lag, t = params.peak_lag_h, ltst_h % 24.0
    t_rise, t_set = 12.0 - d / 2, 12.0 + d / 2
    if t_rise <= t <= t_set:
        return math.sin(math.pi * (t - t_rise) / (d + 2 * lag))
    at_sunset = math.sin(math.pi * d / (d + 2 * lag))
    since_sunset = (t - t_set) % 24.0
    e, e_end = math.exp(-params.night_decay * since_sunset / (24.0 - d)), math.exp(-params.night_decay)
    return at_sunset * (e - e_end) / (1 - e_end)


def lit_and_incidence(normal, horizons: np.ndarray, lat_deg: float, ls_deg: float,
                      ltst_h: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(sunlit mask, cosine of the Sun's incidence angle on the ground, sin of Sun elevation).

    A cell is sunlit when the Sun is above its local horizon and the ground faces it.
    """
    east, north, up = sun_vector(lat_deg, ls_deg, ltst_h)
    n_e, n_n, n_u = normal
    if up <= 0:
        zero = np.zeros(n_u.shape, np.float32)
        return zero.astype(bool), zero, up
    cos_i = np.clip(n_e * east + n_n * north + n_u * up, 0, None)
    lit = (horizon_toward(horizons, math.atan2(east, north)) < math.asin(up)) & (cos_i > 0)
    return lit, cos_i, up


def insolation_ratio(normal, horizons: np.ndarray, sky_view: np.ndarray, lat_deg: float,
                     ls_deg: float, tau: float, params: TemperatureParams = TemperatureParams(),
                     step_h: float = 0.5) -> np.ndarray:
    """Sunlight a cell collects over the sol, relative to open flat ground (1 = same).

    Direct light depends on slope, aspect and cast shadows; scattered light only
    on how much sky is visible. Their mix is set by the dust optical depth.
    """
    d = day_length_h(lat_deg, ls_deg)
    direct = np.zeros(sky_view.shape, np.float64)
    flat_direct = flat_clear = 0.0
    for t in np.arange(12.0 - d / 2 + step_h / 2, 12.0 + d / 2, step_h):
        lit, cos_i, up = lit_and_incidence(normal, horizons, lat_deg, ls_deg, float(t))
        if up <= 0:
            continue
        through_dust = math.exp(-tau / up)
        direct += through_dust * cos_i * lit
        flat_direct += through_dust * up
        flat_clear += up
    if flat_direct <= 0:
        return sky_view.astype(np.float32)
    direct_share = flat_direct / flat_clear        # of the light reaching flat ground
    ratio = np.minimum(direct / flat_direct, params.max_insolation_ratio)
    return (direct_share * ratio + (1 - direct_share) * sky_view).astype(np.float32)


def surface_temperature_map(temp_min_c: float, temp_max_c: float, ltst_h: float,
                            day_len_h: float, thermal_inertia: np.ndarray,
                            insolation: np.ndarray,
                            params: TemperatureParams = TemperatureParams()) -> np.ndarray:
    """Ground temperature in every cell, degrees C (float32).

    Open flat ground with the site's median thermal inertia follows the series
    exactly: temp_min_c at sunrise, temp_max_c at the afternoon peak.
    """
    swing = (thermal_inertia / float(np.median(thermal_inertia))) ** params.inertia_exponent
    swing = np.clip(swing, params.inertia_factor_min, params.inertia_factor_max)
    mid, half = (temp_max_c + temp_min_c) / 2, (temp_max_c - temp_min_c) / 2
    cell_min = mid - half * swing                    # night: ground type only
    day_rise = 2 * half * swing * insolation         # day: ground type and sunlight
    return (cell_min + day_rise * diurnal_shape(ltst_h, day_len_h, params)).astype(np.float32)
