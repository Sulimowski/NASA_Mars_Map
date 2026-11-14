"""H3.2: the course of one sol built from per-sol series alone.

When a site has series (dose per sol, daily ground temperature extremes) but no
terrain-following environment model, ``SeriesEnvironment`` supplies the
simulator with conditions that are uniform over the site and vary with time.

Sun (latitude ``phi``, declination ``delta``, hour angle ``h`` = 0 at local noon)::

    sin(elevation) = sin(phi) sin(delta) + cos(phi) cos(delta) cos(h)
    day length     = sol / pi * acos(-tan(phi) tan(delta))

Ground temperature: half-sine from the minimum at sunrise to the maximum
``peak_lag_s`` after noon, then an exponential fall through the night.

Dose: the sol's dose spread evenly over the sol. On a particle-event sol the
part above the quiet level is reported as the event's dose rate.

Mission time 0 is ``start_time_of_sol_s`` seconds after local midnight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..core.config import SolSelectionConfig
from ..core.constants import (MARS_MEAN_SOLAR_FLUX_W_M2, MARS_OBLIQUITY_RAD,
                              MARS_REFERENCE_PRESSURE_PA, SOL_S)
from ..core.exceptions import EnvironmentValidationError
from ..core.schemas import Cell, EnvSample
from ..providers.site_adapter import series_value


def declination_rad(solar_longitude_rad: float) -> float:
    """Solar declination for a season (solar longitude Ls), rad."""
    return math.asin(math.sin(MARS_OBLIQUITY_RAD) * math.sin(solar_longitude_rad))


def sun_elevation_rad(latitude_rad: float, declination: float, time_of_sol_s: float) -> float:
    """Sun elevation at ``time_of_sol_s`` seconds after local midnight, rad."""
    hour_angle = 2.0 * math.pi * ((time_of_sol_s % SOL_S) / SOL_S - 0.5)
    sin_el = (math.sin(latitude_rad) * math.sin(declination)
              + math.cos(latitude_rad) * math.cos(declination) * math.cos(hour_angle))
    return math.asin(max(-1.0, min(1.0, sin_el)))


def day_length_s(latitude_rad: float, declination: float) -> float:
    """Time from sunrise to sunset on flat ground, s (0 in polar night, a full sol in polar day)."""
    x = -math.tan(latitude_rad) * math.tan(declination)
    return SOL_S / math.pi * math.acos(max(-1.0, min(1.0, x)))


def diurnal_shape(time_of_sol_s: float, day_s: float, peak_lag_s: float,
                  night_decay: float) -> float:
    """Position within the daily temperature swing: 0 at the sunrise minimum, 1 at the peak."""
    day = min(max(day_s, 0.02 * SOL_S), 0.98 * SOL_S)
    t = time_of_sol_s % SOL_S
    sunrise, sunset = (SOL_S - day) / 2.0, (SOL_S + day) / 2.0
    span = day + 2.0 * peak_lag_s
    if sunrise <= t <= sunset:
        return math.sin(math.pi * (t - sunrise) / span)
    at_sunset = math.sin(math.pi * day / span)
    since_sunset = (t - sunset) % SOL_S
    decay = math.exp(-night_decay * since_sunset / (SOL_S - day))
    floor = math.exp(-night_decay)
    return at_sunset * (decay - floor) / (1.0 - floor)


@dataclass(frozen=True, slots=True)
class SeriesEnvironment:
    """``EnvironmentProvider`` for one sol from per-sol values, uniform over the site.

    Attributes:
        ground_temp_min_k, ground_temp_max_k: daily extremes of the ground temperature.
        dose_sv_per_sol: dose equivalent of the sol.
        quiet_dose_sv_per_sol: dose of a sol without a particle event; the excess of
            ``dose_sv_per_sol`` over it is the event's contribution.
        particle_event: the sol carries a solar particle event.
        latitude_rad, declination: site latitude and solar declination.
        start_time_of_sol_s: local time of mission time 0, seconds after midnight.
        pressure_pa: surface pressure.
    """

    ground_temp_min_k: float
    ground_temp_max_k: float
    dose_sv_per_sol: float
    quiet_dose_sv_per_sol: float
    particle_event: bool = False
    latitude_rad: float = 0.0
    declination: float = 0.0
    start_time_of_sol_s: float = 0.4 * SOL_S
    pressure_pa: float = MARS_REFERENCE_PRESSURE_PA
    cfg: SolSelectionConfig = SolSelectionConfig()

    def __post_init__(self) -> None:
        if self.ground_temp_max_k < self.ground_temp_min_k:
            raise EnvironmentValidationError("ground_temp_max_k is below ground_temp_min_k")
        if self.dose_sv_per_sol < 0.0 or self.quiet_dose_sv_per_sol < 0.0:
            raise EnvironmentValidationError("dose per sol must be non-negative")

    @classmethod
    def from_site(cls, site: Any, sol: int, *, start_ltst_h: float = 10.0,
                  solar_longitude_rad: float = 0.0,
                  cfg: SolSelectionConfig = SolSelectionConfig()) -> SeriesEnvironment:
        """Build from a ``Site`` of the data contract, using only its series.

        Ground temperatures come from ``ground_temp_*`` when present, else from
        ``temp_*``, else from the configured defaults. A missing radiation series
        gives the typical dose.

        Raises:
            EnvironmentValidationError: ``sol`` is not in the site series.
        """
        from .sol_selection import sol_series_from_site       # local: avoids an import cycle

        series = sol_series_from_site(site, cfg)
        if sol not in series.sols:
            raise EnvironmentValidationError(f"sol {sol} is not in the series of the site")
        index = series.sols.index(sol)
        dose = series.dose_sv[index]
        quiet = sorted(series.dose_sv)[len(series.dose_sv) // 2]
        event = dose > cfg.spe_dose_factor * quiet
        t_min = (series.ground_temp_min_k[index] if series.ground_temp_min_k is not None
                 else cfg.default_ground_temp_min_k)
        t_max = (series.ground_temp_max_k[index] if series.ground_temp_max_k is not None
                 else cfg.default_ground_temp_max_k)
        pressure = series_value(site, "pressure_pa", sol)
        return cls(ground_temp_min_k=t_min, ground_temp_max_k=t_max, dose_sv_per_sol=dose,
                   quiet_dose_sv_per_sol=min(dose, quiet) if event else dose,
                   particle_event=event, latitude_rad=math.radians(float(site.center_lat)),
                   declination=declination_rad(solar_longitude_rad),
                   start_time_of_sol_s=start_ltst_h / 24.0 * SOL_S,
                   pressure_pa=pressure if pressure is not None else MARS_REFERENCE_PRESSURE_PA,
                   cfg=cfg)

    def time_of_sol_s(self, t_s: float) -> float:
        """Seconds after local midnight at mission time ``t_s``."""
        return (self.start_time_of_sol_s + t_s) % SOL_S

    def get_env(self, cell: Cell, t_s: float) -> EnvSample:
        """Conditions at mission time ``t_s``; the cell is ignored."""
        cfg = self.cfg
        local = self.time_of_sol_s(t_s)
        elevation = sun_elevation_rad(self.latitude_rad, self.declination, local)
        shape = diurnal_shape(local, day_length_s(self.latitude_rad, self.declination),
                              cfg.peak_lag_s, cfg.night_decay)
        t_ground = self.ground_temp_min_k + (self.ground_temp_max_k - self.ground_temp_min_k) * shape
        if elevation > 0.0:
            sin_el = math.sin(elevation)
            beam = MARS_MEAN_SOLAR_FLUX_W_M2 * cfg.beam_transmittance ** (1.0 / max(sin_el, 0.05))
            diffuse = cfg.diffuse_share * MARS_MEAN_SOLAR_FLUX_W_M2 * sin_el
            t_air, t_sky, uv = t_ground - 15.0 * sin_el, 170.0 + 10.0 * sin_el, 10.0 * sin_el
        else:
            beam = diffuse = uv = 0.0
            t_air, t_sky = t_ground + 3.0, 150.0
        return EnvSample(
            t_ground_k=t_ground, t_air_k=t_air, t_sky_k=t_sky, pressure_pa=self.pressure_pa,
            solar_direct_w_m2=beam, solar_diffuse_w_m2=diffuse, sun_elevation_rad=elevation,
            dose_rate_gcr_sv_s=self.quiet_dose_sv_per_sol / SOL_S,
            spe_active=self.particle_event,
            spe_dose_rate_sv_s=max(0.0, self.dose_sv_per_sol - self.quiet_dose_sv_per_sol) / SOL_S,
            uv_index=uv)

