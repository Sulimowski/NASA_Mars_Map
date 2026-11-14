"""Radiation, temperature and sunlight maps of a Site, for a chosen sol and time.

    from mars_map import Environment, load_stub

    env = Environment(load_stub())            # ~5 s: scans the horizon of every cell
    dose = env.dose_rate(sol=17)              # microsieverts per Earth hour
    temp = env.temperature(sol=0, ltst_h=13)  # degrees C, ground
    lit  = env.sunlit(sol=0, ltst_h=8)        # bool, False = in shadow
    sun  = env.direct_sunlight(sol=0, ltst_h=8)   # W/m^2 on the ground

Every map has the shape of the site layers, so it is indexed the same way:
map[row, col] with site.to_pixel(x_m, y_m).
"""

from __future__ import annotations

import math

import numpy as np

from .radiation import RadiationParams, dose_rate_map
from .site import Site
from .sun import LS_AT_SOL0_DEG, day_length_h, ls_of_sol, solar_flux_toa
from .temperature import (TemperatureParams, insolation_ratio, lit_and_incidence,
                          surface_temperature_map)
from .terrain import horizon_angles, sky_view_factor, surface_normal

# Used when a site has no value of its own (real sites may lack some series).
TYPICAL_DOSE_MSV_PER_SOL = 0.67      # MSL/RAD surface average in Gale
TYPICAL_DUST_TAU = 0.6


class Environment:
    def __init__(self, site: Site, n_azimuth: int = 16, max_distance_m: float = 400.0,
                 radiation: RadiationParams | None = None,
                 temperature: TemperatureParams | None = None,
                 ls_at_sol0: float = LS_AT_SOL0_DEG):
        self.site = site
        self.radiation_params = radiation or RadiationParams()
        self.temperature_params = temperature or TemperatureParams()
        self.ls_at_sol0 = ls_at_sol0
        elevation = site.layers["elevation"]
        self.horizons = horizon_angles(elevation, site.resolution_m, n_azimuth, max_distance_m)
        self.sky_view = sky_view_factor(self.horizons)       # 0..1, 1 = open plain
        self.normal = surface_normal(elevation, site.resolution_m)
        self._insolation: dict[int, np.ndarray] = {}

    # ---- per-sol values --------------------------------------------------
    @property
    def sols(self) -> list[int]:
        """Sols the site has series for (empty when it has none)."""
        series = self.site.series
        if series.get("sol"):
            return list(series["sol"])
        return list(range(max((len(v) for v in series.values()), default=0)))

    def series_value(self, key: str, sol: int, default: float | None = None) -> float:
        """Value of a site.series entry for a sol.

        If the site has no such series at all, `default` is returned when given.
        """
        if key not in self.site.series:
            if default is not None:
                return default
            raise KeyError(f"site '{self.site.name}' has no series '{key}'")
        sols = self.site.series.get("sol")
        try:
            index = sols.index(sol) if sols else int(sol)
            return float(self.site.series[key][index])
        except (ValueError, IndexError):
            raise ValueError(f"sol {sol} is not in the series of site '{self.site.name}' "
                             f"({len(self.site.series[key])} sols)") from None

    def dust_tau(self, sol: int) -> float:
        return self.series_value("dust_opacity_tau", sol, TYPICAL_DUST_TAU)

    @property
    def has_temperature(self) -> bool:
        return self._temperature_keys() is not None

    def _temperature_keys(self) -> tuple[str, str] | None:
        """Ground temperature series when the site has them, else the generic ones."""
        for pair in (("ground_temp_min_c", "ground_temp_max_c"), ("temp_min_c", "temp_max_c")):
            if all(k in self.site.series for k in pair):
                return pair
        return None

    def ls(self, sol: int) -> float:
        """Season (solar longitude, degrees) for a sol."""
        return ls_of_sol(sol, self.ls_at_sol0)

    def is_particle_event(self, sol: int, factor: float = 2.0) -> bool:
        """True when the sol's dose is far above the usual level (solar particle event)."""
        doses = self.site.series.get("radiation_msv_per_sol")
        if not doses:
            return False
        return self.series_value("radiation_msv_per_sol", sol) > factor * float(np.median(doses))

    # ---- maps ------------------------------------------------------------
    def dose_rate(self, sol: int) -> np.ndarray:
        """Radiation dose rate, microsieverts per Earth hour."""
        dose = self.series_value("radiation_msv_per_sol", sol, TYPICAL_DOSE_MSV_PER_SOL)
        return dose_rate_map(dose, self.site.layers["elevation"], self.sky_view,
                             self.radiation_params)

    def insolation(self, sol: int) -> np.ndarray:
        """Sunlight collected over the sol relative to open flat ground (1 = same)."""
        if sol not in self._insolation:
            self._insolation[sol] = insolation_ratio(
                self.normal, self.horizons, self.sky_view, self.site.center_lat, self.ls(sol),
                self.dust_tau(sol), self.temperature_params)
        return self._insolation[sol]

    def temperature(self, sol: int, ltst_h: float) -> np.ndarray:
        """Ground temperature in degrees C at local true solar time ltst_h (0..24)."""
        keys = self._temperature_keys()
        if keys is None:
            raise KeyError(f"site '{self.site.name}' has no temperature series")
        return surface_temperature_map(
            self.series_value(keys[0], sol), self.series_value(keys[1], sol),
            ltst_h, day_length_h(self.site.center_lat, self.ls(sol)),
            self.site.layers["thermal_inertia"], self.insolation(sol), self.temperature_params)

    def sunlit(self, sol: int, ltst_h: float) -> np.ndarray:
        """True where the Sun shines on the ground, False in shadow or at night."""
        return lit_and_incidence(self.normal, self.horizons, self.site.center_lat,
                                 self.ls(sol), ltst_h)[0]

    def direct_sunlight(self, sol: int, ltst_h: float) -> np.ndarray:
        """Direct solar beam on the ground in W/m^2, after dust; 0 in shadow.

        Light scattered by the dust is not included, so this understates the
        total light, most of all under a dusty sky.
        """
        lit, cos_i, up = lit_and_incidence(self.normal, self.horizons, self.site.center_lat,
                                           self.ls(sol), ltst_h)
        if up <= 0:
            return cos_i
        through_dust = math.exp(-self.dust_tau(sol) / up)
        return (solar_flux_toa(self.ls(sol)) * through_dust * cos_i * lit).astype(np.float32)
