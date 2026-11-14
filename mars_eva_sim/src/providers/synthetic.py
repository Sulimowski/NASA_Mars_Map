"""Synthetic testbeds: small terrains and environments with a known answer.

Terrain factories   flat_terrain, slope_terrain, hill_terrain, wall_terrain,
                    forbidden_zone_terrain
Environments        ConstantEnvironment, DiurnalEnvironment, SpeEnvironment
Reference samples   reference_day_sample, reference_night_sample
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from ..core.constants import MARS_MEAN_SOLAR_FLUX_W_M2, SOL_S
from ..core.exceptions import TerrainValidationError
from ..core.schemas import Cell, EnvSample
from .grid import GridTerrain
from .interfaces import EnvironmentProvider


def _shelter_mask(shape: tuple[int, int], shelters: Iterable[Cell]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.bool_)
    for row, col in shelters:
        if not (0 <= row < shape[0] and 0 <= col < shape[1]):
            raise TerrainValidationError(f"shelter {(row, col)} is outside the grid {shape}")
        mask[row, col] = True
    return mask


def flat_terrain(rows: int, cols: int, cell_size_m: float = 10.0, eta: float = 1.2,
                 rock_hazard: float = 0.0, shelters: Iterable[Cell] = ((0, 0),)) -> GridTerrain:
    """Level plain with uniform ground."""
    shape = (rows, cols)
    return GridTerrain(elevation=np.zeros(shape), cell_size_m=cell_size_m,
                       walk_eta=np.full(shape, eta), rock_hazard=np.full(shape, rock_hazard),
                       is_shelter=_shelter_mask(shape, shelters))


def slope_terrain(rows: int, cols: int, cell_size_m: float = 10.0, grade: float = 0.10,
                  eta: float = 1.2, shelters: Iterable[Cell] = ((0, 0),)) -> GridTerrain:
    """Inclined plane rising toward increasing column index.

    Args:
        grade: rise over run along the column axis (0.10 = +10 %). Negative = downhill.
    """
    shape = (rows, cols)
    elevation = np.tile(np.arange(cols, dtype=np.float64) * cell_size_m * grade, (rows, 1))
    return GridTerrain(elevation=elevation, cell_size_m=cell_size_m,
                       walk_eta=np.full(shape, eta), is_shelter=_shelter_mask(shape, shelters))


def hill_terrain(rows: int, cols: int, cell_size_m: float = 10.0, height_m: float = 20.0,
                 sigma_m: float = 60.0, center: Cell | None = None, eta: float = 1.2,
                 shelters: Iterable[Cell] = ((0, 0),)) -> GridTerrain:
    """Gaussian hill (negative ``height_m`` gives a pit)."""
    shape = (rows, cols)
    c_row, c_col = center if center is not None else (rows // 2, cols // 2)
    rr, cc = np.mgrid[0:rows, 0:cols]
    dist2 = ((rr - c_row) ** 2 + (cc - c_col) ** 2) * cell_size_m ** 2
    elevation = height_m * np.exp(-dist2 / (2.0 * sigma_m ** 2))
    return GridTerrain(elevation=elevation, cell_size_m=cell_size_m,
                       walk_eta=np.full(shape, eta), is_shelter=_shelter_mask(shape, shelters))


def wall_terrain(rows: int, cols: int, cell_size_m: float = 10.0, wall_col: int | None = None,
                 wall_angle_rad: float = math.radians(30.0), half_width_cells: int = 3,
                 gap_rows: range | None = None, eta: float = 1.2,
                 shelters: Iterable[Cell] = ((0, 0),)) -> GridTerrain:
    """Plain cut by a ridge whose faces are steeper than a suited astronaut can climb.

    The ridge runs along ``wall_col`` with a triangular cross-section: each face
    rises at ``wall_angle_rad`` over ``half_width_cells`` cells. ``gap_rows`` is a
    level pass through it (keep it at least 5 rows wide so its middle is flat).
    """
    shape = (rows, cols)
    wall_col = cols // 2 if wall_col is None else wall_col
    step_m = cell_size_m * math.tan(wall_angle_rad)
    offsets = np.abs(np.arange(cols) - wall_col)
    profile = np.clip(half_width_cells - offsets, 0, None).astype(np.float64) * step_m
    elevation = np.tile(profile, (rows, 1))
    if gap_rows is not None:
        elevation[gap_rows.start:gap_rows.stop, :] = 0.0
    return GridTerrain(elevation=elevation, cell_size_m=cell_size_m,
                       walk_eta=np.full(shape, eta), is_shelter=_shelter_mask(shape, shelters))


def forbidden_zone_terrain(rows: int, cols: int, cell_size_m: float = 10.0,
                           zone: tuple[int, int, int, int] = (0, 0, 0, 0), eta: float = 1.2,
                           shelters: Iterable[Cell] = ((0, 0),)) -> GridTerrain:
    """Level plain with a rectangular no-entry zone.

    Args:
        zone: ``(row_start, row_stop, col_start, col_stop)``, stops exclusive.
    """
    shape = (rows, cols)
    forbidden = np.zeros(shape, dtype=np.bool_)
    r0, r1, c0, c1 = zone
    forbidden[r0:r1, c0:c1] = True
    return GridTerrain(elevation=np.zeros(shape), cell_size_m=cell_size_m,
                       walk_eta=np.full(shape, eta), forbidden=forbidden,
                       is_shelter=_shelter_mask(shape, shelters))


def reference_day_sample() -> EnvSample:
    """Clear equatorial mid-day: the reference conditions of the calibration tests."""
    return EnvSample()


def reference_night_sample() -> EnvSample:
    """Clear equatorial night: cold ground, no sunlight."""
    return EnvSample(t_ground_k=195.0, t_air_k=190.0, t_sky_k=150.0, pressure_pa=700.0,
                     solar_direct_w_m2=0.0, solar_diffuse_w_m2=0.0,
                     sun_elevation_rad=math.radians(-30.0), uv_index=0.0)


@dataclass(frozen=True, slots=True)
class ConstantEnvironment:
    """The same conditions everywhere and at all times."""

    sample: EnvSample = EnvSample()

    def __post_init__(self) -> None:
        self.sample.validate()

    def get_env(self, cell: Cell, t_s: float) -> EnvSample:
        """Return the fixed sample."""
        return self.sample


@dataclass(frozen=True, slots=True)
class DiurnalEnvironment:
    """Equatorial day/night cycle, uniform over the site.

    Mission time 0 is local midnight. The Sun follows
    ``sin(elevation) = cos(hour angle)`` (equator at equinox). The ground follows a
    half-sine by day and an exponential decay toward ``t_ground_min_k`` at night.

    Attributes:
        t_ground_min_k, t_ground_max_k: daily extremes of the ground temperature.
        peak_lag_s: delay of the warmest moment after local noon.
        beam_transmittance: share of the direct beam reaching the ground with the Sun
            at zenith; lower Sun, longer path, less light.
        diffuse_share: diffuse light on the ground as a share of the flux above the air.
    """

    t_ground_min_k: float = 195.0
    t_ground_max_k: float = 261.0
    peak_lag_s: float = 3_700.0
    beam_transmittance: float = 0.6
    diffuse_share: float = 0.25
    pressure_pa: float = 700.0
    dose_rate_gcr_sv_s: float = 0.64e-3 / SOL_S
    sol_s: float = SOL_S

    def sun_elevation_rad(self, t_s: float) -> float:
        """Sun elevation at mission time ``t_s``."""
        hour_angle = 2.0 * math.pi * ((t_s % self.sol_s) / self.sol_s - 0.5)
        return math.asin(max(-1.0, min(1.0, math.cos(hour_angle))))

    def _ground_temperature_k(self, t_s: float) -> float:
        day = self.sol_s / 2.0
        t = t_s % self.sol_s
        sunrise, sunset = self.sol_s / 4.0, 3.0 * self.sol_s / 4.0
        swing = self.t_ground_max_k - self.t_ground_min_k
        span = day + 2.0 * self.peak_lag_s
        if sunrise <= t <= sunset:
            return self.t_ground_min_k + swing * math.sin(math.pi * (t - sunrise) / span)
        at_sunset = math.sin(math.pi * day / span)
        since = (t - sunset) % self.sol_s
        decay = math.exp(-3.0 * since / (self.sol_s - day))
        floor = math.exp(-3.0)
        return self.t_ground_min_k + swing * at_sunset * (decay - floor) / (1.0 - floor)

    def get_env(self, cell: Cell, t_s: float) -> EnvSample:
        """Conditions at mission time ``t_s``; the cell is ignored."""
        elevation = self.sun_elevation_rad(t_s)
        t_ground = self._ground_temperature_k(t_s)
        if elevation > 0.0:
            sin_el = math.sin(elevation)
            beam = MARS_MEAN_SOLAR_FLUX_W_M2 * self.beam_transmittance ** (
                1.0 / max(sin_el, 0.05))
            diffuse = self.diffuse_share * MARS_MEAN_SOLAR_FLUX_W_M2 * sin_el
            t_air, t_sky, uv = t_ground - 15.0 * sin_el, 170.0 + 10.0 * sin_el, 10.0 * sin_el
        else:
            beam = diffuse = uv = 0.0
            t_air, t_sky = t_ground + 3.0, 150.0
        return EnvSample(t_ground_k=t_ground, t_air_k=t_air, t_sky_k=t_sky,
                         pressure_pa=self.pressure_pa,
                         solar_direct_w_m2=beam, solar_diffuse_w_m2=diffuse,
                         sun_elevation_rad=elevation,
                         dose_rate_gcr_sv_s=self.dose_rate_gcr_sv_s, uv_index=uv)


@dataclass(frozen=True, slots=True)
class SpeEnvironment:
    """Wraps another environment and adds a solar particle event to it.

    Attributes:
        base: the undisturbed environment.
        start_s, end_s: mission-time window of the event.
        spe_dose_rate_sv_s: extra dose-equivalent rate during the event.
    """

    base: EnvironmentProvider
    start_s: float = 0.0
    end_s: float = math.inf
    spe_dose_rate_sv_s: float = 3.0 * 0.64e-3 / SOL_S

    def get_env(self, cell: Cell, t_s: float) -> EnvSample:
        """Base conditions, with the event flag and dose rate set inside the window."""
        sample = self.base.get_env(cell, t_s)
        if self.start_s <= t_s < self.end_s:
            return replace(sample, spe_active=True, spe_dose_rate_sv_s=self.spe_dose_rate_sv_s)
        return sample
