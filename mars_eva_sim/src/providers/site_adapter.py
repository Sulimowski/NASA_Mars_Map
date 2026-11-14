"""Adapter from the project's ``Site`` data contract to the simulator's providers.

The data team delivers a ``Site`` (``DATA_CONTRACT.md``), the environment team an
``Environment`` model built on it, the terrain team a ground model
(``AstronautTerrainModel``). This module is the input boundary: it converts their
units (degrees, degrees Celsius, microsieverts per hour, Mars hours) to SI and
their layers to the fields the simulator needs. Nothing in ``physiology/`` or
``routing/`` knows about ``Site``. The objects are duck-typed, so this module
does not import ``mars_map``.

Terrain
-------
With the terrain team's ground model (preferred, so both planners walk the same ground)::

    walk_eta    = terramech_factor          (Pandolf terrain coefficient)
    rock_hazard = rock_impedance
    passable    = share of ``impassable`` cells in the block below 50 %
    forbidden   = ``no_go`` mask

Without it, from the site layers alone::

    walk_eta    = 1.0 + 0.8 * sand + 0.5 * boulders + 0.6 * rock_abundance + 0.3 * softness
                  (+0.2 where high-resolution data is missing), clipped to [1.0, 2.6]
    softness    = clip((ti_firm - thermal_inertia) / (ti_firm - ti_sand), 0, 1)
    rock_hazard = clip(0.65 * rock_abundance + 0.35 * roughness / roughness_ref, 0, 1)
    passable    = terrain class 3 or the optional ``blocked`` layer under 50 % of the block
    forbidden   = inside a no-go polygon

In both cases the slope comes from the (downsampled) elevation, x1.25 where
high-resolution data is missing, and the hazard x1.5 there.

Environment
-----------
Only series every site has are required. Optional series are used when present:
``pressure_pa`` (else the barometric formula) and separate air temperatures
``temp_min_c`` / ``temp_max_c`` next to ``ground_temp_*`` (else a fixed offset
from the ground temperature). There is no dust or wind input: the sites carry
no such data.

``downsample`` merges ``n x n`` source cells into one, which shrinks the search
by ``n^2`` and is the practical way to run the pure-Python search on a
1000 x 1000 site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from ..core.constants import (HOUR_S, MARS_MEAN_SOLAR_FLUX_W_M2, MARS_OBLIQUITY_RAD,
                              MARS_REFERENCE_PRESSURE_PA, MARS_SCALE_HEIGHT_M, SOL_S,
                              ZERO_CELSIUS_K)
from ..core.exceptions import EnvironmentValidationError, TerrainValidationError
from ..core.schemas import Cell, EnvSample
from .grid import GridTerrain, slope_from_elevation

MARS_HOUR_S = SOL_S / 24.0
_IMPASSABLE_CLASS = 3
_SAND_CLASS = 1
_BOULDER_CLASS = 2


@dataclass(frozen=True, slots=True)
class SiteAdapterConfig:
    """Coefficients of the layer mapping (engineering estimates, to be calibrated)."""

    eta_sand: float = 0.8
    eta_boulders: float = 0.5
    eta_rock: float = 0.6
    eta_softness: float = 0.3
    eta_nodata: float = 0.2
    eta_max: float = 2.6
    thermal_inertia_firm: float = 400.0
    thermal_inertia_sand: float = 150.0
    hazard_rock: float = 0.65
    hazard_roughness: float = 0.35
    roughness_ref_m: float = 0.45
    nodata_hazard_factor: float = 1.5
    nodata_slope_factor: float = 1.25
    impassable_share: float = 0.5
    air_offset_day_k: float = 15.0
    air_offset_night_k: float = 3.0
    sky_air_fraction: float = 0.6
    sky_offset_k: float = 9.0
    diffuse_share: float = 0.25
    uv_index_zenith: float = 10.0
    time_bucket_h: float = 0.5


def series_value(site: Any, key: str, sol: int) -> float | None:
    """Value of ``site.series[key]`` for a sol, or ``None`` when the site lacks the series.

    Raises:
        EnvironmentValidationError: the series exists but has no entry for ``sol``.
    """
    series = getattr(site, "series", None) or {}
    values = series.get(key)
    if not values:
        return None
    sols = series.get("sol")
    try:
        index = list(sols).index(sol) if sols else int(sol)
        if index < 0:
            raise IndexError(index)
        return float(values[index])
    except (ValueError, IndexError):
        raise EnvironmentValidationError(
            f"sol {sol} is not in series '{key}' of site '{getattr(site, 'name', '?')}'") from None


def site_sols(site: Any) -> list[int]:
    """Sols the site has series for (empty when it has none)."""
    series = getattr(site, "series", None) or {}
    if series.get("sol"):
        return [int(s) for s in series["sol"]]
    return list(range(max((len(v) for v in series.values()), default=0)))


def _block_mean(layer: NDArray[Any], factor: int, fill: float = 0.0) -> NDArray[np.float64]:
    """Mean over ``factor x factor`` blocks; NaN replaced by ``fill``; trailing cells dropped."""
    data = np.nan_to_num(np.asarray(layer, dtype=np.float64), nan=fill, posinf=fill, neginf=fill)
    if factor == 1:
        return data
    rows, cols = data.shape[0] // factor, data.shape[1] // factor
    trimmed = data[:rows * factor, :cols * factor]
    return trimmed.reshape(rows, factor, cols, factor).mean(axis=(1, 3))


def rasterize_polygons(shape: tuple[int, int], cell_size_m: float,
                       polygons: Iterable[Iterable[tuple[float, float]]]) -> NDArray[np.bool_]:
    """Mask of the cells whose centre lies inside any polygon (even-odd rule).

    Args:
        polygons: vertex lists in metres, ``(x east, y south)`` of the NW corner.
    """
    rows, cols = shape
    xs = (np.arange(cols) + 0.5) * cell_size_m
    ys = (np.arange(rows) + 0.5) * cell_size_m
    grid_x, grid_y = np.meshgrid(xs, ys)
    mask = np.zeros(shape, dtype=np.bool_)
    for polygon in polygons:
        vertices = list(polygon)
        if len(vertices) < 3:
            continue
        inside = np.zeros(shape, dtype=np.bool_)
        j = len(vertices) - 1
        for i in range(len(vertices)):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            if yi != yj:
                crosses = ((yi > grid_y) != (yj > grid_y)) & (
                    grid_x < (xj - xi) * (grid_y - yi) / (yj - yi) + xi)
                inside ^= crosses
            j = i
        mask |= inside
    return mask


def terrain_from_site(site: Any, *, downsample: int = 1,
                      shelters_m: Iterable[tuple[float, float]] | None = None,
                      ground_model: Any | None = None,
                      cfg: SiteAdapterConfig = SiteAdapterConfig()) -> GridTerrain:
    """Build a ``GridTerrain`` from a ``Site``.

    Args:
        site: object with ``layers`` (dict of rasters), ``resolution_m``, ``no_go``
            and ``start``, as defined by the data contract.
        downsample: merge this many source cells per side into one.
        shelters_m: shelter positions in metres; the site start by default.
        ground_model: the terrain team's ``AstronautTerrainModel`` for this site
            (``terramech_factor``, ``rock_impedance``, ``impassable``, optional
            ``no_go``). When given, its rasters replace the built-in mapping.

    Raises:
        TerrainValidationError: a required layer is missing, ``downsample`` is
            invalid, or the ground model does not match the site grid.
    """
    if downsample < 1:
        raise TerrainValidationError(f"downsample={downsample} must be >= 1")
    layers = site.layers
    if "elevation" not in layers:
        raise TerrainValidationError("site has no 'elevation' layer")
    source_shape = np.asarray(layers["elevation"]).shape
    cell = float(site.resolution_m) * downsample
    elevation = _block_mean(layers["elevation"], downsample,
                            fill=float(np.nanmedian(layers["elevation"])))
    shape = (int(elevation.shape[0]), int(elevation.shape[1]))
    nodata = (_block_mean(np.asarray(layers["nodata"], dtype=bool), downsample) > 0.0
              if "nodata" in layers else np.zeros(shape, dtype=np.bool_))

    if ground_model is not None:
        for name in ("terramech_factor", "rock_impedance", "impassable"):
            raster = getattr(ground_model, name, None)
            if raster is None or np.asarray(raster).shape != source_shape:
                raise TerrainValidationError(
                    f"ground model raster '{name}' is missing or does not match the site grid "
                    f"{source_shape}")
        eta = _block_mean(ground_model.terramech_factor, downsample, fill=1.0)
        hazard = _block_mean(ground_model.rock_impedance, downsample)
        blocked = _block_mean(np.asarray(ground_model.impassable, dtype=bool), downsample)
        model_no_go = getattr(ground_model, "no_go", None)
        forbidden = (_block_mean(np.asarray(model_no_go, dtype=bool), downsample)
                     >= cfg.impassable_share) if model_no_go is not None else None
    else:
        required = ("roughness", "rock_abundance", "thermal_inertia", "terrain_class")
        missing = [name for name in required if name not in layers]
        if missing:
            raise TerrainValidationError(f"site is missing layers: {missing}")
        rock = np.clip(_block_mean(layers["rock_abundance"], downsample), 0.0, 1.0)
        roughness = _block_mean(layers["roughness"], downsample)
        inertia = _block_mean(layers["thermal_inertia"], downsample,
                              fill=cfg.thermal_inertia_firm)
        classes = np.asarray(layers["terrain_class"])
        sand = _block_mean(classes == _SAND_CLASS, downsample)
        boulders = _block_mean(classes == _BOULDER_CLASS, downsample)
        hard = classes == _IMPASSABLE_CLASS
        if "blocked" in layers:
            hard = hard | np.asarray(layers["blocked"], dtype=bool)
        blocked = _block_mean(hard, downsample)
        softness = np.clip((cfg.thermal_inertia_firm - inertia)
                           / (cfg.thermal_inertia_firm - cfg.thermal_inertia_sand), 0.0, 1.0)
        eta = (1.0 + cfg.eta_sand * sand + cfg.eta_boulders * boulders + cfg.eta_rock * rock
               + cfg.eta_softness * softness + cfg.eta_nodata * nodata)
        hazard = cfg.hazard_rock * rock + cfg.hazard_roughness * roughness / cfg.roughness_ref_m
        forbidden = None

    if forbidden is None:
        forbidden = rasterize_polygons(shape, cell, getattr(site, "no_go", None) or [])
    hazard = np.where(nodata, hazard * cfg.nodata_hazard_factor, hazard)
    slope = slope_from_elevation(elevation, cell)
    slope = np.where(nodata, np.arctan(np.tan(slope) * cfg.nodata_slope_factor), slope)

    shelter_mask = np.zeros(shape, dtype=np.bool_)
    for x_m, y_m in (shelters_m if shelters_m is not None else [tuple(site.start)]):
        row = min(max(int(y_m / cell), 0), shape[0] - 1)
        col = min(max(int(x_m / cell), 0), shape[1] - 1)
        shelter_mask[row, col] = True
    return GridTerrain(
        elevation=elevation, cell_size_m=cell, slope=slope,
        walk_eta=np.clip(eta, 1.0, cfg.eta_max), rock_hazard=np.clip(hazard, 0.0, 1.0),
        passable_walk=blocked < cfg.impassable_share, forbidden=forbidden,
        is_shelter=shelter_mask)


class SiteEnvironment:
    """``EnvironmentProvider`` backed by the environment team's model for one sol.

    Mission time ``t_s = 0`` is local true solar time ``start_ltst_h`` of ``sol``.
    Maps are requested from the model once per ``time_bucket_h`` and kept.

    The model gives ground temperature, dose rate and direct sunlight. The rest
    is filled in here: air temperature from the air series when the site has
    them (else a fixed offset from the ground), sky temperature from the air
    temperature, pressure from the pressure series or the barometric formula,
    diffuse light as a fixed share of the solar flux.
    """

    def __init__(self, site: Any, model: Any, *, sol: int, start_ltst_h: float,
                 downsample: int = 1, cfg: SiteAdapterConfig = SiteAdapterConfig()) -> None:
        """
        Args:
            site: the ``Site``.
            model: the ``Environment`` built on that site (``temperature``, ``dose_rate``,
                ``direct_sunlight``, ``ls``, ``is_particle_event``).
            sol: sol of the EVA, an entry of the site series.
            start_ltst_h: local true solar time at mission time 0, Mars hours (0..24).
            downsample: the same factor as passed to ``terrain_from_site``.

        Raises:
            EnvironmentValidationError: ``start_ltst_h`` is outside 0..24, the sol is
                not in the series, or the site has no temperature series (use
                ``planning.diurnal.SeriesEnvironment`` with explicit temperatures then).
        """
        if not 0.0 <= start_ltst_h <= 24.0:
            raise EnvironmentValidationError(f"start_ltst_h={start_ltst_h} outside [0, 24]")
        if not getattr(model, "has_temperature", True):
            raise EnvironmentValidationError(
                f"site '{getattr(site, 'name', '?')}' has no temperature series")
        self._site = site
        self._model = model
        self._sol = sol
        self._start_ltst_h = start_ltst_h
        self._factor = downsample
        self._cfg = cfg
        self._lat = math.radians(float(site.center_lat))
        self._declination = math.asin(math.sin(MARS_OBLIQUITY_RAD)
                                      * math.sin(math.radians(float(model.ls(sol)))))
        self._elevation = _block_mean(site.layers["elevation"], downsample,
                                      fill=float(np.nanmedian(site.layers["elevation"])))
        self._median_elevation = float(np.median(self._elevation))
        self._pressure_series = series_value(site, "pressure_pa", sol)

        ground = (series_value(site, "ground_temp_min_c", sol),
                  series_value(site, "ground_temp_max_c", sol))
        air = (series_value(site, "temp_min_c", sol), series_value(site, "temp_max_c", sol))
        self._ground_range: tuple[float, float] | None = None
        self._air_range: tuple[float, float] | None = None
        if None not in ground and None not in air:      # both: the generic pair is air
            self._ground_range = (ground[0] + ZERO_CELSIUS_K, ground[1] + ZERO_CELSIUS_K)  # type: ignore[operator]
            self._air_range = (air[0] + ZERO_CELSIUS_K, air[1] + ZERO_CELSIUS_K)  # type: ignore[operator]

        try:
            dose_sv_s = _block_mean(model.dose_rate(sol), downsample) * 1.0e-6 / HOUR_S
        except ValueError as error:
            raise EnvironmentValidationError(str(error)) from None
        doses = [float(v) for v in (getattr(site, "series", {}) or {}).get(
            "radiation_msv_per_sol", [])]
        today = series_value(site, "radiation_msv_per_sol", sol)
        self._spe = bool(model.is_particle_event(sol))
        quiet_share = 1.0
        if self._spe and doses and today:
            quiet_share = min(1.0, float(np.median(doses)) / today)
        self._gcr = dose_sv_s * quiet_share
        self._spe_rate = dose_sv_s * (1.0 - quiet_share)
        self._temp_cache: dict[int, NDArray[np.float64]] = {}
        self._sun_cache: dict[int, NDArray[np.float64]] = {}
        self._air_shift_cache: dict[int, float | None] = {}

    @property
    def sol(self) -> int:
        """Sol the environment describes."""
        return self._sol

    @property
    def particle_event(self) -> bool:
        """True when the sol carries a solar particle event."""
        return self._spe

    def ltst_h(self, t_s: float) -> float:
        """Local true solar time at mission time ``t_s``, Mars hours."""
        return (self._start_ltst_h + t_s / MARS_HOUR_S) % 24.0

    def sun_elevation_rad(self, ltst_h: float) -> float:
        """Sun elevation at the site centre for a local true solar time."""
        hour_angle = math.radians(15.0 * (ltst_h - 12.0))
        sin_el = (math.sin(self._lat) * math.sin(self._declination)
                  + math.cos(self._lat) * math.cos(self._declination) * math.cos(hour_angle))
        return math.asin(max(-1.0, min(1.0, sin_el)))

    def _load_bucket(self, index: int, bucket_h: float) -> None:
        temp = _block_mean(self._model.temperature(self._sol, bucket_h), self._factor,
                           fill=-60.0) + ZERO_CELSIUS_K
        self._temp_cache[index] = temp
        self._sun_cache[index] = _block_mean(
            self._model.direct_sunlight(self._sol, bucket_h), self._factor)
        shift: float | None = None
        if self._ground_range is not None and self._air_range is not None:
            g_min, g_max = self._ground_range
            a_min, a_max = self._air_range
            span = g_max - g_min
            phase = (min(1.0, max(0.0, (float(np.median(temp)) - g_min) / span))
                     if span > 0.0 else 0.5)
            shift = (a_min + (a_max - a_min) * phase) - (g_min + span * phase)
        self._air_shift_cache[index] = shift

    def get_env(self, cell: Cell, t_s: float) -> EnvSample:
        """Conditions at ``cell`` (of the downsampled grid) at mission time ``t_s``."""
        cfg = self._cfg
        ltst = self.ltst_h(t_s)
        index = int(round(ltst / cfg.time_bucket_h))
        bucket_h = (index * cfg.time_bucket_h) % 24.0
        if index not in self._temp_cache:
            self._load_bucket(index, bucket_h)
        row, col = cell
        t_ground = float(self._temp_cache[index][row, col])
        ground_beam = float(self._sun_cache[index][row, col])
        elevation = self.sun_elevation_rad(bucket_h)
        sin_el = math.sin(elevation)
        shift = self._air_shift_cache[index]
        if sin_el > 0.0:
            t_air = t_ground + shift if shift is not None else (
                t_ground - cfg.air_offset_day_k * sin_el)
            beam = ground_beam / max(sin_el, 0.1)
            diffuse = cfg.diffuse_share * MARS_MEAN_SOLAR_FLUX_W_M2 * sin_el
            uv = cfg.uv_index_zenith * sin_el
        else:
            t_air = t_ground + shift if shift is not None else t_ground + cfg.air_offset_night_k
            beam = diffuse = uv = 0.0
        height = float(self._elevation[row, col])
        if self._pressure_series is not None:
            pressure = self._pressure_series * math.exp(
                -(height - self._median_elevation) / MARS_SCALE_HEIGHT_M)
        else:
            pressure = MARS_REFERENCE_PRESSURE_PA * math.exp(-height / MARS_SCALE_HEIGHT_M)
        return EnvSample(
            t_ground_k=t_ground, t_air_k=t_air,
            t_sky_k=cfg.sky_air_fraction * t_air + cfg.sky_offset_k,
            pressure_pa=pressure, solar_direct_w_m2=beam, solar_diffuse_w_m2=diffuse,
            sun_elevation_rad=elevation, dose_rate_gcr_sv_s=float(self._gcr[row, col]),
            spe_active=self._spe, spe_dose_rate_sv_s=float(self._spe_rate[row, col]), uv_index=uv)
