"""Where the Sun is in the Martian sky, and how much of its light gets through the dust.

Time conventions:
    sol      -- index into site.series
    ltst_h   -- local true solar time, 0..24 "Mars hours" (12 = local noon).
                One Mars hour = SOL_HOURS / 24 = 1.0275 Earth hours.
    ls_deg   -- solar longitude, the Martian season: 0 = northern spring equinox,
                90 = northern summer solstice, 251 = perihelion (southern summer).
"""

from __future__ import annotations

import math

SOL_HOURS = 24.6597              # Earth hours in one sol
SOLS_PER_YEAR = 668.6
OBLIQUITY_DEG = 25.19
ECCENTRICITY = 0.0934
SEMI_MAJOR_AXIS_AU = 1.5237
LS_PERIHELION_DEG = 251.0
SOLAR_CONSTANT_1AU = 1361.0      # W/m^2

# The stub series has no season, so sol 0 is pinned to Ls 150 (late southern
# winter, the season Curiosity landed in). Replace once real data carries Ls.
LS_AT_SOL0_DEG = 150.0


def ls_of_sol(sol: float, ls_at_sol0: float = LS_AT_SOL0_DEG) -> float:
    """Season for a sol index. Uses the mean motion, so it drifts by a few degrees
    over a Mars year (the real Ls runs faster near perihelion)."""
    return (ls_at_sol0 + 360.0 / SOLS_PER_YEAR * sol) % 360.0


def declination_rad(ls_deg: float) -> float:
    return math.asin(math.sin(math.radians(OBLIQUITY_DEG)) * math.sin(math.radians(ls_deg)))


def sun_distance_au(ls_deg: float) -> float:
    nu = math.radians(ls_deg - LS_PERIHELION_DEG)
    return SEMI_MAJOR_AXIS_AU * (1 - ECCENTRICITY ** 2) / (1 + ECCENTRICITY * math.cos(nu))


def solar_flux_toa(ls_deg: float) -> float:
    """Sunlight above the atmosphere, W/m^2 (about 490 at aphelion, 715 at perihelion)."""
    return SOLAR_CONSTANT_1AU / sun_distance_au(ls_deg) ** 2


def day_length_h(lat_deg: float, ls_deg: float) -> float:
    """Sunrise to sunset on flat ground, in Mars hours."""
    x = -math.tan(math.radians(lat_deg)) * math.tan(declination_rad(ls_deg))
    return 24.0 / math.pi * math.acos(min(1.0, max(-1.0, x)))


def sun_vector(lat_deg: float, ls_deg: float, ltst_h: float) -> tuple[float, float, float]:
    """Unit vector toward the Sun as (east, north, up). up < 0 means night."""
    lat, dec = math.radians(lat_deg), declination_rad(ls_deg)
    h = math.radians(15.0 * (ltst_h - 12.0))          # hour angle, positive after noon
    east = -math.cos(dec) * math.sin(h)
    north = math.cos(lat) * math.sin(dec) - math.sin(lat) * math.cos(dec) * math.cos(h)
    up = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(h)
    return east, north, up


def sun_elevation_azimuth(lat_deg: float, ls_deg: float, ltst_h: float) -> tuple[float, float]:
    """(elevation, azimuth) in radians; azimuth clockwise from north."""
    east, north, up = sun_vector(lat_deg, ls_deg, ltst_h)
    return math.asin(max(-1.0, min(1.0, up))), math.atan2(east, north) % (2 * math.pi)


def direct_transmittance(tau: float, sin_elevation: float) -> float:
    """Share of the direct solar beam that reaches the ground through dust of
    optical depth tau (Beer-Lambert). Scattered light is not included."""
    if sin_elevation <= 0:
        return 0.0
    return math.exp(-tau / sin_elevation)
