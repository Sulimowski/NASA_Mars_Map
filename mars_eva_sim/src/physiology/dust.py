"""A12: dust carried on the suit, airlock cleaning time, perchlorate flag.

Martian dust is fine (1-3 micrometres), electrostatically sticky, and carries
0.5-1 % perchlorates by mass, which are toxic to the thyroid. The dust picked up
grows with the distance walked and the fineness of the soil::

    d_load = load_per_m * distance * (1 + fine_fraction)                  [kg]
    t_clean = base_cleaning_time + cleaning_time_per_kg * load            [s]
    perchlorate = load * perchlorate_mass_fraction                        [kg]

``fine_fraction`` is estimated from the terrain factor: loose sand (eta = 2.1)
is all fines, firm ground (eta = 1.0) has none. Dust does not enter the route
cost; it adds to the post-EVA timeline and raises a habitat contamination flag.
"""

from __future__ import annotations

from ..core.config import DustConfig

_ETA_FIRM = 1.0
_ETA_LOOSE_SAND = 2.1


def soil_fine_fraction(walk_eta: float) -> float:
    """Share of fines in the soil, 0..1, estimated from the terrain factor."""
    return min(1.0, max(0.0, (walk_eta - _ETA_FIRM) / (_ETA_LOOSE_SAND - _ETA_FIRM)))


def dust_increment_kg(distance_m: float, fine_fraction: float, cfg: DustConfig) -> float:
    """Dust picked up over ``distance_m``, kg."""
    return cfg.load_per_m_kg * distance_m * (1.0 + fine_fraction)


def cleaning_time_s(dust_load_kg: float, cfg: DustConfig) -> float:
    """Time needed to clean the suit in the airlock, s."""
    return cfg.base_cleaning_time_s + cfg.cleaning_time_per_kg_s * max(0.0, dust_load_kg)


def perchlorate_mass_kg(dust_load_kg: float, cfg: DustConfig) -> float:
    """Perchlorate carried in with the dust, kg."""
    return max(0.0, dust_load_kg) * cfg.perchlorate_mass_fraction


def perchlorate_flag(dust_load_kg: float, cfg: DustConfig) -> bool:
    """True when the perchlorate brought in exceeds the contamination threshold."""
    return perchlorate_mass_kg(dust_load_kg, cfg) > cfg.perchlorate_flag_kg
