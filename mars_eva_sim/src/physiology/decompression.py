"""A11: nitrogen washout before the EVA (prebreathe).

The suit runs at a lower pressure than the habitat. Nitrogen dissolved in the
tissues would form bubbles (decompression sickness) unless part of it is washed
out by breathing pure oxygen first. The risk is measured by the tissue ratio::

    R = pN2_tissue / P_suit,      pN2_tissue = P_hab * (1 - fO2_hab)   when saturated

Breathing pure oxygen, tissue nitrogen falls exponentially with the half-time
``T_half`` of the slowest compartment (360 min in NASA protocols)::

    pN2(t) = pN2_0 * 2^(-t / T_half)

so the prebreathe time that brings ``R`` down to the allowed value is::

    t = T_half * log2(R_0 / R_allowed)      for R_0 > R_allowed, else 0

Checks: ISS air (101.3 kPa, 21 % O2) with a 29.6 kPa suit gives R_0 = 2.70 and
several hours of prebreathe; the exploration atmosphere (56.5 kPa, 34 % O2)
gives R_0 = 1.26 and none.

The prebreathe happens before the suit oxygen is used, so it adds to the
timeline but not to the consumables.
"""

from __future__ import annotations

import math

from ..core.config import DecompressionConfig
from ..core.exceptions import ValidationError
from ..core.schemas import HabitatAtmosphere


def tissue_ratio(habitat: HabitatAtmosphere, suit_pressure_pa: float) -> float:
    """Tissue nitrogen partial pressure over suit pressure for a saturated crew member.

    Raises:
        ValidationError: the suit pressure is not positive.
    """
    if suit_pressure_pa <= 0.0:
        raise ValidationError("suit_pressure_pa must be positive")
    return habitat.pressure_pa * (1.0 - habitat.o2_fraction) / suit_pressure_pa


def prebreathe_time_s(habitat: HabitatAtmosphere, suit_pressure_pa: float,
                      cfg: DecompressionConfig) -> float:
    """Pure-oxygen prebreathe time needed before the EVA, s (0 when none is needed)."""
    ratio = tissue_ratio(habitat, suit_pressure_pa)
    if ratio <= cfg.allowed_tissue_ratio:
        return 0.0
    return cfg.n2_half_time_s * math.log2(ratio / cfg.allowed_tissue_ratio)
