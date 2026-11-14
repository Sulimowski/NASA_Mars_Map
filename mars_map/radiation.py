"""Radiation dose-rate map.

The site-wide dose for a sol comes from the data (site.series). This module only
spreads it over the terrain, using the two effects that vary within a few km:

* sky shielding  -- cliffs and crater walls block part of the sky, and cosmic rays
                    come from the sky, so the dose drops next to them;
* air column     -- lower ground has more atmosphere overhead, so slightly less dose.

Both effects are small (a few percent). On a site this size radiation is mostly a
question of WHICH SOL and HOW LONG, not WHERE: a solar particle event multiplies
the whole map.

Dose here is dose equivalent (Sv), the sum of everything the series was built
from: galactic cosmic rays and their secondaries (neutrons, gammas), plus solar
protons during an event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sun import SOL_HOURS


@dataclass
class RadiationParams:
    """Working assumptions, to be replaced by the real radiation data."""
    # Pressure scale height of the Martian atmosphere.
    scale_height_m: float = 11_100.0
    # d ln(dose) / d ln(air column). Order of magnitude of the pressure/dose
    # anticorrelation MSL/RAD sees over a sol; negative = more air, less dose.
    column_sensitivity: float = -0.5
    # Share of the dose that is lost when the (cosine-weighted) sky is blocked.
    # Below 1 because the ground and the cliff itself send back secondary
    # neutrons and gammas. 0.5 gives a drop of a few percent beside a butte,
    # the size of the effect MSL/RAD measured.
    shielding_efficiency: float = 0.5


def dose_rate_usv_h(dose_msv_per_sol: float) -> float:
    """Site-wide dose for a sol -> microsieverts per Earth hour."""
    return dose_msv_per_sol * 1000.0 / SOL_HOURS


def dose_rate_map(dose_msv_per_sol: float, elevation: np.ndarray, sky_view: np.ndarray,
                  params: RadiationParams = RadiationParams()) -> np.ndarray:
    """Dose rate in every cell, microsieverts per Earth hour (float32).

    A flat, open cell at the site's median elevation gets exactly the series value.
    """
    z_ref = float(np.median(elevation))
    column = np.exp(-(elevation - z_ref) / params.scale_height_m)   # relative air column
    altitude = column ** params.column_sensitivity
    shielding = 1.0 - params.shielding_efficiency * (1.0 - sky_view)
    return (dose_rate_usv_h(dose_msv_per_sol) * altitude * shielding).astype(np.float32)
