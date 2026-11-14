"""Mars season from an Earth date.

`mars_ls` follows Allison & McEwen (2000), the algorithm behind NASA's Mars24. Ls is the solar
longitude in degrees: 0 = northern spring equinox, 90 = northern summer solstice, 180 = autumn
equinox, 270 = winter solstice. The dusty season is roughly Ls 180 to 360.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

SOL_SECONDS = 88775.244            # one Mars solar day
MARS_HOUR_IN_EARTH_HOURS = SOL_SECONDS / 24.0 / 3600.0   # 1.0275

_J2000_UTC = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
_TT_MINUS_UTC_S = 69.184           # since 2017; 5 s less in 2012, which moves Ls by 0.00003 deg

# Perturbations by the other planets: amplitude (deg), period (Julian years), phase (deg).
_PBS = np.array([
    (0.0071, 2.2353, 49.409), (0.0057, 2.7543, 168.173), (0.0039, 1.1177, 191.837),
    (0.0037, 15.7866, 21.736), (0.0021, 2.1354, 15.704), (0.0020, 2.4694, 95.528),
    (0.0018, 32.8493, 49.095),
])


def mars_ls(utc) -> np.ndarray | float:
    """Solar longitude Ls, degrees 0..360, for a UTC datetime or an array / Series of them."""
    scalar = isinstance(utc, datetime)
    times = np.atleast_1d(np.asarray(utc, dtype="datetime64[ms]") if not scalar else np.datetime64(
        utc.astimezone(timezone.utc).replace(tzinfo=None), "ms"))
    j2000 = np.datetime64(_J2000_UTC.replace(tzinfo=None), "ms")
    days = ((times - j2000) / np.timedelta64(1, "ms") / 1000.0 + _TT_MINUS_UTC_S - 64.184) / 86400.0

    m = np.radians(19.3871 + 0.52402073 * days)                       # mean anomaly
    alpha_fms = 270.3871 + 0.524038496 * days                         # fictitious mean Sun
    pbs = sum(a * np.cos(np.radians(0.985626 * days / tau + phi)) for a, tau, phi in _PBS)
    equation_of_center = ((10.691 + 3.0e-7 * days) * np.sin(m) + 0.623 * np.sin(2 * m)
                          + 0.050 * np.sin(3 * m) + 0.005 * np.sin(4 * m) + 0.0005 * np.sin(5 * m) + pbs)
    ls = (alpha_fms + equation_of_center) % 360.0
    return float(ls[0]) if scalar else ls
