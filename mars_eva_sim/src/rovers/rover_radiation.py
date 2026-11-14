"""R3: radiation and the rover's electronics.

A rover needs no shelter, and the cosmic-ray background over one traverse does
not threaten it. What matters is a solar particle event: energetic protons flip
bits in memory and processors (single-event upsets), which can send the rover
into safe mode. During an event

* the upset probability over a step is ``rate * (event dose rate / reference) * dt``,
* autonomous navigation runs slower (``spe_speed_factor``), or driving stops
  altogether when ``spe_hold`` is set.

The absorbed dose in the electronics is accumulated for the record. The
environment gives dose equivalent (Sv); it is converted with the mean quality
factor of the Martian surface field measured by MSL/RAD.
"""

from __future__ import annotations

from ..core.config import RoverConfig
from ..core.schemas import EnvSample


def upset_risk(env: EnvSample, dt_s: float, cfg: RoverConfig) -> float:
    """Probability of a radiation upset during ``dt_s`` (0 outside a particle event)."""
    if not env.spe_active or cfg.seu_reference_dose_rate_sv_s <= 0.0:
        return 0.0
    relative = env.spe_dose_rate_sv_s / cfg.seu_reference_dose_rate_sv_s
    return min(1.0, cfg.seu_rate_per_s * relative * dt_s)


def speed_factor(env: EnvSample, cfg: RoverConfig) -> float:
    """Multiplier of the driving speed: lower during a particle event, 0 under a hold."""
    if not env.spe_active:
        return 1.0
    return 0.0 if cfg.spe_hold else cfg.spe_speed_factor


def absorbed_dose_gy(env: EnvSample, dt_s: float, cfg: RoverConfig) -> float:
    """Dose absorbed by the electronics over ``dt_s``, Gy."""
    rate = env.dose_rate_gcr_sv_s + (env.spe_dose_rate_sv_s if env.spe_active else 0.0)
    return rate * cfg.electronics_gy_per_sv * dt_s
