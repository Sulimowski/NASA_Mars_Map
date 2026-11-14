"""A10: contingency scenarios and the safety class of a cell.

Three failures are considered at every cell:

* Primary oxygen lost or suit punctured. The secondary supply lasts
  ``o2_secondary_duration_s``; ``reaction_time_s`` of it goes to recognising the
  failure. A shelter must be reachable at emergency pace in what is left.
* Cooling lost. Heat accumulates at ``Q = M + Q_env``; the crew member can keep
  walking for ``S_limit / Q``. If resting is thermally neutral or cold
  (``M_rest + Q_env <= 0``) the crew member can also stop and wait for help, and
  the scenario does not limit the distance.
* Battery lost. Ventilation and CO2 removal stop; the secondary oxygen purge flow
  takes over for ``battery_failure_tolerance_s``.

Classes::

    SAFE_EMERGENCY  every scenario fits, with the reaction margin
    MARGINAL        fits only without the reaction margin, or only the cooling scenario fails
    UNSAFE          a shelter cannot be reached on the secondary oxygen
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..core.config import EmergencyConfig
from ..core.schemas import SafetyFlag, SuitProfile


def emergency_time_limit_s(suit: SuitProfile, cfg: EmergencyConfig) -> float:
    """Time available to reach a shelter after an oxygen or battery failure, s.

    The shorter of the secondary oxygen and the battery-failure tolerance, minus
    the reaction time.
    """
    available = min(suit.o2_secondary_duration_s, cfg.battery_failure_tolerance_s)
    return max(0.0, available - cfg.reaction_time_s)


def cooling_failure_tolerance_s(walking_metabolic_w: float, resting_metabolic_w: float,
                                q_env_w: float, heat_storage_limit_j: float,
                                heat_storage_j: float = 0.0) -> float:
    """Time the crew member can keep walking after the cooling fails, s.

    Args:
        walking_metabolic_w: metabolic power of the walk back.
        resting_metabolic_w: metabolic power at rest.
        q_env_w: heat entering the body from the environment (negative = loss).
        heat_storage_limit_j: tolerated stored heat.
        heat_storage_j: heat already stored.

    Returns:
        ``math.inf`` when resting does not heat the body (the crew member can wait
        for help), otherwise the time until the stored heat reaches its limit while walking.
    """
    if resting_metabolic_w + q_env_w <= 0.0:
        return math.inf
    heating = walking_metabolic_w + q_env_w
    if heating <= 0.0:
        return math.inf
    return max(0.0, heat_storage_limit_j - heat_storage_j) / heating


def classify_cell(emergency_time_s: float, return_time_s: float, cooling_tolerance_s: float,
                  suit: SuitProfile, cfg: EmergencyConfig) -> SafetyFlag:
    """Safety class of a cell.

    Args:
        emergency_time_s: time to the nearest shelter at emergency pace.
        return_time_s: time to the nearest shelter at nominal pace.
        cooling_tolerance_s: from ``cooling_failure_tolerance_s``.
    """
    hard_limit = min(suit.o2_secondary_duration_s, cfg.battery_failure_tolerance_s)
    if not math.isfinite(emergency_time_s) or emergency_time_s > hard_limit:
        return SafetyFlag.UNSAFE
    cooling_ok = return_time_s <= cooling_tolerance_s
    if emergency_time_s <= emergency_time_limit_s(suit, cfg) and cooling_ok:
        return SafetyFlag.SAFE_EMERGENCY
    return SafetyFlag.MARGINAL


def emergency_flag_map(emergency_time_s: NDArray[np.float64], return_time_s: NDArray[np.float64],
                       cooling_tolerance_s: float, suit: SuitProfile,
                       cfg: EmergencyConfig) -> NDArray[np.int8]:
    """``classify_cell`` applied to whole rasters. Values are ``SafetyFlag`` integers."""
    hard_limit = min(suit.o2_secondary_duration_s, cfg.battery_failure_tolerance_s)
    soft_limit = emergency_time_limit_s(suit, cfg)
    flags = np.full(emergency_time_s.shape, int(SafetyFlag.MARGINAL), dtype=np.int8)
    unsafe = ~np.isfinite(emergency_time_s) | (emergency_time_s > hard_limit)
    safe = (emergency_time_s <= soft_limit) & (return_time_s <= cooling_tolerance_s) & ~unsafe
    flags[safe] = int(SafetyFlag.SAFE_EMERGENCY)
    flags[unsafe] = int(SafetyFlag.UNSAFE)
    return flags
