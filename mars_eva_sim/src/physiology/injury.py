"""A9: probability of a fall and of its consequences.

Logistic model, per metre walked::

    x = b0 + b_slope * |slope| + b_rock * rock_hazard + b_sens * (1 - sensorimotor)
        + b_fat * F + b_vis * (1 - visibility) + b_steep * [grade < steep_downhill_grade]
        + b_speed * (V / V_pref)
    p_per_m = base_rate_per_m / (1 + exp(-x))
    P_fall  = 1 - (1 - p_per_m)^length

Consequences::

    P(injury | fall) = injury_given_fall * bone_fracture_mult
    P(breach | fall) = breach_given_fall * rock_hazard
    expected_harm    = P_fall * (w_injury * P(injury | fall) + w_breach * P(breach | fall))

There is no NASA data for falls on Mars. Apollo crews fell often and harmlessly,
which is why the coefficients are calibration parameters and not constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.config import InjuryConfig


@dataclass(frozen=True, slots=True)
class FallRisk:
    """Fall probability over one edge and what a fall would lead to."""

    p_fall: float
    p_injury_given_fall: float
    p_breach_given_fall: float
    expected_harm: float


def fall_probability_per_m(slope_rad: float, grade: float, rock_hazard: float,
                           sensorimotor: float, fatigue: float, visibility: float,
                           speed_ratio: float, cfg: InjuryConfig) -> float:
    """Probability of a fall per metre walked."""
    x = (cfg.intercept
         + cfg.slope_per_rad * abs(slope_rad)
         + cfg.rock_hazard * rock_hazard
         + cfg.sensorimotor_deficit * (1.0 - sensorimotor)
         + cfg.fatigue * fatigue
         + cfg.visibility_deficit * (1.0 - visibility)
         + (cfg.steep_downhill if grade < cfg.steep_downhill_grade else 0.0)
         + cfg.speed_ratio * speed_ratio)
    return cfg.base_rate_per_m / (1.0 + math.exp(-x))


def fall_risk(length_m: float, slope_rad: float, grade: float, rock_hazard: float,
              sensorimotor: float, fatigue: float, visibility: float, speed_ratio: float,
              bone_fracture_mult: float, cfg: InjuryConfig) -> FallRisk:
    """Fall risk over an edge of ``length_m`` metres.

    Args:
        slope_rad: slope angle along the direction of travel.
        grade: rise over run along the direction of travel (negative downhill).
        rock_hazard: 0..1.
        sensorimotor: balance factor from A1, 0..1.
        fatigue: whole-body fatigue F, 0..1.
        visibility: 0..1 (1 = clear daylight).
        speed_ratio: walking speed over the preferred speed.
        bone_fracture_mult: fracture risk multiplier from A1.
    """
    per_m = fall_probability_per_m(slope_rad, grade, rock_hazard, sensorimotor, fatigue,
                                   visibility, speed_ratio, cfg)
    p_fall = -math.expm1(length_m * math.log1p(-per_m))
    p_injury = min(1.0, cfg.injury_given_fall * bone_fracture_mult)
    p_breach = min(1.0, cfg.breach_given_fall * rock_hazard)
    harm = p_fall * (cfg.weight_injury * p_injury + cfg.weight_breach * p_breach)
    return FallRisk(p_fall=p_fall, p_injury_given_fall=p_injury, p_breach_given_fall=p_breach,
                    expected_harm=harm)
