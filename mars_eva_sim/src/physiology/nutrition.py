"""A8: water and energy balance during the EVA.

Water loss (1 kg of water = 1 litre)::

    sweat       = (base + gain * clip((M - onset) / span, 0, 1)) * sweat_rate_coef    [kg/s]
    respiratory = constant                                                             [kg/s]
    deficit    += (sweat + respiratory) * dt

The crew member drinks from the in-suit bag once the deficit passes
``drink_trigger_kg``. With the bag empty and the deficit above
``dehydration_fraction`` of body mass, the power cap is multiplied by
``dehydration_power_factor``.

Energy: after ``fasting_limit_s`` in the suit without food the power cap is
multiplied by ``fasting_power_factor``.
"""

from __future__ import annotations

from ..core.config import NutritionConfig
from ..core.schemas import AstronautProfile


def sweat_rate_kg_s(metabolic_w: float, astronaut: AstronautProfile, cfg: NutritionConfig) -> float:
    """Sweat rate at metabolic power ``M``, kg/s."""
    share = min(1.0, max(0.0, (metabolic_w - cfg.sweat_onset_w) / cfg.sweat_span_w))
    return (cfg.sweat_base_kg_s + cfg.sweat_gain_kg_s * share) * astronaut.sweat_rate_coef


def water_loss_kg(metabolic_w: float, dt_s: float, astronaut: AstronautProfile,
                  cfg: NutritionConfig) -> float:
    """Water lost through sweat and breathing over ``dt_s``, kg."""
    return (sweat_rate_kg_s(metabolic_w, astronaut, cfg) + cfg.respiratory_loss_kg_s) * dt_s


def hydration_step(deficit_kg: float, drink_left_kg: float, metabolic_w: float, dt_s: float,
                   astronaut: AstronautProfile, cfg: NutritionConfig) -> tuple[float, float]:
    """Advance the water balance by ``dt_s``.

    Args:
        deficit_kg: water deficit at the start of the step.
        drink_left_kg: water left in the drink bag.

    Returns:
        ``(new_deficit_kg, water_drunk_kg)``.
    """
    deficit = deficit_kg + water_loss_kg(metabolic_w, dt_s, astronaut, cfg)
    drunk = 0.0
    if deficit > cfg.drink_trigger_kg and drink_left_kg > 0.0:
        drunk = min(deficit, drink_left_kg)
        deficit -= drunk
    return deficit, drunk


def dehydration_factor(deficit_kg: float, drink_left_kg: float, body_mass_kg: float,
                       cfg: NutritionConfig) -> float:
    """Multiplier of the power cap from dehydration (1.0 = no effect)."""
    if drink_left_kg <= 0.0 and deficit_kg > cfg.dehydration_fraction * body_mass_kg:
        return cfg.dehydration_power_factor
    return 1.0


def fasting_factor(time_in_suit_s: float, cfg: NutritionConfig) -> float:
    """Multiplier of the power cap after a long time in the suit without food."""
    return cfg.fasting_power_factor if time_in_suit_s > cfg.fasting_limit_s else 1.0
