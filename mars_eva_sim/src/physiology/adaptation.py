"""A1: deconditioning after the transit and re-adaptation on the surface.

Months of microgravity reduce aerobic capacity and muscle strength and disturb
balance and blood-pressure regulation. On the surface each quantity recovers
exponentially; aerobic capacity and strength level off below the pre-flight
value because 0.38 g loads the body less than Earth does.

With ``d`` = time on the surface and ``T`` = transit duration (both in seconds)::

    loss0     = min(loss_max, loss_per_transit_s * T)
    plateau   = min(plateau_loss, loss0)
    aerobic   = (1 - loss0) + (loss0 - plateau) * (1 - exp(-d / tau_aerobic))
    strength  = same form with its own constants
    sensorimotor = 1 - 0.5 * exp(-d / tau_sensorimotor)
    orthostatic  = 1 - 0.4 * exp(-d / tau_orthostatic)
    bone_fracture_mult = 1 + gain_per_transit_s * T
"""

from __future__ import annotations

import math

from ..core.config import AdaptationConfig
from ..core.exceptions import ValidationError
from ..core.schemas import Fitness


def _recovering_capacity(surface_s: float, transit_s: float, loss_per_s: float, loss_max: float,
                         plateau_loss: float, tau_s: float) -> float:
    loss0 = min(loss_max, loss_per_s * transit_s)
    plateau = min(plateau_loss, loss0)
    return (1.0 - loss0) + (loss0 - plateau) * (1.0 - math.exp(-surface_s / tau_s))


def compute_fitness(time_on_surface_s: float, transit_duration_s: float,
                    cfg: AdaptationConfig) -> Fitness:
    """Condition of the crew member relative to pre-flight.

    Args:
        time_on_surface_s: time since landing, s.
        transit_duration_s: time spent in microgravity before landing, s.
        cfg: A1 parameters.

    Returns:
        Fitness factors (1.0 = pre-flight) and the bone fracture risk multiplier.

    Raises:
        ValidationError: a duration is negative or not finite.
    """
    for name, value in (("time_on_surface_s", time_on_surface_s),
                        ("transit_duration_s", transit_duration_s)):
        if not math.isfinite(value) or value < 0.0:
            raise ValidationError(f"{name}={value} must be a non-negative finite number")
    aerobic = _recovering_capacity(time_on_surface_s, transit_duration_s,
                                   cfg.aerobic_loss_per_transit_s, cfg.aerobic_loss_max,
                                   cfg.aerobic_plateau_loss, cfg.aerobic_recovery_tau_s)
    strength = _recovering_capacity(time_on_surface_s, transit_duration_s,
                                    cfg.strength_loss_per_transit_s, cfg.strength_loss_max,
                                    cfg.strength_plateau_loss, cfg.strength_recovery_tau_s)
    sensorimotor = 1.0 - cfg.sensorimotor_initial_deficit * math.exp(
        -time_on_surface_s / cfg.sensorimotor_tau_s)
    orthostatic = 1.0 - cfg.orthostatic_initial_deficit * math.exp(
        -time_on_surface_s / cfg.orthostatic_tau_s)
    bone = 1.0 + cfg.bone_fracture_gain_per_transit_s * transit_duration_s
    return Fitness(aerobic=aerobic, strength=strength, sensorimotor=sensorimotor,
                   orthostatic=orthostatic, bone_fracture_mult=bone)


def eva_permitted(time_on_surface_s: float, cfg: AdaptationConfig) -> bool:
    """False during the first sol after landing, when no EVA is allowed."""
    return time_on_surface_s >= cfg.min_surface_time_for_eva_s


def slope_limits_rad(fitness: Fitness, cfg: AdaptationConfig, up_limit_rad: float,
                     down_limit_rad: float, reduced_limit_rad: float) -> tuple[float, float]:
    """Steepest slopes the crew member may walk, as (uphill, downhill) in rad.

    While balance is impaired (``sensorimotor`` below the threshold) both limits
    drop to ``reduced_limit_rad``.
    """
    if fitness.sensorimotor < cfg.sensorimotor_slope_threshold:
        return min(up_limit_rad, reduced_limit_rad), min(down_limit_rad, reduced_limit_rad)
    return up_limit_rad, down_limit_rad
