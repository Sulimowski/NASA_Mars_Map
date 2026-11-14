"""A2: metabolic power of walking in a suit and of stationary tasks.

Walking uses the Pandolf load-carriage equation (Pandolf et al., 1977) rescaled
to Mars gravity, with a Santee-style correction for descents (Santee et al., 2003).
With body mass ``W`` (kg), suit mass ``L`` (kg), speed ``V`` (m/s), grade ``G``
(percent), terrain factor ``eta`` and ``k_g = g / g_earth``::

    M_stand = 1.5 * W
    M_level = k_lvl * (2 * (W + L) * (L / W)^2 + eta * (W + L) * 1.5 * V^2)
    M_grade = k_g * eta * (W + L) * 0.35 * V * G_eff
    M       = stiffness * max(M_stand, M_stand + M_level + M_grade)          [W]

    k_lvl = 0.3 + 0.65 * k_g
    G_eff = G                                         for G >= 0
    G_eff = 0.5 * G + 0.6 * max(0, |G| - 10)          for G < 0

The climbing term scales exactly with gravity (potential energy is ``m g h``).
Level walking scales less than gravity, which is what suited reduced-gravity
trials show. Going down, half of the grade term is recovered and braking costs
energy beyond a 10 % descent, so the cheapest descent is near -10 %.

For a fixed edge ``M`` is a quadratic in ``V``; ``pandolf_coefficients`` returns
that quadratic so the speed search of A3 does not recompute the terrain terms.
"""

from __future__ import annotations

from ..core.config import MetabolismConfig
from ..core.constants import G_EARTH_M_S2, G_MARS_M_S2
from ..core.exceptions import ValidationError
from ..core.schemas import TaskType


def effective_grade_percent(grade: float, cfg: MetabolismConfig) -> float:
    """Grade in percent as it enters the Pandolf grade term.

    Args:
        grade: rise over run (0.10 = +10 %).

    Returns:
        ``G_eff`` in percent; negative on gentle descents, rising again on steep ones.
    """
    percent = 100.0 * grade
    if percent >= 0.0:
        return percent
    braking = max(0.0, -percent - 100.0 * cfg.braking_onset_grade)
    return cfg.downhill_recovery * percent + cfg.braking_factor * braking


def pandolf_coefficients(body_mass_kg: float, load_mass_kg: float, eta: float, grade: float,
                         stiffness: float, cfg: MetabolismConfig,
                         gravity_m_s2: float = G_MARS_M_S2) -> tuple[float, float, float, float]:
    """Coefficients of ``M(V) = max(floor, c0 + c1 * V + c2 * V^2)`` in watts.

    Args:
        body_mass_kg: body mass W.
        load_mass_kg: carried mass L (suit and life support).
        eta: terrain factor, >= 1.
        grade: rise over run of the edge.
        stiffness: suit stiffness multiplier, >= 1.
        cfg: A2 parameters.
        gravity_m_s2: surface gravity.

    Returns:
        ``(c0, c1, c2, floor)``; ``floor`` is the standing power.

    Raises:
        ValidationError: a mass or the terrain factor is not positive.
    """
    if body_mass_kg <= 0.0 or load_mass_kg < 0.0 or eta <= 0.0:
        raise ValidationError(
            f"non-physical walking inputs: W={body_mass_kg}, L={load_mass_kg}, eta={eta}")
    total = body_mass_kg + load_mass_kg
    k_g = gravity_m_s2 / G_EARTH_M_S2
    k_level = cfg.level_scale_intercept + cfg.level_scale_slope * k_g
    standing = cfg.standing_coeff * body_mass_kg
    c0 = standing + k_level * cfg.load_coeff * total * (load_mass_kg / body_mass_kg) ** 2
    c1 = k_g * eta * total * cfg.grade_coeff * effective_grade_percent(grade, cfg)
    c2 = k_level * eta * total * cfg.speed_coeff
    return stiffness * c0, stiffness * c1, stiffness * c2, stiffness * standing


def power_from_coefficients(c0: float, c1: float, c2: float, floor: float,
                            speed_m_s: float) -> float:
    """Evaluate the walking power quadratic, W."""
    return max(floor, c0 + c1 * speed_m_s + c2 * speed_m_s * speed_m_s)


def walking_power_w(body_mass_kg: float, load_mass_kg: float, speed_m_s: float, grade: float,
                    eta: float, stiffness: float, cfg: MetabolismConfig,
                    gravity_m_s2: float = G_MARS_M_S2) -> float:
    """Metabolic power of suited walking, W.

    Args:
        speed_m_s: walking speed along the ground, m/s.
        grade: rise over run (positive uphill).

    Raises:
        ValidationError: the speed is negative or an input is non-physical.
    """
    if speed_m_s < 0.0:
        raise ValidationError(f"speed_m_s={speed_m_s} must be non-negative")
    c0, c1, c2, floor = pandolf_coefficients(body_mass_kg, load_mass_kg, eta, grade, stiffness,
                                             cfg, gravity_m_s2)
    return power_from_coefficients(c0, c1, c2, floor, speed_m_s)


def mechanical_efficiency(grade: float, cfg: MetabolismConfig) -> float:
    """Share of the metabolic power that leaves the body as external work, 0..max.

    Zero on level ground and downhill (all the power becomes heat), rising
    linearly with the uphill grade up to ``mech_efficiency_max``.
    """
    if grade <= 0.0:
        return 0.0
    return min(cfg.mech_efficiency_max, cfg.mech_efficiency_per_grade * grade)


def task_power_w(task: TaskType, fatigue: float, cfg: MetabolismConfig) -> float:
    """Metabolic power of a stationary task, W.

    ``M = table[task] * (1 + penalty * F)``: a tired crew member works less
    efficiently. Rest is not penalised.
    """
    base = cfg.power_of(task)
    if task is TaskType.REST:
        return base
    return base * (1.0 + cfg.task_fatigue_penalty * min(max(fatigue, 0.0), 1.0))
