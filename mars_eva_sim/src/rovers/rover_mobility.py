"""R1: can the rover take this step, how much do its wheels slip, how fast does it go.

Geometry
--------
A step is refused when

* an obstacle in the cell is taller than the rover may pass over (``CLEARANCE``),
* the ground of the cell is steeper than the tilt limit (``TILT``),
* the slope along the direction of travel is steeper than the climb limit (``CLIMB``).

Wheel slip (simplified Bekker-Wong terramechanics)
--------------------------------------------------
Soil parameters are interpolated by the terrain factor between firm ground and
loose sand. For one wheel of diameter ``D`` and width ``b`` carrying ``W``::

    sinkage      z   = (3 W / ((3 - n) (k_c + b k_phi) sqrt(D))) ^ (2 / (2 n + 1))
    resistance   R_c = (k_c + b k_phi) z^(n+1) / (n + 1)
    contact      l   = max(sqrt(D z), floor * D) * contact_length_factor
    thrust       H(s) = traction (b l c + W tan(phi)) (1 - K/(s l) (1 - exp(-s l / K)))

The slip ``s`` is where the thrust of all wheels equals what the step demands::

    N H(s) = m g sin(theta) + N R_c

The step is refused (``SLIP``) when no slip below the critical value gives enough
thrust: the wheels would dig in. Ground speed is the wheel speed times ``1 - s``.

Wheel-walking
-------------
A rover that can walk its wheels (Rosalind Franklin) switches to that mode when
rolling fails on slip: thrust rises by ``wheel_walking_thrust_factor``, the
critical slip rises, and the pace falls to the wheel-walking speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.config import RoverConfig
from ..core.constants import G_MARS_M_S2
from .profiles import RoverProfile
from .types import DriveMode, RoverBlock


@dataclass(frozen=True, slots=True)
class Soil:
    """Bekker-Wong soil parameters (SI)."""

    cohesion_pa: float
    friction_rad: float
    k_c: float
    k_phi: float
    sinkage_exponent: float
    shear_modulus_m: float


@dataclass(frozen=True, slots=True)
class MobilityResult:
    """Outcome of ``assess_step``.

    Attributes:
        block: ``RoverBlock.NONE`` when the step can be taken.
        slip: wheel slip, 0..1.
        mode: rolling or wheel-walking.
        wheel_speed_m_s: speed of the wheel rim.
        ground_speed_m_s: speed over the ground, ``wheel speed * (1 - slip)``.
        sinkage_m: wheel sinkage.
        thrust_n: traction force the step demands from all wheels (0 on a descent).
    """

    block: RoverBlock
    slip: float = 0.0
    mode: DriveMode = DriveMode.ROLLING
    wheel_speed_m_s: float = 0.0
    ground_speed_m_s: float = 0.0
    sinkage_m: float = 0.0
    thrust_n: float = 0.0

    @property
    def passable(self) -> bool:
        """True when the step can be taken."""
        return self.block is RoverBlock.NONE


def soil_from_terrain_factor(eta: float, cfg: RoverConfig) -> Soil:
    """Soil parameters for a terrain factor, between firm ground and loose sand.

    Stiffness is interpolated on a logarithmic scale (it spans more than a decade),
    the other parameters linearly.
    """
    loose = min(1.0, max(0.0, (eta - cfg.eta_firm) / (cfg.eta_loose - cfg.eta_firm)))
    firm = 1.0 - loose

    def mix(firm_value: float, loose_value: float) -> float:
        return firm * firm_value + loose * loose_value

    k_phi = math.exp(mix(math.log(cfg.k_phi_firm), math.log(cfg.k_phi_loose)))
    return Soil(cohesion_pa=mix(cfg.cohesion_firm_pa, cfg.cohesion_loose_pa),
                friction_rad=mix(cfg.friction_firm_rad, cfg.friction_loose_rad),
                k_c=cfg.k_c, k_phi=k_phi,
                sinkage_exponent=mix(cfg.sinkage_exponent_firm, cfg.sinkage_exponent_loose),
                shear_modulus_m=mix(cfg.shear_modulus_firm_m, cfg.shear_modulus_loose_m))


def wheel_sinkage_m(load_n: float, diameter_m: float, width_m: float, soil: Soil) -> float:
    """Static sinkage of a rigid wheel (Bekker), m."""
    n = soil.sinkage_exponent
    stiffness = soil.k_c + width_m * soil.k_phi
    return (3.0 * load_n / ((3.0 - n) * stiffness * math.sqrt(diameter_m))) ** (2.0 / (2.0 * n + 1.0))


def compaction_resistance_n(sinkage_m: float, width_m: float, soil: Soil) -> float:
    """Force needed to compact the rut of one wheel (Bekker), N."""
    n = soil.sinkage_exponent
    return (soil.k_c + width_m * soil.k_phi) * sinkage_m ** (n + 1.0) / (n + 1.0)


def wheel_thrust_n(slip: float, load_n: float, contact_length_m: float, width_m: float,
                   soil: Soil, traction_factor: float) -> float:
    """Thrust one wheel develops at a slip (Janosi-Hanamoto shear law), N."""
    if slip <= 0.0:
        return 0.0
    ratio = slip * contact_length_m / soil.shear_modulus_m
    mobilised = 1.0 - (1.0 - math.exp(-ratio)) / ratio
    limit = width_m * contact_length_m * soil.cohesion_pa + load_n * math.tan(soil.friction_rad)
    return traction_factor * limit * mobilised


def solve_slip(demand_n: float, load_n: float, contact_length_m: float, profile: RoverProfile,
               soil: Soil, traction_factor: float, slip_limit: float,
               tolerance: float) -> float | None:
    """Slip at which all wheels together give ``demand_n``, or ``None`` beyond the limit."""
    if demand_n <= 0.0:
        return 0.0

    def total(slip: float) -> float:
        return profile.wheel_count * wheel_thrust_n(slip, load_n, contact_length_m,
                                                    profile.wheel_width_m, soil, traction_factor)

    if total(slip_limit) < demand_n:
        return None
    low, high = 0.0, slip_limit
    while high - low > tolerance:
        mid = 0.5 * (low + high)
        if total(mid) >= demand_n:
            high = mid
        else:
            low = mid
    return high


def assess_step(profile: RoverProfile, cfg: RoverConfig, *, slope_rad: float,
                cell_slope_rad: float, eta: float, obstacle_height_m: float,
                wheel_speed_m_s: float) -> MobilityResult:
    """Judge one step of a rover.

    Args:
        slope_rad: slope along the direction of travel (positive uphill).
        cell_slope_rad: steepest slope of the cell being entered.
        eta: terrain factor of the ground (1.0 firm .. 2.1 loose sand).
        obstacle_height_m: height of the obstacles in the cell.
        wheel_speed_m_s: wheel rim speed commanded for rolling.
    """
    if obstacle_height_m > profile.max_obstacle_height_m:
        return MobilityResult(RoverBlock.CLEARANCE)
    if cell_slope_rad > profile.max_tilt_rad:
        return MobilityResult(RoverBlock.TILT)
    if abs(slope_rad) > profile.max_climb_rad:
        return MobilityResult(RoverBlock.CLIMB)

    soil = soil_from_terrain_factor(eta, cfg)
    load = profile.mass_kg * G_MARS_M_S2 * math.cos(slope_rad) / profile.wheel_count
    sinkage = wheel_sinkage_m(load, profile.wheel_diameter_m, profile.wheel_width_m, soil)
    contact = profile.contact_length_factor * max(
        math.sqrt(profile.wheel_diameter_m * sinkage),
        cfg.contact_length_floor * profile.wheel_diameter_m)
    resistance = profile.wheel_count * compaction_resistance_n(sinkage, profile.wheel_width_m, soil)
    demand = profile.mass_kg * G_MARS_M_S2 * math.sin(slope_rad) + resistance

    slip = solve_slip(demand, load, contact, profile, soil, profile.traction_factor,
                      cfg.slip_critical, cfg.slip_tolerance)
    if slip is not None:
        return MobilityResult(RoverBlock.NONE, slip, DriveMode.ROLLING, wheel_speed_m_s,
                              wheel_speed_m_s * (1.0 - slip), sinkage, max(0.0, demand))
    if profile.wheel_walking:
        slip = solve_slip(demand, load, contact, profile, soil,
                          profile.traction_factor * cfg.wheel_walking_thrust_factor,
                          cfg.slip_critical_wheel_walking, cfg.slip_tolerance)
        if slip is not None:
            ground = profile.wheel_walk_speed_m_s
            return MobilityResult(RoverBlock.NONE, slip, DriveMode.WHEEL_WALKING,
                                  ground / (1.0 - slip), ground, sinkage, max(0.0, demand))
    return MobilityResult(RoverBlock.SLIP, sinkage_m=sinkage, thrust_n=max(0.0, demand))
