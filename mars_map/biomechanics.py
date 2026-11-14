"""How fast a suited astronaut walks and what it costs: time, oxygen, battery.

One vectorised implementation, used for every graph edge during the search and
again for the final route, so the planner and the report can never disagree.

Models:
* speed       -- Tobler's hiking function (direction matters: a step up is slower
                 than the same step down), scaled so flat firm ground gives
                 MissionParams.walk_speed_flat_kmh, times the terrain speed factor.
* metabolism  -- Pandolf load-carriage equation with the terrain coefficient from
                 astronaut_terrain, weight terms scaled to Mars gravity, and the
                 Santee downhill correction.
* oxygen      -- metabolic power / energy per litre of O2.
* battery     -- avionics + cooling loop that follows the metabolic heat + heater
                 that follows the ground temperature.

Every number in SuitParams is a working assumption, like MissionParams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .site import MissionParams

MARS_GRAVITY = 3.72
EARTH_GRAVITY = 9.81


@dataclass
class SuitParams:
    """Astronaut and life-support assumptions. Not NASA requirements."""
    body_mass_kg: float = 75.0
    suit_mass_kg: float = 70.0
    # Boot sole pressing on the ground during single support (Bekker-Wong sinkage).
    boot_area_m2: float = 0.035
    boot_width_m: float = 0.11

    # Walking
    min_speed_ms: float = 0.15            # never slower than this, however bad the ground
    suit_joint_overhead_w: float = 60.0   # extra work against the pressurised suit joints
    min_metabolic_w: float = 150.0
    work_metabolic_w: float = 220.0       # sampling / working at a stop
    # Braking on a descent costs at most this share of lifting on the same grade.
    downhill_cost_cap: float = 0.3

    # Oxygen
    o2_tank_l: float = 480.0              # litres at standard conditions
    o2_joule_per_l: float = 20_100.0      # energy released per litre of O2 consumed
    o2_reserve_frac: float = 0.20         # must still be in the tank at the airlock

    # Battery
    battery_wh: float = 1200.0
    battery_reserve_frac: float = 0.20
    base_electric_w: float = 70.0         # avionics, comms, fan, pump
    cooling_w_per_metabolic_w: float = 0.12
    heater_w_per_k: float = 0.6           # per kelvin of ground temperature below ...
    heater_on_below_c: float = -20.0      # ... this

    @property
    def total_mass_kg(self) -> float:
        return self.body_mass_kg + self.suit_mass_kg


# Tobler: speed ~ exp(-3.5 * |grade + 0.05|), fastest on a gentle 5 % descent.
_TOBLER_K = 3.5
_TOBLER_FLAT = float(np.exp(-_TOBLER_K * 0.05))


def tobler_factor(grade):
    """Walking speed relative to flat ground for a step with slope = rise / run."""
    return np.exp(-_TOBLER_K * np.abs(grade + 0.05)) / _TOBLER_FLAT


def metabolic_power(speed_ms, grade, eta, suit: SuitParams = SuitParams()):
    """Metabolic power in watts while walking.

    speed_ms -- walking speed; grade -- rise / run (negative downhill);
    eta      -- Pandolf terrain coefficient (1 = firm ground, ~2.1 = loose sand).
    """
    w, load = suit.body_mass_kg, suit.suit_mass_kg
    total = w + load
    g = MARS_GRAVITY / EARTH_GRAVITY
    pct = grade * 100.0
    up, down = np.maximum(pct, 0.0), np.maximum(-pct, 0.0)

    standing = 1.5 * w + 2.0 * total * (load / w) ** 2 * g
    level = eta * total * 1.5 * speed_ms ** 2 * g
    lifting = eta * total * 0.35 * speed_ms * up * g
    # Santee et al. 2001, written so that it is zero on level ground. A positive
    # value is energy saved by the descent, a negative one is braking effort.
    santee = eta * g * (down * total * speed_ms / 3.5 - (total / w) * (down ** 2 + 12.0 * down))
    braking_cap = suit.downhill_cost_cap * eta * total * 0.35 * speed_ms * down * g
    descent = np.maximum(santee, -braking_cap)

    power = standing + level + lifting - descent + suit.suit_joint_overhead_w
    return np.maximum(power, suit.min_metabolic_w)


def electric_power(metabolic_w, heater_w, suit: SuitParams = SuitParams()):
    """Battery drain in watts."""
    return suit.base_electric_w + suit.cooling_w_per_metabolic_w * metabolic_w + heater_w


def heater_power(ground_temp_c, suit: SuitParams = SuitParams()):
    """Suit heater power in watts for a ground temperature."""
    return suit.heater_w_per_k * np.maximum(suit.heater_on_below_c - ground_temp_c, 0.0)


def locomotion(dist_m, rise_m, eta, speed_mult, mission: MissionParams,
               suit: SuitParams = SuitParams(), heater_w=0.0):
    """One step (or one array of steps) on foot.

    dist_m -- horizontal length; rise_m -- height change (negative downhill);
    eta, speed_mult -- terrain coefficients of the cell being stepped onto.
    Returns (time_s, metabolic_w, o2_litres, energy_wh).
    """
    grade = rise_m / dist_m
    speed = np.maximum(mission.walk_speed_flat_kmh / 3.6 * tobler_factor(grade) * speed_mult,
                       suit.min_speed_ms)
    time_s = dist_m / speed
    met = metabolic_power(speed, grade, eta, suit)
    o2 = met * time_s / suit.o2_joule_per_l
    wh = electric_power(met, heater_w, suit) * time_s / 3600.0
    return time_s, met, o2, wh


def dwell(minutes: float, heater_w: float = 0.0, suit: SuitParams = SuitParams()
          ) -> tuple[float, float, float]:
    """Working at a stop: (time_s, o2_litres, energy_wh)."""
    time_s = minutes * 60.0
    o2 = suit.work_metabolic_w * time_s / suit.o2_joule_per_l
    wh = float(electric_power(suit.work_metabolic_w, heater_w, suit)) * time_s / 3600.0
    return time_s, o2, wh
