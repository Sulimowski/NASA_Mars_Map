"""How fast a suited astronaut walks and what it costs: time, oxygen, battery.

The numbers come from the astronaut model in ``mars_eva_sim`` (algorithms A2-A4:
Pandolf load carriage rescaled to Mars, speed limited by the power the crew
member can hold, oxygen and battery from the metabolic power). This module is
the adapter the router calls: it keeps the router's units (litres of O2,
watt-hours, degrees Celsius) and array interface, and gets every value from
that one model, so the planner, the report and the simulator cannot disagree.

What is taken from where:
* ground (terrain coefficient, speed factor, sinkage)  -- ``astronaut_terrain``
* radiation and ground temperature                    -- ``environment``
* speed, metabolic power, oxygen, battery, heaters    -- ``mars_eva_sim``

One vectorised implementation, used for every graph edge during the search and
again for the final route.

A step can be physically impossible: on a steep climb over loose ground even
the slowest walk needs more power than a person can sustain. ``locomotion``
returns infinite time for such a step and the router treats it as blocked.
"""

from __future__ import annotations

import sys
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mars_eva_sim.src.core.config import DEFAULT_CONFIG, SimConfig
from mars_eva_sim.src.core.constants import HOUR_S, O2_DENSITY_STP_KG_M3, SOL_S
from mars_eva_sim.src.core.schemas import AstronautProfile, SuitProfile, TaskType
from mars_eva_sim.src.physiology import vectorized

if __package__ in (None, ""):
    from mars_map.site import MissionParams
else:
    from .site import MissionParams

MARS_GRAVITY = 3.72
EARTH_GRAVITY = 9.81

_ASTRONAUT = AstronautProfile()
_SUIT = SuitProfile()


@dataclass
class SuitParams:
    """Astronaut and suit as the router sees them, in the router's units.

    Defaults are the profiles of the astronaut model. ``astronaut()``, ``suit()``
    and ``config`` turn the values back into that model's inputs, so a changed
    field here changes the physics everywhere.
    """
    body_mass_kg: float = _ASTRONAUT.body_mass_kg
    suit_mass_kg: float = _SUIT.mass_kg
    # Boot sole pressing on the ground during single support (Bekker-Wong sinkage).
    boot_area_m2: float = 0.035
    boot_width_m: float = 0.11

    # Oxygen
    o2_tank_l: float = float(vectorized.o2_kg_to_litres(_SUIT.o2_primary_kg))   # litres, standard conditions
    o2_reserve_frac: float = DEFAULT_CONFIG.routing.reserve_fraction   # must still be in the tank at the airlock

    # Battery
    battery_wh: float = _SUIT.battery_j / HOUR_S
    battery_reserve_frac: float = DEFAULT_CONFIG.routing.reserve_fraction

    # Crew condition and work
    sols_on_surface: float = 30.0         # time since landing; sets the fitness of the crew member
    stop_task: str = TaskType.SAMPLING.value   # kind of work at a target stop

    @property
    def total_mass_kg(self) -> float:
        return self.body_mass_kg + self.suit_mass_kg

    @property
    def config(self) -> SimConfig:
        """Parameters of the astronaut model."""
        return DEFAULT_CONFIG

    def astronaut(self) -> AstronautProfile:
        """Crew member profile of the astronaut model."""
        return AstronautProfile(body_mass_kg=self.body_mass_kg)

    def suit(self) -> SuitProfile:
        """Suit profile of the astronaut model."""
        return SuitProfile(mass_kg=self.suit_mass_kg,
                           o2_primary_kg=self.o2_tank_l * 1.0e-3 * O2_DENSITY_STP_KG_M3,
                           battery_j=self.battery_wh * HOUR_S)

    def power_cap_w(self) -> float:
        """Metabolic power the crew member may hold while walking, W."""
        return vectorized.planning_power_cap_w(self.astronaut(), self.config,
                                               self.sols_on_surface * SOL_S)


_MODEL_CACHE: dict[tuple, tuple[AstronautProfile, SuitProfile, float]] = {}


def _model(suit: SuitParams) -> tuple[AstronautProfile, SuitProfile, float]:
    """(astronaut, suit, power cap) of the astronaut model for these parameters, cached:
    the router asks for them on every one of its many small calls."""
    key = astuple(suit)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = (suit.astronaut(), suit.suit(), suit.power_cap_w())
    return _MODEL_CACHE[key]


def electric_power(metabolic_w, heater_w, suit: SuitParams = SuitParams()):
    """Battery drain in watts: avionics, fan (scaled by the workload) and heaters."""
    return vectorized.electric_power_w(metabolic_w, heater_w, _model(suit)[1], suit.config)


def heater_power(ground_temp_c, suit: SuitParams = SuitParams()):
    """Glove and boot heater power in watts for a ground temperature."""
    return vectorized.extremity_heater_w(vectorized.celsius_to_kelvin(ground_temp_c),
                                         _model(suit)[1], suit.config)


def locomotion(dist_m, rise_m, eta, speed_mult, mission: MissionParams,
               suit: SuitParams = SuitParams(), heater_w=0.0):
    """One step (or one array of steps) on foot.

    dist_m -- horizontal length; rise_m -- height change (negative downhill);
    eta, speed_mult -- terrain coefficients of the cell being stepped onto.
    Returns (time_s, metabolic_w, o2_litres, energy_wh).

    The speed is the astronaut's preferred pace times ``speed_mult``, lowered
    until the metabolic power fits what the crew member can hold. A step that
    does not fit even at the slowest pace comes back with infinite time, oxygen
    and energy. ``mission`` is kept for the call signature; the walking speed is
    no longer a mission constant.
    """
    astronaut, suit_profile, cap = _model(suit)
    steps = vectorized.walking_steps(dist_m, rise_m, eta, speed_mult, heater_w,
                                     astronaut=astronaut, suit=suit_profile,
                                     cfg=suit.config, power_cap_w=cap)
    return (steps.time_s, steps.metabolic_w, vectorized.o2_kg_to_litres(steps.o2_kg),
            steps.battery_j / HOUR_S)


def dwell(minutes: float, heater_w: float = 0.0, suit: SuitParams = SuitParams()
          ) -> tuple[float, float, float]:
    """Working at a stop: (time_s, o2_litres, energy_wh)."""
    time_s = minutes * 60.0
    _, o2_kg, battery_j = vectorized.stationary_work(time_s, TaskType(suit.stop_task), heater_w,
                                                     suit=_model(suit)[1], cfg=suit.config)
    return time_s, float(vectorized.o2_kg_to_litres(o2_kg)), battery_j / HOUR_S
