"""Records shared by the rover physics and the route search. All fields are SI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..core.schemas import Cell


class PowerSource(str, Enum):
    """Where the rover's electricity comes from."""

    MMRTG = "mmrtg"      # radioisotope generator: steady output day and night
    SOLAR = "solar"      # solar arrays: output follows the Sun


class DriveMode(str, Enum):
    """How the wheels move the rover."""

    ROLLING = "rolling"
    WHEEL_WALKING = "wheel_walking"   # wheels step forward one after another on movable legs


class RoverBlock(str, Enum):
    """Why a rover cannot take a step."""

    NONE = "none"
    OUT_OF_BOUNDS = "out_of_bounds"
    FORBIDDEN = "forbidden"
    IMPASSABLE = "impassable"
    CLEARANCE = "clearance"     # an obstacle is taller than the rover can pass over
    TILT = "tilt"               # the ground is steeper than the tilt limit
    CLIMB = "climb"             # the step is steeper than the climb limit
    SLIP = "slip"               # the wheels would slip past the embedding limit
    NIGHT = "night"             # no driving without daylight
    SPE_HOLD = "spe_hold"       # driving suspended during a solar particle event


class RoverRule(str, Enum):
    """Hard constraints on a rover's state during a traverse."""

    BATTERY_RESERVE = "battery_reserve"
    THERMAL_WINDOW = "thermal_window"
    TIME_BUDGET = "time_budget"
    RISK_BUDGET = "risk_budget"


@dataclass(slots=True)
class RoverTransition:
    """One drive step or one parked interval.

    Attributes:
        dt_s: duration.
        distance_m: ground distance covered (0 when parked).
        speed_m_s: ground speed, after slip.
        slope_rad: slope along the direction of travel.
        slip: wheel slip, 0..1.
        mode: rolling or wheel-walking.
        drive_w, heater_w, load_w: electric power of the drive, the heaters, and in total.
        generation_w: electric power generated.
        d_battery_j: change of the stored energy (negative when discharging).
        heater_deficit_w: heater power demanded beyond what the heaters can give.
        d_dose_gy: absorbed dose in the electronics.
        risk: probability of a radiation upset during the step.
        battery_low_j: lowest stored energy reached during the step.
    """

    dt_s: float
    distance_m: float
    speed_m_s: float
    slope_rad: float
    slip: float
    mode: DriveMode
    drive_w: float
    heater_w: float
    load_w: float
    generation_w: float
    d_battery_j: float
    heater_deficit_w: float
    d_dose_gy: float
    risk: float
    battery_low_j: float


@dataclass(slots=True, eq=False)
class RoverLabel:
    """Search state of a rover: position, time, stored energy and accumulated risk.

    ``eq=False`` gives identity comparison, which the search relies on.
    """

    row: int
    col: int
    t_s: float
    battery_j: float
    min_battery_j: float = float("inf")   # lowest stored energy since the start
    distance_m: float = 0.0
    cost: float = 0.0
    risk: float = 0.0
    dose_gy: float = 0.0
    energy_used_j: float = 0.0
    wait_s: float = 0.0
    wheel_walk_m: float = 0.0
    max_slip: float = 0.0
    heater_deficit_w: float = 0.0
    action: str = "start"
    transition: RoverTransition | None = None
    parent: RoverLabel | None = None
    alive: bool = True

    def __repr__(self) -> str:
        return (f"RoverLabel(cell=({self.row}, {self.col}), t_s={self.t_s:.0f}, "
                f"dist_m={self.distance_m:.0f}, battery_j={self.battery_j:.0f}, "
                f"cost={self.cost:.1f}, action={self.action!r})")

    @property
    def cell(self) -> Cell:
        """Grid cell of the label."""
        return (self.row, self.col)

    def chain(self) -> list[RoverLabel]:
        """Labels from the start of the traverse to this one, in travel order."""
        out: list[RoverLabel] = []
        node: RoverLabel | None = self
        while node is not None:
            out.append(node)
            node = node.parent
        out.reverse()
        return out
