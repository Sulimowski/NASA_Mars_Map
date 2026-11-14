"""H11 and H0: reference tables that answer the questions of the project README.

    terrain_slope_table        speed, power and oxygen per km by ground type and slope
    continuous_walking_table   how long one can walk without a rest
    max_distance_table         how far one EVA reaches, and what limits it
    limiting_resource_table    which consumable runs out first at a given workload
    recommended_mission_params values for the planner's ``MissionParams``

All tables are computed from the same functions the simulator walks with
(A1-A4, A7), for a rested crew member in daylight with the suit cooling not
limiting. Rows are plain dictionaries in report units (km/h, degrees, W, g,
minutes, hours, km); ``to_markdown`` renders them.

Run ``python -m src.reporting.reference_tables`` to print every table.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..core.config import DEFAULT_CONFIG, SimConfig
from ..core.constants import HOUR_S, MINUTE_S, SOL_S, ms_to_kmh
from ..core.schemas import AstronautProfile, CoolingMode, SuitProfile
from ..physiology import adaptation, capacity, fatigue, life_support, metabolism

GROUND_TYPES: Mapping[str, float] = {"bedrock": 1.0, "firm regolith": 1.2,
                                     "boulder field": 1.5, "loose sand": 2.1}
"""Pandolf terrain factor of the ground types used in the tables."""

Row = dict[str, Any]


def _sustainable_w(astronaut: AstronautProfile, cfg: SimConfig, time_on_surface_s: float) -> float:
    fitness = adaptation.compute_fitness(time_on_surface_s, astronaut.transit_duration_s,
                                         cfg.adaptation)
    return capacity.sustainable_power_w(astronaut, fitness, cfg.capacity)


def walking_point(astronaut: AstronautProfile, suit: SuitProfile, cfg: SimConfig, eta: float,
                  grade: float, power_cap_w: float) -> tuple[float, float] | None:
    """Speed (m/s) and metabolic power (W) on a grade, or ``None`` when impassable."""
    coeffs = metabolism.pandolf_coefficients(astronaut.body_mass_kg, suit.mass_kg, eta, grade,
                                             suit.stiffness_factor, cfg.metabolism)
    speed = capacity.select_speed(*coeffs, grade, astronaut.preferred_speed_m_s, power_cap_w,
                                  cfg.capacity)
    if speed is None:
        return None
    return speed, metabolism.power_from_coefficients(*coeffs, speed)


def terrain_slope_table(astronaut: AstronautProfile, suit: SuitProfile,
                        cfg: SimConfig = DEFAULT_CONFIG, *, time_on_surface_s: float = 30 * SOL_S,
                        slopes_deg: Sequence[float] = (-20, -15, -10, -5, 0, 5, 10, 15, 20, 25),
                        ground: Mapping[str, float] = GROUND_TYPES) -> list[Row]:
    """Walking speed, power and oxygen use for every ground type and slope.

    A cell reads ``None`` where the slope is beyond the walking limits or even
    the slowest walk needs more than the sustainable power.
    """
    cap = _sustainable_w(astronaut, cfg, time_on_surface_s)
    up_limit, down_limit = cfg.routing.slope_up_limit_rad, cfg.routing.slope_down_limit_rad
    rows: list[Row] = []
    for name, eta in ground.items():
        for slope_deg in slopes_deg:
            slope = math.radians(slope_deg)
            row: Row = {"ground": name, "slope_deg": float(slope_deg), "speed_kmh": None,
                        "power_w": None, "time_min_per_km": None, "o2_g_per_km": None}
            point = None
            if -down_limit - 1e-9 <= slope <= up_limit + 1e-9:
                point = walking_point(astronaut, suit, cfg, eta, math.tan(slope), cap)
            if point is not None:
                speed, power = point
                seconds = 1000.0 / speed
                row.update(speed_kmh=ms_to_kmh(speed), power_w=power,
                           time_min_per_km=seconds / MINUTE_S,
                           o2_g_per_km=1000.0 * life_support.o2_consumption_kg(power, seconds, suit))
            rows.append(row)
    return rows


def continuous_walking_table(astronaut: AstronautProfile, suit: SuitProfile,
                             cfg: SimConfig = DEFAULT_CONFIG, *,
                             time_on_surface_s: float = 30 * SOL_S,
                             slopes_deg: Sequence[float] = (0, 5, 10, 15),
                             ground: Mapping[str, float] = GROUND_TYPES) -> list[Row]:
    """Time of non-stop walking until a rest is due and until only the way home is allowed."""
    cap = _sustainable_w(astronaut, cfg, time_on_surface_s)
    rows: list[Row] = []
    for name, eta in ground.items():
        for slope_deg in slopes_deg:
            point = walking_point(astronaut, suit, cfg, eta, math.tan(math.radians(slope_deg)), cap)
            if point is None:
                continue
            speed, power = point
            to_rest = fatigue.time_to_fatigue_s(0.0, cfg.fatigue.rest_trigger, power, cap,
                                                cfg.fatigue)
            to_return = fatigue.time_to_fatigue_s(0.0, cfg.fatigue.return_only, power, cap,
                                                  cfg.fatigue)
            rows.append({
                "ground": name, "slope_deg": float(slope_deg), "power_w": power,
                "share_of_sustainable": power / cap,
                "until_rest_h": to_rest / HOUR_S, "until_return_only_h": to_return / HOUR_S,
                "distance_until_rest_km": speed * to_rest / 1000.0,
            })
    return rows


def _usable(amount: float, cfg: SimConfig) -> float:
    return amount * (1.0 - cfg.routing.reserve_fraction)


def max_distance_table(astronaut: AstronautProfile, suit: SuitProfile,
                       cfg: SimConfig = DEFAULT_CONFIG, *, time_on_surface_s: float = 30 * SOL_S,
                       work_hours: Sequence[float] = (0.0, 1.0, 2.0, 3.0), work_power_w: float = 300.0,
                       eta: float = 1.2) -> list[Row]:
    """Farthest point of an out-and-back EVA on level ground, by time worked there.

    For oxygen and for time: what is left after the work and the reserve must
    cover the walk out plus ``return_margin`` times the walk back, so::

        radius = usable / (cost per metre * (1 + return_margin))
    """
    cap = _sustainable_w(astronaut, cfg, time_on_surface_s)
    point = walking_point(astronaut, suit, cfg, eta, 0.0, cap)
    if point is None:
        return []
    speed, power = point
    o2_per_m = life_support.o2_consumption_kg(power, 1.0 / speed, suit)
    margin = 1.0 + cfg.routing.return_margin
    rows: list[Row] = []
    for hours in work_hours:
        work_s = hours * HOUR_S
        o2_left = _usable(suit.o2_primary_kg, cfg) - life_support.o2_consumption_kg(
            work_power_w, work_s, suit)
        time_left = _usable(suit.max_eva_s, cfg) - work_s
        by_o2 = max(0.0, o2_left) / (o2_per_m * margin)
        by_time = max(0.0, time_left) * speed / margin
        radius = min(by_o2, by_time)
        rows.append({"work_h": float(hours), "radius_km": radius / 1000.0,
                     "round_trip_km": 2.0 * radius / 1000.0,
                     "limited_by": "oxygen" if by_o2 <= by_time else "EVA time",
                     "radius_by_o2_km": by_o2 / 1000.0, "radius_by_time_km": by_time / 1000.0})
    return rows


def limiting_resource_table(suit: SuitProfile, cfg: SimConfig = DEFAULT_CONFIG, *,
                            powers_w: Sequence[float] = (150, 250, 350, 450),
                            env_heat_w: float = -150.0, heater_w: float = 0.0) -> list[Row]:
    """Hours until each consumable reaches its reserve at a steady workload.

    Args:
        env_heat_w: heat entering the body from the environment (negative = loss);
            sets how much the cooling loop has to reject.
        heater_w: electric heater power.
    """
    rows: list[Row] = []
    for power in powers_w:
        cooling = max(0.0, power + env_heat_w)
        fan = max(cfg.life_support.fan_min_factor, power / cfg.life_support.fan_reference_metabolic_w)
        electric = suit.base_power_w + suit.fan_power_w * fan + heater_w
        hours: dict[str, float] = {
            "oxygen": _usable(suit.o2_primary_kg, cfg)
            / life_support.o2_consumption_kg(power, 1.0, suit) / HOUR_S,
            "co2_scrubber": _usable(suit.co2_capacity_kg, cfg)
            / life_support.co2_production_kg(power, 1.0, cfg.life_support) / HOUR_S,
            "battery": _usable(suit.battery_j, cfg) / electric / HOUR_S,
            "eva_time": _usable(suit.max_eva_s, cfg) / HOUR_S,
        }
        if suit.cooling_mode is CoolingMode.MEMBRANE_EVAPORATOR and cooling > 0.0:
            hours["cooling_water"] = (_usable(suit.cooling_water_kg, cfg)
                                      / life_support.cooling_water_kg(cooling, 1.0, suit,
                                                                      cfg.life_support) / HOUR_S)
        first = min(hours, key=lambda key: hours[key])
        rows.append({"power_w": float(power), **{f"{k}_h": v for k, v in hours.items()},
                     "first_to_run_out": first})
    return rows


def recommended_mission_params(astronaut: AstronautProfile, suit: SuitProfile,
                               cfg: SimConfig = DEFAULT_CONFIG, *,
                               time_on_surface_s: float = 30 * SOL_S) -> Row:
    """Values for the planner's ``MissionParams`` derived from the model.

    Keys match the fields of ``MissionParams`` in the routing team's code, in
    its units (degrees, km/h, hours, mSv, km).
    """
    cap = _sustainable_w(astronaut, cfg, time_on_surface_s)
    point = walking_point(astronaut, suit, cfg, 1.0, 0.0, cap)
    speed = point[0] if point is not None else 0.0
    reach = max_distance_table(astronaut, suit, cfg, time_on_surface_s=time_on_surface_s,
                               work_hours=(1.0,), eta=1.2)
    return {
        "max_slope_deg": 15.0,
        "hard_max_slope_deg": math.degrees(cfg.routing.slope_up_limit_rad),
        "walk_speed_flat_kmh": round(ms_to_kmh(speed), 2),
        "eva_max_hours": suit.max_eva_s / HOUR_S,
        "return_reserve_frac": cfg.routing.reserve_fraction,
        "max_dose_msv": cfg.radiation.per_eva_limit_sv * 1000.0,
        "max_distance_km": round(reach[0]["round_trip_km"], 1) if reach else 0.0,
    }


def to_markdown(rows: Sequence[Row], digits: int = 2) -> str:
    """Render rows as a Markdown table (``None`` shows as a dash)."""
    if not rows:
        return ""
    columns = list(rows[0])

    def cell(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return "∞" if math.isinf(value) else f"{value:.{digits}f}"
        return str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines.extend("| " + " | ".join(cell(row.get(col)) for col in columns) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    """Print every reference table for the default crew member and suit."""
    astronaut, suit = AstronautProfile(), SuitProfile()
    sections = (
        ("Speed, power and oxygen by ground and slope", terrain_slope_table(astronaut, suit)),
        ("Continuous walking", continuous_walking_table(astronaut, suit)),
        ("Reach of one EVA on level ground", max_distance_table(astronaut, suit)),
        ("Hours until each consumable reaches its reserve", limiting_resource_table(suit)),
        ("Recommended MissionParams", [recommended_mission_params(astronaut, suit)]),
    )
    for title, rows in sections:
        print(f"\n## {title}\n\n{to_markdown(rows)}")


if __name__ == "__main__":
    main()
