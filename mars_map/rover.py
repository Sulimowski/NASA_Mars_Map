"""A wheeled rover as a second kind of traveller for the route planner.

    from mars_map import load_stub
    from mars_map.rover import RoverParams, plan_rover

    plans = plan_rover(load_stub())                      # {"eco": ..., "science": ...}
    plans["eco"].energy                                  # where the battery energy went

The rover is planned by the same router as the astronaut (astronaut_router.EvaRouter):
this module only answers the router's questions for wheels instead of legs.

What is modelled:
* ground   -- Bekker wheel sinkage and compaction resistance from the same soil
              moduli the boot model uses; rocks add rolling resistance and, above a
              cover the wheels cannot cross, block the cell.
* driving  -- force along the slope = weight * (rolling resistance + sine of the slope).
              Uphill the speed is limited by motor power, traction by the slope the
              wheels can hold on that soil.
* battery  -- drive energy through the drivetrain efficiency, a constant "hotel" load,
              and two ways to charge:
              - solar panels, from the direct-sunlight map (nothing in shadow);
              - regenerative braking: on a descent steeper than the rolling
                resistance the slope pushes the rover, and the motors, run as
                generators, return part of that energy.
              A full battery takes no more charge; the router tracks that along the
              route, so charge offered to a full battery is counted as lost.

Regeneration is real but modest: rolling resistance eats the first 4 to 10 degrees of
any descent, and only `regen_efficiency` of the rest comes back. Every number in
RoverParams is a working assumption; the chassis is sized after the Apollo lunar
rover, the only crewed rover that has driven on another world.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import mars_map  # noqa: F401  (run directly: make relative imports work)
    __package__ = "mars_map"

from .astronaut_router import EvaRouter, MissionPlan, Strategy
from .astronaut_terrain import (K_C, SINKAGE_REF_CM, AstronautTerrainModel, rasterize_nogo,
                                soil_moduli, uniform_dose_rate_msv_h)
from .biomechanics import MARS_GRAVITY
from .site import Site

ROVER_STRATEGIES: dict[str, Strategy] = {
    # Lowest battery use: descents are worth taking, loose and rocky ground is not.
    "eco": Strategy("eco", "Minimum battery use", time_weight=0.1, o2_weight=1.0,
                    looseness_weight=0.5, steep_weight=0.3, budget_use=0.85,
                    saving="energy", saving_weight=40.0),
    # Fastest routes, as many targets as the limits allow.
    "science": Strategy("science", "Maximum science", science_weight=5.0),
}


@dataclass
class RoverParams:
    """Rover assumptions and, through its methods, the mover the router plans with."""
    # Chassis
    mass_kg: float = 210.0
    payload_kg: float = 250.0           # crew member or cargo, tools, samples
    n_wheels: int = 4
    wheel_diameter_m: float = 0.82
    wheel_width_m: float = 0.23

    # Driving
    max_speed_kmh: float = 8.0
    min_speed_ms: float = 0.05          # slower than this counts as stuck
    motor_power_w: float = 750.0        # mechanical power at the wheels, all motors
    drive_efficiency: float = 0.75      # battery -> wheels
    internal_rolling: float = 0.02      # bearings and wheel flexing, as a share of weight
    rock_rolling: float = 0.10          # added rolling resistance at full rock cover
    traction_coeff: float = 0.60        # gross traction / weight
    max_slope_deg: float = 15.0         # comfortable; steeper costs extra in the search
    hard_max_slope_deg: float = 20.0    # never driven
    max_rock_abundance: float = 0.30    # rock cover the wheels cannot cross
    descent_speed_frac: float = 0.5     # share of top speed kept on the steepest descent

    # Regenerative braking
    regen_enabled: bool = True
    regen_efficiency: float = 0.55      # wheels -> battery
    regen_max_w: float = 600.0          # charging power the battery accepts

    # Battery
    battery_wh: float = 4000.0
    battery_reserve_frac: float = 0.20  # never go below this
    start_charge_frac: float = 1.0

    # Solar panels
    solar_enabled: bool = True
    panel_area_m2: float = 2.5
    panel_efficiency: float = 0.27
    panel_dust_factor: float = 0.80     # share of the output left under settled dust

    # Other loads
    hotel_w: float = 120.0              # avionics, navigation, radio, heaters: always on
    instrument_w: float = 60.0          # science instruments while parked at a target

    # One sortie
    max_drive_hours: float = 6.0
    max_distance_km: float = 20.0

    # ---- what the router reads -------------------------------------------------
    kind = "rover"
    uses_oxygen = False
    o2_tank_l = None
    strategies = ROVER_STRATEGIES

    @property
    def total_mass_kg(self) -> float:
        return self.mass_kg + self.payload_kg

    @property
    def weight_n(self) -> float:
        return self.total_mass_kg * MARS_GRAVITY

    @property
    def soft_slope_deg(self) -> float:
        return self.max_slope_deg

    @property
    def hard_slope_deg(self) -> float:
        return self.hard_max_slope_deg

    @property
    def energy_floor_wh(self) -> float:
        """Energy drawn since the start cannot fall below this: the room left in the battery."""
        return -self.battery_wh * (1.0 - self.start_charge_frac)

    @property
    def _effort_ref_w(self) -> float:
        """Battery power of cruising on firm level ground: turns effort joules into seconds."""
        cruise = self.weight_n * 0.08 * self.max_speed_kmh / 3.6 / self.drive_efficiency
        return self.hotel_w + cruise

    def solar_power_w(self, sun_w_m2):
        """Electric power of the panels for sunlight falling on them, W."""
        if not self.solar_enabled:
            return np.zeros_like(np.asarray(sun_w_m2, dtype=np.float32))
        return (np.asarray(sun_w_m2, dtype=np.float32)
                * np.float32(self.panel_area_m2 * self.panel_efficiency * self.panel_dust_factor))

    def step(self, dist_m, rise_m, rolling, speed_mult, aux_w=0.0):
        """One step (or an array of steps) on wheels.

        dist_m -- horizontal length; rise_m -- height change (negative downhill);
        rolling, speed_mult -- rolling-resistance coefficient and speed factor of the
        cell driven onto; aux_w -- minus the solar power available there.
        Returns (time_s, oxygen = 0, net battery Wh, effort, parts). A step the rover
        cannot make (no traction, or no power to move at all) has infinite time.
        """
        d = np.asarray(dist_m, dtype=np.float64)
        dh = np.asarray(rise_m, dtype=np.float64)
        rolling = np.asarray(rolling, dtype=np.float64)
        aux = np.asarray(aux_w, dtype=np.float64)
        length = np.hypot(d, dh)
        sin, cos = dh / length, d / length
        force = self.weight_n * (rolling * cos + sin)      # along the slope; negative = pushed

        speed = self.max_speed_kmh / 3.6 * np.asarray(speed_mult, dtype=np.float64)
        steepness = np.clip(-sin / math.sin(math.radians(self.hard_max_slope_deg)), 0.0, 1.0)
        speed = speed * (1.0 - (1.0 - self.descent_speed_frac) * steepness)   # careful downhill
        speed = np.minimum(speed, self.motor_power_w / np.maximum(force, 1e-9))  # motor limit
        stuck = (speed < self.min_speed_ms) | (sin > (self.traction_coeff - rolling) * cos)
        time_s = length / np.maximum(speed, 1e-9)

        drive_j = np.maximum(force, 0.0) * length / self.drive_efficiency
        regen_j = np.zeros_like(drive_j)
        if self.regen_enabled:
            regen_j = np.minimum(np.maximum(-force, 0.0) * length * self.regen_efficiency,
                                 self.regen_max_w * time_s)
        solar_j = np.maximum(-aux, 0.0) * time_s
        hotel_j = (self.hotel_w + np.maximum(aux, 0.0)) * time_s
        net_wh = (drive_j + hotel_j - regen_j - solar_j) / 3600.0

        # Effort for the search. Net energy can be negative on a descent, which a
        # shortest-path search cannot use. Adding k * weight * drop to every step (k =
        # regen efficiency) shifts every route between two points by the same amount
        # and makes each step non-negative, so the cheapest route is unchanged.
        k = self.regen_efficiency if self.regen_enabled else 0.0
        effort_j = np.maximum(drive_j - regen_j - k * self.weight_n * dh + hotel_j - solar_j, 0.0)

        parts = {"drive_wh": drive_j / 3600.0, "hotel_wh": hotel_j / 3600.0,
                 "regen_wh": regen_j / 3600.0, "solar_wh": solar_j / 3600.0}
        if stuck.any():
            net_wh = np.where(stuck, 0.0, net_wh)
            effort_j = np.where(stuck, 0.0, effort_j)
            parts = {name: np.where(stuck, 0.0, v) for name, v in parts.items()}
        return (np.where(stuck, np.inf, time_s), np.zeros_like(net_wh), net_wh,
                effort_j / self._effort_ref_w, parts)

    def dwell(self, minutes: float, aux_w: float = 0.0):
        """Parked at a target with the instruments on: (time_s, oxygen = 0, net Wh, parts)."""
        time_s = minutes * 60.0
        solar_wh = max(-aux_w, 0.0) * time_s / 3600.0
        load_wh = (self.hotel_w + self.instrument_w + max(aux_w, 0.0)) * time_s / 3600.0
        return time_s, 0.0, load_wh - solar_wh, {"solar_wh": solar_wh, "hotel_wh": load_wh}

    def budgets(self, use: float = 1.0) -> list[float]:
        """Limits in the router's order: time, distance, oxygen, battery, dose."""
        return [self.max_drive_hours * 3600.0 * use,
                self.max_distance_km * 1000.0 * use,
                math.inf,
                self.battery_wh * (self.start_charge_frac - self.battery_reserve_frac) * use,
                math.inf]

    def limits(self) -> dict:
        return {"max_drive_hours": self.max_drive_hours, "max_distance_km": self.max_distance_km,
                "battery_wh": self.battery_wh, "battery_reserve_frac": self.battery_reserve_frac,
                "start_charge_frac": self.start_charge_frac,
                "max_slope_deg": self.max_slope_deg, "hard_max_slope_deg": self.hard_max_slope_deg,
                "max_rock_abundance": self.max_rock_abundance,
                "regen_enabled": self.regen_enabled, "solar_enabled": self.solar_enabled}

    def build_terrain(self, site: Site, **weather) -> AstronautTerrainModel:
        return build_rover_terrain(site, self, **weather)

    def to_dict(self) -> dict:
        return asdict(self)


def wheel_sinkage_m(thermal_inertia: np.ndarray, sand: np.ndarray, rover: RoverParams
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Bekker sinkage of one rigid wheel (m) and its compaction resistance as a share of
    the wheel load: z = (3 W / ((3 - n)(k_c + b k_phi) sqrt(D))) ^ (2 / (2 n + 1)),
    R = b (k_c / b + k_phi) z^(n + 1) / (n + 1)."""
    k_phi, n = soil_moduli(thermal_inertia, sand)
    load_kn = rover.weight_n / rover.n_wheels / 1000.0
    b, dia = rover.wheel_width_m, rover.wheel_diameter_m
    z = (3.0 * load_kn / ((3.0 - n) * (K_C + b * k_phi) * math.sqrt(dia))) ** (2.0 / (2.0 * n + 1.0))
    resistance_kn = b * (K_C / b + k_phi) * z ** (n + 1.0) / (n + 1.0)
    return z, resistance_kn / load_kn


def build_rover_terrain(site: Site, rover: RoverParams | None = None,
                        nodata_risk_weight: float = 0.25,
                        dose_rate_msv_h: np.ndarray | None = None,
                        ground_temp_c: np.ndarray | None = None,
                        sun_w_m2: np.ndarray | None = None) -> AstronautTerrainModel:
    """Ground model of a site for a rover, in the form the router reads.

    The fields keep their names but hold wheel values:
    terramech_factor -- rolling-resistance coefficient (force / weight);
    speed_multiplier -- share of top speed the ground allows;
    heater_w         -- minus the solar power of the panels in that cell, W.
    sun_w_m2 is the direct-sunlight map (Environment.direct_sunlight); without it the
    panels give nothing.
    """
    rover = rover or RoverParams()
    L = site.layers
    ti = L["thermal_inertia"].astype(np.float32)
    rock = L["rock_abundance"].astype(np.float32)
    rough = L["roughness"].astype(np.float32)
    slope = L["slope"].astype(np.float32)
    tclass = L["terrain_class"]

    z, soil_rolling = wheel_sinkage_m(ti, tclass == 1, rover)
    sinkage_cm = (z * 100.0).astype(np.float32)
    sinkage = np.clip(sinkage_cm / SINKAGE_REF_CM, 0.0, 1.0)

    rock_impedance = np.clip(0.65 * rock + 0.35 * np.clip(rough / 0.45, 0.0, 1.0), 0.0, 1.0)
    looseness = np.clip(0.58 * sinkage + 0.42 * rock_impedance, 0.0, 1.0)
    rolling = rover.internal_rolling + soil_rolling + rover.rock_rolling * rock_impedance
    speed_multiplier = np.clip(1.0 - 0.3 * sinkage - 0.6 * rock_impedance, 0.25, 1.0)

    no_go = rasterize_nogo(site.shape, site.resolution_m, site.no_go)
    impassable = ((tclass == 3) | (slope > rover.hard_max_slope_deg) | no_go
                  | (rock > rover.max_rock_abundance))
    if "blocked" in L:
        impassable |= L["blocked"].astype(bool)

    if dose_rate_msv_h is None:
        dose = np.full(site.shape, uniform_dose_rate_msv_h(site), np.float32)
    else:
        dose = np.asarray(dose_rate_msv_h, dtype=np.float32)
    solar = (rover.solar_power_w(sun_w_m2) if sun_w_m2 is not None
             else np.zeros(site.shape, np.float32))
    if solar.shape != site.shape or dose.shape != site.shape:
        raise ValueError("weather maps do not match the site shape")

    return AstronautTerrainModel(
        sinkage=sinkage.astype(np.float32),
        rock_impedance=rock_impedance.astype(np.float32),
        looseness=looseness.astype(np.float32),
        terramech_factor=rolling.astype(np.float32),
        speed_multiplier=speed_multiplier.astype(np.float32),
        impassable=impassable,
        nodata_penalty=np.where(L["nodata"].astype(bool), 1.0 + nodata_risk_weight,
                                1.0).astype(np.float32),
        sinkage_cm=sinkage_cm,
        no_go=no_go,
        dose_rate_msv_h=dose,
        ground_temp_c=None if ground_temp_c is None else np.asarray(ground_temp_c, np.float32),
        heater_w=-solar,
        radiation_is_map=dose_rate_msv_h is not None,
    )


def plan_rover(site: Site, rover: RoverParams | None = None,
               sun_w_m2: np.ndarray | None = None, router: EvaRouter | None = None,
               check_radio: bool = True) -> dict[str, MissionPlan]:
    """Rover plans of a site: start -> chosen targets -> start, one per rover strategy.

    For plans with the radiation, temperature and sunlight maps of a sol use
    pipeline.plan_site(site, vehicle="rover").
    """
    rover = rover or RoverParams()
    router = router or EvaRouter(site, build_rover_terrain(site, rover, sun_w_m2=sun_w_m2), rover)
    return router.plan_pareto(check_radio=check_radio)


def regeneration_gain(router: EvaRouter, plan: MissionPlan) -> dict:
    """What regenerative braking is worth on a finished plan: the same route driven again
    with regeneration switched off."""
    rover = router.suit
    plain = EvaRouter(router.site, router.terrain, replace(rover, regen_enabled=False),
                      router.radio_params)
    legs = [plain._segment([router.site.to_pixel(x, y) for x, y in seg.vertices_m])
            for seg in plan.segments]
    without = sum(seg.energy_wh for seg in legs)
    with_regen = sum(seg.energy_wh for seg in plan.segments)
    return {"driving_net_wh_with_regen": round(with_regen, 1),
            "driving_net_wh_without_regen": round(without, 1),
            "saved_wh": round(without - with_regen, 1),
            "saved_pct": round(100.0 * (without - with_regen) / without, 1) if without > 0 else 0.0}
