"""B1: physics of one step. Turns a move between two cells, or a stationary task,
into a ``Transition`` by chaining the physiology algorithms.

Order of evaluation for a walking edge ``a -> b``::

    1. terrain gates    passable, not forbidden, cell slope and directional slope limits
    2. environment      sample at b and time t (cached per cell and time bucket)
    3. A1 / A3 / A5     power cap  P_cap = min(P_base, P_thermal) * modifiers
    4. A2 / A3          speed: highest V <= V_upper with M(V) <= P_cap
    5. A2               metabolic power M at that speed, step time dt = length / V
    6. A5               cooling, heaters, stored heat
    7. A4               oxygen, CO2, cooling water, battery
    8. A6, A7, A8, A12  dose, fatigue, hydration, dust
    9. A9               fall risk

``V_upper = V_pref * rock_factor * visibility`` (never below the minimum speed).
The edge is impassable (``None``) when a gate fails or no speed fits the cap.
"""

from __future__ import annotations

import math
from typing import Protocol

from ..core.config import DEFAULT_CONFIG, RoutingConfig, SimConfig
from ..core.exceptions import InfeasibleStateError, TerrainValidationError
from ..core.schemas import (AstronautProfile, Cell, EnvSample, Label, Pace, PhysState,
                            SuitProfile, SuitResources, TaskType, Transition)
from ..physiology import (adaptation, capacity, dust, fatigue, injury, life_support, metabolism,
                          nutrition, radiation, thermal)
from ..providers.interfaces import NEIGHBOR_OFFSETS, EnvironmentProvider, TerrainProvider


class WalkerState(Protocol):
    """The part of the state the edge physics reads. ``Label`` satisfies it."""

    t_s: float
    fatigue: float
    hand_fatigue: float
    hydration_deficit_kg: float
    drink_water_kg: float
    heat_storage_j: float
    eva_remaining_s: float


def visibility_factor(sun_elevation_rad: float, cfg: RoutingConfig) -> float:
    """How well the crew member sees the ground, 0..1 (1 = daylight).

    ``night_factor`` when the Sun is below the horizon (helmet lights only).
    """
    return cfg.visibility_night_factor if sun_elevation_rad <= 0.0 else 1.0


def transition_cost(tr: Transition, cfg: RoutingConfig) -> float:
    """Scalar cost of a step for the route search (seconds-equivalent).

    ``w_t * dt + w_o2 * dO2 + w_dose * dDose + w_F * max(0, dF) + w_risk * risk``.
    Every term is non-negative, so ``w_t * distance / V_max`` is an admissible heuristic.
    """
    return (cfg.weight_time_per_s * tr.dt_s + cfg.weight_o2_per_kg * tr.d_o2_kg
            + cfg.weight_dose_per_sv * tr.d_dose_sv
            + cfg.weight_fatigue * max(0.0, tr.d_fatigue) + cfg.weight_risk * tr.risk)


class EdgeEvaluator:
    """Edge physics for one crew member, one suit and one site.

    Built once per EVA: the fitness of the crew member (A1) and the power limits
    derived from it (A3) are fixed for the duration of an EVA.
    """

    def __init__(self, terrain: TerrainProvider, environment: EnvironmentProvider,
                 astronaut: AstronautProfile, suit: SuitProfile,
                 config: SimConfig = DEFAULT_CONFIG, *, time_on_surface_s: float,
                 sleep_debt_s: float = 0.0) -> None:
        """
        Args:
            terrain: terrain rasters.
            environment: environment as a function of cell and time.
            astronaut, suit: static profiles.
            config: tunable parameters.
            time_on_surface_s: time since landing at the start of the EVA, s.
            sleep_debt_s: accumulated sleep debt, s.

        Raises:
            TerrainValidationError: the terrain fails its own validation.
        """
        terrain.validate()
        self.terrain = terrain
        self.environment = environment
        self.astronaut = astronaut
        self.suit = suit
        self.config = config
        self.time_on_surface_s = time_on_surface_s
        self.sleep_debt_s = sleep_debt_s

        self.fitness = adaptation.compute_fitness(time_on_surface_s, astronaut.transit_duration_s,
                                                  config.adaptation)
        self.p_aerobic_w = capacity.aerobic_power_w(astronaut, self.fitness)
        self.p_sustain_w = capacity.sustainable_power_w(astronaut, self.fitness, config.capacity)
        self.p_peak_w = capacity.peak_power_w(astronaut, self.fitness, config.capacity)
        self.readiness = capacity.readiness_factor(sleep_debt_s, config.capacity)
        routing = config.routing
        self.slope_up_limit_rad, self.slope_down_limit_rad = adaptation.slope_limits_rad(
            self.fitness, config.adaptation, routing.slope_up_limit_rad,
            routing.slope_down_limit_rad, routing.reduced_slope_limit_rad)
        self.heat_storage_limit_j = thermal.heat_storage_limit_j(suit, config.thermal)

        self.rows, self.cols = terrain.shape
        self.cell_size_m = float(terrain.cell_size_m)
        self._elev: list[list[float]] = terrain.elevation.tolist()
        self._eta: list[list[float]] = terrain.walk_eta.tolist()
        self._rock: list[list[float]] = terrain.rock_hazard.tolist()
        walkable = (terrain.passable_walk & ~terrain.forbidden
                    & (terrain.slope <= routing.cell_slope_hard_limit_rad))
        self._walkable: list[list[bool]] = walkable.tolist()
        self._offsets: tuple[tuple[int, int, float], ...] = tuple(
            (d_row, d_col, factor * self.cell_size_m) for d_row, d_col, factor in NEIGHBOR_OFFSETS)
        self._env_cache: dict[tuple[int, int, int], EnvSample] = {}
        self._heat_cache: dict[tuple[int, int, int], thermal.EnvHeat] = {}

    # ---- grid helpers ---------------------------------------------------
    def in_bounds(self, cell: Cell) -> bool:
        """True when the cell lies inside the grid."""
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def is_walkable(self, cell: Cell) -> bool:
        """True when the cell may be entered (passable, not forbidden, not too steep)."""
        return self.in_bounds(cell) and self._walkable[cell[0]][cell[1]]

    def neighbors(self, cell: Cell) -> list[tuple[Cell, float]]:
        """In-bounds 8-neighbours with the horizontal distance to each, m."""
        row, col = cell
        out: list[tuple[Cell, float]] = []
        for d_row, d_col, dist in self._offsets:
            n_row, n_col = row + d_row, col + d_col
            if 0 <= n_row < self.rows and 0 <= n_col < self.cols:
                out.append(((n_row, n_col), dist))
        return out

    def require_walkable(self, cell: Cell, role: str) -> None:
        """Raise when a start, goal or base cell cannot be stood on.

        Raises:
            TerrainValidationError: the cell is outside the grid.
            InfeasibleStateError: the cell is not walkable.
        """
        if not self.in_bounds(cell):
            raise TerrainValidationError(f"{role} cell {cell} is outside the grid "
                                         f"{(self.rows, self.cols)}")
        if not self._walkable[cell[0]][cell[1]]:
            raise InfeasibleStateError(f"{role} cell {cell} is not walkable "
                                       "(impassable, forbidden or too steep)")

    # ---- environment ----------------------------------------------------
    def _bucket(self, t_s: float) -> int:
        return int(t_s // self.config.routing.env_cache_bucket_s)

    def env_at(self, cell: Cell, t_s: float) -> EnvSample:
        """Environment at a cell, sampled at the middle of the time bucket holding ``t_s``.

        Sampling at the bucket middle makes the result independent of the order
        in which the search visits the cell.
        """
        bucket = self._bucket(t_s)
        key = (cell[0], cell[1], bucket)
        sample = self._env_cache.get(key)
        if sample is None:
            mid_t = (bucket + 0.5) * self.config.routing.env_cache_bucket_s
            sample = self.environment.get_env(cell, mid_t)
            sample.validate()
            self._env_cache[key] = sample
        return sample

    def heat_at(self, cell: Cell, t_s: float) -> thermal.EnvHeat:
        """Workload-independent heat exchange at a cell (cached like ``env_at``)."""
        key = (cell[0], cell[1], self._bucket(t_s))
        heat = self._heat_cache.get(key)
        if heat is None:
            heat = thermal.environmental_heat(self.astronaut, self.suit, self.env_at(cell, t_s),
                                              self.config.thermal, self.config.life_support)
            self._heat_cache[key] = heat
        return heat

    # ---- states ---------------------------------------------------------
    def initial_label(self, cell: Cell, t_s: float, resources: SuitResources | None = None,
                      phys: PhysState | None = None) -> Label:
        """Start label of a search: a serviced suit and a rested crew member by default."""
        self.require_walkable(cell, "start")
        resources = resources if resources is not None else SuitResources.full(self.suit)
        phys = phys if phys is not None else PhysState()
        phys.t_s = t_s
        return Label.from_states(cell, phys, resources)

    def standard_label(self, t_s: float, fatigue_level: float) -> Label:
        """Reference state used for the static maps: full suit, given fatigue."""
        return Label.from_states((0, 0), PhysState(t_s=t_s, fatigue=fatigue_level),
                                 SuitResources.full(self.suit))

    # ---- B1: walking edge -----------------------------------------------
    def walk_edge(self, a: Cell, b: Cell, dist_m: float, state: WalkerState,
                  pace: Pace = Pace.NOMINAL) -> Transition | None:
        """Physics of walking from cell ``a`` to its neighbour ``b``.

        Args:
            a, b: adjacent cells.
            dist_m: horizontal distance between their centres, m.
            state: state of the crew member on leaving ``a``.
            pace: nominal (sustainable power, thermal limit) or emergency
                (peak power, thermal limit ignored, higher speed ceiling).

        Returns:
            The transition, or ``None`` when the edge cannot be walked.
        """
        r1, c1 = a
        r2, c2 = b
        if not self._walkable[r2][c2]:
            return None
        rise = self._elev[r2][c2] - self._elev[r1][c1]
        grade = rise / dist_m
        slope = math.atan(grade)
        if slope > self.slope_up_limit_rad or -slope > self.slope_down_limit_rad:
            return None

        cfg = self.config
        eta = 0.5 * (self._eta[r1][c1] + self._eta[r2][c2])
        rock = 0.5 * (self._rock[r1][c1] + self._rock[r2][c2])
        env = self.env_at(b, state.t_s)
        heat = self.heat_at(b, state.t_s)
        mech = metabolism.mechanical_efficiency(grade, cfg.metabolism)

        hydration = nutrition.dehydration_factor(state.hydration_deficit_kg, state.drink_water_kg,
                                                 self.astronaut.body_mass_kg, cfg.nutrition)
        fed = nutrition.fasting_factor(self.suit.max_eva_s - state.eva_remaining_s, cfg.nutrition)
        if pace is Pace.NOMINAL:
            base_power = self.p_sustain_w
            thermal_power = thermal.thermal_limit_power_w(heat.capacity_w, heat.q_fixed_w, mech)
            preferred = self.astronaut.preferred_speed_m_s
        else:
            base_power, thermal_power = self.p_peak_w, math.inf
            preferred = cfg.capacity.emergency_speed_m_s
        cap = capacity.power_cap_w(base_power, thermal_power, state.fatigue, self.readiness,
                                   hydration, fed, cfg.capacity)

        visibility = visibility_factor(env.sun_elevation_rad, cfg.routing)
        rock_factor = max(cfg.routing.rock_speed_min_factor,
                          1.0 - cfg.routing.rock_speed_penalty * rock)
        upper = max(cfg.capacity.min_speed_m_s, preferred * rock_factor * visibility)
        c0, c1q, c2q, floor = metabolism.pandolf_coefficients(
            self.astronaut.body_mass_kg, self.suit.mass_kg, eta, grade,
            self.suit.stiffness_factor, cfg.metabolism)
        speed = capacity.select_speed(c0, c1q, c2q, floor, grade, upper, cap, cfg.capacity)
        if speed is None:
            return None

        metabolic = metabolism.power_from_coefficients(c0, c1q, c2q, floor, speed)
        length = math.hypot(dist_m, rise)
        dt = length / speed
        risk = injury.fall_risk(length, slope, grade, rock, self.fitness.sensorimotor,
                                state.fatigue, visibility,
                                speed / self.astronaut.preferred_speed_m_s,
                                self.fitness.bone_fracture_mult, cfg.injury)
        return self._dynamics(state, env, heat, metabolic, mech, dt, grip_load=0.0,
                              contact_fraction=0.0, distance_m=length, speed_m_s=speed,
                              grade=grade, slope_rad=slope, eta=eta,
                              risk=risk.expected_harm, p_fall=risk.p_fall)

    def explain_blocked_edge(self, a: Cell, b: Cell, dist_m: float, state: WalkerState,
                             pace: Pace = Pace.NOMINAL) -> str:
        """Why ``walk_edge`` refuses the move from ``a`` to ``b``.

        Returns:
            A readable reason: forbidden zone, impassable terrain, cell slope,
            directional slope, or the power the walk would need against the power
            the crew member can hold. "passable" when the edge is in fact walkable.
        """
        if not self.in_bounds(b):
            return "the cell is outside the grid"
        terrain, routing = self.terrain, self.config.routing
        if bool(terrain.forbidden[b]):
            return "the cell is in a forbidden zone"
        if not bool(terrain.passable_walk[b]):
            return "the cell is impassable terrain"
        cell_slope = float(terrain.slope[b])
        if cell_slope > routing.cell_slope_hard_limit_rad:
            return (f"the cell slope of {math.degrees(cell_slope):.0f} deg exceeds the hard limit "
                    f"of {math.degrees(routing.cell_slope_hard_limit_rad):.0f} deg")
        rise = self._elev[b[0]][b[1]] - self._elev[a[0]][a[1]]
        grade = rise / dist_m
        slope = math.atan(grade)
        if slope > self.slope_up_limit_rad:
            return (f"the climb of {math.degrees(slope):.0f} deg exceeds the limit of "
                    f"{math.degrees(self.slope_up_limit_rad):.0f} deg")
        if -slope > self.slope_down_limit_rad:
            return (f"the descent of {math.degrees(-slope):.0f} deg exceeds the limit of "
                    f"{math.degrees(self.slope_down_limit_rad):.0f} deg")
        if self.walk_edge(a, b, dist_m, state, pace) is not None:
            return "passable"
        cfg = self.config
        eta = 0.5 * (self._eta[a[0]][a[1]] + self._eta[b[0]][b[1]])
        heat = self.heat_at(b, state.t_s)
        mech = metabolism.mechanical_efficiency(grade, cfg.metabolism)
        base_power = self.p_sustain_w if pace is Pace.NOMINAL else self.p_peak_w
        thermal_power = (thermal.thermal_limit_power_w(heat.capacity_w, heat.q_fixed_w, mech)
                         if pace is Pace.NOMINAL else math.inf)
        cap = capacity.power_cap_w(
            base_power, thermal_power, state.fatigue, self.readiness,
            nutrition.dehydration_factor(state.hydration_deficit_kg, state.drink_water_kg,
                                         self.astronaut.body_mass_kg, cfg.nutrition),
            nutrition.fasting_factor(self.suit.max_eva_s - state.eva_remaining_s, cfg.nutrition),
            cfg.capacity)
        needed = metabolism.walking_power_w(
            self.astronaut.body_mass_kg, self.suit.mass_kg, cfg.capacity.min_speed_m_s, grade,
            eta, self.suit.stiffness_factor, cfg.metabolism)
        limit = "cooling" if thermal_power < base_power else "sustainable power"
        return (f"a {math.degrees(slope):.0f} deg slope on ground with terrain factor {eta:.2f} "
                f"needs {needed:.0f} W even at the slowest pace, above the {cap:.0f} W the crew "
                f"member can hold ({limit})")

    # ---- stationary step ------------------------------------------------
    def stationary(self, cell: Cell, state: WalkerState, dt_s: float,
                   task: TaskType) -> Transition:
        """Physics of staying at ``cell`` for ``dt_s`` doing ``task`` (rest included).

        Raises:
            InfeasibleStateError: ``dt_s`` is not positive.
        """
        if dt_s <= 0.0:
            raise InfeasibleStateError(f"stationary step needs dt_s > 0, got {dt_s}")
        cfg = self.config
        env = self.env_at(cell, state.t_s)
        heat = self.heat_at(cell, state.t_s)
        metabolic = metabolism.task_power_w(task, state.fatigue, cfg.metabolism)
        return self._dynamics(state, env, heat, metabolic, 0.0, dt_s,
                              grip_load=cfg.metabolism.grip_load_of(task),
                              contact_fraction=cfg.metabolism.contact_fraction_of(task),
                              distance_m=0.0, speed_m_s=0.0, grade=0.0, slope_rad=0.0,
                              eta=self._eta[cell[0]][cell[1]], risk=0.0, p_fall=0.0)

    # ---- shared dynamics (A4-A8, A12) -----------------------------------
    def _dynamics(self, state: WalkerState, env: EnvSample, heat: thermal.EnvHeat,
                  metabolic_w: float, mech_efficiency: float, dt_s: float, *, grip_load: float,
                  contact_fraction: float, distance_m: float, speed_m_s: float, grade: float,
                  slope_rad: float, eta: float, risk: float, p_fall: float) -> Transition:
        cfg = self.config
        suit = self.suit
        q_env = heat.q_fixed_w + contact_fraction * heat.q_glove_w
        balance = thermal.thermal_balance(
            metabolic_w, mech_efficiency, q_env, heat.capacity_w, state.heat_storage_j, dt_s,
            suit.heater_max_w - heat.extremity_heater_w, cfg.thermal)
        heater_total = balance.heater_w + heat.extremity_heater_w
        spent = life_support.consume(metabolic_w, balance.cooling_w, heater_total, dt_s, suit, env,
                                     cfg.life_support)
        new_fatigue = fatigue.update_fatigue(state.fatigue, metabolic_w, self.p_sustain_w, dt_s,
                                             cfg.fatigue)
        new_hand = fatigue.update_hand_fatigue(state.hand_fatigue, grip_load, suit.pressure_pa,
                                               dt_s, cfg.fatigue)
        new_deficit, drunk = nutrition.hydration_step(
            state.hydration_deficit_kg, state.drink_water_kg, metabolic_w, dt_s, self.astronaut,
            cfg.nutrition)
        return Transition(
            dt_s=dt_s, distance_m=distance_m, speed_m_s=speed_m_s, grade=grade,
            slope_rad=slope_rad, metabolic_w=metabolic_w, energy_j=metabolic_w * dt_s,
            heart_rate_hz=capacity.heart_rate_hz(metabolic_w, self.astronaut, self.fitness,
                                                 cfg.capacity),
            cooling_w=balance.cooling_w, heater_w=heater_total,
            d_o2_kg=spent.o2_kg, d_co2_kg=spent.co2_kg,
            d_cooling_water_kg=spent.cooling_water_kg, d_drink_kg=drunk,
            d_battery_j=spent.battery_j,
            d_dose_sv=radiation.dose_increment_sv(env, dt_s, suit.dose_factor),
            d_fatigue=new_fatigue - state.fatigue,
            d_hand_fatigue=new_hand - state.hand_fatigue,
            d_heat_j=balance.heat_storage_j - state.heat_storage_j,
            d_hydration_kg=new_deficit - state.hydration_deficit_kg,
            d_dust_kg=dust.dust_increment_kg(distance_m, dust.soil_fine_fraction(eta), cfg.dust),
            d_uv=radiation.uv_exposure_increment(env, dt_s, cfg.radiation),
            risk=risk, p_fall=p_fall)

    # ---- state update ---------------------------------------------------
    def advance(self, label: Label, cell: Cell, tr: Transition, action: str) -> Label:
        """Label after applying a transition that ends at ``cell``."""
        return Label(
            row=cell[0], col=cell[1], t_s=label.t_s + tr.dt_s,
            o2_kg=label.o2_kg - tr.d_o2_kg,
            co2_capacity_kg=label.co2_capacity_kg - tr.d_co2_kg,
            cooling_water_kg=label.cooling_water_kg - tr.d_cooling_water_kg,
            drink_water_kg=label.drink_water_kg - tr.d_drink_kg,
            battery_j=label.battery_j - tr.d_battery_j,
            eva_remaining_s=label.eva_remaining_s - tr.dt_s,
            fatigue=label.fatigue + tr.d_fatigue,
            hand_fatigue=label.hand_fatigue + tr.d_hand_fatigue,
            hydration_deficit_kg=label.hydration_deficit_kg + tr.d_hydration_kg,
            heat_storage_j=label.heat_storage_j + tr.d_heat_j,
            dose_sv=label.dose_sv + tr.d_dose_sv,
            risk=label.risk + tr.risk,
            cost=label.cost + transition_cost(tr, self.config.routing),
            distance_m=label.distance_m + tr.distance_m,
            energy_j=label.energy_j + tr.energy_j,
            dust_load_kg=label.dust_load_kg + tr.d_dust_kg,
            uv_exposure=label.uv_exposure + tr.d_uv,
            action=action, transition=tr, parent=label)
