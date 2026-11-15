"""Physics of one rover action, for the generic route search.

``RoverEdgeEvaluator`` implements the ``TraversalModel`` protocol for a rover:
it chains mobility (R1), power and heat (R2) and radiation (R3) into a drive
step or a parked interval, exactly as ``EdgeEvaluator`` does for an astronaut.

A drive step ``a -> b``::

    1. terrain gates    in bounds, passable, not forbidden
    2. operations       daylight (rovers navigate by camera), particle-event hold
    3. R1 mobility      clearance, tilt, climb, wheel slip; wheel-walking fallback
    4. timing           dt = slope length / ground speed
    5. R2 power         drive + avionics + heaters against the generator; battery
    6. R3 radiation     upset risk and absorbed dose

The wheel speed is ``top speed * autonomous-navigation factor * particle-event factor``.

A rover may also stay parked: through the night until it can drive again, or by
day to recharge. A stay is one action however long it lasts, so the search does
not fill up with chains of short waits.
"""

from __future__ import annotations

import math

from ..core.config import DEFAULT_CONFIG, SimConfig
from ..core.constants import SOL_S
from ..core.exceptions import InfeasibleStateError, TerrainValidationError, ValidationError
from ..core.schemas import Cell, EnvSample
from ..providers.interfaces import NEIGHBOR_OFFSETS, EnvironmentProvider, TerrainProvider
from ..rovers import rover_mobility, rover_power_thermal, rover_radiation
from ..rovers.profiles import RoverProfile
from ..rovers.rover_mobility import MobilityResult
from ..rovers.types import DriveMode, RoverBlock, RoverLabel, RoverTransition


class RoverEdgeEvaluator:
    """Drive and park physics of one rover on one site in one mission year."""

    def __init__(self, terrain: TerrainProvider, environment: EnvironmentProvider,
                 profile: RoverProfile, config: SimConfig = DEFAULT_CONFIG, *,
                 mission_year: float, sols_since_panel_cleaning: float = 0.0,
                 allow_night_driving: bool = False) -> None:
        """
        Args:
            terrain: terrain rasters, including ``obstacle_height_m``.
            environment: environment as a function of cell and time.
            profile: the rover.
            config: tunable parameters.
            mission_year: calendar year of the traverse; sets the MMRTG output.
            sols_since_panel_cleaning: dust settled on the solar arrays.
            allow_night_driving: drive without daylight (off by default: Mars rovers
                navigate by camera and drive by day).

        Raises:
            ValidationError: the rover has not landed by ``mission_year``.
            TerrainValidationError: the terrain fails its own validation.
        """
        terrain.validate()
        self.terrain = terrain
        self.environment = environment
        self.profile = profile
        self.config = config
        self.mission_year = mission_year
        self.age_years = profile.mission_age_years(mission_year)
        self.dust_factor = rover_power_thermal.panel_dust_factor(sols_since_panel_cleaning,
                                                                 config.rover)
        self.allow_night_driving = allow_night_driving
        self.block_counts: dict[RoverBlock, int] = {}

        self.rows, self.cols = terrain.shape
        self.cell_size_m = float(terrain.cell_size_m)
        self._elev: list[list[float]] = terrain.elevation.tolist()
        self._eta: list[list[float]] = terrain.walk_eta.tolist()
        self._slope: list[list[float]] = terrain.slope.tolist()
        self._obstacle: list[list[float]] = terrain.obstacle_height_m.tolist()
        self._open: list[list[bool]] = (terrain.passable_walk & ~terrain.forbidden).tolist()
        self._forbidden: list[list[bool]] = terrain.forbidden.tolist()
        self._offsets: tuple[tuple[int, int, float], ...] = tuple(
            (d_row, d_col, factor * self.cell_size_m) for d_row, d_col, factor in NEIGHBOR_OFFSETS)
        self._env_cache: dict[tuple[int, int, int], EnvSample] = {}

    # ---- TraversalModel protocol ------------------------------------------
    @property
    def max_speed_m_s(self) -> float:
        """Top speed of the rover, m/s (no step is faster)."""
        return self.profile.max_speed_m_s

    @property
    def time_weight_per_s(self) -> float:
        """Cost of one second of the traverse."""
        return self.config.rover.weight_time_per_s

    def in_bounds(self, cell: Cell) -> bool:
        """True when the cell lies inside the grid."""
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def cell_block(self, cell: Cell) -> RoverBlock:
        """Why the rover cannot stand in a cell, whatever the direction of approach."""
        if not self.in_bounds(cell):
            return RoverBlock.OUT_OF_BOUNDS
        row, col = cell
        if self._forbidden[row][col]:
            return RoverBlock.FORBIDDEN
        if not self._open[row][col]:
            return RoverBlock.IMPASSABLE
        if self._obstacle[row][col] > self.profile.max_obstacle_height_m:
            return RoverBlock.CLEARANCE
        if self._slope[row][col] > self.profile.max_tilt_rad:
            return RoverBlock.TILT
        return RoverBlock.NONE

    def is_walkable(self, cell: Cell) -> bool:
        """True when the rover can stand in the cell."""
        return self.cell_block(cell) is RoverBlock.NONE

    def require_walkable(self, cell: Cell, role: str) -> None:
        """Raise when a start or goal cell cannot hold the rover.

        Raises:
            TerrainValidationError: the cell is outside the grid.
            InfeasibleStateError: the rover cannot stand in the cell.
        """
        block = self.cell_block(cell)
        if block is RoverBlock.OUT_OF_BOUNDS:
            raise TerrainValidationError(f"{role} cell {cell} is outside the grid "
                                         f"{(self.rows, self.cols)}")
        if block is not RoverBlock.NONE:
            raise InfeasibleStateError(f"{self.profile.name} cannot stand in the {role} cell "
                                       f"{cell}: {block.value}")

    def neighbors(self, cell: Cell) -> list[tuple[Cell, float]]:
        """In-bounds 8-neighbours with the horizontal distance to each, m."""
        row, col = cell
        out: list[tuple[Cell, float]] = []
        for d_row, d_col, dist in self._offsets:
            n_row, n_col = row + d_row, col + d_col
            if 0 <= n_row < self.rows and 0 <= n_col < self.cols:
                out.append(((n_row, n_col), dist))
        return out

    def dominates(self, y: RoverLabel, x: RoverLabel, tol: float) -> bool:
        """True when rover state ``y`` is no worse than ``x`` in cost, energy, time and risk.

        A stay in place is never dominated by the state it started from: being
        earlier is no advantage to a rover that cannot drive until morning.
        """
        if x.action == "wait" and x.parent is y:
            return False
        up, down = 1.0 + tol, 1.0 - tol
        return (y.cost <= x.cost * up + tol
                and y.battery_j >= x.battery_j * down - tol
                and y.t_s <= x.t_s * up + tol
                and y.risk <= x.risk * up + tol)

    def successors(self, label: RoverLabel) -> tuple[list[tuple[Cell, RoverLabel]], int]:
        """Drives to the eight neighbours, plus one stay in place when it is useful.

        The rover stays when it cannot drive now (night, particle-event hold) or
        when its battery is below the recharge trigger and parking would charge it.

        Returns:
            ``(candidates, number of drives the physics refused)``.
        """
        cell = (label.row, label.col)
        out: list[tuple[Cell, RoverLabel]] = []
        refused = 0
        halted = self.operations_block(cell, label.t_s)
        if halted is RoverBlock.NONE:
            for nb, dist in self.neighbors(cell):
                tr = self.drive_edge(cell, nb, dist, label)
                if tr is None:
                    refused += 1
                    continue
                out.append((nb, self.advance(label, nb, tr, "drive")))
        low = label.battery_j < self.config.rover.recharge_trigger_fraction * self.profile.battery_j
        if halted is not RoverBlock.NONE or low:
            stay = self.park(cell, label)
            if stay is not None:
                out.append((cell, self.advance(label, cell, stay, "wait")))
        return out, refused

    def edge_metrics(self, a: Cell, b: Cell, dist_m: float, state: RoverLabel
                     ) -> dict[str, float] | None:
        """Named costs of one drive for a reference state, or ``None`` when refused."""
        tr = self.drive_edge(a, b, dist_m, state, count_blocks=False)
        if tr is None:
            return None
        return {"time_s": tr.dt_s, "energy_j": tr.load_w * tr.dt_s, "slip": tr.slip,
                "length_m": tr.distance_m,
                "wheel_walking": 1.0 if tr.mode is DriveMode.WHEEL_WALKING else 0.0}

    # ---- environment and state ----------------------------------------------
    def env_at(self, cell: Cell, t_s: float) -> EnvSample:
        """Environment at a cell, sampled at the middle of the time bucket holding ``t_s``."""
        bucket_s = self.config.routing.env_cache_bucket_s
        bucket = int(t_s // bucket_s)
        key = (cell[0], cell[1], bucket)
        sample = self._env_cache.get(key)
        if sample is None:
            sample = self.environment.get_env(cell, (bucket + 0.5) * bucket_s)
            sample.validate()
            self._env_cache[key] = sample
        return sample

    def initial_label(self, cell: Cell, t_s: float, battery_fraction: float = 1.0) -> RoverLabel:
        """Start state of a traverse.

        Raises:
            ValidationError: ``battery_fraction`` is outside 0..1.
        """
        if not 0.0 <= battery_fraction <= 1.0:
            raise ValidationError(f"battery_fraction={battery_fraction} outside [0, 1]")
        self.require_walkable(cell, "start")
        energy = battery_fraction * self.profile.battery_j
        return RoverLabel(row=cell[0], col=cell[1], t_s=t_s, battery_j=energy,
                          min_battery_j=energy)

    # ---- gates --------------------------------------------------------------
    def operations_block(self, cell: Cell, t_s: float) -> RoverBlock:
        """Why the rover may not drive at all right now: night, or a particle-event hold."""
        env = self.env_at(cell, t_s)
        if (not self.allow_night_driving
                and env.sun_elevation_rad < self.config.rover.min_sun_elevation_drive_rad):
            return RoverBlock.NIGHT
        if rover_radiation.speed_factor(env, self.config.rover) <= 0.0:
            return RoverBlock.SPE_HOLD
        return RoverBlock.NONE

    def mobility(self, a: Cell, b: Cell, dist_m: float, t_s: float) -> MobilityResult:
        """R1 verdict on the drive from ``a`` to ``b`` (terrain gates included)."""
        block = self.cell_block(b)
        if block in (RoverBlock.OUT_OF_BOUNDS, RoverBlock.FORBIDDEN, RoverBlock.IMPASSABLE):
            return MobilityResult(block)
        rise = self._elev[b[0]][b[1]] - self._elev[a[0]][a[1]]
        env = self.env_at(b, t_s)
        wheel_speed = (self.profile.max_speed_m_s * self.profile.autonav_speed_factor
                       * rover_radiation.speed_factor(env, self.config.rover))
        return rover_mobility.assess_step(
            self.profile, self.config.rover, slope_rad=math.atan(rise / dist_m),
            cell_slope_rad=self._slope[b[0]][b[1]],
            eta=0.5 * (self._eta[a[0]][a[1]] + self._eta[b[0]][b[1]]),
            obstacle_height_m=self._obstacle[b[0]][b[1]], wheel_speed_m_s=wheel_speed)

    def block_reason(self, a: Cell, b: Cell, dist_m: float, t_s: float) -> RoverBlock:
        """Why the drive from ``a`` to ``b`` at time ``t_s`` is refused (``NONE`` if it is not)."""
        halted = self.operations_block(a, t_s)
        return halted if halted is not RoverBlock.NONE else self.mobility(a, b, dist_m, t_s).block

    # ---- actions ------------------------------------------------------------
    def drive_edge(self, a: Cell, b: Cell, dist_m: float, state: RoverLabel,
                   count_blocks: bool = True) -> RoverTransition | None:
        """Drive from cell ``a`` to its neighbour ``b``.

        Returns:
            The transition, or ``None`` when the step is refused. Refusals are
            counted per reason in ``block_counts`` unless ``count_blocks`` is False.
        """
        halted = self.operations_block(a, state.t_s)
        result = MobilityResult(halted) if halted is not RoverBlock.NONE else self.mobility(
            a, b, dist_m, state.t_s)
        if not result.passable or result.ground_speed_m_s <= 0.0:
            if count_blocks:
                self.block_counts[result.block] = self.block_counts.get(result.block, 0) + 1
            return None
        rise = self._elev[b[0]][b[1]] - self._elev[a[0]][a[1]]
        length = math.hypot(dist_m, rise)
        dt = length / result.ground_speed_m_s
        env = self.env_at(b, state.t_s)
        drive_w = rover_power_thermal.drive_power_w(self.profile, result)
        return self._power_step(state, env, dt, base_w=self.profile.avionics_w, drive_w=drive_w,
                                moving=True, distance_m=length, speed=result.ground_speed_m_s,
                                slope=math.atan(rise / dist_m), slip=result.slip,
                                mode=result.mode)

    def park(self, cell: Cell, state: RoverLabel) -> RoverTransition | None:
        """Stay in place until the rover can drive again and has recharged.

        The stay lasts until driving is allowed and the battery has been charged:
        it runs on while the rover cannot drive, then until charging has begun (a
        solar rover waits for sunrise) and has finished or stopped, and one sol at
        the latest. Power is integrated in steps of ``wait_step_s``; the lowest
        battery level on the way is kept in the result.

        Returns:
            One transition for the whole stay, or ``None`` when staying would last
            no time at all.
        """
        cfg, profile = self.config.rover, self.profile
        step = cfg.wait_step_s
        t, battery, lowest = state.t_s, state.battery_j, state.battery_j
        total = heat_sum = gen_sum = load_sum = dose = risk = 0.0
        deficit = 0.0
        charged = False
        while total < SOL_S:
            env = self.env_at(cell, t)
            heat = rover_power_thermal.heater_demand(profile, env, profile.sleep_w, moving=False)
            load = profile.sleep_w + heat.power_w
            gen = rover_power_thermal.generation_w(profile, env, self.age_years,
                                                   self.dust_factor, cfg)
            can_drive = self.operations_block(cell, t) is RoverBlock.NONE
            full = battery >= 0.999 * profile.battery_j
            charging = gen > load and not full
            charged = charged or charging
            if total > 0.0 and can_drive and not charging and (charged or full):
                break
            battery = rover_power_thermal.battery_after(battery, gen, load, step, profile, cfg)
            lowest = min(lowest, battery)
            deficit = max(deficit, heat.deficit_w)
            heat_sum += heat.power_w * step
            gen_sum += gen * step
            load_sum += load * step
            dose += rover_radiation.absorbed_dose_gy(env, step, cfg)
            risk += rover_radiation.upset_risk(env, step, cfg)
            total += step
            t += step
        if total <= 0.0:
            return None
        tr = RoverTransition(
            dt_s=total, distance_m=0.0, speed_m_s=0.0, slope_rad=0.0, slip=0.0,
            mode=DriveMode.ROLLING, drive_w=0.0, heater_w=heat_sum / total,
            load_w=load_sum / total, generation_w=gen_sum / total,
            d_battery_j=battery - state.battery_j, heater_deficit_w=deficit, d_dose_gy=dose,
            risk=min(1.0, risk), battery_low_j=lowest)
        return tr

    def _power_step(self, state: RoverLabel, env: EnvSample, dt: float, *, base_w: float,
                    drive_w: float, moving: bool, distance_m: float, speed: float, slope: float,
                    slip: float, mode: DriveMode) -> RoverTransition:
        cfg, profile = self.config.rover, self.profile
        heat = rover_power_thermal.heater_demand(profile, env, base_w, moving)
        load = drive_w + base_w + heat.power_w
        gen = rover_power_thermal.generation_w(profile, env, self.age_years, self.dust_factor, cfg)
        battery = rover_power_thermal.battery_after(state.battery_j, gen, load, dt, profile, cfg)
        return RoverTransition(
            dt_s=dt, distance_m=distance_m, speed_m_s=speed, slope_rad=slope, slip=slip,
            mode=mode, drive_w=drive_w, heater_w=heat.power_w, load_w=load, generation_w=gen,
            d_battery_j=battery - state.battery_j, heater_deficit_w=heat.deficit_w,
            d_dose_gy=rover_radiation.absorbed_dose_gy(env, dt, cfg),
            risk=rover_radiation.upset_risk(env, dt, cfg),
            battery_low_j=min(state.battery_j, battery))

    def transition_cost(self, tr: RoverTransition) -> float:
        """Scalar cost of an action (seconds-equivalent).

        ``w_t * dt + w_e * energy drawn from the battery + w_risk * risk
        + w_slip * slip * distance``. Every term is non-negative.
        """
        cfg = self.config.rover
        return (cfg.weight_time_per_s * tr.dt_s
                + cfg.weight_energy_per_j * max(0.0, -tr.d_battery_j)
                + cfg.weight_risk * tr.risk + cfg.weight_slip_per_m * tr.slip * tr.distance_m)

    def advance(self, label: RoverLabel, cell: Cell, tr: RoverTransition,
                action: str) -> RoverLabel:
        """State after applying a transition that ends at ``cell``."""
        battery = label.battery_j + tr.d_battery_j
        lowest = min(label.min_battery_j, tr.battery_low_j)
        walking = tr.distance_m if tr.mode is DriveMode.WHEEL_WALKING else 0.0
        return RoverLabel(
            row=cell[0], col=cell[1], t_s=label.t_s + tr.dt_s, battery_j=battery,
            min_battery_j=lowest, distance_m=label.distance_m + tr.distance_m,
            cost=label.cost + self.transition_cost(tr), risk=label.risk + tr.risk,
            dose_gy=label.dose_gy + tr.d_dose_gy,
            energy_used_j=label.energy_used_j + tr.load_w * tr.dt_s,
            wait_s=label.wait_s + (tr.dt_s if action == "wait" else 0.0),
            wheel_walk_m=label.wheel_walk_m + walking,
            max_slip=max(label.max_slip, tr.slip), heater_deficit_w=tr.heater_deficit_w,
            action=action, transition=tr, parent=label)
