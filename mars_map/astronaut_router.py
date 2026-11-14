
from __future__ import annotations

import heapq
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mars_map.astronaut_terrain import AstronautTerrainModel, build_astronaut_terrain
    from mars_map.site import MissionParams, Site, Target
else:
    from .astronaut_terrain import AstronautTerrainModel, build_astronaut_terrain
    from .site import MissionParams, Site, Target

MARS_GRAVITY = 3.72
EARTH_GRAVITY = 9.81
ASTRONAUT_MASS_KG = 75.0
SUIT_MASS_KG = 70.0
O2_TANK_LITERS = 480.0
BATTERY_WH = 1200.0


@dataclass
class RouteSegment:

    points_m: list[tuple[float, float]]   # координаты маршрута [(x_m, y_m), ...]
    pixels: list[tuple[int, int]]         # координаты пикселей [(row, col), ...]
    distance_m: float
    elevation_gain_m: float
    elevation_drop_m: float
    max_slope_deg: float
    mean_looseness: float
    duration_min: float
    o2_liters: float
    energy_wh: float
    dose_msv: float
    profile: list[dict] = field(default_factory=list)


@dataclass
class MissionPlan:

    feasible: bool
    warnings: list[str]
    visited_targets: list[str]
    segments: list[RouteSegment]
    total_distance_km: float
    total_elevation_gain_m: float
    max_slope_deg: float
    total_time_hours: float
    o2_remaining_pct: float
    battery_remaining_pct: float
    total_dose_msv: float

    def to_json(self, path: str | Path) -> Path:
        """export."""
        out = Path(path)
        data = {
            "feasible": self.feasible,
            "warnings": self.warnings,
            "visited_targets": self.visited_targets,
            "summary": {
                "total_distance_km": round(self.total_distance_km, 3),
                "total_elevation_gain_m": round(self.total_elevation_gain_m, 1),
                "max_slope_deg": round(self.max_slope_deg, 1),
                "total_time_hours": round(self.total_time_hours, 2),
                "o2_remaining_pct": round(self.o2_remaining_pct, 1),
                "battery_remaining_pct": round(self.battery_remaining_pct, 1),
                "total_dose_msv": round(self.total_dose_msv, 3),
            },
            "waypoints": [
                pt for seg in self.segments for pt in seg.profile
            ],
        }
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return out


def step_biomechanics(
    dist_m: float,
    delta_h_m: float,
    terramech_factor: float,
    speed_mult: float,
    mission: MissionParams,
) -> tuple[float, float, float, float]:
    """
    Step biomechanics
    Returns: (time_sec, o2_liters, energy_wh, slope_deg)
    """
    slope_rad = math.atan2(delta_h_m, dist_m)
    slope_deg = math.degrees(slope_rad)
    tan_s = math.tan(slope_rad)

    v_flat = mission.walk_speed_flat_kmh / 3.6

    slope_speed_factor = math.exp(-3.1 * abs(tan_s + 0.05))
    speed_ms = max(0.18, v_flat * slope_speed_factor * speed_mult)
    time_sec = dist_m / speed_ms

    total_mass = ASTRONAUT_MASS_KG + SUIT_MASS_KG
    g_ratio = MARS_GRAVITY / EARTH_GRAVITY

    standing_w = 1.5 * ASTRONAUT_MASS_KG + 2.0 * total_mass * g_ratio
    horizontal_w = terramech_factor * total_mass * g_ratio * (1.5 * speed_ms * speed_ms)

    grade_pct = tan_s * 100.0
    if grade_pct >= 0.0:
        vertical_w = 0.35 * total_mass * g_ratio * speed_ms * grade_pct * terramech_factor
    elif slope_deg < -10.0:
        
        vertical_w = 0.22 * total_mass * speed_ms * abs(grade_pct + 10.0)
    else:
        vertical_w = -0.06 * total_mass * g_ratio * speed_ms * abs(grade_pct)

    metabolic_w = max(140.0, standing_w + horizontal_w + vertical_w)


    o2_liters = (metabolic_w / 20100.0) * time_sec
    electric_w = 180.0 + 0.35 * metabolic_w
    energy_wh = (electric_w * time_sec) / 3600.0

    return time_sec, o2_liters, energy_wh, slope_deg


def _snap_to_walkable(row: int, col: int, impassable: np.ndarray, max_radius: int = 15) -> tuple[int, int]:
    """Если точка попала на край обрыва/no_go, сдвигает её на ближайший проходимый пиксель."""
    rows, cols = impassable.shape
    if not impassable[row, col]:
        return row, col
    for r in range(1, max_radius + 1):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols and not impassable[nr, nc]:
                    return nr, nc
    raise ValueError(f"Точка ({row}, {col}) полностью заблокирована непроходимой зоной.")


def route_segment(
    site: Site,
    start_m: tuple[float, float],
    goal_m: tuple[float, float],
    terrain: AstronautTerrainModel | None = None,
    radiation_rate_msv_h: np.ndarray | None = None,
    looseness_weight: float = 1.6,
    radiation_weight: float = 0.0,
) -> RouteSegment:
    """
    Строит оптимальный путь A* между двумя точками (x_m, y_m) с учетом
    перепадов высот, рыхлости грунта, запретных зон и (опционально) радиации.
    """
    if terrain is None:
        terrain = build_astronaut_terrain(site)

    rows, cols = site.shape
    res = site.resolution_m
    mission = site.mission

    sr, sc = _snap_to_walkable(*site.to_pixel(*start_m), terrain.impassable)
    gr, gc = _snap_to_walkable(*site.to_pixel(*goal_m), terrain.impassable)

    # A* 1000x1000 grid
    elev_flat = site.layers["elevation"].ravel()
    imp_flat = terrain.impassable.ravel()
    terra_flat = terrain.terramech_factor.ravel()
    speed_flat = terrain.speed_multiplier.ravel()
    loose_flat = terrain.looseness.ravel()
    nodata_flat = terrain.nodata_penalty.ravel()
    rad_flat = (
        radiation_rate_msv_h.ravel()
        if radiation_rate_msv_h is not None
        else np.full(rows * cols, site.series["radiation_msv_per_sol"][0] / 24.6, dtype=np.float32)
    )

    start_id = sr * cols + sc
    goal_id = gr * cols + gc

    g_score = np.full(rows * cols, np.inf, dtype=np.float64)
    parent = np.full(rows * cols, -1, dtype=np.int32)
    closed = np.zeros(rows * cols, dtype=np.uint8)

    g_score[start_id] = 0.0
    v_flat = mission.walk_speed_flat_kmh / 3.6
    hard_limit = mission.hard_max_slope_deg
    soft_limit = mission.max_slope_deg

    neighbors = [
        (-1, 0, res), (1, 0, res), (0, -1, res), (0, 1, res),
        (-1, -1, res * 1.4142135), (-1, 1, res * 1.4142135),
        (1, -1, res * 1.4142135), (1, 1, res * 1.4142135),
    ]

    open_heap: list[tuple[float, int]] = [(0.0, start_id)]

    while open_heap:
        _, curr_id = heapq.heappop(open_heap)
        if curr_id == goal_id:
            break
        if closed[curr_id]:
            continue
        closed[curr_id] = 1

        cr = curr_id // cols
        cc = curr_id - cr * cols
        h_curr = float(elev_flat[curr_id])
        g_curr = g_score[curr_id]

        for dr, dc, dist_m in neighbors:
            nr = cr + dr
            nc = cc + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            nid = nr * cols + nc
            if closed[nid] or imp_flat[nid]:
                continue

            dh = float(elev_flat[nid]) - h_curr
            slope_rad = math.atan2(dh, dist_m)
            slope_deg = math.degrees(slope_rad)
            abs_slope = abs(slope_deg)

            if abs_slope > hard_limit:
                continue


            tan_s = math.tan(slope_rad)
            speed_ms = max(0.18, v_flat * math.exp(-3.1 * abs(tan_s + 0.05)) * float(speed_flat[nid]))
            time_sec = dist_m / speed_ms


            loose_penalty = 1.0 + looseness_weight * float(loose_flat[nid])
            steep_penalty = 1.0 + (0.15 * (abs_slope - soft_limit) ** 2 if abs_slope > soft_limit else 0.0)
            rad_cost = float(rad_flat[nid]) * time_sec * radiation_weight

            step_cost = (time_sec * loose_penalty * steep_penalty * float(nodata_flat[nid])) + rad_cost
            tentative_g = g_curr + step_cost

            if tentative_g < g_score[nid]:
                g_score[nid] = tentative_g
                parent[nid] = curr_id
                h_est = (math.hypot(gr - nr, gc - nc) * res) / v_flat
                heapq.heappush(open_heap, (tentative_g + h_est, nid))

    if parent[goal_id] == -1 and start_id != goal_id:
        raise RuntimeError(f"Маршрут от {start_m} до {goal_m} не найден (цель изолирована обрывами или no_go).")


    path_ids = []
    cur = goal_id
    while cur != -1:
        path_ids.append(cur)
        cur = int(parent[cur])
    path_ids.reverse()

    pixels: list[tuple[int, int]] = []
    points_m: list[tuple[float, float]] = []
    profile: list[dict] = []

    tot_dist = 0.0
    tot_gain = 0.0
    tot_drop = 0.0
    max_s = 0.0
    tot_time = 0.0
    tot_o2 = 0.0
    tot_wh = 0.0
    tot_dose = 0.0
    loose_sum = 0.0

    for i, nid in enumerate(path_ids):
        r, c = divmod(nid, cols)
        xm, ym = site.to_metres(r, c)
        pixels.append((r, c))
        points_m.append((xm, ym))
        loose_val = float(loose_flat[nid])
        loose_sum += loose_val

        if i == 0:
            profile.append({
                "x_m": round(xm, 1), "y_m": round(ym, 1),
                "elev_m": round(float(elev_flat[nid]), 2),
                "dist_m": 0.0, "slope_deg": 0.0, "looseness": round(loose_val, 3),
            })
            continue

        prev_id = path_ids[i - 1]
        pr, pc = divmod(prev_id, cols)
        d_m = math.hypot(r - pr, c - pc) * res
        dh = float(elev_flat[nid]) - float(elev_flat[prev_id])

        t_sec, o2_l, e_wh, s_deg = step_biomechanics(
            d_m, dh, float(terra_flat[nid]), float(speed_flat[nid]), mission
        )
        tot_dist += d_m
        if dh > 0:
            tot_gain += dh
        else:
            tot_drop += abs(dh)
        max_s = max(max_s, abs(s_deg))
        tot_time += t_sec
        tot_o2 += o2_l
        tot_wh += e_wh
        tot_dose += float(rad_flat[nid]) * (t_sec / 3600.0)

        profile.append({
            "x_m": round(xm, 1), "y_m": round(ym, 1),
            "elev_m": round(float(elev_flat[nid]), 2),
            "dist_m": round(tot_dist, 1),
            "slope_deg": round(s_deg, 1),
            "looseness": round(loose_val, 3),
        })

    return RouteSegment(
        points_m=points_m,
        pixels=pixels,
        distance_m=tot_dist,
        elevation_gain_m=tot_gain,
        elevation_drop_m=tot_drop,
        max_slope_deg=max_s,
        mean_looseness=loose_sum / max(1, len(path_ids)),
        duration_min=tot_time / 60.0,
        o2_liters=tot_o2,
        energy_wh=tot_wh,
        dose_msv=tot_dose,
        profile=profile,
    )


def plan_eva_mission(
    site: Site,
    targets: list[Target] | None = None,
    radiation_rate_msv_h: np.ndarray | None = None,
    looseness_weight: float = 1.6,
    radiation_weight: float = 0.0,
) -> MissionPlan:
    """
    Планирует круговой выход EVA: Start -> выбранные Targets -> возврат на Start.
    Проверяет все ограничения из site.mission.
    """
    terrain = build_astronaut_terrain(site)
    mission = site.mission

    if targets is None:
        # По умолчанию берем обязательные цели (priority == 1) в порядке приближения
        targets = sorted([t for t in site.targets if t.priority == 1], key=lambda t: t.priority)
        if not targets:
            targets = site.targets[:2]

    waypoints: list[tuple[str, tuple[float, float], float]] = [
        ("Start", site.start, 0.0),
        *[(t.name, (t.x_m, t.y_m), t.dwell_min) for t in targets],
        ("Return Base", site.start, 0.0),
    ]

    segments: list[RouteSegment] = []
    dwell_total_min = 0.0
    dwell_o2_l = 0.0
    dwell_wh = 0.0
    dwell_dose = 0.0

    base_rad_rate = (
        float(np.mean(radiation_rate_msv_h))
        if radiation_rate_msv_h is not None
        else site.series["radiation_msv_per_sol"][0] / 24.6
    )

    for i in range(len(waypoints) - 1):
        _, p0, _ = waypoints[i]
        _, p1, dwell_min = waypoints[i + 1]
        seg = route_segment(
            site, p0, p1,
            terrain=terrain,
            radiation_rate_msv_h=radiation_rate_msv_h,
            looseness_weight=looseness_weight,
            radiation_weight=radiation_weight,
        )
        segments.append(seg)

        # Затраты во время работы астронавта на научной точке (отбор проб)
        dwell_total_min += dwell_min
        dwell_o2_l += (220.0 / 20100.0) * (dwell_min * 60.0)
        dwell_wh += (260.0 * dwell_min) / 60.0
        dwell_dose += base_rad_rate * (dwell_min / 60.0)

    tot_dist_km = sum(s.distance_m for s in segments) / 1000.0
    tot_gain_m = sum(s.elevation_gain_m for s in segments)
    max_slope = max(s.max_slope_deg for s in segments)
    tot_hours = (sum(s.duration_min for s in segments) + dwell_total_min) / 60.0
    tot_o2 = sum(s.o2_liters for s in segments) + dwell_o2_l
    tot_wh = sum(s.energy_wh for s in segments) + dwell_wh
    tot_dose = sum(s.dose_msv for s in segments) + dwell_dose

    o2_left_pct = max(0.0, 100.0 * (1.0 - tot_o2 / O2_TANK_LITERS))
    bat_left_pct = max(0.0, 100.0 * (1.0 - tot_wh / BATTERY_WH))

    warnings: list[str] = []
    max_allowed_hours = mission.eva_max_hours * (1.0 - mission.return_reserve_frac)
    if tot_hours > max_allowed_hours:
        warnings.append(f"Превышен лимит времени с учетом резерва: {tot_hours:.2f} ч > {max_allowed_hours:.2f} ч")
    if tot_dist_km > mission.max_distance_km:
        warnings.append(f"Превышена макс. дистанция EVA: {tot_dist_km:.2f} км > {mission.max_distance_km:.2f} км")
    if tot_dose > mission.max_dose_msv:
        warnings.append(f"Превышен лимит радиации: {tot_dose:.2f} мЗв > {mission.max_dose_msv:.2f} мЗв")
    if o2_left_pct < 20.0 or bat_left_pct < 20.0:
        warnings.append(f"Низкий остаток СЖО после возврата (O2: {o2_left_pct:.1f}%, BAT: {bat_left_pct:.1f}%)")

    return MissionPlan(
        feasible=len(warnings) == 0,
        warnings=warnings,
        visited_targets=[t.name for t in targets],
        segments=segments,
        total_distance_km=tot_dist_km,
        total_elevation_gain_m=tot_gain_m,
        max_slope_deg=max_slope,
        total_time_hours=tot_hours,
        o2_remaining_pct=o2_left_pct,
        battery_remaining_pct=bat_left_pct,
        total_dose_msv=tot_dose,
    )


def render_route_preview(site: Site, plan: MissionPlan, out_path: str | Path) -> Path:
    """Визуализация рыхлости грунта, профиля высот и построенного маршрута."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    terrain = build_astronaut_terrain(site)
    w, h = site.extent_m
    extent = (0, w, h, 0)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    #  Looseness map
    im0 = axes[0].imshow(terrain.looseness, cmap="YlOrBr", vmin=0, vmax=1, extent=extent)
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    axes[0].set_title("Soil Looseness (0 = Bedrock, 1 = Loose Sand)")


    im1 = axes[1].imshow(site.layers["elevation"], cmap="terrain", extent=extent)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    for poly in site.no_go:
        pxs, pys = zip(*(poly + poly[:1]))
        axes[1].plot(pxs, pys, color="red", lw=1.2)
    colors = ["cyan", "lime", "orange", "magenta"]
    for idx, seg in enumerate(plan.segments):
        xs, ys = zip(*seg.points_m)
        axes[1].plot(xs, ys, color=colors[idx % len(colors)], lw=2.2, label=f"Leg {idx+1}")
    axes[1].plot(*site.start, marker="^", color="white", mec="k", ms=9)
    for t in site.targets:
        axes[1].plot(t.x_m, t.y_m, marker="*", color="yellow", mec="k", ms=10)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_title(f"EVA Route ({plan.total_distance_km:.2f} km, {plan.total_time_hours:.1f} h)")

    # Height profile 
    full_prof = [pt for seg in plan.segments for pt in seg.profile]
    dists = np.cumsum([0.0] + [
        math.hypot(full_prof[i]["x_m"] - full_prof[i - 1]["x_m"], full_prof[i]["y_m"] - full_prof[i - 1]["y_m"])
        for i in range(1, len(full_prof))
    ])
    elevs = [p["elev_m"] for p in full_prof]
    looses = [p["looseness"] for p in full_prof]

    ax2 = axes[2]
    ax2.plot(dists, elevs, color="#fb923c", lw=2, label="Elevation (m)")
    ax2.set_xlabel("Cumulative distance, m")
    ax2.set_ylabel("Elevation, m", color="#fb923c")
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(dists, 0, looses, color="#38bdf8", alpha=0.25, label="Soil Looseness")
    ax2_twin.set_ylabel("Looseness (0..1)", color="#38bdf8")
    ax2_twin.set_ylim(0, 1.1)
    ax2.set_title(f"O2 left: {plan.o2_remaining_pct:.0f}% | Max slope: {plan.max_slope_deg:.1f}°")

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    from mars_map import STUB_DIR, load_stub

    site = load_stub()
    plan = plan_eva_mission(site)
    print(f"Цели: {plan.visited_targets}")
    print(f"Дистанция: {plan.total_distance_km:.2f} км | Время: {plan.total_time_hours:.2f} ч")
    print(f"Набор высоты: +{plan.total_elevation_gain_m:.1f} м | Макс. уклон: {plan.max_slope_deg:.1f}°")
    print(f"Остаток O2: {plan.o2_remaining_pct:.1f}% | Батарея: {plan.battery_remaining_pct:.1f}%")
    print(f"Статус миссии: {'OK' if plan.feasible else plan.warnings}")

    img_path = render_route_preview(site, plan, STUB_DIR / "astronaut_route_preview.png")
    plan.to_json(STUB_DIR / "astronaut_plan.json")
    print(f"Сохранено: {img_path}")