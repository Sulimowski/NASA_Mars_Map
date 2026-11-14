"""Ground model for an astronaut on foot: how deep the boot sinks, how much the
terrain slows the walk and raises its energy cost, and where nobody may step.

Everything is a raster of site.shape. The router reads these rasters and nothing
else, so weather (radiation, ground temperature) is folded in here as well.

Soil: Bekker-Wong pressure-sinkage  p = (k_c / b + k_phi) * z^n.  The boot of a
suited astronaut presses with p = weight on Mars / sole area; the soil moduli are
interpolated from thermal inertia (low inertia = loose dust or sand, high =
cemented soil or rock). The sinkage z then sets the Pandolf terrain coefficient.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mars_map.biomechanics import MARS_GRAVITY, SuitParams, heater_power
    from mars_map.environment import TYPICAL_DOSE_MSV_PER_SOL
    from mars_map.site import Site
    from mars_map.sun import SOL_HOURS
else:
    from .biomechanics import MARS_GRAVITY, SuitParams, heater_power
    from .environment import TYPICAL_DOSE_MSV_PER_SOL
    from .site import Site
    from .sun import SOL_HOURS

# Thermal inertia (J m^-2 K^-1 s^-1/2) of the loosest and the firmest soil.
TI_LOOSE, TI_FIRM = 100.0, 450.0
# Bekker-Wong moduli for those two ends: k_phi in kN/m^(n+2), exponent n.
K_PHI_LOOSE, K_PHI_FIRM = 400.0, 6000.0
N_LOOSE, N_FIRM = 1.1, 1.0
K_C = 1.0                       # cohesive modulus, kN/m^(n+1): nearly cohesionless regolith
SINKAGE_REF_CM = 5.0            # sinkage of loose sand, where the looseness index reaches 1


@dataclass
class AstronautTerrainModel:
    """Rasters of site.shape describing the ground for a walking astronaut."""
    sinkage: np.ndarray            # 0..1: risk of getting stuck in loose sand / dust
    rock_impedance: np.ndarray     # 0..1: boulders and micro-relief in the way
    looseness: np.ndarray          # 0..1: overall soil resistance index
    terramech_factor: np.ndarray   # 1.0..2.6: Pandolf terrain coefficient (energy multiplier)
    speed_multiplier: np.ndarray   # 0.45..1.0: walking speed relative to firm ground
    impassable: np.ndarray         # bool: cliffs, terrain class 3, no-go polygons
    nodata_penalty: np.ndarray     # 1.0, or 1 + weight where only coarse data exists
    # --- added with the Bekker-Wong model and the weather coupling ---
    sinkage_cm: np.ndarray | None = None        # boot sinkage in centimetres
    no_go: np.ndarray | None = None             # bool: inside a no-go polygon
    dose_rate_msv_h: np.ndarray | None = None   # radiation dose rate, mSv per Earth hour
    ground_temp_c: np.ndarray | None = None     # None when the site has no temperature
    heater_w: np.ndarray | None = None          # suit heater power the ground temperature asks for
    radiation_is_map: bool = False              # False = uniform fallback from the series


def rasterize_nogo(
    shape: tuple[int, int],
    resolution_m: float,
    no_go: list[list[tuple[float, float]]],
) -> np.ndarray:
    """Boolean mask of the cells whose centre lies inside a no-go polygon."""
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=bool)
    if not no_go:
        return mask

    xs = (np.arange(cols, dtype=np.float32) + 0.5) * resolution_m
    ys = (np.arange(rows, dtype=np.float32) + 0.5) * resolution_m

    for poly in no_go:
        if len(poly) < 3:
            continue
        px = np.array([p[0] for p in poly], dtype=np.float32)
        py = np.array([p[1] for p in poly], dtype=np.float32)

        c0 = max(0, int(px.min() / resolution_m) - 1)
        c1 = min(cols, int(px.max() / resolution_m) + 2)
        r0 = max(0, int(py.min() / resolution_m) - 1)
        r1 = min(rows, int(py.max() / resolution_m) + 2)
        if c0 >= c1 or r0 >= r1:
            continue

        gx, gy = np.meshgrid(xs[c0:c1], ys[r0:r1])
        inside = np.zeros(gx.shape, dtype=bool)
        j = len(poly) - 1
        for i in range(len(poly)):
            yi, yj = py[i], py[j]
            xi, xj = px[i], px[j]
            cond = ((yi > gy) != (yj > gy)) & (
                gx < (xj - xi) * (gy - yi) / (yj - yi + 1e-12) + xi
            )
            inside ^= cond
            j = i
        mask[r0:r1, c0:c1] |= inside

    return mask


def bekker_sinkage_cm(thermal_inertia: np.ndarray, sand: np.ndarray,
                      suit: SuitParams = SuitParams()) -> np.ndarray:
    """Static boot sinkage in centimetres from the Bekker-Wong pressure-sinkage law."""
    firmness = np.clip((thermal_inertia - TI_LOOSE) / (TI_FIRM - TI_LOOSE), 0.0, 1.0)
    firmness = np.where(sand, np.minimum(firmness, 0.1), firmness)    # mapped sand is never firm
    k_phi = np.exp(np.log(K_PHI_LOOSE) + firmness * (np.log(K_PHI_FIRM) - np.log(K_PHI_LOOSE)))
    n = N_LOOSE + firmness * (N_FIRM - N_LOOSE)
    pressure_kpa = suit.total_mass_kg * MARS_GRAVITY / suit.boot_area_m2 / 1000.0
    z_m = (pressure_kpa / (K_C / suit.boot_width_m + k_phi)) ** (1.0 / n)
    return (z_m * 100.0).astype(np.float32)


def uniform_dose_rate_msv_h(site: Site, sol_index: int = 0) -> float:
    """Site-wide dose rate from the series, mSv per Earth hour (typical value if absent)."""
    series = site.series.get("radiation_msv_per_sol")
    per_sol = float(series[sol_index]) if series else TYPICAL_DOSE_MSV_PER_SOL
    return per_sol / SOL_HOURS


def build_astronaut_terrain(
    site: Site,
    nodata_risk_weight: float = 0.25,
    suit: SuitParams | None = None,
    dose_rate_msv_h: np.ndarray | None = None,
    ground_temp_c: np.ndarray | None = None,
) -> AstronautTerrainModel:
    """Build the ground model of a site.

    dose_rate_msv_h -- radiation map in mSv per Earth hour (Environment.dose_rate() / 1000).
                       Without it the site-wide series value is used everywhere.
    ground_temp_c   -- ground temperature map in degrees C (Environment.temperature()).
                       Without it the suit heater is assumed off.
    """
    suit = suit or SuitParams()
    L = site.layers
    ti = L["thermal_inertia"].astype(np.float32)
    rock = L["rock_abundance"].astype(np.float32)
    rough = L["roughness"].astype(np.float32)
    slope = L["slope"].astype(np.float32)
    tclass = L["terrain_class"]
    nodata = L["nodata"].astype(bool)

    # 1. Sinkage: how deep the boot goes, then as a 0..1 index.
    sinkage_cm = bekker_sinkage_cm(ti, tclass == 1, suit)
    sinkage = np.clip(sinkage_cm / SINKAGE_REF_CM, 0.0, 1.0)

    # 2. Rocks and micro-relief.
    rough_norm = np.clip(rough / 0.45, 0.0, 1.0)
    rock_impedance = np.clip(0.65 * rock + 0.35 * rough_norm, 0.0, 1.0)
    rock_impedance = np.where(tclass == 2, np.maximum(rock_impedance, 0.55), rock_impedance)

    looseness = np.clip(0.58 * sinkage + 0.42 * rock_impedance, 0.0, 1.0).astype(np.float32)

    # 3. Coefficients for the walk. Pandolf: firm ground 1.0, loose sand 2.1; a
    #    boulder field adds up to 0.5 (comparable to heavy brush, 1.5).
    terramech_factor = np.clip(1.0 + 1.1 * sinkage + 0.5 * rock_impedance, 1.0, 2.6)
    speed_multiplier = np.clip(1.0 - 0.50 * looseness, 0.45, 1.0)

    # 4. Where nobody may step.
    nogo_mask = rasterize_nogo(site.shape, site.resolution_m, site.no_go)
    impassable = (tclass == 3) | (slope > site.mission.hard_max_slope_deg) | nogo_mask
    if "blocked" in L:                               # hard-limit mask from the data pipeline
        impassable |= L["blocked"].astype(bool)

    nodata_penalty = np.where(nodata, 1.0 + nodata_risk_weight, 1.0)

    # 5. Weather.
    radiation_is_map = dose_rate_msv_h is not None
    if radiation_is_map:
        dose = np.asarray(dose_rate_msv_h, dtype=np.float32)
        if dose.shape != site.shape:
            raise ValueError(f"dose-rate map {dose.shape} does not match the site {site.shape}")
    else:
        dose = np.full(site.shape, uniform_dose_rate_msv_h(site), np.float32)
    if ground_temp_c is not None:
        ground_temp_c = np.asarray(ground_temp_c, dtype=np.float32)
        if ground_temp_c.shape != site.shape:
            raise ValueError(f"temperature map {ground_temp_c.shape} does not match the site {site.shape}")
        heater = heater_power(ground_temp_c, suit).astype(np.float32)
    else:
        heater = np.zeros(site.shape, np.float32)

    return AstronautTerrainModel(
        sinkage=sinkage.astype(np.float32),
        rock_impedance=rock_impedance.astype(np.float32),
        looseness=looseness,
        terramech_factor=terramech_factor.astype(np.float32),
        speed_multiplier=speed_multiplier.astype(np.float32),
        impassable=impassable,
        nodata_penalty=nodata_penalty.astype(np.float32),
        sinkage_cm=sinkage_cm,
        no_go=nogo_mask,
        dose_rate_msv_h=dose,
        ground_temp_c=ground_temp_c,
        heater_w=heater,
        radiation_is_map=radiation_is_map,
    )
