"""The training table for the dust devil model: one row per station, season bin and local hour.

    table = build_table(["insight", "perseverance"])      # labels + climate inputs

Columns:
    station, ls_from, hour          where and when (season bin of 30 deg, local true solar hour)
    count, exposure_h               vortices >= MIN_DROP_PA seen, Earth hours the sensor listened
    <inputs>                        climate from the Mars Climate Database at the middle of the
                                    season bin and the middle of the hour (see FEATURES)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import rems
from .mcd import mcd_year
from .sites import SITES
from .vortices import COMMON_MIN_DROP_PA, coverage_of, load, rate_table

LS_BIN = 30.0
MIN_DROP_PA = COMMON_MIN_DROP_PA

# Only daytime hours are labels. Convective vortices need a ground warmer than the air; at
# night the REMS and MEDA pressure shows other short drops (slope winds, turbulence, sensor
# noise) that a pressure detector cannot tell apart. Night rate is taken as zero.
DAY_HOURS = range(8, 18)

# Candidate model inputs (the network uses a subset, see model.DEFAULT_FEATURES). All of them
# exist for any point on Mars, which is what lets the model move to a place nobody measured.
# Hour and season are deliberately not inputs: the model must explain the daily and yearly
# cycle through the physics, not memorise the stations' clocks.
FEATURES = {
    "heat_up_wm2": "sensible heat flux from the ground into the air (W/m2): the engine of convection",
    "pbl_km": "height of the convective boundary layer (km): how tall a vortex can grow",
    "wstar_ms": "strongest updraft in the boundary layer (m/s)",
    "dT_k": "ground minus air temperature 2 m up (K): how unstable the lowest air is",
    "wind_ms": "wind 2 m above ground (m/s): shear that can tilt or start vortices",
    "ps_hpa": "surface pressure (hPa): denser air gives deeper pressure drops",
    "dust_tau": "dust optical depth: dust shades the ground and weakens heating",
}


def climate_by_hour(site: str) -> pd.DataFrame:
    """MCD inputs for every season bin and hour bin of a site (value at the middle of the hour)."""
    s = SITES[site]
    year = mcd_year(s["lat"], s["lon"], LS_BIN, verbose=False)
    rows = []
    for ls, day in year.groupby("ls_deg"):
        day = day.sort_values("hour").reset_index(drop=True)
        nxt = day.shift(-1).fillna(day.iloc[0])                        # hour 23 -> 0 wraps around
        mid = (day.drop(columns=["lat", "lon", "ls_deg"]) + nxt.drop(columns=["lat", "lon", "ls_deg"])) / 2
        rows.append(pd.DataFrame({
            "ls_from": ls - LS_BIN / 2, "hour": day["hour"].astype(int),
            "heat_up_wm2": -mid["heat_flux_wm2"],                      # MCD counts downward as positive
            "pbl_km": mid["pbl_m"] / 1000.0,
            "wstar_ms": mid["wstar_ms"],
            "dT_k": mid["tsurf_k"] - mid["tair_k"],
            "wind_ms": mid["wind_ms"],
            "ps_hpa": mid["ps_pa"] / 100.0,
            "dust_tau": mid["dust_tau"],
        }))
    return pd.concat(rows, ignore_index=True)


def curiosity_rates(sols: list[int], site: str = "curiosity_boxwork") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect vortices in the downloaded REMS sols and bin them like `rate_table`."""
    events, coverage = rems.detect_all(sols)
    exp = rems.hourly_exposure(coverage)
    exp["ls_from"] = (exp["ls_deg"] // LS_BIN) * LS_BIN
    table = exp.groupby(["ls_from", "hour"]).agg(exposure_h=("exposure_h", "sum"),
                                                 n_sols=("sol", "nunique")).reset_index()
    strong = events[events["drop_pa"] >= MIN_DROP_PA] if len(events) else events
    if len(strong):
        counts = (strong.assign(ls_from=(strong["ls_deg"] // LS_BIN) * LS_BIN, hour=strong["ltst_h"].astype(int))
                  .groupby(["ls_from", "hour"]).size().rename("count").reset_index())
        table = table.merge(counts, on=["ls_from", "hour"], how="left")
    else:
        table["count"] = 0
    table["count"] = table["count"].fillna(0).astype(int)
    table = table[(table["exposure_h"] > 0.05) & table["hour"].isin(DAY_HOURS)].copy()   # 3 min is not a measurement
    table["station"] = site
    table["rate_per_h"] = table["count"] / table["exposure_h"]
    return table, events


def station_rates(station: str) -> pd.DataFrame:
    events = load(station)
    return rate_table(events, coverage_of(station, events), LS_BIN, MIN_DROP_PA)


def build_table(stations: list[str], curiosity_sols: list[int] | None = None) -> pd.DataFrame:
    """Labels of the given stations joined with their climate inputs."""
    parts = []
    for st in stations:
        rates = station_rates(st)[["station", "ls_from", "hour", "count", "exposure_h"]]
        parts.append(rates.merge(climate_by_hour(st), on=["ls_from", "hour"], how="left"))
    if curiosity_sols:
        rates, _ = curiosity_rates(curiosity_sols)
        rates = rates[["station", "ls_from", "hour", "count", "exposure_h"]]
        parts.append(rates.merge(climate_by_hour("curiosity_boxwork"), on=["ls_from", "hour"], how="left"))
    table = pd.concat(parts, ignore_index=True)
    table = table[table["hour"].isin(DAY_HOURS)].reset_index(drop=True)
    missing = table[list(FEATURES)].isna().any(axis=1)
    if missing.any():
        raise RuntimeError(f"{int(missing.sum())} rows have no climate inputs")
    return table
