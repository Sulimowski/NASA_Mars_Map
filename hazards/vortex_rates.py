"""First look at the dust devil data: how often a vortex hits a point, by season and hour.

    .venv\\Scripts\\python -m hazards.vortex_rates

Downloads the vortex catalogs on the first run (InSight 96 KB from NASA PDS, Perseverance
0.8 MB from Zenodo) and writes into hazards/data/:

    vortex_rates.csv                 every station, season bin and local hour
    <station>_vortex_rates.png       season x hour table of one station
    vortex_rates_by_hour.png         the stations side by side over the day
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):                    # run as a file: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "hazards"

from .marstime import mars_ls                                       # noqa: E402
from .vortices import (COMMON_MIN_DROP_PA, HAZARD_DIR, STATIONS, coverage_of,   # noqa: E402
                       load, rate_table)

HOURS = range(7, 19)
COLOURS = {"insight": "#b5651d", "perseverance": "#2f6f73"}
NOTES = {
    "insight": "Lorenz et al., NASA PDS, doi:10.17189/71eb-n087. Assumes the sensor recorded all day on every sol.",
    "perseverance": "Hueso et al. 2023, doi:10.5281/zenodo.7315863. Recording time is estimated from the catalog; "
                    "rates are lower bounds.",
}


def check_ls(events) -> float:
    """Largest difference between our Ls and the Ls written in the catalog, degrees."""
    ours = mars_ls(events["utc"].dt.tz_localize(None).to_numpy())
    diff = (ours - events["ls_deg"].to_numpy() + 180.0) % 360.0 - 180.0
    return float(np.abs(diff).max())


def plot_station(station: str, rates, events, out: Path) -> None:
    first = events.sort_values("sol")["ls_deg"].iloc[0] // 30 * 30
    seasons = sorted(rates["ls_from"].unique(), key=lambda v: (v - first) % 360.0)      # in mission order
    cell = rates.set_index(["ls_from", "hour"])
    grid = np.array([[cell.loc[(s, h), "rate_per_h"] for h in HOURS] for s in seasons])
    counts = np.array([[cell.loc[(s, h), "count"] for h in HOURS] for s in seasons])
    n_sols = [int(cell.loc[(s, 12), "n_sols"]) for s in seasons]
    per_sol = np.array([rates[rates.ls_from == s]["rate_per_h"].sum() * 1.0275 for s in seasons])

    fig, (ax, side) = plt.subplots(1, 2, figsize=(11.5, 0.5 * len(seasons) + 1.6), width_ratios=[4.2, 1], sharey=True)
    im = ax.imshow(grid, cmap="YlOrBr", aspect="auto", vmin=0)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if counts[i, j]:
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if grid[i, j] > 0.6 * grid.max() else "#3a2a1a")
    ax.set_xticks(range(len(HOURS)), [f"{h}" for h in HOURS])
    ax.set_yticks(range(len(seasons)), [f"Ls {int(s)}–{int(s) + 30}   ({n} sols)" for s, n in zip(seasons, n_sols)])
    ax.set_xlabel("Local true solar time, hour")
    info = STATIONS[station]
    ax.set_title(f"{info['name']}, {info['place']}: vortex encounters per hour "
                 f"(pressure drop ≥ {COMMON_MIN_DROP_PA} Pa)", loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, pad=0.015, label="encounters per Earth hour")
    side.barh(range(len(seasons)), per_sol, color=COLOURS[station])
    side.set_xlabel("encounters per sol")
    side.spines[["top", "right"]].set_visible(False)
    strong = events[events.drop_pa >= COMMON_MIN_DROP_PA]
    fig.text(0.01, 0.01, f"{len(strong)} events, sols {events.sol.min()}–{events.sol.max()}. {NOTES[station]}",
             fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_by_hour(rates, out: Path) -> None:
    """All seasons together: is the shape of the day the same at different places?"""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for station, part in rates.groupby("station"):
        by_hour = part.groupby("hour").agg(count=("count", "sum"), exposure_h=("exposure_h", "sum")).loc[list(HOURS)]
        rate = by_hour["count"] / by_hour["exposure_h"]
        err = np.sqrt(by_hour["count"].clip(lower=1)) / by_hour["exposure_h"]
        ax.errorbar(by_hour.index + 0.5, rate, yerr=err, marker="o", ms=4, lw=1.6, capsize=2,
                    color=COLOURS[station], label=f"{STATIONS[station]['name']}, {STATIONS[station]['place']}")
    ax.set_xlabel("Local true solar time, hour")
    ax.set_ylabel("vortex encounters per Earth hour")
    ax.set_title(f"Vortices with a pressure drop ≥ {COMMON_MIN_DROP_PA} Pa passing over one point", loc="left", fontsize=11)
    ax.set_xticks(list(HOURS))
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.text(0.01, 0.01, "All observed seasons together. Perseverance: recording time estimated, rates are lower bounds.",
             fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    HAZARD_DIR.mkdir(parents=True, exist_ok=True)
    tables = []
    for station in STATIONS:
        events = load(station)
        coverage = coverage_of(station, events)
        rates = rate_table(events, coverage)
        tables.append(rates)
        plot_station(station, rates, events, HAZARD_DIR / f"{station}_vortex_rates.png")

        strong = events[events.drop_pa >= COMMON_MIN_DROP_PA]
        listened = coverage["listened"]
        print(f"\n== {STATIONS[station]['name']}, {STATIONS[station]['place']}")
        print(f"drops in the catalog {len(events)}, of them >= {COMMON_MIN_DROP_PA} Pa: {len(strong)}; "
              f"sols {events.sol.min()}..{events.sol.max()} ({len(coverage)} sols)")
        print(f"season: Ls {events.sort_values('sol').ls_deg.iloc[0]:.0f} -> {events.sort_values('sol').ls_deg.iloc[-1]:.0f}; "
              f"our Ls against the catalog: max difference {check_ls(events):.2f} deg")
        print(f"listened: {listened.mean():.0%} of the time on average (min {listened.min():.0%}, max {listened.max():.0%})")
        print(f"drop, Pa (>= {COMMON_MIN_DROP_PA}): median {strong.drop_pa.median():.2f}, max {strong.drop_pa.max():.2f}")
        dusty = strong["dust_dip_pct"].gt(0).sum()
        if dusty:
            print(f"carried dust (sunlight dimmed): {dusty} of {len(strong)} = {dusty / len(strong):.0%}")
        by_season = rates.groupby("ls_from").agg(n_sols=("n_sols", "first"), count=("count", "sum"),
                                                 per_sol=("rate_per_h", lambda r: r.sum() * 1.0275))
        print(by_season.round(2).to_string())

    rates = pd.concat(tables, ignore_index=True)
    rates.to_csv(HAZARD_DIR / "vortex_rates.csv", index=False)
    plot_by_hour(rates, HAZARD_DIR / "vortex_rates_by_hour.png")
    print(f"\n{HAZARD_DIR / 'vortex_rates_by_hour.png'}")


if __name__ == "__main__":
    main()
