"""Train and test the dust devil model, then predict for our map region.

    .venv\\Scripts\\python -m hazards.train_dust_devils

Tests, all on stations the model did not see while training:
  1. train on InSight, predict Perseverance; and the other way round
  2. train on both, predict Curiosity at Gale (if its pressure data is downloaded, see rems.py)

Writes into hazards/data/:
  dust_devil_scores.csv          every test, model against the baselines
  dust_devil_tests.png           observed against predicted, by hour and by season
  gale_boxwork_dust_devils.csv   the product: expected vortices per hour for every season and hour
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "hazards"

from .dataset import (DAY_HOURS, FEATURES, MIN_DROP_PA, build_table, climate_by_hour,   # noqa: E402
                      curiosity_rates)
from .marstime import MARS_HOUR_IN_EARTH_HOURS                                               # noqa: E402
from .model import DiurnalTable, DustDevilModel, PhysicsIndex, score                         # noqa: E402
from .rems import STORE, pick_sols                                                           # noqa: E402
from .sites import SITES                                                                     # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
MODELS = {"network": DustDevilModel, "diurnal table": DiurnalTable, "physics index": PhysicsIndex}
COLOURS = {"observed": "#222222", "network": "#c0392b", "diurnal table": "#999999", "physics index": "#2f6f73"}


def curiosity_sols() -> list[int]:
    return [s for s in pick_sols(4176, 4843, every=6) if (STORE / f"sol{s:05d}.npz").exists()]


def run_test(name: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[dict], dict]:
    rows, preds = [], {}
    for label, cls in MODELS.items():
        rate = cls().fit(train).predict(test)
        preds[label] = rate
        rows.append({"test": name, "model": label, **score(test, rate)})
    return rows, preds


def by_hour(test, rate=None):
    e = test.groupby("hour")["exposure_h"].sum()
    c = test.groupby("hour")["count"].sum() if rate is None else (test.assign(x=rate * test["exposure_h"])
                                                                   .groupby("hour")["x"].sum())
    return (c / e).reindex(range(24))


def by_season(test, rate=None):
    """Vortices per sol in each season bin (rate summed over the 24 hours of a sol)."""
    per_hour = test.assign(r=(test["count"] / test["exposure_h"]) if rate is None else rate)
    return per_hour.groupby("ls_from")["r"].sum() * MARS_HOUR_IN_EARTH_HOURS


def plot_tests(tests: list[tuple[str, pd.DataFrame, dict]], out: Path) -> None:
    fig, axes = plt.subplots(2, len(tests), figsize=(4.6 * len(tests), 7.2), squeeze=False)
    for j, (name, test, preds) in enumerate(tests):
        ax = axes[0, j]
        hours = np.arange(24) + 0.5
        obs = by_hour(test)
        err = np.sqrt(test.groupby("hour")["count"].sum().clip(lower=1)) / test.groupby("hour")["exposure_h"].sum()
        ax.errorbar(hours, obs, yerr=err.reindex(range(24)), fmt="o", ms=3.5, color=COLOURS["observed"],
                    label="observed", zorder=5)
        for label, rate in preds.items():
            ax.plot(hours, by_hour(test, rate), lw=1.8 if label == "network" else 1.2, color=COLOURS[label], label=label)
        ax.set_xlim(6, 20)
        ax.set_ylim(bottom=0)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel("local solar hour")
        if j == 0:
            ax.set_ylabel("vortices per Earth hour")
            ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        ax = axes[1, j]
        seasons = by_season(test)
        ax.plot(seasons.index + 15, seasons.values, "o", ms=4, color=COLOURS["observed"])
        for label, rate in preds.items():
            s = by_season(test, rate)
            ax.plot(s.index + 15, s.values, lw=1.8 if label == "network" else 1.2, color=COLOURS[label])
        ax.set_xlim(0, 360)
        ax.set_xticks(range(0, 361, 90))
        ax.set_ylim(bottom=0)
        ax.set_xlabel("season, Ls (deg)")
        if j == 0:
            ax.set_ylabel("vortices per sol")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Dust devil model on stations it never saw (pressure drop ≥ {MIN_DROP_PA} Pa)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    table = build_table(["insight", "perseverance"])
    insight, jezero = table[table.station == "insight"], table[table.station == "perseverance"]
    print(f"training rows: InSight {len(insight)} ({insight['count'].sum()} vortices), "
          f"Perseverance {len(jezero)} ({jezero['count'].sum()} vortices)")

    scores, tests = [], []
    for name, train, test in [("InSight → Perseverance", insight, jezero), ("Perseverance → InSight", jezero, insight)]:
        rows, preds = run_test(name, train, test)
        scores += rows
        tests.append((name, test, preds))

    sols = curiosity_sols()
    gale = None
    if sols:
        gale, events = curiosity_rates(sols)
        gale = gale.merge(climate_by_hour("curiosity_boxwork"), on=["ls_from", "hour"], how="left")
        day = events[events.ltst_h.astype(int).isin(DAY_HOURS)] if len(events) else events
        print(f"Curiosity: {len(sols)} sols, {gale['exposure_h'].sum():.0f} daytime hours listened, "
              f"{len(day)} daytime drops >= 0.3 Pa, {int((day.drop_pa >= MIN_DROP_PA).sum())} >= {MIN_DROP_PA} Pa")
        rows, preds = run_test("both → Curiosity (Gale)", table, gale)
        scores += rows
        tests.append(("both → Curiosity (Gale)", gale, preds))

    scores = pd.DataFrame(scores)
    scores.to_csv(OUT / "dust_devil_scores.csv", index=False)
    print(scores.round(3).to_string(index=False))
    plot_tests(tests, OUT / "dust_devil_tests.png")

    # The product. The network trained on InSight and Perseverance gives the shape over hours and
    # seasons; Curiosity's own count at Gale sets the level, because no model here sees that the
    # crater suppresses convection (all of them predicted about three times too many).
    final = DustDevilModel().fit(table)
    region = climate_by_hour("curiosity_boxwork")
    raw = np.where(region["hour"].isin(DAY_HOURS), final.predict(region), 0.0)
    calibration = 1.0
    if gale is not None and gale["count"].sum() > 0:
        calibration = gale["count"].sum() / (final.predict(gale) * gale["exposure_h"]).sum()
    region["vortices_per_h_uncalibrated"] = raw
    region["vortices_per_h"] = raw * calibration
    region["site"] = "gale_boxwork"
    print(f"\nGale calibration factor (observed / predicted): {calibration:.2f}")
    keep = ["site", "ls_from", "hour", "vortices_per_h", "vortices_per_h_uncalibrated", *FEATURES]
    region[keep].round(4).to_csv(OUT / "gale_boxwork_dust_devils.csv", index=False)
    peak = region.loc[region["vortices_per_h"].idxmax()]
    print(f"\ngale_boxwork: peak {peak.vortices_per_h:.3f} vortices/h at Ls {peak.ls_from:.0f}-{peak.ls_from + 30:.0f}, "
          f"hour {int(peak.hour)}; per sol {by_season(region.assign(count=0, exposure_h=1), region.vortices_per_h).round(2).to_dict()}")
    print(OUT / "dust_devil_tests.png")


if __name__ == "__main__":
    main()
