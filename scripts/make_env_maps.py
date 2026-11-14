"""Render the radiation and temperature maps of a site.

    python scripts/make_env_maps.py                    # data/gale_stub/env_preview.png
    python scripts/make_env_maps.py --sol 23           # temperatures for another sol
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mars_map import Environment, Site  # noqa: E402
from mars_map.preview import _overlay  # noqa: E402
from mars_map.site import resolve_path  # noqa: E402

TIMES_H = [5.5, 9.0, 13.0, 19.0]          # local true solar time of the temperature panels
SHADOW_TIME_H = 8.0


def _panel(fig, ax, site, data, title, cmap, **kw):
    w, h = site.extent_m
    im = ax.imshow(data, cmap=cmap, extent=(0, w, h, 0), **kw)
    fig.colorbar(im, ax=ax, shrink=0.85)
    _overlay(ax, site)
    ax.set_title(title)
    ax.set_xlabel("x, m (east)")
    ax.set_ylabel("y, m (south)")


def render(site: Site, env: Environment, sol: int, out: Path) -> Path:
    doses = site.series["radiation_msv_per_sol"]
    sols = site.series.get("sol") or list(range(len(doses)))
    quiet_sol = sols[int(np.argsort(doses)[len(doses) // 2])]
    event_sol = sols[int(np.argmax(doses))]

    fig, axes = plt.subplots(2, 4, figsize=(21, 10.5))
    top, bottom = axes

    _panel(fig, top[0], site, env.sky_view, "Sky view (1 = open sky)", "viridis")
    for ax, s, label in ((top[1], quiet_sol, "typical sol"), (top[2], event_sol, "worst sol")):
        _panel(fig, ax, site, env.dose_rate(s),
               f"Dose rate, µSv/h — sol {s} ({label})", "inferno")
    _panel(fig, top[3], site, env.direct_sunlight(sol, SHADOW_TIME_H),
           f"Direct sunlight, W/m² — sol {sol}, {SHADOW_TIME_H:04.1f} h", "gray")

    for ax, t in zip(bottom, TIMES_H):
        _panel(fig, ax, site, env.temperature(sol, t),
               f"Ground temperature, °C — sol {sol}, {t:04.1f} h", "coolwarm")

    fig.suptitle(f"{site.name} — {site.source}  |  radiation and temperature model  |  "
                 f"season Ls {env.ls(sol):.0f}°, dust tau {env.series_value('dust_opacity_tau', sol)}"
                 f"  |  hours are local true solar time", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="data/gale_stub")
    ap.add_argument("--sol", type=int, default=0, help="sol for the temperature and sunlight panels")
    args = ap.parse_args()

    folder = resolve_path(args.site)
    site = Site.load(folder)
    env = Environment(site)
    png = render(site, env, args.sol, folder / "env_preview.png")
    print(f"preview {png}")

    dose = env.dose_rate(args.sol)
    print(f"sol {args.sol}: Ls {env.ls(args.sol):.1f}, "
          f"particle event: {env.is_particle_event(args.sol)}")
    print(f"  sky view        {env.sky_view.min():.2f} .. {env.sky_view.max():.2f}")
    print(f"  dose rate       {dose.min():.1f} .. {dose.max():.1f} µSv/h")
    for t in TIMES_H:
        temp = env.temperature(args.sol, t)
        print(f"  temperature {t:04.1f} h  {temp.min():6.1f} .. {temp.max():6.1f} °C "
              f"(median {np.median(temp):.1f})")
    for t in site.targets:
        r, c = site.to_pixel(t.x_m, t.y_m)
        print(f"  {t.name:22s} sky {env.sky_view[r, c]:.2f}  dose {dose[r, c]:.1f} µSv/h  "
              f"T {env.temperature(args.sol, 5.5)[r, c]:.0f} / "
              f"{env.temperature(args.sol, 13.0)[r, c]:.0f} °C (dawn / 13 h)")


if __name__ == "__main__":
    main()
