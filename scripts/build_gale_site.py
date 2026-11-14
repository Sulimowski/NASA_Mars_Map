"""Parts A and B together on real data:

    data/catalog.sqlite + data/processed/  (written by scripts/ingest_gale.py)
        -> fusion -> derived layers -> data/gale_real/  (a Site, same format as the stub)

    .venv\\Scripts\\python scripts\\ingest_gale.py       # part A, once
    .venv\\Scripts\\python scripts\\build_gale_site.py   # part B on its output
"""

import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mars_map import DATA_DIR, TERRAIN_CLASSES  # noqa: E402
from mars_map.catalog.db import CATALOG_DB  # noqa: E402
from mars_map.fusion import add_derived, fuse_region  # noqa: E402
from mars_map.grid import GALE_MOUNT_SHARP  # noqa: E402

OUT = DATA_DIR / "gale_real"
CLASS_CMAP = ListedColormap(["#c9b18f", "#f2d36b", "#8a5a44", "#222222"])


def pick_start(site):
    """No base location is defined yet: take the easiest walkable cell near the centre."""
    cost = site.layers["cost"]
    rows, cols = np.indices(site.shape)
    w, h = site.extent_m
    dist = np.hypot((cols + 0.5) * site.resolution_m - w / 2, (rows + 0.5) * site.resolution_m - h / 2)
    score = np.where(np.isfinite(cost) & (dist < 0.15 * w), cost + dist / (0.15 * w), np.inf)
    r, c = np.unravel_index(np.argmin(score), score.shape)
    return site.to_metres(int(r), int(c))


def hillshade(elev, res, az=315.0, alt=35.0):
    dy, dx = np.gradient(elev.astype(np.float64), res)
    slope, aspect = np.arctan(np.hypot(dx, dy)), np.arctan2(dy, -dx)
    a, z = np.radians(360 - az + 90), np.radians(90 - alt)
    return np.clip(np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect), 0, 1)


def figure(site, path):
    L = site.layers
    w, h = site.extent_m
    ext = (0, w, h, 0)
    cost = np.where(np.isfinite(L["cost"]), L["cost"], np.nan)
    panels = [
        ("Elevation, m (hillshaded)", L["elevation"], "terrain", {}),
        ("Slope, deg", L["slope"], "magma", {"vmax": 40}),
        ("Elevation source resolution, m", L["source_res"], "viridis_r", {}),
        ("Terrain class", L["terrain_class"], CLASS_CMAP, {"vmin": -0.5, "vmax": 3.5}),
        ("Cost (x slower than flat), blocked = black", cost, "YlOrRd", {"vmax": np.nanpercentile(cost, 99)}),
        ("Roughness, m", L["roughness"], "viridis", {"vmax": np.percentile(L["roughness"], 99)}),
    ]
    for key, title in (("themis_night_ir", "THEMIS night IR, DN (warm = rock)"),
                       ("themis_day_ir", "THEMIS day IR, DN"), ("albedo", "TES albedo")):
        if key in L and len(panels) < 8:
            panels.append((title, L[key], "inferno" if "themis" in key else "gray", {}))
    while len(panels) < 8:
        panels.append((None, None, None, None))

    fig, axes = plt.subplots(2, 4, figsize=(21, 10.5))
    shade = hillshade(L["elevation"], site.resolution_m)
    for ax, (title, data, cmap, kw) in zip(axes.flat, panels):
        if title is None:
            ax.axis("off")
            continue
        if cmap == "YlOrRd":
            ax.set_facecolor("black")
        im = ax.imshow(data, cmap=cmap, extent=ext, **kw)
        if title.startswith("Elevation, m"):
            ax.imshow(shade, cmap="gray", extent=ext, alpha=0.45, vmin=0, vmax=1)
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        if title == "Terrain class":
            cb.set_ticks(list(TERRAIN_CLASSES))
            cb.ax.set_yticklabels(list(TERRAIN_CLASSES.values()))
        ax.plot(*site.start, marker="^", color="cyan", ms=9, mec="k")
        ax.set_title(title)
        ax.set_xlabel("x, m (east)")
        ax.set_ylabel("y, m (south)")
    fig.suptitle(f"{site.name}  |  {site.source}  |  {site.resolution_m} m/px, {w:.0f}x{h:.0f} m  |  "
                 f"centre {site.center_lat:.3f}N {site.center_lon:.3f}E  |  ^ start", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    if not CATALOG_DB.exists():
        sys.exit(f"{CATALOG_DB} not found: run scripts/ingest_gale.py first")
    g = GALE_MOUNT_SHARP

    db = sqlite3.connect(CATALOG_DB)
    print("layers in the catalog:")
    for name, src, res, cov in db.execute(
            "SELECT l.name, s.name, l.native_res_m, l.coverage_frac FROM layers l "
            "JOIN sources s ON s.id=l.source_id JOIN regions r ON r.id=l.region_id "
            "WHERE r.name=? ORDER BY l.name, l.native_res_m", (g.name,)):
        print(f"  {name:16s} {src:14s} {res:7.1f} m  cover {cov:6.1%}")
    db.close()

    centre = (g.size_m / 2, g.size_m / 2)
    site = add_derived(fuse_region(CATALOG_DB, g.name, start=centre, targets=[]))
    site.start = pick_start(site)
    site.save(OUT)
    figure(site, OUT / "preview.png")

    L = site.layers
    print(f"\nsite saved: {OUT}")
    print(f"  layers: {', '.join(sorted(L))}")
    print(f"  elevation {L['elevation'].min():.1f} .. {L['elevation'].max():.1f} m "
          f"(relief {np.ptp(L['elevation']):.1f} m)")
    print(f"  slope: median {np.median(L['slope']):.1f}, p90 {np.percentile(L['slope'], 90):.1f}, "
          f"p99 {np.percentile(L['slope'], 99):.1f} deg")
    res, counts = np.unique(L["source_res"], return_counts=True)
    print("  elevation source: " + ", ".join(f"{r:.0f} m {c / L['slope'].size:.1%}" for r, c in zip(res, counts)))
    print("  terrain: " + ", ".join(f"{n} {(L['terrain_class'] == k).mean():.1%}"
                                    for k, n in TERRAIN_CLASSES.items()))
    ok = np.isfinite(L["cost"])
    print(f"  blocked {L['blocked'].mean():.1%} | cost median {np.median(L['cost'][ok]):.2f}, "
          f"p95 {np.percentile(L['cost'][ok], 95):.2f}")
    print(f"  start {tuple(round(v) for v in site.start)} m, targets: {len(site.targets)}")
    for k, v in site.series.items():
        if k != "sol":
            print(f"  series {k}: {len(v)} sols, {min(v):.2f} .. {max(v):.2f}")
    if site.series:
        print(f"  common sols: {site.series['sol'][0]} .. {site.series['sol'][-1]}")


if __name__ == "__main__":
    main()
