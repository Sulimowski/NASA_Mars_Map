"""Parts A and B together on real data:

    data/catalog.sqlite + data/processed/  (written by scripts/ingest_gale.py)
        -> fusion -> derived layers -> base + targets -> data/<region>/  (a Site, same format as the stub)

    .venv\\Scripts\\python scripts\\ingest_gale.py       # part A, once
    .venv\\Scripts\\python scripts\\build_gale_site.py   # part B on its output
"""

import argparse
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

import mars_map.ingest  # noqa: F401,E402  (OS certificate store for the rover feeds)
from mars_map import DATA_DIR, TERRAIN_CLASSES, Target  # noqa: E402
from mars_map.catalog.db import CATALOG_DB  # noqa: E402
from mars_map.fusion import add_derived, fuse_region  # noqa: E402
from mars_map.fusion.derived import reachable_mask, snap_to  # noqa: E402
from mars_map.fusion.science import MINERAL_NAMES, mineral_targets, pick_base  # noqa: E402
from mars_map.grid import REGIONS  # noqa: E402
from mars_map.ingest.fetch import FetchError  # noqa: E402
from mars_map.ingest.rover import curiosity_stops, curiosity_track, surface_image  # noqa: E402

CLASS_CMAP = ListedColormap(["#c9b18f", "#f2d36b", "#8a5a44", "#222222"])
MINERAL_CMAP = ListedColormap(["#e8e4dc", "#e6c229", "#f17105", "#2a9d8f", "#3a86ff", "#8338ec",
                               "#d1495b", "#6a994e", "#4a4e69"])


def rover_targets(grid, max_stops=4, min_sols=10):
    """Curiosity's longest stops in the region as targets, with a photo taken there."""
    try:
        stops = curiosity_stops(grid, min_sols=min_sols)[:max_stops]
        track = curiosity_track(grid)
    except FetchError as e:
        print(f"  rover track unavailable: {e}")
        return [], []
    out = []
    for i, s in enumerate(stops):
        photo = surface_image(s.sol, s.sol + s.sols_here)
        note = f"Curiosity worked here for {s.sols_here} sols (sol {s.sol} to {s.sol + s.sols_here})."
        if photo:
            note += f" Photo from the spot: {photo}"
        out.append(Target(f"Curiosity stop, sol {s.sol}", round(s.x_m, 1), round(s.y_m, 1),
                          priority=1 if i < 2 else 2, dwell_min=40, note=note))
    return out, track


def hillshade(elev, res, az=315.0, alt=35.0):
    dy, dx = np.gradient(elev.astype(np.float64), res)
    slope, aspect = np.arctan(np.hypot(dx, dy)), np.arctan2(dy, -dx)
    a, z = np.radians(360 - az + 90), np.radians(90 - alt)
    return np.clip(np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect), 0, 1)


def stretch(a, lo=1, hi=99):
    p0, p1 = np.nanpercentile(a, lo), np.nanpercentile(a, hi)
    return np.clip((a - p0) / max(p1 - p0, 1e-9), 0, 1)


def figure(site, track, path):
    L = site.layers
    w, h = site.extent_m
    ext = (0, w, h, 0)
    cost = np.where(np.isfinite(L["cost"]), L["cost"], np.nan)
    panels = []          # (title, data, cmap, imshow kwargs, kind)
    if "optical" in L:
        panels.append(("HiRISE orthoimage (1 m)", stretch(L["optical"]), "gray", {}, "image"))
    if all(k in L for k in ("optical_ir", "optical_red", "optical_bg")):
        ok = np.isfinite(L["optical_ir"])        # the colour strip is narrower than the region

        def band(k):
            p0, p1 = np.percentile(L[k][ok], 1), np.percentile(L[k][ok], 99)
            return np.clip((L[k] - p0) / max(p1 - p0, 1e-9), 0, 1)

        rgb = np.dstack([band(k) for k in ("optical_ir", "optical_red", "optical_bg")])
        rgb[~ok] = 0.15
        panels.append(("HiRISE colour (IR, red, blue-green)", rgb, None, {}, "image"))
    panels += [
        ("Elevation, m (hillshaded)", L["elevation"], "terrain", {}, "elev"),
        ("Slope, deg", L["slope"], "magma", {"vmax": 40}, "map"),
        ("Terrain class", L["terrain_class"], CLASS_CMAP, {"vmin": -0.5, "vmax": 3.5}, "class"),
        ("Cost (x slower than flat), blocked = black", cost, "YlOrRd",
         {"vmax": np.nanpercentile(cost, 99)}, "cost"),
        ("Thermal inertia (THEMIS 100 m)", L["thermal_inertia"], "inferno", {}, "map"),
        ("Rock proxy from image texture", L["rock_abundance"], "copper", {"vmax": 0.5}, "map"),
        ("Science score (CRISM minerals)", L["science_score"], "YlGn", {"vmin": 0, "vmax": 1}, "map"),
    ]
    if "mineral_dominant" in L:
        panels.append(("Dominant mineral", L["mineral_dominant"], MINERAL_CMAP,
                       {"vmin": -0.5, "vmax": 8.5}, "mineral"))
    for key, title in (("mineral_sulfate", "Sulfate index (SINDEX2)"),
                       ("ridge_index", "Ridge index, m (boxwork = ridges)")):
        if key in L:
            v = np.nanpercentile(np.abs(L[key]), 99)
            kw = {"vmin": -v, "vmax": v} if key == "ridge_index" else {"vmin": 0, "vmax": v}
            panels.append((title, L[key], "RdBu_r" if key == "ridge_index" else "viridis", kw, "map"))
    panels = panels[:12]

    fig, axes = plt.subplots(3, 4, figsize=(23, 17))
    shade = hillshade(L["elevation"], site.resolution_m)
    for ax, (title, data, cmap, kw, kind) in zip(axes.flat, panels):
        if kind == "cost":
            ax.set_facecolor("black")
        im = ax.imshow(data, cmap=cmap, extent=ext, **kw)
        if kind == "elev":
            ax.imshow(shade, cmap="gray", extent=ext, alpha=0.45, vmin=0, vmax=1)
        if kind != "image":
            cb = fig.colorbar(im, ax=ax, shrink=0.8)
            if kind == "class":
                cb.set_ticks(list(TERRAIN_CLASSES))
                cb.ax.set_yticklabels(list(TERRAIN_CLASSES.values()))
            if kind == "mineral":
                cb.set_ticks(range(len(MINERAL_NAMES)))
                cb.ax.set_yticklabels([n.split(" (")[0] for n in MINERAL_NAMES], fontsize=7)
        if track:
            ax.plot([t.x_m for t in track], [t.y_m for t in track], color="cyan", lw=0.8, alpha=0.9)
        for t in site.targets:
            rover = t.name.startswith("Curiosity")
            ax.plot(t.x_m, t.y_m, marker="*" if rover else "D", color="white" if rover else "yellow",
                    ms=11 if rover else 6, mec="k")
        ax.plot(*site.start, marker="^", color="lime", ms=10, mec="k")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(title)
        ax.set_xlabel("x, m (east)")
        ax.set_ylabel("y, m (south)")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{site.name}  |  {site.resolution_m} m/px, {w:.0f}x{h:.0f} m  |  centre "
                 f"{site.center_lat:.3f}N {site.center_lon:.3f}E  |  ^ base, * Curiosity stops, "
                 f"diamonds = mineral targets, cyan = Curiosity track", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="gale_boxwork", choices=sorted(REGIONS))
    args = ap.parse_args()
    if not CATALOG_DB.exists():
        sys.exit(f"{CATALOG_DB} not found: run scripts/ingest_gale.py first")
    g = REGIONS[args.region]
    out = DATA_DIR / g.name

    db = sqlite3.connect(CATALOG_DB)
    rows = db.execute(
        "SELECT l.name, s.name, l.native_res_m, l.coverage_frac FROM layers l "
        "JOIN sources s ON s.id=l.source_id JOIN regions r ON r.id=l.region_id "
        "WHERE r.name=? ORDER BY l.name, l.native_res_m", (g.name,)).fetchall()
    db.close()
    if not rows:
        sys.exit(f"no processed layers for {g.name}: run scripts/ingest_gale.py --region {g.name}")
    print(f"layers in the catalog for {g.name}:")
    for name, src, res, cov in rows:
        print(f"  {name:22s} {src:14s} {res:7.1f} m  cover {cov:6.1%}")

    centre = (g.size_m / 2, g.size_m / 2)
    site = add_derived(fuse_region(CATALOG_DB, g.name, start=centre, targets=[]))
    L = site.layers

    # The area a crew can walk without leaving comfortable slopes. It is seeded where the
    # rover worked (proven ground) and the largest such area wins; with no rover track it
    # is seeded at the flattest patch of the region.
    rover, track = rover_targets(g)
    seeds = [(t.x_m, t.y_m) for t in rover] or [pick_base(site)]
    reach = max((reachable_mask(site, *s) for s in seeds), key=lambda m: m.sum())
    L["reachable"] = reach
    site.start = tuple(round(v, 1) for v in pick_base(site, allowed=reach))

    for t in rover:
        p = snap_to(reach, site, t.x_m, t.y_m, max_m=30.0)
        if p:
            t.x_m, t.y_m = p
        else:
            t.priority = 3
            t.note += " Not connected to the base by slopes within the comfortable limit."
    minerals = mineral_targets(site, count=4, avoid=[(t.x_m, t.y_m) for t in rover], allowed=reach)
    site.targets = rover + minerals

    print(f"\nreachable on foot from the base: {reach.mean():.1%} of the region")
    print("targets:")
    for t in site.targets:
        r, c = site.to_pixel(t.x_m, t.y_m)
        print(f"  P{t.priority} {t.name:34s} ({t.x_m:6.0f}, {t.y_m:6.0f}) m  slope {L['slope'][r, c]:4.1f}  "
              f"science {L['science_score'][r, c]:.2f}  reachable {bool(reach[r, c])}")

    site.save(out)
    figure(site, track, out / "preview.png")

    print(f"\nsite saved: {out}")
    print(f"  source: {site.source}")
    print(f"  {len(L)} layers: {', '.join(sorted(L))}")
    print(f"  elevation {L['elevation'].min():.1f} .. {L['elevation'].max():.1f} m "
          f"(relief {np.ptp(L['elevation']):.1f} m)")
    print(f"  slope: median {np.median(L['slope']):.1f}, p90 {np.percentile(L['slope'], 90):.1f}, "
          f"p99 {np.percentile(L['slope'], 99):.1f} deg")
    print(f"  thermal inertia {L['thermal_inertia'].min():.0f} .. {L['thermal_inertia'].max():.0f}")
    print("  terrain: " + ", ".join(f"{n} {(L['terrain_class'] == k).mean():.1%}"
                                    for k, n in TERRAIN_CLASSES.items()))
    if "mineral_dominant" in L:
        share = np.bincount(L["mineral_dominant"].ravel(), minlength=len(MINERAL_NAMES)) / L["slope"].size
        print("  minerals: " + ", ".join(f"{n.split(' (')[0]} {s:.1%}" for n, s in zip(MINERAL_NAMES, share)))
    ok = np.isfinite(L["cost"])
    print(f"  blocked {L['blocked'].mean():.1%} | cost median {np.median(L['cost'][ok]):.2f}, "
          f"p95 {np.percentile(L['cost'][ok], 95):.2f}")
    print(f"  base {site.start} m = {tuple(round(v, 4) for v in site.to_latlon(*site.start))}")
    for k, v in site.series.items():
        if k != "sol":
            print(f"  series {k}: {len(v)} sols, {min(v):.2f} .. {max(v):.2f}")
    if site.series:
        print(f"  common sols: {site.series['sol'][0]} .. {site.series['sol'][-1]}")


if __name__ == "__main__":
    main()
