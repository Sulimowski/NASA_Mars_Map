"""Quick-look PNG of all layers of a Site."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if __package__ in (None, ""):          # run directly (PyCharm Run / python preview.py)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mars_map.site import TERRAIN_CLASSES, Site
else:
    from .site import TERRAIN_CLASSES, Site

PANELS = [
    ("elevation", "terrain", "Elevation, m"),
    ("slope", "magma", "Slope, deg"),
    ("roughness", "viridis", "Roughness, m"),
    ("rock_abundance", "copper", "Rock abundance"),
    ("thermal_inertia", "inferno", "Thermal inertia"),
    ("terrain_class", None, "Terrain class"),
    ("science_score", "YlGn", "Science score"),
    ("nodata", "Greys", "No high-res data"),
]

CLASS_CMAP = ListedColormap(["#c9b18f", "#f2d36b", "#8a5a44", "#222222"])


def _overlay(ax, site: Site):
    for poly in site.no_go:
        xs, ys = zip(*(poly + poly[:1]))
        ax.plot(xs, ys, color="red", lw=1)
    ax.plot(*site.start, marker="^", color="cyan", ms=8, mec="k")
    for t in site.targets:
        ax.plot(t.x_m, t.y_m, marker="*", color="white", ms=10, mec="k")


def render(site: Site, out: str | Path) -> Path:
    w, h = site.extent_m
    extent = (0, w, h, 0)                         # y grows south, like the grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    for ax, (key, cmap, title) in zip(axes.flat, PANELS):
        data = site.layers[key]
        if key == "terrain_class":
            im = ax.imshow(data, cmap=CLASS_CMAP, vmin=-0.5, vmax=3.5, extent=extent)
            cb = fig.colorbar(im, ax=ax, ticks=list(TERRAIN_CLASSES))
            cb.ax.set_yticklabels(list(TERRAIN_CLASSES.values()))
        else:
            vmax = 40 if key == "slope" else None
            im = ax.imshow(data, cmap=cmap, extent=extent, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.85)
        _overlay(ax, site)
        ax.set_title(title)
        ax.set_xlabel("x, m (east)")
        ax.set_ylabel("y, m (south)")
    fig.suptitle(f"{site.name} — {site.source}  |  {site.resolution_m} m/px, "
                 f"{w:.0f}x{h:.0f} m  |  ^ start, * targets, red = no-go", fontsize=12)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


if __name__ == "__main__":
    # Render the stub site (generated on first run) -> data/gale_stub/preview.png
    from mars_map import STUB_DIR, load_stub
    print(f"preview saved: {render(load_stub(), STUB_DIR / 'preview.png')}")
