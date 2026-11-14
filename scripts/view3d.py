"""Build an interactive 3D view of a site: one self-contained HTML file.

    .venv\\Scripts\\python scripts\\view3d.py                      # data/gale_boxwork -> view3d.html
    .venv\\Scripts\\python scripts\\view3d.py data/gale_stub       # any Site folder
    .venv\\Scripts\\python scripts\\view3d.py --open               # and open it in the browser
    .venv\\Scripts\\python scripts\\view3d.py --serve              # or serve it on localhost:8765

Everything is inside the file (terrain, every layer as a texture, targets, series, the
3D library), so `data/<site>/view3d.html` can be opened directly or sent to someone.
Fonts and the rover photos come from the internet; without it the page still works.

Shown automatically when the files are there:
* `<site>/astronaut_plan.json`  the EVA route written by the Alg router;
* Curiosity's track, for regions the rover drove through.
"""

import argparse
import base64
import io
import json
import re
import sys
import webbrowser
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, features

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mars_map import DATA_DIR, TERRAIN_CLASSES, Site  # noqa: E402
from mars_map.site import resolve_path  # noqa: E402

TEMPLATE = ROOT / "mars_map" / "viewer" / "template.html"
THREE_VERSION = "0.160.0"
THREE_URL = f"https://cdnjs.cloudflare.com/ajax/libs/three.js/{THREE_VERSION}/three.module.min.js"
MESH_MAX = 500                       # terrain mesh is at most MESH_MAX x MESH_MAX vertices
MARS_RADIUS_M = 3_396_190.0

# Mars-toned ramp for the grayscale photo: the image is real, the colour is a tint.
DUST = LinearSegmentedColormap.from_list("dust", ["#150c08", "#4a2a1c", "#8e5637", "#c98f60", "#ecd2ae", "#fbf1df"])
RELIEF = LinearSegmentedColormap.from_list("relief", ["#2f4b5c", "#4f7a6c", "#a9a56b", "#d9b77a", "#b7754a", "#7c4a36", "#f1e6d8"])
CLASS_COLORS = ["#c9a57b", "#f0cf63", "#8a4f38", "#1b1512"]
MINERAL_COLORS = ["#00000000", "#f2c230", "#f0721b", "#2bb3a0", "#3d8bfd", "#9b5de5", "#d6455d", "#7aa63c", "#59607a"]


def encode(rgba: np.ndarray, lossless: bool = False) -> str:
    """HxWx4 uint8 -> data URI (WebP: small and keeps transparency; PNG if WebP is missing)."""
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    if features.check("webp"):
        img.save(buf, "WEBP", lossless=lossless, quality=88, method=4)
        mime = "image/webp"
    else:
        img.save(buf, "PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def colorize(data, cmap, vmin, vmax, alpha=None) -> np.ndarray:
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    t = np.clip((np.nan_to_num(data, nan=vmin) - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    rgba = (cm(t) * 255).astype(np.uint8)
    if alpha is not None:
        rgba[..., 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    rgba[~np.isfinite(data), 3] = 0
    return rgba


def categorical(data, colors) -> np.ndarray:
    table = np.array([matplotlib.colors.to_rgba(c) for c in colors])
    return (table[np.clip(data.astype(int), 0, len(colors) - 1)] * 255).astype(np.uint8)


def ramp(cmap, n=9) -> list[str]:
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    return [matplotlib.colors.to_hex(cm(i / (n - 1))) for i in range(n)]


def hillshade(elev, res, az=315.0, alt=40.0):
    dy, dx = np.gradient(elev.astype(np.float64), res)
    slope, aspect = np.arctan(np.hypot(dx, dy)), np.arctan2(dy, -dx)
    a, z = np.radians(360 - az + 90), np.radians(90 - alt)
    return np.clip(np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect), 0, 1)


def stretch(a, lo=1.0, hi=99.0):
    p0, p1 = np.nanpercentile(a, lo), np.nanpercentile(a, hi)
    return np.clip((a - p0) / max(p1 - p0, 1e-12), 0, 1)


def build_layers(site: Site) -> tuple[str, str, list[dict]]:
    """(base texture, base caption, overlay layers). Layers the site lacks are skipped."""
    L = site.layers
    out: list[dict] = []

    if "optical" in L:
        base = colorize(stretch(L["optical"]) ** 0.9, DUST, 0, 1)
        caption = "HiRISE orthoimage, 1 m. Brightness is real, the colour is a tint."
    else:
        base = colorize(hillshade(L["elevation"], site.resolution_m), DUST, 0, 1)
        caption = "No image for this site: shaded relief, tinted."

    def cont(key, title, group, cmap, unit, lo=1, hi=99, note="", alpha=None, vmin=None, vmax=None, fmt=".1f"):
        if key not in L:
            return
        d = L[key].astype(np.float64)
        d[~np.isfinite(d)] = np.nan
        a = float(np.nanpercentile(d, lo)) if vmin is None else vmin
        b = float(np.nanpercentile(d, hi)) if vmax is None else vmax
        out.append({"id": key, "title": title, "group": group, "kind": "ramp", "unit": unit, "note": note,
                    "min": format(a, fmt), "max": format(b, fmt), "stops": ramp(cmap),
                    "tex": encode(colorize(d, cmap, a, b, alpha))})

    def cat(key, title, group, colors, names, note=""):
        if key not in L:
            return
        used = sorted(int(v) for v in np.unique(L[key]) if int(v) < len(names))
        out.append({"id": key, "title": title, "group": group, "kind": "classes", "note": note,
                    "classes": [{"color": colors[i][:7], "name": names[i]} for i in used
                                if not colors[i].endswith("00") or len(colors[i]) == 7],
                    "tex": encode(categorical(L[key], colors), lossless=True)})

    # --- surface -----------------------------------------------------------------
    if all(k in L for k in ("optical_ir", "optical_red", "optical_bg")):
        ok = np.isfinite(L["optical_ir"])
        rgb = np.dstack([stretch(np.where(ok, L[k], np.nan)) for k in ("optical_ir", "optical_red", "optical_bg")])
        rgba = np.dstack([np.nan_to_num(rgb), ok.astype(float)])
        out.append({"id": "colour", "title": "False colour", "group": "Surface", "kind": "note",
                    "note": f"HiRISE near-infrared, red, blue-green. The colour strip covers {ok.mean():.0%} "
                            "of the site; the rest shows the photo.",
                    "tex": encode((rgba * 255).astype(np.uint8))})
    cont("rock_abundance", "Rock / roughness proxy", "Surface", "copper", "0..1", vmin=0, vmax=0.5, fmt=".2f",
         note="From image texture on real sites: ranks smooth to rugged, does not count rocks.")
    cont("thermal_inertia", "Thermal inertia", "Surface", "inferno", "J m⁻² K⁻¹ s⁻½", fmt=".0f",
         note="Low = loose sand and dust, high = rock. THEMIS, 100 m per pixel.")

    # --- terrain -----------------------------------------------------------------
    cont("elevation", "Elevation", "Terrain", RELIEF, "m", lo=0, hi=100, fmt=".0f")
    cont("slope", "Slope", "Terrain", "magma", "°", vmin=0, vmax=40, fmt=".0f")
    cat("terrain_class", "Terrain class", "Terrain", CLASS_COLORS, list(TERRAIN_CLASSES.values()))
    if "ridge_index" in L:
        v = float(np.nanpercentile(np.abs(L["ridge_index"]), 99))
        cont("ridge_index", "Ridges and troughs", "Terrain", "RdBu_r", "m", vmin=-v, vmax=v,
             note="Height above the surrounding 30 m. Red = ridge (boxwork), blue = trough.")

    # --- routing -----------------------------------------------------------------
    if "cost" in L:
        c = np.where(np.isfinite(L["cost"]), L["cost"], np.nan)
        hi = float(np.nanpercentile(c, 99))
        tex = colorize(c, "YlOrRd", 1.0, hi)
        tex[~np.isfinite(c)] = (18, 12, 10, 255)
        out.append({"id": "cost", "title": "Walking cost", "group": "Routing", "kind": "ramp", "unit": "× flat ground",
                    "note": "How many times slower than firm flat ground. Black = blocked.",
                    "min": "1.0", "max": f"{hi:.1f}", "stops": ramp("YlOrRd"), "tex": encode(tex)})
    if "reachable" in L:
        r = L["reachable"].astype(bool)
        tex = np.zeros(r.shape + (4,), np.uint8)
        tex[r] = (64, 196, 150, 190)
        tex[~r] = (20, 14, 12, 150)
        out.append({"id": "reachable", "title": "Reachable from base", "group": "Routing", "kind": "classes",
                    "note": f"Connected to the base on slopes within {site.mission.max_slope_deg:.0f}°: "
                            f"{r.mean():.0%} of the site.",
                    "classes": [{"color": "#40c496", "name": "reachable"}, {"color": "#3a2c26", "name": "cut off"}],
                    "tex": encode(tex, lossless=True)})

    # --- science -----------------------------------------------------------------
    if "science_score" in L and np.any(L["science_score"] > 0):
        s = L["science_score"]
        cont("science_score", "Science score", "Science", "YlGn", "0..1", vmin=0, vmax=1, alpha=np.clip(s * 3, 0, 1), fmt=".0f",
             note="Strength of the clearest mineral detection. Transparent where nothing is detected.")
    if "mineral_dominant" in L:
        from mars_map.fusion.science import MINERAL_NAMES
        cat("mineral_dominant", "Dominant mineral", "Science", MINERAL_COLORS, MINERAL_NAMES,
            note="Which mineral group has the clearest CRISM signature. Transparent = none detected.")
    for key, title in (("mineral_sulfate", "Sulfate index"), ("mineral_hydration", "Hydration index"),
                       ("mineral_clay", "Clay index"), ("mineral_silica", "Silica index")):
        if key in L:
            hi = float(np.nanpercentile(L[key], 99.5))
            cont(key, title, "Science", "viridis", "index", vmin=0, vmax=max(hi, 1e-6), fmt=".3f",
                 note="CRISM spectral index, unitless. Higher = stronger signature, noisy near zero.")
    return encode(base), caption, out


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def probe(data, scale, offset=0.0, missing=255) -> dict:
    """A layer at mesh resolution as uint8, for the cursor readout: value = raw * scale + offset."""
    q = np.round((np.nan_to_num(data.astype(np.float64), nan=np.nan, posinf=np.nan) - offset) / scale)
    q = np.where(np.isfinite(q), np.clip(q, 0, 254), missing).astype(np.uint8)
    return {"scale": scale, "offset": offset, "data": b64(q)}


def rover_track(site: Site) -> list[list[float]]:
    """Curiosity's track inside the site, if this is a known region and the feed is reachable."""
    try:
        import mars_map.ingest  # noqa: F401  (OS certificate store)
        from mars_map.grid import REGIONS
        from mars_map.ingest.rover import curiosity_track
        grid = REGIONS.get(site.name)
        if grid is None:
            return []
        return [[round(w.x_m, 1), round(w.y_m, 1), w.sol] for w in curiosity_track(grid)]
    except Exception as e:                                   # no network, no cache: just no track
        print(f"  rover track skipped: {e}")
        return []


def three_source() -> str | None:
    """The 3D library, cached in data/cache so the page works offline. None = load from the CDN."""
    path = DATA_DIR / "cache" / f"three-{THREE_VERSION}.module.min.js"
    try:
        if not path.exists():
            import mars_map.ingest  # noqa: F401
            from mars_map.ingest.fetch import fetch
            fetch(THREE_URL, path)
        return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  three.js not cached ({e}); the page will load it from the CDN")
        return None


def build(folder: Path) -> Path:
    site = Site.load(folder)
    L = site.layers
    rows, cols = site.shape
    step = max(1, int(np.ceil(max(rows, cols) / MESH_MAX)))
    elev = L["elevation"][::step, ::step].astype(np.float64)
    emin, emax = float(L["elevation"].min()), float(L["elevation"].max())
    heights = np.round((elev - emin) / max(emax - emin, 1e-9) * 65535).astype("<u2")

    print(f"{site.name}: {rows}x{cols} cells, mesh {elev.shape[0]}x{elev.shape[1]}")
    base, caption, layers = build_layers(site)
    print(f"  {len(layers)} overlay layers: " + ", ".join(l["id"] for l in layers))

    sub = lambda k: L[k][::step, ::step]  # noqa: E731
    probes = {"slope": probe(sub("slope"), 0.5), "terrain_class": probe(sub("terrain_class"), 1)}
    if "cost" in L:
        probes["cost"] = probe(sub("cost"), 0.1)
    if "thermal_inertia" in L:
        probes["thermal_inertia"] = probe(sub("thermal_inertia"), 10)
    if "science_score" in L:
        probes["science_score"] = probe(sub("science_score"), 1 / 254)
    if "mineral_dominant" in L:
        probes["mineral_dominant"] = probe(sub("mineral_dominant"), 1)
    if "reachable" in L:
        probes["reachable"] = probe(sub("reachable"), 1)
    blocked = L["blocked"] if "blocked" in L else (L["terrain_class"] == 3)
    mask = np.zeros(blocked.shape + (4,), np.uint8)
    mask[blocked] = (255, 255, 255, 255)

    targets = []
    for t in site.targets:
        url = re.search(r"https?://\S+", t.note or "")
        kind = "rover" if t.name.lower().startswith("curiosity") else (
            "mineral" if "exposure" in t.name.lower() else "site")
        targets.append({"name": t.name, "x": t.x_m, "y": t.y_m, "priority": t.priority, "dwell": t.dwell_min,
                        "kind": kind, "photo": url.group(0).rstrip(".") if url else None,
                        "note": re.sub(r"\s*Photo from the spot:\s*https?://\S+", "", t.note or "").strip()})

    route = None
    plan_path = folder / "astronaut_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        pts = plan.get("waypoints", [])
        keep = max(1, len(pts) // 600)
        route = {"points": [[p["x_m"], p["y_m"]] for p in pts[::keep]] + ([[pts[-1]["x_m"], pts[-1]["y_m"]]] if pts else []),
                 "summary": plan.get("summary", {}), "feasible": plan.get("feasible"),
                 "warnings": plan.get("warnings", []), "targets": plan.get("visited_targets", [])}
        print(f"  route: {len(pts)} points from {plan_path.name}")

    mineral_names: list[str] = []
    if "mineral_dominant" in L:
        from mars_map.fusion.science import MINERAL_NAMES as mineral_names

    m = site.mission
    data = {
        "name": site.name, "source": site.source, "res": site.resolution_m,
        "width": cols * site.resolution_m, "height": rows * site.resolution_m,
        "lat": site.center_lat, "lon": site.center_lon, "radius": MARS_RADIUS_M,
        "mesh": {"rows": int(elev.shape[0]), "cols": int(elev.shape[1]), "emin": emin, "emax": emax,
                 "heights": b64(heights)},
        "base": base, "baseCaption": caption, "mask": encode(mask, lossless=True), "layers": layers,
        "probes": probes, "terrainNames": list(TERRAIN_CLASSES.values()),
        "mineralNames": mineral_names,
        "start": list(site.start), "targets": targets, "noGo": site.no_go,
        "track": rover_track(site), "route": route, "series": site.series,
        "mission": {"maxSlope": m.max_slope_deg, "hardSlope": m.hard_max_slope_deg, "evaHours": m.eva_max_hours,
                    "maxDose": m.max_dose_msv, "maxKm": m.max_distance_km, "speed": m.walk_speed_flat_kmh},
        "stats": {"blocked": float(blocked.mean()),
                  "reachable": float(L["reachable"].mean()) if "reachable" in L else None,
                  "slopeMedian": float(np.median(L["slope"])), "relief": emax - emin},
        "threeUrl": THREE_URL,
    }

    html = TEMPLATE.read_text(encoding="utf-8")
    three = three_source()
    html = html.replace("/*__THREE_SOURCE__*/", (three or "").replace("</script", "<\\/script"))
    html = html.replace("/*__SITE_DATA__*/", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    out = folder / "view3d.html"
    out.write_text(html, encoding="utf-8")
    print(f"  written {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", nargs="?", default="data/gale_boxwork", help="Site folder (default: data/gale_boxwork)")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    ap.add_argument("--serve", action="store_true",
                    help="serve the page on http://127.0.0.1:8765 and open it (if double-click does not work)")
    args = ap.parse_args()
    out = build(resolve_path(args.site))
    if args.serve:
        import functools
        import http.server
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out.parent))
        with http.server.ThreadingHTTPServer(("127.0.0.1", 8765), handler) as srv:
            url = "http://127.0.0.1:8765/view3d.html"
            print(f"  serving {url}  (Ctrl+C to stop)")
            webbrowser.open(url)
            try:
                srv.serve_forever()
            except KeyboardInterrupt:
                pass
    elif args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
