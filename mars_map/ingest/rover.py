"""Curiosity's real track: where the rover drove and where it stopped to work.

Source: the MSL waypoints feed of the mission map (mars.nasa.gov, GeoJSON, ~0.5 MB).
One point per end-of-drive position with the sol, lon/lat and elevation.

Used for two things: placing regions on ground the rover actually studied, and
turning its long stops into science targets (a place where the team parked the rover
for weeks is a place worth visiting).

    track = curiosity_track(grid)              # points inside the region, site metres
    stops = curiosity_stops(grid, min_sols=10)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..grid import RegionGrid
from .fetch import RAW_DIR, fetch

WAYPOINTS_URL = "https://mars.nasa.gov/mmgis-maps/MSL/Layers/json/MSL_waypoints.json"
RAW_IMAGES_API = "https://mars.nasa.gov/api/v1/raw_image_items/"
# cameras that show the scene around the rover, best first
SCENE_CAMERAS = ("MAST_LEFT", "MAST_RIGHT", "NAV_LEFT_B", "NAV_RIGHT_B", "NAV_LEFT_A", "NAV_RIGHT_A")


@dataclass
class Waypoint:
    sol: int
    lon: float
    lat: float
    elev_m: float
    x_m: float               # site metres, east from the NW corner
    y_m: float               # site metres, south from the NW corner
    sols_here: int = 0       # sols until the rover drove on (0 for the last point)


def all_waypoints(raw_dir: str | Path = RAW_DIR, refresh: bool = False) -> list[dict]:
    """Every waypoint of the mission as {sol, lon, lat, elev_m}, in driving order."""
    path = Path(raw_dir) / "msl" / "MSL_waypoints.json"
    if refresh:
        path.unlink(missing_ok=True)
    fetch(WAYPOINTS_URL, path)
    feats = json.loads(path.read_text(encoding="utf-8"))["features"]
    pts = [{"sol": int(f["properties"]["sol"]), "lon": float(f["properties"]["lon"]),
            "lat": float(f["properties"]["lat"]), "elev_m": float(f["properties"]["elev_geoid"])}
           for f in feats]
    pts.sort(key=lambda p: p["sol"])
    return pts


def curiosity_track(grid: RegionGrid, raw_dir: str | Path = RAW_DIR) -> list[Waypoint]:
    """Waypoints inside the region, with how long the rover stayed at each."""
    pts = all_waypoints(raw_dir)
    out = []
    for p, nxt in zip(pts, pts[1:] + [None]):
        x, y = grid.lonlat_to_xy(p["lon"], p["lat"])
        if 0 <= x < grid.size_m and 0 <= y < grid.size_m:
            stay = (nxt["sol"] - p["sol"]) if nxt else 0
            out.append(Waypoint(p["sol"], p["lon"], p["lat"], p["elev_m"], x, y, stay))
    return out


def surface_image(sol_from: int, sol_to: int, raw_dir: str | Path = RAW_DIR) -> str | None:
    """Link to one full-size photo the rover took in these sols (what the place looks like).

    Asks the public raw-images API; the answer is cached. Returns None when there is
    no scene-camera image or the API cannot be reached.
    """
    import urllib.parse

    from .fetch import FetchError

    for sol in range(sol_from, min(sol_to, sol_from + 4) + 1):       # a few sols are enough
        q = {"order": "sol desc", "per_page": 100, "page": 0,
             "condition_1": "msl:mission", "condition_2": f"{sol}:sol:in"}
        path = Path(raw_dir) / "msl" / f"raw_images_sol{sol}.json"
        try:
            fetch(RAW_IMAGES_API + "?" + urllib.parse.urlencode(q), path)
            items = json.loads(path.read_text(encoding="utf-8")).get("items", [])
        except (FetchError, ValueError):
            continue
        full = [i for i in items if not i.get("is_thumbnail") and i.get("https_url")]
        for cam in SCENE_CAMERAS:
            hit = [i for i in full if i.get("instrument") == cam]
            if hit:
                return hit[0]["https_url"]
    return None


def curiosity_stops(grid: RegionGrid, min_sols: int = 10, min_spacing_m: float = 40.0,
                    raw_dir: str | Path = RAW_DIR) -> list[Waypoint]:
    """Places where the rover stayed at least `min_sols`, longest first, not closer than
    `min_spacing_m` to a longer stop (repositioning by a few metres is the same place)."""
    stops: list[Waypoint] = []
    for w in sorted(curiosity_track(grid, raw_dir), key=lambda w: -w.sols_here):
        if w.sols_here < min_sols:
            break
        if all((w.x_m - s.x_m) ** 2 + (w.y_m - s.y_m) ** 2 >= min_spacing_m ** 2 for s in stops):
            stops.append(w)
    return stops
