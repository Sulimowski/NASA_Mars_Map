"""A rover's real track: where it drove and where it stopped to work.

Two rovers, picked by where the region is: Curiosity (Gale crater) and Perseverance
(Jezero crater and Nili Planum). Source: the waypoints feed of each mission map
(mars.nasa.gov, GeoJSON, ~0.5 MB): one point per end-of-drive position with the sol,
lon/lat and elevation.

Used for two things: placing regions on ground a rover actually studied, and turning
its long stops into science targets (a place where the team parked the rover for
weeks is a place worth visiting).

    rover = rover_for(grid)                    # None when no rover ever came near
    track = rover_track(grid)                  # points inside the region, site metres
    stops = rover_stops(grid, min_sols=10)
"""

from __future__ import annotations

import json
import math
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from ..grid import MARS_SPHERE_RADIUS_M, RegionGrid
from .fetch import RAW_DIR, FetchError, fetch

MAX_ROVER_DISTANCE_KM = 150.0        # a region further than this from the landing site has no rover


@dataclass(frozen=True)
class Rover:
    key: str                         # also the folder under data/raw/
    name: str
    landing_lat: float
    landing_lon: float
    waypoints_url: str
    scene_cameras: tuple[str, ...]   # cameras that show the scene around the rover, best first


CURIOSITY = Rover(
    "msl", "Curiosity", -4.5895, 137.4417,
    "https://mars.nasa.gov/mmgis-maps/MSL/Layers/json/MSL_waypoints.json",
    ("MAST_LEFT", "MAST_RIGHT", "NAV_LEFT_B", "NAV_RIGHT_B", "NAV_LEFT_A", "NAV_RIGHT_A"))
PERSEVERANCE = Rover(
    "m20", "Perseverance", 18.4446, 77.4509,
    "https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_waypoints.json",
    ("NAVCAM_LEFT", "NAVCAM_RIGHT", "MCZ_LEFT", "MCZ_RIGHT", "FRONT_HAZCAM_LEFT_A"))
ROVERS = {r.key: r for r in (CURIOSITY, PERSEVERANCE)}

MSL_IMAGES_API = "https://mars.nasa.gov/api/v1/raw_image_items/"
M20_IMAGES_API = "https://mars.nasa.gov/rss/api/"


@dataclass
class Waypoint:
    sol: int
    lon: float
    lat: float
    elev_m: float
    x_m: float               # site metres, east from the NW corner
    y_m: float               # site metres, south from the NW corner
    sols_here: int = 0       # sols until the rover drove on (0 for the last point)


def rover_for(grid: RegionGrid) -> Rover | None:
    """The rover whose landing site is nearest to the region, None if none is near."""
    def km(r: Rover) -> float:
        dlat = math.radians(grid.center_lat - r.landing_lat)
        dlon = math.radians((grid.center_lon - r.landing_lon + 180.0) % 360.0 - 180.0)
        dlon *= math.cos(math.radians(grid.center_lat))
        return math.hypot(dlat, dlon) * MARS_SPHERE_RADIUS_M / 1000.0
    best = min(ROVERS.values(), key=km)
    return best if km(best) <= MAX_ROVER_DISTANCE_KM else None


def all_waypoints(rover: Rover = CURIOSITY, raw_dir: str | Path = RAW_DIR,
                  refresh: bool = False) -> list[dict]:
    """Every waypoint of the mission as {sol, lon, lat, elev_m}, in driving order."""
    path = Path(raw_dir) / rover.key / url_name(rover.waypoints_url)
    if refresh:
        path.unlink(missing_ok=True)
    fetch(rover.waypoints_url, path)
    feats = json.loads(path.read_text(encoding="utf-8"))["features"]
    pts = [{"sol": int(f["properties"]["sol"]), "lon": float(f["properties"]["lon"]),
            "lat": float(f["properties"]["lat"]), "elev_m": float(f["properties"]["elev_geoid"])}
           for f in feats]
    pts.sort(key=lambda p: p["sol"])
    return pts


def url_name(url: str) -> str:
    return url.rsplit("/", 1)[1]


def rover_track(grid: RegionGrid, raw_dir: str | Path = RAW_DIR) -> list[Waypoint]:
    """Waypoints inside the region, with how long the rover stayed at each."""
    rover = rover_for(grid)
    if rover is None:
        return []
    pts = all_waypoints(rover, raw_dir)
    out = []
    for p, nxt in zip(pts, pts[1:] + [None]):
        x, y = grid.lonlat_to_xy(p["lon"], p["lat"])
        if 0 <= x < grid.size_m and 0 <= y < grid.size_m:
            stay = (nxt["sol"] - p["sol"]) if nxt else 0
            out.append(Waypoint(p["sol"], p["lon"], p["lat"], p["elev_m"], x, y, stay))
    return out


def rover_stops(grid: RegionGrid, min_sols: int = 10, min_spacing_m: float = 40.0,
                raw_dir: str | Path = RAW_DIR) -> list[Waypoint]:
    """Places where the rover stayed at least `min_sols`, longest first, not closer than
    `min_spacing_m` to a longer stop (repositioning by a few metres is the same place)."""
    stops: list[Waypoint] = []
    for w in sorted(rover_track(grid, raw_dir), key=lambda w: -w.sols_here):
        if w.sols_here < min_sols:
            break
        if all((w.x_m - s.x_m) ** 2 + (w.y_m - s.y_m) ** 2 >= min_spacing_m ** 2 for s in stops):
            stops.append(w)
    return stops


def _scene_images(rover: Rover, sol: int, raw_dir: str | Path) -> list[tuple[str, str]]:
    """[(camera, url)] of the full-size images of one sol. The two missions have different APIs."""
    path = Path(raw_dir) / rover.key / f"raw_images_sol{sol}.json"
    if rover.key == "msl":
        q = {"order": "sol desc", "per_page": 100, "page": 0,
             "condition_1": "msl:mission", "condition_2": f"{sol}:sol:in"}
        fetch(MSL_IMAGES_API + "?" + urllib.parse.urlencode(q), path)
        items = json.loads(path.read_text(encoding="utf-8")).get("items", [])
        return [(i.get("instrument"), i["https_url"]) for i in items
                if not i.get("is_thumbnail") and i.get("https_url")]
    q = {"feed": "raw_images", "category": "mars2020", "feedtype": "json",
         "sol": sol, "num": 100, "page": 0}
    fetch(M20_IMAGES_API + "?" + urllib.parse.urlencode(q), path)
    items = json.loads(path.read_text(encoding="utf-8")).get("images", [])
    return [(i.get("camera", {}).get("instrument"), i["image_files"]["full_res"]) for i in items
            if i.get("sample_type") == "Full" and i.get("image_files", {}).get("full_res")]


def surface_image(sol_from: int, sol_to: int, raw_dir: str | Path = RAW_DIR,
                  rover: Rover = CURIOSITY) -> str | None:
    """Link to one full-size photo the rover took in these sols (what the place looks like).

    Asks the public raw-images API; the answer is cached. Returns None when there is
    no scene-camera image or the API cannot be reached.
    """
    for sol in range(sol_from, min(sol_to, sol_from + 4) + 1):       # a few sols are enough
        try:
            images = _scene_images(rover, sol, raw_dir)
        except (FetchError, ValueError, KeyError):
            continue
        for cam in rover.scene_cameras:
            hit = [url for c, url in images if c == cam]
            if hit:
                return hit[0]
    return None


# Old names, from when Curiosity was the only rover. They follow the region now.
curiosity_track = rover_track
curiosity_stops = rover_stops
