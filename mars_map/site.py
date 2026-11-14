"""Site data contract: what every map algorithm receives.

A Site is one exploration area on a regular metric grid:

* layers      -- 2-D rasters, all the same shape, float32 (or uint8 for classes)
* series      -- per-sol time series that apply to the whole site
* targets     -- named points of interest (science stops)
* start       -- where the walk begins and ends (habitat / rover)
* no_go       -- polygons the crew must not enter
* mission     -- limits of the crew and suits

Grid convention (same as numpy image indexing):
    layer[row, col], row 0 = NORTH edge, col 0 = WEST edge.
    Cell (row, col) covers x = col*res .. (col+1)*res metres east of the west edge,
    y = row*res .. (row+1)*res metres south of the north edge.
Points are stored as (x_m, y_m) = metres east / metres south of the NW corner,
so that col = x_m / res and row = y_m / res.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MARS_RADIUS_M = 3_389_500.0

# Repo root, so "data/..." works no matter which folder a script is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def resolve_path(folder: str | Path) -> Path:
    """Relative paths are taken from the repo root, not the current working dir."""
    folder = Path(folder)
    return folder if folder.is_absolute() else PROJECT_ROOT / folder

# Terrain classes stored in layers["terrain_class"] (uint8).
TERRAIN_CLASSES = {
    0: "bedrock",      # firm, walkable
    1: "sand",         # loose; slow, risk of sinking (low thermal inertia)
    2: "boulders",     # rock field; slow, trip hazard
    3: "impassable",   # cliffs / too steep; never route through
}

# Required layers and their meaning. Algorithms may rely on all of these existing.
LAYER_SPEC = {
    "elevation":       ("m", "Height above Mars datum"),
    "slope":           ("deg", "Terrain slope, derived from elevation"),
    "roughness":       ("m", "Local relief (std of elevation in a small window)"),
    "rock_abundance":  ("0..1", "Fraction of area covered by rocks"),
    "thermal_inertia": ("J m^-2 K^-1 s^-1/2", "Low = loose sand/dust, high = rock"),
    "terrain_class":   ("class", "See TERRAIN_CLASSES"),
    "science_score":   ("0..1", "How interesting the spot is for the mission goals"),
    "nodata":          ("bool", "True where real data is missing"),
}


@dataclass
class Target:
    name: str
    x_m: float
    y_m: float
    priority: int = 1          # 1 = must visit, 2 = nice to have, 3 = optional
    dwell_min: float = 20.0    # minutes the crew spends working there
    note: str = ""


@dataclass
class MissionParams:
    """Crew / suit limits. Values are working assumptions, not NASA requirements."""
    max_slope_deg: float = 15.0          # comfortable walking limit in a suit
    hard_max_slope_deg: float = 25.0     # never exceed
    walk_speed_flat_kmh: float = 2.0     # suited walking speed on firm flat ground
    eva_max_hours: float = 6.0           # total time outside, incl. stops
    return_reserve_frac: float = 0.25    # share of EVA time kept as reserve
    max_dose_msv: float = 1.0            # dose budget for one EVA
    max_distance_km: float = 8.0         # total walk length, there and back


@dataclass
class Site:
    name: str
    resolution_m: float
    center_lat: float
    center_lon: float                    # east-positive, degrees
    layers: dict[str, np.ndarray]
    series: dict[str, list[float]]       # per-sol values, all same length
    start: tuple[float, float]           # (x_m, y_m)
    targets: list[Target]
    no_go: list[list[tuple[float, float]]] = field(default_factory=list)
    mission: MissionParams = field(default_factory=MissionParams)
    source: str = "synthetic"            # "synthetic" or a description of real data

    # ---- grid helpers -------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.layers["elevation"].shape

    @property
    def extent_m(self) -> tuple[float, float]:
        """(width_m, height_m)"""
        rows, cols = self.shape
        return cols * self.resolution_m, rows * self.resolution_m

    def to_pixel(self, x_m: float, y_m: float) -> tuple[int, int]:
        """(x_m, y_m) -> (row, col), clipped to the grid."""
        rows, cols = self.shape
        col = min(max(int(x_m / self.resolution_m), 0), cols - 1)
        row = min(max(int(y_m / self.resolution_m), 0), rows - 1)
        return row, col

    def to_metres(self, row: int, col: int) -> tuple[float, float]:
        """(row, col) -> (x_m, y_m) of the cell centre."""
        return (col + 0.5) * self.resolution_m, (row + 0.5) * self.resolution_m

    def to_latlon(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Local metres -> planetocentric lat/lon (local equirectangular approx)."""
        w, h = self.extent_m
        dx, dy = x_m - w / 2, -(y_m - h / 2)          # east, north from centre
        lat = self.center_lat + math.degrees(dy / MARS_RADIUS_M)
        lon = self.center_lon + math.degrees(
            dx / (MARS_RADIUS_M * math.cos(math.radians(self.center_lat))))
        return lat, lon

    def validate(self) -> None:
        missing = set(LAYER_SPEC) - set(self.layers)
        if missing:
            raise ValueError(f"missing layers: {sorted(missing)}")
        shapes = {k: v.shape for k, v in self.layers.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"layers differ in shape: {shapes}")
        lengths = {k: len(v) for k, v in self.series.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"series differ in length: {lengths}")
        w, h = self.extent_m
        for p in [self.start] + [(t.x_m, t.y_m) for t in self.targets]:
            if not (0 <= p[0] < w and 0 <= p[1] < h):
                raise ValueError(f"point {p} outside site extent {w}x{h} m")

    # ---- I/O -----------------------------------------------------------
    def save(self, folder: str | Path) -> Path:
        """Writes <folder>/layers.npz + <folder>/site.json."""
        self.validate()
        folder = resolve_path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "layers.npz", **self.layers)
        meta = {
            "name": self.name,
            "source": self.source,
            "resolution_m": self.resolution_m,
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "shape": list(self.shape),
            "grid_convention": "layer[row, col]; row 0 = north, col 0 = west; "
                               "points are (x_m east, y_m south) from NW corner",
            "layer_spec": {k: {"unit": u, "meaning": m} for k, (u, m) in LAYER_SPEC.items()},
            "terrain_classes": TERRAIN_CLASSES,
            "series": self.series,
            "start": list(self.start),
            "targets": [asdict(t) for t in self.targets],
            "no_go": [[list(p) for p in poly] for poly in self.no_go],
            "mission": asdict(self.mission),
        }
        (folder / "site.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "Site":
        folder = resolve_path(folder)
        if not (folder / "site.json").exists():
            raise FileNotFoundError(
                f"no site at {folder}. Generate it first:  "
                f"python scripts/make_stub_site.py   (or use mars_map.load_stub())")
        meta = json.loads((folder / "site.json").read_text(encoding="utf-8"))
        with np.load(folder / "layers.npz") as z:
            layers = {k: z[k] for k in z.files}
        site = cls(
            name=meta["name"],
            source=meta.get("source", "unknown"),
            resolution_m=meta["resolution_m"],
            center_lat=meta["center_lat"],
            center_lon=meta["center_lon"],
            layers=layers,
            series=meta["series"],
            start=tuple(meta["start"]),
            targets=[Target(**t) for t in meta["targets"]],
            no_go=[[tuple(p) for p in poly] for poly in meta["no_go"]],
            mission=MissionParams(**meta["mission"]),
        )
        site.validate()
        return site


def slope_deg(elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    """Slope in degrees from an elevation grid."""
    dy, dx = np.gradient(elevation.astype(np.float64), resolution_m)
    return np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)
