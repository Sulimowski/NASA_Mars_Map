"""``GridTerrain``: the in-memory reference implementation of ``TerrainProvider``.

Synthetic testbeds and the site adapter both produce a ``GridTerrain``; a data
team can also fill one directly from its own rasters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..core.exceptions import TerrainValidationError
from ..core.schemas import Cell
from .interfaces import NEIGHBOR_OFFSETS, BoolGrid, FloatGrid


def slope_from_elevation(elevation: FloatGrid, cell_size_m: float) -> FloatGrid:
    """Steepest slope of every cell, rad, from central differences of the elevation."""
    d_row, d_col = np.gradient(np.asarray(elevation, dtype=np.float64), cell_size_m)
    return np.arctan(np.hypot(d_row, d_col))


@dataclass
class GridTerrain:
    """Terrain rasters on a regular square grid.

    Only ``elevation`` and ``cell_size_m`` are required. Missing layers take neutral
    defaults (firm ground, no rocks, everything passable, nothing forbidden, no
    shelter) and ``slope`` is derived from the elevation.

    Raises:
        TerrainValidationError: from ``__post_init__`` when the layers are inconsistent.
    """

    elevation: FloatGrid
    cell_size_m: float
    slope: FloatGrid | None = None
    walk_eta: FloatGrid | None = None
    rock_hazard: FloatGrid | None = None
    passable_walk: BoolGrid | None = None
    forbidden: BoolGrid | None = None
    is_shelter: BoolGrid | None = None

    def __post_init__(self) -> None:
        self.elevation = np.asarray(self.elevation, dtype=np.float64)
        if self.elevation.ndim != 2 or self.elevation.size == 0:
            raise TerrainValidationError("elevation must be a non-empty 2-D array")
        if not (np.isfinite(self.cell_size_m) and self.cell_size_m > 0.0):
            raise TerrainValidationError(f"cell_size_m={self.cell_size_m} must be positive")
        shape = self.elevation.shape
        if self.slope is None:
            self.slope = slope_from_elevation(self.elevation, self.cell_size_m)
        self.slope = np.asarray(self.slope, dtype=np.float64)
        self.walk_eta = self._float_layer(self.walk_eta, 1.2, shape)
        self.rock_hazard = self._float_layer(self.rock_hazard, 0.0, shape)
        self.passable_walk = self._bool_layer(self.passable_walk, True, shape)
        self.forbidden = self._bool_layer(self.forbidden, False, shape)
        self.is_shelter = self._bool_layer(self.is_shelter, False, shape)
        self.validate()

    @staticmethod
    def _float_layer(layer: FloatGrid | None, default: float, shape: tuple[int, ...]) -> FloatGrid:
        if layer is None:
            return np.full(shape, default, dtype=np.float64)
        return np.asarray(layer, dtype=np.float64)

    @staticmethod
    def _bool_layer(layer: BoolGrid | None, default: bool, shape: tuple[int, ...]) -> BoolGrid:
        if layer is None:
            return np.full(shape, default, dtype=np.bool_)
        return np.asarray(layer, dtype=np.bool_)

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, cols)."""
        rows, cols = self.elevation.shape
        return int(rows), int(cols)

    def in_bounds(self, cell: Cell) -> bool:
        """True when the cell lies inside the grid."""
        rows, cols = self.elevation.shape
        return 0 <= cell[0] < rows and 0 <= cell[1] < cols

    def neighbors(self, cell: Cell) -> Iterator[tuple[Cell, float]]:
        """In-bounds 8-neighbours with the horizontal distance to each, m."""
        if not self.in_bounds(cell):
            raise TerrainValidationError(f"cell {cell} is outside the grid {self.shape}")
        row, col = cell
        for d_row, d_col, factor in NEIGHBOR_OFFSETS:
            nb = (row + d_row, col + d_col)
            if self.in_bounds(nb):
                yield nb, factor * self.cell_size_m

    def to_cell(self, x_m: float, y_m: float) -> Cell:
        """Local metres (x east, y south of the NW corner) -> cell, clipped to the grid."""
        rows, cols = self.shape
        col = min(max(int(x_m / self.cell_size_m), 0), cols - 1)
        row = min(max(int(y_m / self.cell_size_m), 0), rows - 1)
        return row, col

    def to_metres(self, cell: Cell) -> tuple[float, float]:
        """Cell -> (x_m, y_m) of its centre."""
        return (cell[1] + 0.5) * self.cell_size_m, (cell[0] + 0.5) * self.cell_size_m

    def validate(self) -> None:
        """Check shapes, finiteness and value ranges of every layer.

        Raises:
            TerrainValidationError: a layer has the wrong shape or a non-physical value.
        """
        shape = self.elevation.shape
        layers = {"slope": self.slope, "walk_eta": self.walk_eta,
                  "rock_hazard": self.rock_hazard, "passable_walk": self.passable_walk,
                  "forbidden": self.forbidden, "is_shelter": self.is_shelter}
        for name, layer in layers.items():
            if layer is None or layer.shape != shape:
                got = None if layer is None else layer.shape
                raise TerrainValidationError(f"layer '{name}' has shape {got}, expected {shape}")
        for name in ("elevation", "slope", "walk_eta", "rock_hazard"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise TerrainValidationError(f"layer '{name}' contains NaN or infinity")
        assert self.slope is not None and self.walk_eta is not None and self.rock_hazard is not None
        if np.any(self.slope < 0.0) or np.any(self.slope >= np.pi / 2):
            raise TerrainValidationError("slope must be in [0, pi/2) rad")
        if np.any(self.walk_eta < 1.0):
            raise TerrainValidationError("walk_eta must be >= 1.0 (1.0 = firm level ground)")
        if np.any(self.rock_hazard < 0.0) or np.any(self.rock_hazard > 1.0):
            raise TerrainValidationError("rock_hazard must be in [0, 1]")
