"""Contracts the physiology and routing layers depend on.

``TerrainProvider`` and ``EnvironmentProvider`` are structural protocols: any
object with these members works, so a synthetic testbed can be swapped for the
real data of the Terrain and Environment teams without touching ``physiology/``
or ``routing/``.

Grid convention: ``layer[row, col]``; a cell is ``(row, col)``; all cells are
squares of ``cell_size_m``. Neighbours are 8-connected, at distance
``cell_size_m`` (edge) or ``cell_size_m * sqrt(2)`` (corner).
"""

from __future__ import annotations

import math
from typing import Final, Iterator, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..core.schemas import Cell, EnvSample

FloatGrid = NDArray[np.float64]
BoolGrid = NDArray[np.bool_]

NEIGHBOR_OFFSETS: Final[tuple[tuple[int, int, float], ...]] = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
)
"""(d_row, d_col, distance in cells). The index in this tuple is the direction
index of every ``[8, rows, cols]`` edge-cost cube."""


@runtime_checkable
class TerrainProvider(Protocol):
    """Raster terrain of one exploration site.

    Layers (all ``[rows, cols]``):
        elevation       m
        slope           rad, steepest slope of the cell (direction-free)
        walk_eta        Pandolf terrain factor, 1.0 firm ground .. 2.1 loose sand
        rock_hazard     0..1, tripping hazard from rocks and micro-relief
        passable_walk   bool, False where a suited astronaut cannot walk
        forbidden       bool, True where entry is not allowed (no-go, planetary protection)
        is_shelter      bool, True for the habitat and any other safe haven
    """

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def cell_size_m(self) -> float: ...

    @property
    def elevation(self) -> FloatGrid: ...

    @property
    def slope(self) -> FloatGrid: ...

    @property
    def walk_eta(self) -> FloatGrid: ...

    @property
    def rock_hazard(self) -> FloatGrid: ...

    @property
    def passable_walk(self) -> BoolGrid: ...

    @property
    def forbidden(self) -> BoolGrid: ...

    @property
    def is_shelter(self) -> BoolGrid: ...

    def in_bounds(self, cell: Cell) -> bool:
        """True when the cell lies inside the grid."""
        ...

    def neighbors(self, cell: Cell) -> Iterator[tuple[Cell, float]]:
        """In-bounds 8-neighbours of a cell with the horizontal distance to each, m."""
        ...

    def validate(self) -> None:
        """Raise ``TerrainValidationError`` when the data is inconsistent."""
        ...


@runtime_checkable
class EnvironmentProvider(Protocol):
    """Environment as a function of place and time."""

    def get_env(self, cell: Cell, t_s: float) -> EnvSample:
        """Conditions at ``cell`` at mission time ``t_s`` (seconds)."""
        ...
