"""What the graph algorithms need from whoever travels: an astronaut or a rover.

The route search (``pareto_astar``), the reachability analysis and the cost
cubes are written once, against these protocols. ``EdgeEvaluator`` implements
them for a suited astronaut, ``RoverEdgeEvaluator`` for a rover. Adding another
vehicle means implementing the protocols, not touching the algorithms.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Protocol, TypeVar, runtime_checkable

from ..core.config import SimConfig
from ..core.schemas import Cell


class SearchLabel(Protocol):
    """State the search carries: where, when, at what cost, and how it got there."""

    row: int
    col: int
    t_s: float
    cost: float
    distance_m: float
    action: str
    alive: bool
    parent: Any

    @property
    def cell(self) -> Cell: ...

    def chain(self) -> list[Any]: ...


L = TypeVar("L", bound=SearchLabel)


@runtime_checkable
class TraversalModel(Protocol[L]):
    """Physics of moving over the grid for one kind of traveller."""

    rows: int
    cols: int
    cell_size_m: float
    config: SimConfig

    @property
    def max_speed_m_s(self) -> float:
        """Upper bound of the ground speed, for the search heuristic."""
        ...

    @property
    def time_weight_per_s(self) -> float:
        """Cost of one second of travel, for the search heuristic."""
        ...

    def in_bounds(self, cell: Cell) -> bool: ...

    def is_walkable(self, cell: Cell) -> bool: ...

    def require_walkable(self, cell: Cell, role: str) -> None: ...

    def successors(self, label: L) -> tuple[list[tuple[Cell, L]], int]:
        """States reachable in one action, and how many moves the physics refused.

        Actions are moves to the eight neighbours and, where the traveller has
        one, a stay in place (an astronaut resting, a rover recharging).
        """
        ...

    def dominates(self, y: L, x: L, tol: float) -> bool:
        """True when ``y`` is at least as good as ``x`` in cost and in every state variable."""
        ...

    def edge_metrics(self, a: Cell, b: Cell, dist_m: float, state: L) -> Mapping[str, float] | None:
        """Named costs of one move for a reference state, or ``None`` when it is refused."""
        ...


class Constraints(Protocol[L]):
    """Hard limits on a state."""

    def violated(self, label: L, prev_cell: Cell | None = None) -> tuple[Enum, str] | None:
        """First rule the state breaks, with a short detail, or ``None``."""
        ...

    def describe(self, rule: Enum, detail: str, distance_km: float) -> str:
        """Readable sentence for a rule that blocked the search."""
        ...
