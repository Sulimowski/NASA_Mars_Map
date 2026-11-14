"""H9: where the crew member can talk to the base, and for how long a route is out of contact.

Coverage
--------
A cell is in contact when the straight ray from the base antenna to the
crew member's antenna clears the ground everywhere between them. With base
height ``h_b``, crew height ``h_c``, and a point at fraction ``f`` of the way::

    ray(f)    = z_base + h_b + f * (z_cell + h_c - z_base - h_b)
    bulge(f)  = d1 * d2 / (2 * R)            planet curvature, d1 + d2 = distance
    contact   = min over f of (ray - ground - bulge) >= 0

Two sources produce a ``GridCoverage``:

* ``coverage_from_terrain`` computes the line of sight on the simulator's own grid;
* ``coverage_from_radio_map`` takes the radio team's map (0 shadow, 1 degraded,
  2 clear, on its own grid) and resamples it. Degraded counts as contact: with
  antennas this low the ground always intrudes into the Fresnel zone.

Rule
----
The crew member may be out of contact for at most ``max_no_contact_s`` at a
stretch. ``contact_along`` measures the stretches of a walked route. A longer
stretch does not stop the route; it is reported and lowers the verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.config import CommsConfig
from ..core.exceptions import ValidationError
from ..core.schemas import Cell, Label
from ..providers.interfaces import TerrainProvider

_RADIO_SHADOW = 0


class CoverageProvider(Protocol):
    """Anything that can say whether a cell has radio contact with the base."""

    def in_contact(self, cell: Cell) -> bool:
        """True when the crew member at ``cell`` can reach the base."""
        ...


@dataclass(frozen=True, slots=True)
class GridCoverage:
    """Radio contact as a raster on the simulator's grid (True = contact)."""

    mask: NDArray[np.bool_]

    def in_contact(self, cell: Cell) -> bool:
        """True when ``cell`` has contact; cells outside the raster have none."""
        row, col = cell
        rows, cols = self.mask.shape
        return 0 <= row < rows and 0 <= col < cols and bool(self.mask[row, col])

    @property
    def share(self) -> float:
        """Share of the site in contact, 0..1."""
        return float(self.mask.mean())


@dataclass(frozen=True, slots=True)
class ContactGap:
    """One stretch of a route without contact."""

    start_index: int          # index in the label chain of the first step without contact
    end_index: int            # index of the last step without contact
    duration_s: float
    distance_m: float


@dataclass(frozen=True, slots=True)
class ContactReport:
    """Radio contact along a walked route.

    Attributes:
        no_contact_s: total time without contact.
        longest_gap_s: longest single stretch without contact.
        gaps: every stretch, in travel order.
        violates: the longest stretch exceeds ``max_no_contact_s``.
        contact_share: share of the route time with contact, 0..1.
    """

    no_contact_s: float
    longest_gap_s: float
    violates: bool
    contact_share: float
    gaps: tuple[ContactGap, ...] = field(default_factory=tuple)


def line_of_sight_mask(elevation: NDArray[np.float64], cell_size_m: float, base: Cell,
                       cfg: CommsConfig) -> NDArray[np.bool_]:
    """Cells with an unobstructed ray to the base antenna.

    Every cell is tested at once: the ray to each cell is sampled at steps of
    about one cell, and the lowest clearance over the ground is kept.

    Raises:
        ValidationError: the base lies outside the raster.
    """
    rows, cols = elevation.shape
    if not (0 <= base[0] < rows and 0 <= base[1] < cols):
        raise ValidationError(f"base cell {base} is outside the raster {(rows, cols)}")
    z = np.asarray(elevation, dtype=np.float64)
    rr, cc = np.mgrid[0:rows, 0:cols].astype(np.float64)
    d_row, d_col = rr - base[0], cc - base[1]
    dist_m = np.hypot(d_row, d_col) * cell_size_m
    z_tx = float(z[base]) + cfg.base_antenna_height_m
    z_rx = z + cfg.astronaut_antenna_height_m
    steps = max(2, int(math.ceil(float(np.hypot(d_row, d_col).max()))))
    clearance = np.full(z.shape, np.inf)
    for step in range(1, steps):
        frac = step / steps
        sample_row = np.clip(np.rint(base[0] + frac * d_row).astype(np.intp), 0, rows - 1)
        sample_col = np.clip(np.rint(base[1] + frac * d_col).astype(np.intp), 0, cols - 1)
        ray = z_tx + frac * (z_rx - z_tx)
        bulge = (frac * dist_m) * ((1.0 - frac) * dist_m) / (2.0 * cfg.planet_radius_m)
        clear = ray - z[sample_row, sample_col] - bulge
        own_cell = ((sample_row == base[0]) & (sample_col == base[1])) | (
            (sample_row == rr) & (sample_col == cc))      # the antennas' own cells do not block
        np.minimum(clearance, np.where(own_cell, np.inf, clear), out=clearance)
    return clearance >= 0.0


def coverage_from_terrain(terrain: TerrainProvider, base: Cell, cfg: CommsConfig) -> GridCoverage:
    """Line-of-sight coverage of a base antenna on the simulator's grid."""
    return GridCoverage(line_of_sight_mask(terrain.elevation, terrain.cell_size_m, base, cfg))


def coverage_from_radio_map(radio_map: NDArray[np.integer], terrain: TerrainProvider,
                            degraded_is_contact: bool = True) -> GridCoverage:
    """Coverage from the radio team's map (0 shadow, 1 degraded, 2 clear).

    The map may be on any grid covering the same area; each simulator cell takes
    the value of the map cell under its centre.

    Raises:
        ValidationError: the map is not a non-empty 2-D array.
    """
    source = np.asarray(radio_map)
    if source.ndim != 2 or source.size == 0:
        raise ValidationError("radio_map must be a non-empty 2-D array")
    rows, cols = terrain.shape
    row_index = np.minimum(((np.arange(rows) + 0.5) / rows * source.shape[0]).astype(np.intp),
                           source.shape[0] - 1)
    col_index = np.minimum(((np.arange(cols) + 0.5) / cols * source.shape[1]).astype(np.intp),
                           source.shape[1] - 1)
    resampled = source[np.ix_(row_index, col_index)]
    threshold = _RADIO_SHADOW if degraded_is_contact else 1
    return GridCoverage(resampled > threshold)


def contact_along(labels: Sequence[Label], coverage: CoverageProvider,
                  cfg: CommsConfig) -> ContactReport:
    """Radio contact over a walked route.

    Each step of the route (walk, rest or task) counts as without contact when
    the cell it ends in has none.

    Args:
        labels: the label chain of the route (``Label.chain()``), start first.
    """
    gaps: list[ContactGap] = []
    total = no_contact = 0.0
    gap_start, gap_time, gap_dist = -1, 0.0, 0.0
    for index in range(1, len(labels)):
        label = labels[index]
        tr = label.transition
        if tr is None:
            continue
        total += tr.dt_s
        if coverage.in_contact(label.cell):
            if gap_start >= 0:
                gaps.append(ContactGap(gap_start, index - 1, gap_time, gap_dist))
                gap_start, gap_time, gap_dist = -1, 0.0, 0.0
            continue
        no_contact += tr.dt_s
        if gap_start < 0:
            gap_start = index
        gap_time += tr.dt_s
        gap_dist += tr.distance_m
    if gap_start >= 0:
        gaps.append(ContactGap(gap_start, len(labels) - 1, gap_time, gap_dist))
    longest = max((gap.duration_s for gap in gaps), default=0.0)
    return ContactReport(no_contact_s=no_contact, longest_gap_s=longest,
                         violates=longest > cfg.max_no_contact_s,
                         contact_share=1.0 - no_contact / total if total > 0.0 else 1.0,
                         gaps=tuple(gaps))
