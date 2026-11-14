"""H9: radio coverage and contact along a route."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import walk_straight
from src.comms.coverage import (GridCoverage, contact_along, coverage_from_radio_map,
                                coverage_from_terrain, line_of_sight_mask)
from src.core.config import CommsConfig, SimConfig
from src.core.exceptions import ValidationError
from src.providers.synthetic import flat_terrain, hill_terrain, wall_terrain


def test_open_plain_is_fully_covered(cfg: SimConfig) -> None:
    coverage = coverage_from_terrain(flat_terrain(20, 20, 10.0), (10, 10), cfg.comms)
    assert coverage.mask.all() and coverage.share == 1.0
    assert coverage.in_contact((0, 0)) and not coverage.in_contact((-1, 0))
    assert not coverage.in_contact((20, 5))


def test_ridge_casts_a_radio_shadow(cfg: SimConfig) -> None:
    terrain = wall_terrain(21, 41, 10.0, wall_col=20)            # 17 m ridge, 10 m mast
    coverage = coverage_from_terrain(terrain, (10, 5), cfg.comms)
    assert coverage.mask[:, :17].all()                          # the base side
    assert coverage.in_contact((10, 20))                        # the crest sees the base
    assert not coverage.mask[:, 26:].any()                      # everything behind the ridge
    tall = replace(cfg.comms, base_antenna_height_m=200.0)
    assert coverage_from_terrain(terrain, (10, 5), tall).share > coverage.share


def test_a_pass_through_the_ridge_lets_the_signal_through(cfg: SimConfig) -> None:
    terrain = wall_terrain(21, 41, 10.0, wall_col=20, gap_rows=range(8, 13))
    coverage = coverage_from_terrain(terrain, (10, 5), cfg.comms)
    assert coverage.in_contact((10, 35))                        # straight through the pass
    assert not coverage.in_contact((2, 35))                     # behind the ridge beside it


def test_hill_hides_the_far_side_but_not_its_top(cfg: SimConfig) -> None:
    terrain = hill_terrain(31, 61, 10.0, height_m=40.0, sigma_m=25.0, center=(15, 30))
    coverage = coverage_from_terrain(terrain, (15, 5), cfg.comms)
    assert coverage.in_contact((15, 30))
    assert not coverage.in_contact((15, 45))
    assert coverage.in_contact((1, 45))                         # around the side


def test_line_of_sight_rejects_a_base_outside_the_raster(cfg: SimConfig) -> None:
    with pytest.raises(ValidationError):
        line_of_sight_mask(np.zeros((5, 5)), 10.0, (9, 9), cfg.comms)


def test_radio_team_map_is_resampled_to_the_simulator_grid() -> None:
    radio_map = np.array([[2, 2, 1, 0],
                          [2, 1, 0, 0]], dtype=np.uint8)        # 0 shadow, 1 degraded, 2 clear
    terrain = flat_terrain(4, 8, 10.0)
    coverage = coverage_from_radio_map(radio_map, terrain)
    assert coverage.mask.shape == (4, 8)
    assert coverage.mask[0].tolist() == [True] * 6 + [False] * 2
    assert coverage.mask[3].tolist() == [True] * 4 + [False] * 4
    strict = coverage_from_radio_map(radio_map, terrain, degraded_is_contact=False)
    assert strict.mask[0].tolist() == [True] * 4 + [False] * 4
    with pytest.raises(ValidationError):
        coverage_from_radio_map(np.zeros(4), terrain)


def test_contact_along_a_route(make_evaluator, cfg: SimConfig) -> None:
    evaluator = make_evaluator(flat_terrain(3, 40, 10.0))
    labels = walk_straight(evaluator, (1, 0), 30).chain()       # 12 s per 10 m step
    mask = np.ones((3, 40), dtype=bool)
    mask[:, 10:21] = False                                      # 11 steps without contact
    report = contact_along(labels, GridCoverage(mask), cfg.comms)
    assert report.no_contact_s == pytest.approx(11 * 12.0)
    assert report.longest_gap_s == pytest.approx(11 * 12.0)
    assert len(report.gaps) == 1 and not report.violates
    assert report.gaps[0].start_index == 10 and report.gaps[0].end_index == 20
    assert report.gaps[0].distance_m == pytest.approx(110.0)
    assert report.contact_share == pytest.approx(1.0 - 11 / 30)

    strict = contact_along(labels, GridCoverage(mask), CommsConfig(max_no_contact_s=60.0))
    assert strict.violates

    mask[:, 25:] = False                                        # a second gap that runs to the end
    two = contact_along(labels, GridCoverage(mask), cfg.comms)
    assert [g.start_index for g in two.gaps] == [10, 25] and two.gaps[1].end_index == 30
    assert two.longest_gap_s == pytest.approx(11 * 12.0)

    clear = contact_along(labels, GridCoverage(np.ones((3, 40), dtype=bool)), cfg.comms)
    assert clear.no_contact_s == 0.0 and clear.gaps == () and clear.contact_share == 1.0
    assert contact_along(labels[:1], GridCoverage(mask), cfg.comms).contact_share == 1.0
