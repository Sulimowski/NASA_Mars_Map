"""Shared fixtures: reference profiles (W = 80 kg, L = 60 kg, eta = 1.2) and helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.core.config import DEFAULT_CONFIG, SimConfig  # noqa: E402
from src.core.constants import SOL_S  # noqa: E402
from src.core.schemas import AstronautProfile, Cell, Label, SuitProfile  # noqa: E402
from src.providers.interfaces import EnvironmentProvider, TerrainProvider  # noqa: E402
from src.providers.synthetic import (ConstantEnvironment, reference_day_sample,  # noqa: E402
                                     reference_night_sample)
from src.routing.constraints import ConstraintChecker  # noqa: E402
from src.routing.pareto_astar import ParetoAStar  # noqa: E402
from src.routing.shelter_maps import ShelterMaps, compute_shelter_maps  # noqa: E402
from src.routing.transition import EdgeEvaluator  # noqa: E402

REFERENCE_ETA = 1.2
REFERENCE_SPEED_M_S = 3.0 / 3.6
START_TIME_S = 0.4 * SOL_S

EvaluatorFactory = Callable[..., EdgeEvaluator]


@pytest.fixture
def astronaut() -> AstronautProfile:
    return AstronautProfile()


@pytest.fixture
def suit() -> SuitProfile:
    return SuitProfile()


@pytest.fixture
def cfg() -> SimConfig:
    return DEFAULT_CONFIG


@pytest.fixture
def day_env() -> ConstantEnvironment:
    return ConstantEnvironment(reference_day_sample())


@pytest.fixture
def night_env() -> ConstantEnvironment:
    return ConstantEnvironment(reference_night_sample())


@pytest.fixture
def make_evaluator(astronaut: AstronautProfile, suit: SuitProfile,
                   day_env: ConstantEnvironment) -> EvaluatorFactory:
    """Factory: ``make_evaluator(terrain, env=None, sols=30, suit=None, config=None)``."""

    def factory(terrain: TerrainProvider, env: EnvironmentProvider | None = None,
                sols: float = 30.0, suit_override: SuitProfile | None = None,
                config: SimConfig | None = None) -> EdgeEvaluator:
        return EdgeEvaluator(terrain, env or day_env, astronaut, suit_override or suit,
                             config or DEFAULT_CONFIG, time_on_surface_s=sols * SOL_S)

    return factory


def walk_straight(evaluator: EdgeEvaluator, start: Cell, steps: int,
                  step: tuple[int, int] = (0, 1), t_s: float = START_TIME_S) -> Label:
    """Walk ``steps`` edges in a fixed direction with no constraints; returns the last label."""
    label = evaluator.initial_label(start, t_s)
    dist = evaluator.cell_size_m * (step[0] ** 2 + step[1] ** 2) ** 0.5
    for _ in range(steps):
        nxt = (label.row + step[0], label.col + step[1])
        tr = evaluator.walk_edge(label.cell, nxt, dist, label)
        assert tr is not None, f"edge into {nxt} is impassable"
        label = evaluator.advance(label, nxt, tr, "walk")
    return label


def make_search(evaluator: EdgeEvaluator, base: Cell, t_s: float = START_TIME_S,
                spe_watch: bool = False) -> tuple[ParetoAStar, ConstraintChecker, ShelterMaps]:
    """Shelter maps, constraint checker and search engine for a base cell."""
    maps = compute_shelter_maps(evaluator, base, t_s)
    checker = ConstraintChecker(evaluator, maps,
                                dose_budget_sv=evaluator.config.radiation.per_eva_limit_sv,
                                spe_watch=spe_watch)
    return ParetoAStar(evaluator, checker), checker, maps
