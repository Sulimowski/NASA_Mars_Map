"""H3 and H6: sol selection, the series-only environment, the dose budget."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import walk_straight
from src.core.config import SimConfig
from src.core.constants import EARTH_DAY_S, HOUR_S, SOL_S, ZERO_CELSIUS_K
from src.core.exceptions import EnvironmentValidationError, ValidationError
from src.physiology.radiation import DoseLedger
from src.planning import dose_budget
from src.planning.diurnal import (SeriesEnvironment, day_length_s, declination_rad,
                                  diurnal_shape, sun_elevation_rad)
from src.planning.sol_selection import (SolSeries, assess_sols, best_sol, detect_particle_events,
                                        eva_window_s, rank_sols, sol_series_from_site)
from src.providers.synthetic import flat_terrain

QUIET = 0.67e-3


def series_with_event(count: int = 10, event: tuple[int, ...] = (5, 6)) -> SolSeries:
    dose = [QUIET * (4.0 if i in event else 1.0) for i in range(count)]
    return SolSeries(sols=tuple(range(count)), dose_sv=tuple(dose))


# ---- H3.1 sol selection ----------------------------------------------------
def test_particle_event_and_its_guard_sols_are_rejected(cfg: SimConfig) -> None:
    series = series_with_event()
    assert detect_particle_events(series.dose_sv, cfg.sol_selection) == [
        i in (5, 6) for i in range(10)]
    result = assess_sols(series, cfg.sol_selection)
    assert [a.sol for a in result if not a.allowed] == [4, 5, 6, 7]
    assert result[5].particle_event and "particle event" in result[5].reasons[0]
    assert not result[4].particle_event and "guard" in result[4].reasons[0]
    assert all(a.reasons == () for a in result if a.allowed)


def test_guard_uses_sol_numbers_not_positions(cfg: SimConfig) -> None:
    dose = (QUIET, QUIET, QUIET, 4 * QUIET, QUIET)
    series = SolSeries(sols=(0, 1, 2, 10, 11), dose_sv=dose)
    rejected = [a.sol for a in assess_sols(series, cfg.sol_selection) if not a.allowed]
    assert rejected == [10, 11]                      # sol 2 is eight sols before the event


def test_eva_window_follows_the_daylight(cfg: SimConfig) -> None:
    window = eva_window_s(0.0, 0.0, cfg.sol_selection)
    assert window == pytest.approx((SOL_S / 4 + HOUR_S, 3 * SOL_S / 4 - HOUR_S))
    assert eva_window_s(math.radians(80), math.radians(-25), cfg.sol_selection) is None
    polar = assess_sols(series_with_event(), cfg.sol_selection, latitude_rad=math.radians(80),
                        declination_of_sol=lambda sol: math.radians(-25))
    assert not any(a.allowed for a in polar)
    assert all("daylight" in a.reasons[-1] for a in polar)
    assert best_sol(polar) is None


def test_temperature_and_allocation_rules(cfg: SimConfig) -> None:
    cold = SolSeries(sols=(0, 1, 2), dose_sv=(QUIET,) * 3,
                     ground_temp_min_k=(150.0, 190.0, 195.0),
                     ground_temp_max_k=(180.0, 250.0, 262.0))
    result = assess_sols(cold, cfg.sol_selection)
    assert [a.allowed for a in result] == [False, True, True]
    assert "cold" in result[0].reasons[0]
    assert best_sol(result).sol == 2                 # equal dose: the warmer afternoon wins

    tight = assess_sols(series_with_event(event=()), cfg.sol_selection,
                        dose_allocation_sv=0.4e-3)
    assert not any(a.allowed for a in tight)
    assert "allocation" in tight[0].reasons[0]


def test_ranking_prefers_low_dose(cfg: SimConfig) -> None:
    series = SolSeries(sols=(3, 4, 5, 6), dose_sv=(0.70e-3, 0.60e-3, 0.67e-3, 0.60e-3))
    ranked = rank_sols(assess_sols(series, cfg.sol_selection))
    assert [a.sol for a in ranked] == [4, 6, 5, 3]


def test_sol_series_is_validated() -> None:
    with pytest.raises(ValidationError):
        SolSeries(sols=(), dose_sv=())
    with pytest.raises(ValidationError):
        SolSeries(sols=(0, 0), dose_sv=(QUIET, QUIET))
    with pytest.raises(ValidationError):
        SolSeries(sols=(0, 1), dose_sv=(QUIET,))
    with pytest.raises(ValidationError):
        SolSeries(sols=(0,), dose_sv=(-1.0,))
    with pytest.raises(ValidationError):
        SolSeries(sols=(0,), dose_sv=(QUIET,), ground_temp_min_k=(200.0,))


def test_series_are_read_from_a_site_whatever_it_carries(cfg: SimConfig) -> None:
    full = SimpleNamespace(name="real", series={
        "sol": [100, 101], "radiation_msv_per_sol": [0.45, 0.50],
        "temp_min_c": [-80.0, -79.0], "temp_max_c": [-10.0, -8.0],
        "ground_temp_min_c": [-90.0, -88.0], "ground_temp_max_c": [5.0, 7.0]})
    series = sol_series_from_site(full, cfg.sol_selection)
    assert series.sols == (100, 101) and series.dose_sv == pytest.approx((0.45e-3, 0.50e-3))
    assert series.ground_temp_max_k == pytest.approx((5.0 + ZERO_CELSIUS_K, 7.0 + ZERO_CELSIUS_K))

    air_only = SimpleNamespace(name="stub", series={"temp_min_c": [-78.0], "temp_max_c": [-12.0]})
    series = sol_series_from_site(air_only, cfg.sol_selection)
    assert series.sols == (0,) and series.dose_sv == (cfg.sol_selection.typical_dose_sv_per_sol,)
    assert series.ground_temp_min_k == pytest.approx((-78.0 + ZERO_CELSIUS_K,))

    bare = SimpleNamespace(name="bare", series={"radiation_msv_per_sol": [0.6, 0.7]})
    assert sol_series_from_site(bare, cfg.sol_selection).ground_temp_max_k is None
    with pytest.raises(ValidationError):
        sol_series_from_site(SimpleNamespace(name="empty", series={}), cfg.sol_selection)


# ---- H6 dose budget --------------------------------------------------------
def test_allocation_and_eva_share() -> None:
    assert dose_budget.sol_allocation_sv(0.3, 0.1, 100) == pytest.approx(2.0e-3)
    assert dose_budget.sol_allocation_sv(0.3, 0.5, 100) == 0.0
    assert dose_budget.eva_dose_sv(0.67e-3, SOL_S / 4) == pytest.approx(0.1675e-3)
    with pytest.raises(ValidationError):
        dose_budget.sol_allocation_sv(0.3, 0.1, 0)
    with pytest.raises(ValidationError):
        dose_budget.sol_allocation_sv(-0.3, 0.1, 10)


def test_dose_decision(cfg: SimConfig) -> None:
    sel = cfg.sol_selection
    ok = dose_budget.assess_dose(QUIET, sols_remaining=300, received_sv=0.05,
                                 eva_duration_s=6 * HOUR_S, cfg=sel)
    assert ok.allowed and ok.reasons == ()
    assert ok.allocation_sv == pytest.approx((0.3 - 0.05) / 300)
    assert ok.expected_eva_sv == pytest.approx(QUIET * 6 * HOUR_S / SOL_S)

    event = dose_budget.assess_dose(4 * QUIET, sols_remaining=300, received_sv=0.05,
                                    eva_duration_s=6 * HOUR_S, cfg=sel)
    assert not event.allowed and "allocation" in event.reasons[0]

    ledger = DoseLedger(cfg.radiation, career_dose_sv=0.5999)
    limited = dose_budget.assess_dose(QUIET, sols_remaining=300, received_sv=0.0,
                                      eva_duration_s=HOUR_S, cfg=sel, ledger=ledger,
                                      t_s=10 * EARTH_DAY_S)
    assert not limited.allowed and any("career" in r for r in limited.reasons)


# ---- H3.2 series-only environment -----------------------------------------
def test_sun_geometry() -> None:
    assert day_length_s(0.0, 0.0) == pytest.approx(SOL_S / 2)
    assert day_length_s(math.radians(80), math.radians(-25)) == 0.0
    assert day_length_s(math.radians(80), math.radians(25)) == pytest.approx(SOL_S)
    assert sun_elevation_rad(0.0, 0.0, SOL_S / 2) == pytest.approx(math.pi / 2)
    assert sun_elevation_rad(0.0, 0.0, 0.0) == pytest.approx(-math.pi / 2)
    assert sun_elevation_rad(0.0, 0.0, SOL_S / 4) == pytest.approx(0.0, abs=1e-9)
    assert declination_rad(math.pi / 2) == pytest.approx(math.radians(25.19))
    assert declination_rad(0.0) == 0.0


def test_diurnal_shape_runs_from_sunrise_minimum_to_afternoon_peak() -> None:
    day, lag = SOL_S / 2, 3_700.0
    assert diurnal_shape(SOL_S / 4, day, lag, 3.0) == pytest.approx(0.0, abs=1e-12)
    assert diurnal_shape(SOL_S / 2 + lag, day, lag, 3.0) == pytest.approx(1.0)
    samples = [diurnal_shape(t, day, lag, 3.0) for t in np.linspace(0.0, SOL_S, 200)]
    assert min(samples) >= -1e-12 and max(samples) <= 1.0 + 1e-12
    night = [diurnal_shape(t, day, lag, 3.0) for t in np.linspace(0.76 * SOL_S, 1.24 * SOL_S, 30)]
    assert night == sorted(night, reverse=True)       # the ground only cools through the night


def test_series_environment_over_a_sol(cfg: SimConfig) -> None:
    env = SeriesEnvironment(ground_temp_min_k=195.0, ground_temp_max_k=261.0,
                            dose_sv_per_sol=QUIET, quiet_dose_sv_per_sol=QUIET,
                            start_time_of_sol_s=0.0)
    for t in np.linspace(0.0, SOL_S, 97):
        sample = env.get_env((0, 0), float(t))
        sample.validate()
        assert 195.0 - 1e-6 <= sample.t_ground_k <= 261.0 + 1e-6
    noon, midnight = env.get_env((3, 4), SOL_S / 2), env.get_env((3, 4), 0.0)
    assert noon.solar_direct_w_m2 > 0.0 and midnight.solar_direct_w_m2 == 0.0
    assert noon.t_ground_k > midnight.t_ground_k and not noon.spe_active
    assert noon.dose_rate_gcr_sv_s * SOL_S == pytest.approx(QUIET)

    stormy = SeriesEnvironment(195.0, 261.0, dose_sv_per_sol=4 * QUIET,
                               quiet_dose_sv_per_sol=QUIET, particle_event=True)
    sample = stormy.get_env((0, 0), 0.0)
    assert sample.spe_active
    assert (sample.dose_rate_gcr_sv_s + sample.spe_dose_rate_sv_s) * SOL_S == pytest.approx(4 * QUIET)
    with pytest.raises(EnvironmentValidationError):
        SeriesEnvironment(261.0, 195.0, QUIET, QUIET)


def test_series_environment_from_a_site_drives_the_simulator(cfg: SimConfig, make_evaluator) -> None:
    doses = [0.67] * 9 + [2.68]
    site = SimpleNamespace(name="s", center_lat=-4.75, series={
        "sol": list(range(10)), "radiation_msv_per_sol": doses,
        "ground_temp_min_c": [-80.0] * 10, "ground_temp_max_c": [-5.0] * 10,
        "pressure_pa": [740.0] * 10})
    quiet = SeriesEnvironment.from_site(site, 3, start_ltst_h=10.0)
    assert not quiet.particle_event and quiet.pressure_pa == 740.0
    assert quiet.ground_temp_max_k == pytest.approx(-5.0 + ZERO_CELSIUS_K)
    assert quiet.start_time_of_sol_s == pytest.approx(10.0 / 24.0 * SOL_S)
    event = SeriesEnvironment.from_site(site, 9)
    assert event.particle_event and event.quiet_dose_sv_per_sol == pytest.approx(0.67e-3)
    with pytest.raises(EnvironmentValidationError):
        SeriesEnvironment.from_site(site, 42)

    evaluator = make_evaluator(flat_terrain(3, 30, 10.0), env=quiet)
    last = walk_straight(evaluator, (1, 0), 20, t_s=0.0)
    assert last.distance_m == pytest.approx(200.0) and last.dose_sv > 0.0
    assert last.transition is not None and last.transition.speed_m_s == pytest.approx(3.0 / 3.6)
