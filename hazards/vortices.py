"""Convective vortices (dust devils) seen by surface stations: catalogs and encounter rates.

A vortex that passes over a station shows up as a short drop in air pressure. A catalog of such
drops is our label: how often does a vortex of at least a given strength hit one fixed point.
An astronaut or a robot is also one point, so the station rate is the rate they would see.

    events   = load("insight")                   # one row per vortex encounter, see EVENT_COLUMNS
    coverage = coverage_of("insight", events)    # the sols the station listened, and how much
    rates    = rate_table(events, coverage)      # encounters per hour by season and local hour

Stations so far: InSight (Elysium Planitia) and Perseverance (Jezero crater). Curiosity next.

This package does not import `mars_map`, and keeps its downloads in `hazards/data/` (not in git).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .marstime import MARS_HOUR_IN_EARTH_HOURS, SOL_SECONDS, mars_ls

HAZARD_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = HAZARD_DIR / "raw"

# "The Whirlwinds of Elysium", Lorenz et al., PDS Atmospheres node, doi:10.17189/71eb-n087.
# Sols 15 to 390, every pressure drop of 0.8 Pa or more. Public domain (NASA PDS).
INSIGHT_URL = ("https://atmos.nmsu.edu/PDS/data/PDS4/InSight_vortex_bundle/data_vortex/"
               "InSight_Vortex_Catalog_000_390_01.csv")

# Hueso et al. (2023), JGR Planets 128, doi:10.1029/2022JE007516; tables on Zenodo,
# doi:10.5281/zenodo.7315863, CC BY 4.0. Sols 15 to 415, every pressure drop of 0.3 Pa or more.
PERSEVERANCE_URL = "https://zenodo.org/api/records/7315863/files/{}/content"
PERSEVERANCE_DROPS = "Table_S1_Mars_Jezero_MEDA__Pressure_Drops.xlsx"
PERSEVERANCE_VORTICES = "Table_S2_Mars_Jezero_MEDA_Vortices.xlsx"

# The weakest drop every catalog lists in full, so stations can be compared.
COMMON_MIN_DROP_PA = 0.8

STATIONS = {
    "insight": {"name": "InSight", "place": "Elysium Planitia", "lat": 4.502, "lon": 135.623},
    "perseverance": {"name": "Perseverance", "place": "Jezero crater", "lat": 18.445, "lon": 77.451},
}

EVENT_COLUMNS = [
    "station", "sol",
    "ltst_h",            # local true solar time, 0..24: the Sun is highest at 12
    "lmst_h",            # local mean solar time, 0..24: the clock the station is scheduled by
    "ls_deg", "utc",
    "drop_pa",           # peak pressure drop
    "width_s",           # how long the drop lasted (full width at half maximum)
    "wind_max_ms", "wind_before_ms", "pressure_pa", "air_temp_k", "ground_temp_k",
    "dust_dip_pct",      # dimming of sunlight during the drop: above 0 means the vortex carried dust
]

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
_SCLK_EPOCH = pd.Timestamp("2000-01-01 11:58:55.816", tz="UTC")     # spacecraft clock zero


def fetch(url: str, folder: str, name: str | None = None) -> Path:
    """Download a catalog once into hazards/data/raw/<folder>/ and return its path."""
    path = RAW_DIR / folder / (name or url.rsplit("/", 1)[-1])
    if not path.exists():
        import truststore                       # the PDS certificate chain needs the OS store
        truststore.inject_into_ssl()
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    return path


def _seconds(utc: pd.Series) -> pd.Series:
    return (utc - _EPOCH).dt.total_seconds()


def _sol0_seconds(events: pd.DataFrame) -> float:
    """Unix time of local midnight of sol 0, estimated from the events (robust to a few bad dates)."""
    return float(np.median(_seconds(events["utc"]) - (events["sol"] + events["lmst_h"] / 24.0) * SOL_SECONDS))


def _clock_hours(text: pd.Series) -> pd.Series:
    """'13:09:41', '9:58:13' or '00015M14:35:1' -> decimal hours."""
    parts = text.str.extract(r"(\d{1,2}):(\d\d)").astype(float)
    return parts[0] + parts[1] / 60.0


# ---------- catalogs ----------

def load_insight(path: str | Path | None = None) -> pd.DataFrame:
    """InSight vortex encounters, drops of 0.8 Pa and more, sols 15 to 390."""
    raw = pd.read_csv(Path(path) if path else fetch(INSIGHT_URL, "insight"))
    missing = lambda col: raw[col].where(raw[col] > -9.0)          # the catalog uses -9.9 for "no data"
    events = pd.DataFrame({
        "station": "insight",
        "sol": raw["Sol"].astype(int),
        "ltst_h": raw["LTST"].astype(float),
        "lmst_h": _clock_hours(raw["LMST"]),
        "ls_deg": raw["Ls"].astype(float),
        "utc": pd.to_datetime(raw["UTC"], utc=True),
        "drop_pa": raw["DP"].astype(float),
        "width_s": raw["Wdth"].astype(float),
        "wind_max_ms": missing("Wmax"),
        "wind_before_ms": missing("Wbef"),
        "pressure_pa": raw["Press"].astype(float),
        "air_temp_k": missing("MaxT1"),
        "ground_temp_k": np.nan,
        "dust_dip_pct": np.nan,                                     # not estimated in this catalog
    })
    # Two rows of sol 197 carry a date nine months off. Sol and local time are right, so any
    # date further than one sol from where the sol puts it is rebuilt from them.
    expected = _sol0_seconds(events) + (events["sol"] + events["lmst_h"] / 24.0) * SOL_SECONDS
    wrong = (_seconds(events["utc"]) - expected).abs() > SOL_SECONDS
    events.loc[wrong, "utc"] = _EPOCH + pd.to_timedelta(expected[wrong].round(), unit="s")
    return events[EVENT_COLUMNS]


def load_perseverance(folder: str | Path | None = None) -> pd.DataFrame:
    """Perseverance (MEDA) pressure drops of 0.3 Pa and more, sols 15 to 415.

    Needs `openpyxl` to read the two Excel tables. Temperatures and the dust flag come from the
    second table, which covers the drops the authors accepted as vortices; elsewhere they are NaN.
    """
    get = (lambda n: Path(folder) / n) if folder else (lambda n: fetch(PERSEVERANCE_URL.format(n), "perseverance", n))
    drops = pd.read_excel(get(PERSEVERANCE_DROPS))
    extra = pd.read_excel(get(PERSEVERANCE_VORTICES))[
        ["SCLK", "RDS_TOP7decrease(%)", "TIRS_Ground_Temp(K)", "ATS_1.45"]].drop_duplicates("SCLK")
    raw = drops.merge(extra, on="SCLK", how="left")
    return pd.DataFrame({
        "station": "perseverance",
        "sol": raw["Sol"].astype(int),
        "ltst_h": raw["LT(hr)"].astype(float),
        "lmst_h": _clock_hours(raw["LMST"]),
        "ls_deg": raw["Ls"].astype(float),
        "utc": _SCLK_EPOCH + pd.to_timedelta(raw["SCLK"], unit="s"),
        "drop_pa": raw["Amp(Pa)"].astype(float),
        "width_s": raw["G_FWHM(s)"].astype(float),
        "wind_max_ms": np.nan,
        "wind_before_ms": np.nan,
        "pressure_pa": np.nan,
        "air_temp_k": raw["ATS_1.45"],
        "ground_temp_k": raw["TIRS_Ground_Temp(K)"],
        "dust_dip_pct": pd.to_numeric(raw["RDS_TOP7decrease(%)"], errors="coerce"),
    })[EVENT_COLUMNS]


def load(station: str) -> pd.DataFrame:
    return {"insight": load_insight, "perseverance": load_perseverance}[station]()


# ---------- how long each station listened ----------

def continuous_coverage(events: pd.DataFrame, sol_first: int | None = None,
                        sol_last: int | None = None) -> pd.DataFrame:
    """One row per sol the station listened, with the season of that sol.

    Assumes the pressure sensor recorded all day on every sol from `sol_first` to `sol_last`.
    That is close to true for InSight in its first 390 sols. A sol without vortices is absent
    from a catalog, so the season of each sol is computed from the date, not read from events.
    `listened` is the fraction of each hour of the day that was recorded (here 1).
    """
    sol_first = int(events["sol"].min()) if sol_first is None else sol_first
    sol_last = int(events["sol"].max()) if sol_last is None else sol_last
    sols = np.arange(sol_first, sol_last + 1)
    noon = _EPOCH + pd.to_timedelta(_sol0_seconds(events) + (sols + 0.5) * SOL_SECONDS, unit="s")
    return pd.DataFrame({"station": events["station"].iloc[0], "sol": sols,
                         "ls_deg": mars_ls(noon.tz_localize(None).to_numpy()), "listened": 1.0})


def alternating_coverage(events: pd.DataFrame, ls_bin: float = 30.0, min_events: int = 60) -> pd.DataFrame:
    """Coverage of a station that records in one-hour blocks, estimated from the catalog itself.

    MEDA on Perseverance records odd hours on one sol and even hours on the next, plus extra
    blocks when power allows. So drops fall mostly on one parity of (clock hour + sol). If the
    regular hours are recorded in full and the other hours a fraction f of the time, the two
    parities hold events in the ratio 1 : f, and the station listened (1 + f) / 2 of the time.

    This is an estimate, not the real schedule: where the regular hours were not recorded in
    full, coverage is overestimated and the rates come out too low. Replace it with the block
    times from the raw MEDA pressure files when those are ingested.
    """
    cov = continuous_coverage(events)
    parity = (events["lmst_h"].astype(int) + events["sol"]) % 2

    def fraction(mask) -> float:
        n = np.bincount(parity[mask], minlength=2)
        return (1.0 + n.min() / n.max()) / 2.0

    n_bins = int(round(360.0 / ls_bin))
    event_bin = (events["ls_deg"] // ls_bin).astype(int) % n_bins
    overall = fraction(np.ones(len(events), bool))
    by_bin = {b: fraction(event_bin == b) for b in np.unique(event_bin) if (event_bin == b).sum() >= min_events}
    cov["listened"] = [(by_bin.get(b, overall)) for b in (cov["ls_deg"] // ls_bin).astype(int) % n_bins]
    return cov


def coverage_of(station: str, events: pd.DataFrame) -> pd.DataFrame:
    return {"insight": continuous_coverage, "perseverance": alternating_coverage}[station](events)


# ---------- rates ----------

def rate_table(events: pd.DataFrame, coverage: pd.DataFrame, ls_bin: float = 30.0,
               min_drop_pa: float = COMMON_MIN_DROP_PA) -> pd.DataFrame:
    """Vortex encounters per Earth hour for every season bin and local hour.

    One row per (season bin, hour 0..23) that has coverage:
    `station`, `ls_from`, `hour`, `n_sols`, `exposure_h`, `count`, `rate_per_h`, `rate_err`.
    `rate_err` is the Poisson error sqrt(count) / exposure; for an empty cell it is 1 / exposure,
    the rate at which one event would have been expected.
    """
    n_bins = int(round(360.0 / ls_bin))
    ev = events[events["drop_pa"] >= min_drop_pa]
    count = np.zeros((n_bins, 24))
    np.add.at(count, ((ev["ls_deg"] // ls_bin).astype(int) % n_bins, ev["ltst_h"].astype(int) % 24), 1)
    cov_bin = (coverage["ls_deg"] // ls_bin).astype(int) % n_bins
    listened = np.zeros(n_bins)                                     # station hours per hour-of-day bin
    np.add.at(listened, cov_bin, coverage["listened"])
    n_sols = np.bincount(cov_bin, minlength=n_bins)

    rows = []
    for b in np.flatnonzero(listened > 0):
        exposure = listened[b] * MARS_HOUR_IN_EARTH_HOURS
        for hour in range(24):
            c = count[b, hour]
            rows.append({"station": events["station"].iloc[0], "ls_from": b * ls_bin, "hour": hour,
                         "n_sols": int(n_sols[b]), "exposure_h": exposure, "count": int(c),
                         "rate_per_h": c / exposure, "rate_err": max(np.sqrt(c), 1.0) / exposure})
    return pd.DataFrame(rows)
