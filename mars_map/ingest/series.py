"""Per-sol time series from Curiosity (Gale crater) into the `series` table.

* REMS weather: one small JSON feed from mars.nasa.gov with every sol of the mission.
  Series: `temp_min_c`, `temp_max_c` (air), `ground_temp_min_c`, `ground_temp_max_c`,
  `pressure_pa`. The feed has no dust opacity (tau) and no usable wind.
* RAD radiation: PDS PPI data set MSL-M-RAD-3-RDR-V1.0, one ~18 MB text file per sol.
  Each observation in it has the dose rate in the tissue-equivalent plastic detector E
  (microgray / hour). Series: `radiation_dose_ugy_per_sol` (measured, absorbed dose)
  and `radiation_msv_per_sol` (dose equivalent, see RAD_QUALITY_FACTOR).

Sols are Curiosity mission sols. The series belong to a region, and they are only
meaningful for regions near Gale.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from ..catalog.db import add_series, add_source, region_id
from .fetch import RAW_DIR, FetchError, fetch

REMS_URL = "https://mars.nasa.gov/rss/api/?feed=weather&category=msl&feedtype=json"
RAD_URL = "https://pds-ppi.igpp.ucla.edu/data/MSL-M-RAD-3-RDR-V1.0"

SOL_HOURS = 24.6597
# Mean quality factor of the radiation field on the surface in Gale: 0.21 mGy/day absorbed
# vs 0.64 mSv/day dose equivalent (Hassler et al. 2014, Science 343). An average, not a
# per-sol measurement: during a solar particle event the real factor is lower.
RAD_QUALITY_FACTOR = 3.05

# feed field -> (series name, multiplier)
REMS_FIELDS = {
    "min_temp": ("temp_min_c", 1.0),
    "max_temp": ("temp_max_c", 1.0),
    "min_gts_temp": ("ground_temp_min_c", 1.0),
    "max_gts_temp": ("ground_temp_max_c", 1.0),
    "pressure": ("pressure_pa", 1.0),
}


def _number(text) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):                 # the feed writes "--" for missing values
        return None


# --- REMS ------------------------------------------------------------------------

def parse_rems(feed: dict) -> dict[str, dict[int, float]]:
    """Feed JSON -> {series name: {sol: value}}, missing values dropped."""
    out: dict[str, dict[int, float]] = {name: {} for name, _ in REMS_FIELDS.values()}
    for rec in feed.get("soles", []):
        sol = _number(rec.get("sol"))
        if sol is None:
            continue
        for field, (name, k) in REMS_FIELDS.items():
            v = _number(rec.get(field))
            if v is not None:
                out[name][int(sol)] = v * k
    return out


def load_rems(db: sqlite3.Connection, region: str, raw_dir: str | Path = RAW_DIR,
              refresh: bool = False) -> dict[str, int]:
    """REMS weather for all mission sols. Returns {series name: number of sols}."""
    region_id(db, region)
    add_source(db, "rems", "REMS", "mars.nasa.gov", base_url=REMS_URL)
    path = Path(raw_dir) / "rems" / "msl_weather.json"
    if refresh:
        path.unlink(missing_ok=True)
    fetch(REMS_URL, path)
    series = parse_rems(json.loads(path.read_text(encoding="utf-8")))
    return {name: add_series(db, region, name, values, "rems")
            for name, values in series.items() if values}


M20_WEATHER_URL = "https://mars.nasa.gov/rss/api/?feed=weather&category=mars2020&feedtype=json"
M20_FIELDS = {"min_temp": "temp_min_c", "max_temp": "temp_max_c", "pressure": "pressure_pa"}


def parse_m20_weather(feed: dict) -> dict[str, dict[int, float]]:
    """Perseverance (MEDA) weather feed -> {series name: {sol: value}}, missing values dropped."""
    out: dict[str, dict[int, float]] = {name: {} for name in M20_FIELDS.values()}
    for rec in feed.get("sols", []):
        sol = _number(rec.get("sol"))
        if sol is None:
            continue
        for field, name in M20_FIELDS.items():
            v = _number(rec.get(field))
            if v is not None:
                out[name][int(sol)] = v
    return out


def load_m20_weather(db: sqlite3.Connection, region: str, raw_dir: str | Path = RAW_DIR,
                     refresh: bool = False) -> dict[str, int]:
    """Perseverance weather (Jezero). The public feed holds only a handful of sols.

    There is no radiation instrument on Perseverance, so a Jezero region gets no
    radiation series at all rather than Curiosity's numbers from another crater.
    """
    region_id(db, region)
    add_source(db, "meda", "MEDA", "mars.nasa.gov", base_url=M20_WEATHER_URL)
    path = Path(raw_dir) / "m20" / "m20_weather.json"
    if refresh:
        path.unlink(missing_ok=True)
    fetch(M20_WEATHER_URL, path)
    series = parse_m20_weather(json.loads(path.read_text(encoding="utf-8")))
    return {name: add_series(db, region, name, values, "meda")
            for name, values in series.items() if values}


# --- RAD -------------------------------------------------------------------------

_DOSE_E = re.compile(r"^\[DOSIMETRY_TOTAL_DOSE_E:[^\]]*\]\s*\n\s*\n?\s*([-+0-9.eE]+)", re.M)
_RAD_FILE = re.compile(r"/DATA/(SOL_\d+_\d+)/(RAD_RDR_\d{4}_\d{3}_\d{2}_\d{2}_(\d{4})_V\d+)\.LBL")


def rad_index(raw_dir: str | Path = RAW_DIR, refresh: bool = False) -> dict[int, list[str]]:
    """{sol: [URL of the RDR text file, ...]} from the data set index (cached, ~1 MB)."""
    path = Path(raw_dir) / "rad" / "INDEX.TAB"
    if refresh:
        path.unlink(missing_ok=True)
    fetch(f"{RAD_URL}/INDEX/INDEX.TAB", path)
    out: dict[int, list[str]] = {}
    for folder, name, sol in _RAD_FILE.findall(path.read_text(encoding="latin-1")):
        out.setdefault(int(sol), []).append(f"{RAD_URL}/DATA/{folder}/{name}.TXT")
    return out


def parse_rad_dose_rate(text: str) -> float | None:
    """Mean detector-E dose rate over the observations of one RDR file, microgray / hour."""
    rates = [float(v) for v in _DOSE_E.findall(text)]
    rates = [r for r in rates if 0.0 < r < 1e4]             # drop empty / corrupt observations
    return sum(rates) / len(rates) if rates else None


def load_rad(db: sqlite3.Connection, region: str, sols, raw_dir: str | Path = RAW_DIR,
             keep_raw: bool = False, verbose: bool = True) -> dict[int, float]:
    """RAD dose for the given mission sols. Returns {sol: mSv per sol} for sols that have data.

    Sols already in the `series` table are skipped, so the call can be repeated or
    extended cheaply. Raw files are 18 MB each and are deleted after parsing unless
    `keep_raw` (the result lives in the catalog).
    """
    rid = region_id(db, region)
    add_source(db, "rad", "RAD", "PDS PPI", base_url=RAD_URL)
    have = dict(db.execute("SELECT sol, value FROM series WHERE region_id=? AND "
                           "name='radiation_msv_per_sol'", (rid,)))
    index = rad_index(raw_dir)
    out: dict[int, float] = {}
    for sol in sols:
        if sol in have:
            out[sol] = have[sol]
            continue
        rates = []
        for url in index.get(sol, []):
            path = Path(raw_dir) / "rad" / url.rsplit("/", 1)[1]
            try:
                fetch(url, path)
            except FetchError as e:
                if verbose:
                    print(f"  RAD sol {sol}: {e}")
                continue
            rate = parse_rad_dose_rate(path.read_text(encoding="latin-1"))
            if rate is not None:
                rates.append(rate)
            if not keep_raw:
                path.unlink()
        if not rates:
            if verbose:
                print(f"  RAD sol {sol}: no data")
            continue
        ugy = sum(rates) / len(rates) * SOL_HOURS
        out[sol] = ugy * RAD_QUALITY_FACTOR / 1000.0
        add_series(db, region, "radiation_dose_ugy_per_sol", {sol: ugy}, "rad")
        add_series(db, region, "radiation_msv_per_sol", {sol: out[sol]}, "rad")
        if verbose:
            print(f"  RAD sol {sol}: {ugy:.0f} uGy = {out[sol]:.3f} mSv")
    return out
