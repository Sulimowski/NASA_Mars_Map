"""Curiosity (REMS) at Gale crater: raw pressure, vortex detection, recording time.

No public vortex catalog exists for the recent years at Gale, so we find vortices ourselves in
the 1 Hz pressure of REMS (NASA PDS Atmospheres node, MSL-M-REMS-5-MODRDR, public domain).

    sols = pick_sols(4176, 4843, every=6)
    fetch_sols(sols)                         # ~10 MB per sol downloaded, ~0.3 MB kept
    events, coverage = detect_all(sols)      # same formats as vortices.py

Detection: inside each continuous recording block, the slow pressure trend is a running median
over `BASELINE_S`; a vortex is a dip below it of at least `MIN_DROP_PA`, with local minima at
least `SEPARATION_S` apart. The first and last `EDGE_S` of a block are not searched (the median
is unreliable there) and are not counted as listening time either.
"""

from __future__ import annotations

import io
import re
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .marstime import mars_ls

BASE = "https://atmos.nmsu.edu/PDS/data/mslrem_1001/DATA/"
STORE = Path(__file__).resolve().parent / "data" / "raw" / "curiosity"
J2000 = pd.Timestamp("2000-01-01 12:00:00", tz="UTC")     # REMS TIMESTAMP is seconds since this

BASELINE_S = 61
EDGE_S = 40
MIN_DROP_PA = 0.3
SEPARATION_S = 20
MAX_STEP_S = 2.0          # a gap longer than this starts a new block


def pick_sols(first: int, last: int, every: int = 6) -> list[int]:
    return list(range(first, last + 1, every))


# ---------- download ----------

_folders: dict[int, str] | None = None


def _get(url: str, timeout: int = 180) -> bytes:
    import truststore
    truststore.inject_into_ssl()
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _folder_of(sol: int) -> str:
    global _folders
    if _folders is None:
        listing = _get(BASE).decode("latin-1")
        _folders = {}
        for a, b in re.findall(r"SOL_(\d{5})_(\d{5})/", listing):
            for s in range(int(a), int(b) + 1):
                _folders[s] = f"SOL_{a}_{b}/SOL{s:05d}/"
    return _folders[sol]


def sol_path(sol: int) -> Path:
    return STORE / f"sol{sol:05d}.npz"


def fetch_sol(sol: int) -> Path | None:
    """Download the MODRDR table of one sol and keep time, local time, pressure and UV."""
    path = sol_path(sol)
    if path.exists():
        return path
    folder = _folder_of(sol)
    names = re.findall(r'href="(RME_\w+RMD\w+\.TAB)"', _get(BASE + folder).decode("latin-1"))
    if not names:
        return None
    parts = []
    for name in names:
        df = pd.read_csv(io.BytesIO(_get(BASE + folder + name)), header=None, skipinitialspace=True,
                         usecols=[0, 2, 20, 36, 37, 39], dtype={2: str, 36: str, 39: str})
        parts.append(df)
    df = pd.concat(parts).sort_values(0).drop_duplicates(0)
    hms = df[2].str.extract(r" (\d\d):(\d\d):(\d\d)").astype(float)
    good = (df[37] > 0) & df[36].str.strip().str.endswith("H") & df[39].str.strip().str[0].eq("1")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
                        t=df[0].to_numpy(float),                                      # s since J2000
                        ltst=(hms[0] + hms[1] / 60 + hms[2] / 3600).to_numpy(float),  # local true solar time, h
                        p=np.where(good, df[37], np.nan).astype(np.float32),          # Pa, NaN when not trusted
                        uv=df[20].where(df[20] > -900).to_numpy(np.float32))          # UV_ABC, W/m2
    return path


def fetch_sols(sols: list[int], workers: int = 4, verbose: bool = True) -> list[int]:
    """Download several sols in parallel; returns the sols that have data."""
    done = []
    with ThreadPoolExecutor(workers) as pool:
        for sol, path in zip(sols, pool.map(_safe_fetch, sols)):
            if path:
                done.append(sol)
            if verbose:
                print(f"  sol {sol}: {'ok' if path else 'no data'}", flush=True)
    return done


def _safe_fetch(sol: int) -> Path | None:
    try:
        return fetch_sol(sol)
    except Exception as e:                       # one bad sol must not stop the rest
        print(f"  sol {sol}: failed, {type(e).__name__}: {e}", flush=True)
        return None


# ---------- detection ----------

def _running_median(x: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # windows that are all NaN stay NaN
        return np.nanmedian(view, axis=1)


def detect_sol(sol: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Vortex events of one sol, and seconds listened in each hour of local true solar time."""
    d = np.load(sol_path(sol))
    t, ltst, p, uv = d["t"], d["ltst"], d["p"].astype(float), d["uv"].astype(float)
    listened = np.zeros(24)
    rows = []
    breaks = np.flatnonzero(np.diff(t) > MAX_STEP_S)
    for s, e in zip(np.r_[0, breaks + 1], np.r_[breaks + 1, len(t)]):
        if e - s < BASELINE_S + 2 * EDGE_S:
            continue
        pb, tb, lb, ub = p[s:e], t[s:e], ltst[s:e], uv[s:e]
        resid = pb - _running_median(pb, BASELINE_S)
        inner = np.zeros(len(pb), bool)
        inner[EDGE_S:len(pb) - EDGE_S] = True
        inner &= np.isfinite(resid)
        np.add.at(listened, lb[inner].astype(int) % 24, 1.0)
        cand = np.flatnonzero(inner & (resid <= -MIN_DROP_PA))
        while len(cand):
            group = cand[cand <= cand[0] + SEPARATION_S]
            k = group[np.argmin(resid[group])]
            depth = -resid[k]
            above = np.flatnonzero(resid[max(0, k - 60):k + 61] > -depth / 2)
            width = np.nan
            if len(above):
                rel = above - min(k, 60)
                left, right = rel[rel < 0], rel[rel > 0]
                if len(left) and len(right):
                    width = float(right.min() - left.max())
            uv_around = ub[max(0, k - 30):k + 31]
            dip = np.nan
            if np.isfinite(uv_around).sum() > 20 and np.nanmedian(uv_around) > 1:
                dip = float(100 * (1 - np.nanmin(ub[max(0, k - 3):k + 4]) / np.nanmedian(uv_around)))
            rows.append({"t": tb[k], "ltst_h": lb[k], "drop_pa": depth, "width_s": width,
                         "pressure_pa": pb[k] + depth, "dust_dip_pct": dip})
            cand = cand[cand > group[-1] + 0]
            cand = cand[cand > k + SEPARATION_S]
    ev = pd.DataFrame(rows)
    if len(ev):
        ev.insert(0, "sol", sol)
    return ev, listened


def detect_all(sols: list[int], station: str = "curiosity") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Events in the common format of vortices.py, and per-sol coverage with listening per hour."""
    events, cover = [], []
    for sol in sols:
        if not sol_path(sol).exists():
            continue
        ev, listened = detect_sol(sol)
        t_mid = np.load(sol_path(sol))["t"]
        noon = J2000 + pd.to_timedelta(float(np.median(t_mid)), unit="s")
        cover.append({"station": station, "sol": sol, "ls_deg": float(mars_ls(noon.to_pydatetime())),
                      **{f"h{h:02d}": listened[h] / 3600.0 for h in range(24)}})
        if len(ev):
            events.append(ev)
    ev = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    if len(ev):
        utc = J2000 + pd.to_timedelta(ev["t"], unit="s")
        ev = pd.DataFrame({
            "station": station, "sol": ev["sol"], "ltst_h": ev["ltst_h"], "lmst_h": np.nan,
            "ls_deg": mars_ls(utc.dt.tz_localize(None).to_numpy()), "utc": utc,
            "drop_pa": ev["drop_pa"], "width_s": ev["width_s"], "wind_max_ms": np.nan,
            "wind_before_ms": np.nan, "pressure_pa": ev["pressure_pa"], "air_temp_k": np.nan,
            "ground_temp_k": np.nan, "dust_dip_pct": ev["dust_dip_pct"],
        })
    return ev, pd.DataFrame(cover)


def hourly_exposure(coverage: pd.DataFrame) -> pd.DataFrame:
    """Long format: station, sol, ls_deg, hour, exposure_h (Earth hours recorded in that local hour)."""
    long = coverage.melt(id_vars=["station", "sol", "ls_deg"], value_vars=[f"h{h:02d}" for h in range(24)],
                         var_name="hour", value_name="exposure_h")
    long["hour"] = long["hour"].str[1:].astype(int)
    return long
