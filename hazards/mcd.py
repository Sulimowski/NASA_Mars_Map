"""Mars Climate Database (MCD 6.2, LMD / ESA / CNES): the climate of any point, hour by hour.

These are the model inputs: things known for every place and season on Mars, not only where a
station stood. One request returns the whole day (local true solar time 0..24, hourly) for up
to four variables; results are cached in hazards/data/mcd/.

    day = mcd_day(lat=4.502, lon=135.623, ls=60)       # DataFrame, one row per local hour

Scenario: climatology, average solar activity, high-resolution mode (local surface from MOLA).
Air values are taken 2 m above the local surface, where a station sensor or an astronaut is.

Citation: Forget et al. 1999, Millour et al. 2018 (MCD 6.x). Free for research use; the web
interface is shared by everyone, so requests are sequential with a pause.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CGI = "https://www-mars.lmd.jussieu.fr/mcd_python/cgi-bin/mcdcgi.py"
CACHE = Path(__file__).resolve().parent / "data" / "mcd"
PAUSE_S = 1.0

# Our column name -> MCD variable code. Grouped by four, one request per group.
VARIABLES = {
    "tsurf_k": "tsurf",               # ground temperature
    "tair_k": "t",                    # air temperature 2 m above ground
    "heat_flux_wm2": "sensib_flux",   # sensible heat flux from ground to air: what drives convection
    "pbl_m": "zmax",                  # height of the convective boundary layer
    "wstar_ms": "wstar_up",           # strongest updraft in the boundary layer
    "wind_ms": "wind",                # horizontal wind 2 m above ground
    "ps_pa": "ps",                    # surface pressure
    "dust_tau": "dod",                # dust optical depth of the whole column
}


def _request(lat: float, lon: float, ls: float, codes: list[str]) -> str:
    form = dict(datekeyhtml=1, ls=ls, localtime="all", latitude=lat, longitude=lon, altitude=2.0, zkey=3,
                isfixedlt="off", dust=1, hrkey=1, zonmean="off", diumean="off", averaging="off",
                animation="off", dpi=80, islog="off", colorm="jet", minval="", maxval="", proj="cyl",
                plat="", plon="", trans="", iswind="off", latpoint="", lonpoint="", year="", month="",
                day="", hours="", minutes="", seconds="", julian="", martianyear="", sol="", palt="",
                animframes=12, spacecraft="none")
    for i in range(4):
        form[f"var{i + 1}"] = codes[i] if i < len(codes) else "none"
    import truststore
    truststore.inject_into_ssl()
    req = urllib.request.Request(CGI, data=urllib.parse.urlencode(form).encode(), method="POST",
                                 headers={"User-Agent": "python-urllib"})
    html = urllib.request.urlopen(req, timeout=180).read().decode("utf-8", "replace")
    link = re.search(r"href='([^']+\.txt)'", html)
    if not link:
        raise RuntimeError(f"MCD gave no data for lat {lat} lon {lon} Ls {ls}: {html[-300:]}")
    txt_url = urllib.parse.urljoin(CGI, link.group(1))
    return urllib.request.urlopen(txt_url, timeout=120).read().decode("utf-8", "replace")


def _parse(text: str, lat: float, lon: float, ls: float, n: int) -> list[np.ndarray]:
    """Split the ASCII answer into one (hour, value) array per variable, checking the header."""
    blocks = [b for b in re.split(r"#{20,}\n", text) if b.strip() and not b.lstrip().startswith("###")]
    headers = re.findall(r"### Ls ([\d.]+)deg\. Latitude ([-\d.]+)N\. Longitude ([-\d.]+)E", text)
    if len(blocks) != n or len(headers) < n:
        raise RuntimeError(f"expected {n} blocks, got {len(blocks)}")
    for h_ls, h_lat, h_lon in headers:
        if abs(float(h_ls) - ls) > 0.01 or abs(float(h_lat) - lat) > 0.01 or abs(float(h_lon) - lon) > 0.01:
            raise RuntimeError(f"MCD answered for another point: Ls {h_ls} lat {h_lat} lon {h_lon}")
    return [np.loadtxt(b.splitlines()) for b in blocks]


def mcd_day(lat: float, lon: float, ls: float, refresh: bool = False) -> pd.DataFrame:
    """Hourly climate (local true solar time 0..23) at a point and season. Cached on disk."""
    path = CACHE / f"lat{lat:+.3f}_lon{lon:.3f}_ls{ls:05.1f}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    names = list(VARIABLES)
    out = {}
    for start in range(0, len(names), 4):
        group = names[start:start + 4]
        arrays = _parse(_request(lat, lon, ls, [VARIABLES[g] for g in group]), lat, lon, ls, len(group))
        for name, arr in zip(group, arrays):
            out["hour"] = arr[:24, 0]
            out[name] = arr[:24, 1]
        time.sleep(PAUSE_S)
    day = pd.DataFrame(out)
    day.insert(0, "ls_deg", ls)
    day.insert(0, "lon", lon)
    day.insert(0, "lat", lat)
    path.parent.mkdir(parents=True, exist_ok=True)
    day.to_csv(path, index=False)
    return day


def mcd_year(lat: float, lon: float, ls_step: float = 30.0, verbose: bool = True) -> pd.DataFrame:
    """`mcd_day` for the middle of every season bin (Ls 15, 45, ... 345)."""
    days = []
    for ls in np.arange(ls_step / 2, 360.0, ls_step):
        days.append(mcd_day(lat, lon, float(ls)))
        if verbose:
            print(f"  MCD lat {lat:+.2f} lon {lon:.2f} Ls {ls:5.1f}: ok", flush=True)
    return pd.concat(days, ignore_index=True)
