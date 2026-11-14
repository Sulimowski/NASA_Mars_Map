"""Fetcher with a cache: bring a product into `data/raw/` exactly once.

Two ways to get a product, both cached on disk:

* `fetch()` / `fetch_product()` download the whole file. Data goes to `<name>.part`
  first and is renamed only when the size is right, so a file under its final name
  is always complete. A broken download keeps its `.part` and the next call resumes
  it with an HTTP Range request.
* `fetch_window()` reads only the pixels over one region straight from the remote
  raster (HTTP range reads through GDAL) and saves that crop. This is how we use
  global mosaics of many GB, and 500 MB DTM strips, without downloading them.

If the file is already there, neither function touches the network.
"""

from __future__ import annotations

import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..catalog.db import Product, set_local_path
from ..grid import RegionGrid
from ..site import DATA_DIR, resolve_path

RAW_DIR = DATA_DIR / "raw"
USER_AGENT = "mars-map-ingest/1.0"
CHUNK = 1 << 20


class FetchError(RuntimeError):
    """The download failed. A partial `.part` file may be left behind for resuming."""


def url_filename(url: str) -> str:
    return urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]) or "download"


def _total_size(resp, offset: int) -> int | None:
    """Full size of the remote file from the response headers, if the server says."""
    crange = resp.headers.get("Content-Range")            # "bytes 100-999/1000"
    if crange and "/" in crange and not crange.endswith("/*"):
        return int(crange.rsplit("/", 1)[1])
    length = resp.headers.get("Content-Length")
    if length is not None:
        return int(length) + (offset if resp.status == 206 else 0)
    return None


def _download_once(url: str, part: Path, timeout: float, progress: bool) -> int | None:
    """One attempt: append to `part` (resuming if it exists). Returns the expected total size."""
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416 and offset:                      # range not satisfiable
            total = (e.headers.get("Content-Range") or "").rsplit("/", 1)[-1]
            if total.isdigit() and int(total) == offset:  # .part was already complete
                return offset
            part.unlink()                                 # .part is garbage, start over
            raise FetchError(f"stale partial file for {url}, removed") from e
        raise
    with resp:
        if offset and resp.status != 206:                 # server ignored Range: start over
            offset = 0
        total = _total_size(resp, offset)
        t0, done = time.time(), offset
        with open(part, "ab" if offset else "wb") as out:
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if progress and total:
                    mb = (done - offset) / 1e6 / max(time.time() - t0, 1e-3)
                    print(f"\r    {part.name[:-5]}: {done / 1e6:.0f}/{total / 1e6:.0f} MB "
                          f"({mb:.1f} MB/s)", end="", flush=True)
        if progress and total:
            print()
    return total


def fetch(url: str, dest: str | Path, *, expected_size: int | None = None, retries: int = 3,
          timeout: float = 60.0, progress: bool = False) -> Path:
    """Download `url` to `dest`. If `dest` exists, return it without any network access.

    Raises FetchError when the file cannot be completed; the partial data stays in
    `<dest>.part` and the next call continues from there.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            total = _download_once(url, part, timeout, progress)
            size = part.stat().st_size
            want = total if total is not None else expected_size
            if want is not None and size < want:
                raise FetchError(f"connection closed at {size} of {want} bytes")
            if want is not None and size > want:
                part.unlink()
                raise FetchError(f"got {size} bytes, expected {want}; partial file removed")
            os.replace(part, dest)
            return dest
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code not in (408, 429):
                raise FetchError(f"HTTP {e.code} for {url}") from e     # retrying will not help
            last = e
        except (FetchError, urllib.error.URLError, OSError, ConnectionError, TimeoutError) as e:
            last = e
        if attempt + 1 < retries:
            time.sleep(min(2 ** attempt, 10))
    left = f", {part.stat().st_size} bytes kept in {part.name} for resume" if part.exists() else ""
    raise FetchError(f"could not download {url}: {last}{left}") from last


def product_path(product: Product, raw_dir: str | Path = RAW_DIR) -> Path:
    """data/raw/<source>/<file name from the URL>"""
    return Path(raw_dir) / product.source / url_filename(product.url)


def fetch_product(db: sqlite3.Connection, product: Product, raw_dir: str | Path = RAW_DIR,
                  progress: bool = True, **kwargs) -> Path:
    """Download a whole catalog product and record `local_path`. Cached: no network if present."""
    if product.local_path:
        known = resolve_path(product.local_path)
        if known.exists():
            return known
    path = fetch(product.url, product_path(product, raw_dir),
                 expected_size=product.size_bytes, progress=progress, **kwargs)
    set_local_path(db, product, path)
    return path


# --- region-sized crops of remote rasters -------------------------------------------

def window_path(url: str, source: str, grid: RegionGrid, raw_dir: str | Path = RAW_DIR) -> Path:
    """data/raw/<source>/<stem>__<region>.tif"""
    stem = url_filename(url).rsplit(".", 1)[0]
    return Path(raw_dir) / source / f"{stem}__{grid.name}.tif"


def fetch_window(url: str, source: str, grid: RegionGrid, raw_dir: str | Path = RAW_DIR,
                 margin_px: int = 4) -> Path | None:
    """Save the part of a remote (or local) raster that covers `grid`, in its native grid.

    Pixels are untouched: same CRS, resolution, dtype and nodata as the original,
    only cropped (plus `margin_px` source pixels so resampling has neighbours).
    Returns None when the raster does not reach the region. Cached like `fetch`.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds
    from rasterio.windows import transform as window_transform

    dest = window_path(url, source, grid, raw_dir)
    if dest.exists():
        return dest
    x0, y0 = grid.origin
    region = (x0, y0 - grid.size_m, x0 + grid.size_m, y0)             # left, bottom, right, top
    remote = url.startswith(("http://", "https://"))
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_USERAGENT=USER_AGENT,
                       GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
                       CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.img,.IMG,.jp2,.JP2")
    try:
        with env, rasterio.open(("/vsicurl/" + url) if remote else url) as src:
            bounds = transform_bounds(CRS.from_proj4(grid.proj4), src.crs, *region, densify_pts=21)
            win = from_bounds(*bounds, transform=src.transform)
            c0 = max(0, int(win.col_off) - margin_px)
            r0 = max(0, int(win.row_off) - margin_px)
            c1 = min(src.width, int(win.col_off + win.width) + 1 + margin_px)
            r1 = min(src.height, int(win.row_off + win.height) + 1 + margin_px)
            if c1 <= c0 or r1 <= r0:
                return None
            win = Window(c0, r0, c1 - c0, r1 - r0)
            data = src.read(1, window=win)
            profile = dict(driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                           dtype=data.dtype, crs=src.crs, nodata=src.nodata,
                           transform=window_transform(win, src.transform), compress="deflate")
            tags = {"cropped_from": url, "region": grid.name}
    except rasterio.errors.RasterioIOError as e:
        raise FetchError(f"could not read {url}: {e}") from e

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with rasterio.open(part, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(**tags)
    os.replace(part, dest)
    return dest


def clear_partials(raw_dir: str | Path = RAW_DIR) -> int:
    """Delete leftover `.part` files (use when a resume keeps failing). Returns how many."""
    n = 0
    for p in Path(raw_dir).rglob("*.part"):
        p.unlink()
        n += 1
    return n


__all__ = ["RAW_DIR", "FetchError", "fetch", "fetch_product", "fetch_window", "product_path",
           "window_path", "clear_partials", "url_filename"]
