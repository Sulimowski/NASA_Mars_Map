"""Loose sand from orbit: a 0..1 `sand_fraction` map at the site resolution.

Why not thermal inertia alone: THEMIS pixels are ~100 m, while the sand at our sites
lies in strips a few metres to tens of metres wide between ridges. Averaged with the
rock around it, a 100 m pixel never drops below the "sand" threshold (Gale boxwork
reads 300-460, Jezero 240-410), so the old rule found no sand at all.

What we use instead, all at the 1-2 m scale of HiRISE:

* sand sheets  -- darker than the ground around them (basaltic sand), smooth in the
                  image, flat, and lying in local lows where wind drops sand;
* ripple fields -- crests a few metres apart: strong light/shadow contrast in the
                  image while the DTM sees flat, smooth ground (boulders have the
                  contrast too, but also relief). Bright ripples count, crests are loose.

Every cue is turned into 0..1 with a smooth step, the cues of each kind multiply (all
must agree), and the two kinds combine as max. Coarse layers that do measure the
ground (THEMIS thermal inertia, CRISM olivine) only nudge the result by +-20 %, they
cannot create sand where the image shows none.

This is an interpretation, not a measurement. `sand.validate` compares it with
thermal inertia, which it does not use for detection beyond the nudge.
"""

from __future__ import annotations

import numpy as np

from .stack import box_mean

# Working thresholds; each cue reaches 0.5 at the value given.
DARK_SIGMA_HALF = 0.6        # darker than the 150 m surroundings by this many local sigmas
SMOOTH_TEXTURE_HALF = 0.55   # texture below this share of the region median
FLAT_SLOPE_HALF_DEG = 9.0
LOW_RIDGE_HALF_M = 0.3       # at most this far above the 30 m mean (troughs are negative)
RIPPLE_TEXTURE_HALF = 1.45   # image texture above this share of the region median
RIPPLE_ROUGHNESS_HALF_M = 0.2
NUDGE = 0.2                  # how far THEMIS / CRISM may move the result
SAND_CLASS_MIN = 0.5         # sand_fraction from which a cell is terrain class "sand"


def _step(x, half, width, rising=True):
    """Logistic 0..1 step that is 0.5 at `half`; `width` is the 0.27..0.73 span."""
    z = (np.asarray(x, np.float64) - half) / max(width, 1e-9)
    s = 1.0 / (1.0 + np.exp(-4.0 * z))
    return s if rising else 1.0 - s


def _filled(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64)
    bad = ~np.isfinite(a)
    if bad.any():
        a = a.copy()
        a[bad] = np.nanmedian(a) if (~bad).any() else 0.0
    return a


def _px(metres: float, res: float) -> int:
    return max(1, int(round(metres / 2 / res)))


def darkness(optical: np.ndarray, res: float, window_m: float = 150.0) -> np.ndarray:
    """How much darker each cell is than its surroundings, in local standard deviations."""
    o = _filled(optical)
    r = _px(window_m, res)
    mean = box_mean(o, r)
    sd = np.sqrt(np.maximum(box_mean(o * o, r) - mean ** 2, 1e-8))
    o_s = box_mean(o, _px(4.0, res))                 # a sand patch, not a single dark pixel
    return (mean - o_s) / sd


def sand_cues(L: dict, res: float) -> dict[str, np.ndarray]:
    """The single cues, each 0..1, for inspection and for `sand_fraction`."""
    tex = box_mean(_filled(L["optical_texture"]), _px(6.0, res))
    tex_rel = tex / max(float(np.median(tex)), 1e-9)
    slope = box_mean(_filled(L["slope"]), _px(6.0, res))
    ridge = box_mean(_filled(L["ridge_index"]), _px(6.0, res))
    rough = box_mean(_filled(L["roughness"]), _px(10.0, res))
    return {
        "dark": _step(darkness(L["optical"], res), DARK_SIGMA_HALF, 0.5),
        "smooth": _step(tex_rel, SMOOTH_TEXTURE_HALF, 0.25, rising=False),
        "flat": _step(slope, FLAT_SLOPE_HALF_DEG, 4.0, rising=False),
        "low": _step(ridge, LOW_RIDGE_HALF_M, 0.4, rising=False),
        # Ripple fields: strong image contrast (crest / trough light and shadow) on
        # ground the DTM sees as flat and smooth. Boulders have the contrast too, but
        # also relief.
        "ripples": _step(tex_rel, RIPPLE_TEXTURE_HALF, 0.35),
        "smooth_dtm": _step(rough, RIPPLE_ROUGHNESS_HALF_M, 0.08, rising=False),
    }


def _prior(L: dict) -> np.ndarray:
    """-1..1 from the coarse measurements: low thermal inertia / more olivine -> +."""
    out = np.zeros(np.shape(L["optical"]))
    n = 0
    for name, sign in (("thermal_inertia", -1.0), ("mineral_olivine", 1.0)):
        if name in L and np.isfinite(L[name]).any():
            v = _filled(L[name])
            lo, hi = np.percentile(v, 5), np.percentile(v, 95)
            if hi > lo:
                out += sign * np.clip((v - (lo + hi) / 2) / ((hi - lo) / 2), -1, 1)
                n += 1
    return out / max(n, 1)


def _from_image(c: dict) -> np.ndarray:
    """Sand from the image cues alone: sand sheet or ripple field, whichever is stronger."""
    sheet = c["dark"] * c["smooth"] * c["flat"] * (0.4 + 0.6 * c["low"])
    ripples = c["ripples"] * c["smooth_dtm"] * c["flat"] ** 2
    return np.maximum(sheet, ripples)


def sand_fraction(L: dict, res: float) -> np.ndarray | None:
    """0..1 share of loose sand per cell, or None if the site has no orthoimage."""
    if "optical" not in L or "optical_texture" not in L or not np.isfinite(L["optical"]).any():
        return None
    s = _from_image(sand_cues(L, res)) * (1.0 + NUDGE * _prior(L))
    return np.clip(s, 0, 1).astype(np.float32)


def validate(L: dict, res: float, block_m: float = 100.0) -> dict[str, float]:
    """Check against THEMIS: on ~100 m blocks, more mapped sand should mean lower
    thermal inertia (and a colder ground at night, if the night image is there).

    Uses the map without the THEMIS nudge, so the check is not circular.
    """
    raw = _from_image(sand_cues(L, res))
    k = max(1, int(round(block_m / res)))
    n = raw.shape[0] // k

    def blocks(a):
        a = _filled(a)[: n * k, : n * k]
        return a.reshape(n, k, n, k).mean(axis=(1, 3)).ravel()

    out = {"blocks": float(n * n), "mean_fraction": float(raw.mean()),
           "share_class_sand": float((raw >= SAND_CLASS_MIN).mean())}
    s = blocks(raw)
    for name in ("thermal_inertia", "themis_night_ir"):
        if name in L and np.isfinite(L[name]).any():
            out[f"corr_{name}"] = float(np.corrcoef(s, blocks(L[name]))[0, 1])
    return out
