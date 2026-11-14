"""Science and surface interpretation of the fused layers.

* `mineral_scores`   -- each CRISM index turned into 0..1 "how clearly is it detected here"
* `science_score`    -- 0..1, the best weighted mineral detection per cell
* `mineral_dominant` -- which mineral group wins per cell (0 = none detected)
* `rock_proxy`       -- stand-in for rock abundance from image texture (see the warning there)
* `ridge_index`      -- how much a cell stands above its surroundings (boxwork = ridges)
* `pick_base`, `mineral_targets` -- where to start and what to visit

Mineral detection uses fixed thresholds (`DETECTION`), the same in every region, so
scores are comparable between regions.
"""

from __future__ import annotations

import numpy as np

from ..site import Site, Target
from .stack import box_mean

# mineral layer -> (weight in the science score, label for people)
MINERALS = {
    "mineral_sulfate": (1.0, "Hydrated sulfate"),
    "mineral_sulfate_mono": (1.0, "Monohydrated sulfate"),
    "mineral_clay": (1.0, "Clay minerals"),
    "mineral_hydration": (0.8, "Hydrated minerals (bound water)"),
    "mineral_silica": (0.8, "Hydrated silica / Al-OH"),
    "mineral_hematite": (0.7, "Hematite"),
    "mineral_olivine": (0.4, "Olivine"),
    "mineral_pyroxene": (0.4, "Pyroxene"),
}
MINERAL_NAMES = ["none"] + [label for _, label in MINERALS.values()]

# One fixed rule for every region: an index scores 0 at `lo` or below and 1 at `hi` or
# above, linear in between. So a score means the same strength of signal in Gale, in
# Jezero or anywhere else, and two regions can be compared.
#
# (lo, hi) per CRISM index. Working assumptions, not published calibrations. They were
# set from the three regions we have (Gale boxwork, Jezero delta, Nili Planum), after
# smoothing over one CRISM pixel, where the noise of a band depth is about 0.002:
#   * `lo` is above the level every region reaches by noise and scene-to-scene offset;
#   * `hi` is about the strongest value seen in a region where the mineral is known.
# Olivine has a large background (median 0.043 over the Gale boxwork, which is not an
# olivine site; 0.084 over the Jezero delta, which is), so its floor is high.
# Caveat: a CRISM scene can be offset by a few 0.001 as a whole. A region whose scene
# sits low will under-detect; that is the price of a rule that is the same everywhere.
DETECTION = {
    "mineral_sulfate": (0.010, 0.015),
    "mineral_sulfate_mono": (0.007, 0.015),
    "mineral_clay": (0.008, 0.020),
    "mineral_hydration": (0.004, 0.020),
    "mineral_silica": (0.0045, 0.012),
    "mineral_hematite": (0.005, 0.020),
    "mineral_olivine": (0.070, 0.120),
    "mineral_pyroxene": (0.010, 0.050),
}


def detection_score(index: np.ndarray, lo: float, hi: float, smooth_px: int = 0) -> np.ndarray:
    """CRISM index -> 0..1: 0 at `lo` or below, 1 at `hi` or above. No data scores 0.

    `smooth_px` > 0 first averages over a (2*smooth_px+1) window: single CRISM pixels
    are speckle, a real exposure spans several.
    """
    missing = ~np.isfinite(index)
    if missing.all():
        return np.zeros(index.shape, np.float32)
    if smooth_px > 0:
        index = box_mean(np.nan_to_num(index, nan=float(np.nanmedian(index))), smooth_px)
    score = np.clip((np.nan_to_num(index, nan=lo) - lo) / (hi - lo), 0, 1).astype(np.float32)
    score[missing] = 0.0
    return score


def mineral_scores(layers: dict[str, np.ndarray], smooth_px: int = 0) -> dict[str, np.ndarray]:
    return {name: detection_score(layers[name], *DETECTION[name], smooth_px)
            for name in MINERALS if name in layers}


def science_from_minerals(layers: dict[str, np.ndarray],
                          smooth_px: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    """(science_score 0..1, mineral_dominant uint8) or None when there are no mineral layers.

    `smooth_px`: about one native CRISM pixel, in grid cells.
    """
    scores = mineral_scores(layers, smooth_px)
    if not scores:
        return None
    names = list(MINERALS)
    stack = np.stack([scores[n] * MINERALS[n][0] if n in scores else
                      np.zeros(next(iter(scores.values())).shape, np.float32) for n in names])
    best = stack.max(axis=0)
    dominant = np.where(best > 0.05, stack.argmax(axis=0) + 1, 0).astype(np.uint8)
    return best.astype(np.float32), dominant


def rock_proxy(texture: np.ndarray) -> np.ndarray:
    """Image texture -> 0..1 stand-in for rock abundance.

    WARNING: this is not a measured rock abundance. It is the local contrast of the
    1 m orthoimage scaled so that the roughest 1 % of the region reads 0.5. Blocky
    outcrop, boulder fields and sharp shadows are high; sand and dust are low. It
    ranks ground from smooth to rugged, it does not count rocks.
    """
    t = np.nan_to_num(texture, nan=float(np.nanmedian(texture)))
    lo, hi = np.percentile(t, 20), np.percentile(t, 99)
    if hi <= lo:
        return np.zeros(t.shape, np.float32)
    return np.clip(0.5 * (t - lo) / (hi - lo), 0, 1).astype(np.float32)


def ridge_index(elevation: np.ndarray, resolution_m: float, window_m: float = 30.0) -> np.ndarray:
    """Height of a cell above the mean of its `window_m` neighbourhood, metres (+ = ridge)."""
    r = max(1, int(round(window_m / 2 / resolution_m)))
    e = elevation.astype(np.float64)
    return (e - box_mean(e, r)).astype(np.float32)


def pick_base(site: Site, patch_m: float = 40.0, margin_m: float = 150.0,
              allowed: np.ndarray | None = None) -> tuple[float, float]:
    """A flat, firm, unblocked patch to land / park on: the lowest mean slope over `patch_m`.

    Among equally flat patches the one closer to the centre wins. `allowed` restricts
    the choice (for example to the area the targets can be reached from).
    Returns (x_m, y_m).
    """
    L = site.layers
    r = max(1, int(round(patch_m / 2 / site.resolution_m)))
    mean_slope = box_mean(L["slope"], r)
    clear = box_mean((~L["blocked"]).astype(np.float64), r) > 0.999
    if allowed is not None:
        clear &= allowed
    rows, cols = np.indices(site.shape)
    w, h = site.extent_m
    x, y = (cols + 0.5) * site.resolution_m, (rows + 0.5) * site.resolution_m
    inside = (x > margin_m) & (x < w - margin_m) & (y > margin_m) & (y < h - margin_m)
    dist = np.hypot(x - w / 2, y - h / 2) / (w / 2)
    score = np.where(clear & inside, mean_slope + 1.0 * dist, np.inf)
    rr, cc = np.unravel_index(np.argmin(score), score.shape)
    return site.to_metres(int(rr), int(cc))


def mineral_targets(site: Site, count: int = 4, min_spacing_m: float = 250.0,
                    avoid: list[tuple[float, float]] = (), patch_m: float = 60.0,
                    margin_m: float = 100.0, allowed: np.ndarray | None = None) -> list[Target]:
    """The strongest mineral exposures a crew can stand on, spread over the region.

    A target is the centre of a `patch_m` patch (about three CRISM pixels, so one noisy
    pixel cannot make a target) with the highest mean science score, not blocked, at
    least `margin_m` from the edge and `min_spacing_m` from other targets and from
    `avoid` points. `allowed` restricts the choice, e.g. to cells reachable from the base.
    """
    L = site.layers
    if "mineral_dominant" not in L or not np.any(L["science_score"] > 0):
        return []
    r = max(1, int(round(patch_m / 2 / site.resolution_m)))
    score = box_mean(L["science_score"], r)
    score[L["blocked"] | (L["slope"] > site.mission.max_slope_deg)] = -1
    if allowed is not None:
        score[~allowed] = -1
    rows, cols = np.indices(site.shape)
    x, y = (cols + 0.5) * site.resolution_m, (rows + 0.5) * site.resolution_m
    w, h = site.extent_m
    score[(x < margin_m) | (x > w - margin_m) | (y < margin_m) | (y > h - margin_m)] = -1
    for ax, ay in avoid:
        score[np.hypot(x - ax, y - ay) < min_spacing_m] = -1
    out: list[Target] = []
    while len(out) < count:
        rr, cc = np.unravel_index(np.argmax(score), score.shape)
        if score[rr, cc] <= 0.15:
            break
        tx, ty = site.to_metres(int(rr), int(cc))
        # name it after the mineral that dominates the patch
        patch = L["mineral_dominant"][max(rr - r, 0): rr + r + 1, max(cc - r, 0): cc + r + 1]
        kinds = np.bincount(patch.ravel(), minlength=len(MINERAL_NAMES))[1:]
        label = MINERAL_NAMES[int(kinds.argmax()) + 1]
        out.append(Target(f"{label} exposure {len(out) + 1}", round(tx, 1), round(ty, 1), priority=2,
                          dwell_min=25, note=f"CRISM detection, mean score {score[rr, cc]:.2f} "
                                             f"over {patch_m:.0f} m"))
        score[np.hypot(x - tx, y - ty) < min_spacing_m] = -1
    return out
