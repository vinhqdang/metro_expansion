"""Fairness, demand and accessibility objectives for a finished TNDP line."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


def ggi(s: Sequence[float] | np.ndarray, base: float = 2.0) -> float:
    """Generalized Gini welfare: sum_k w_k s_(k) with s ascending and w_k proportional to base^-(k-1); 0 if all zero."""
    v = np.sort(np.asarray(s, dtype=float))
    if v.size == 0 or not np.any(v > 0):
        return 0.0
    w = base ** (-np.arange(v.size, dtype=float))
    w /= w.sum()
    return float(np.dot(w, v))


def gini_or_nan(s: Sequence[float] | np.ndarray) -> float:
    """Standard Gini coefficient over group ratios; NaN when sum(s) == 0 (undefined, never 0)."""
    v = np.sort(np.asarray(s, dtype=float))
    n = v.size
    total = v.sum()
    if n == 0 or total == 0:
        return float("nan")
    idx = np.arange(1, n + 1, dtype=float)
    return float(2.0 * np.dot(idx, v) / (n * total) - (n + 1.0) / n)


def min_max_ratio(s: Sequence[float] | np.ndarray) -> float:
    """Minimum over maximum of group ratios; NaN when max == 0."""
    v = np.asarray(s, dtype=float)
    mx = v.max() if v.size else 0.0
    if mx == 0:
        return float("nan")
    return float(v.min() / mx)


def cv(s: Sequence[float] | np.ndarray) -> float:
    """Coefficient of variation std/mean (population std); NaN when mean == 0."""
    v = np.asarray(s, dtype=float)
    mu = v.mean() if v.size else 0.0
    if mu == 0:
        return float("nan")
    return float(v.std() / mu)


def group_metrics(s: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Per-group satisfaction ratios s to the summary metrics used in the evaluation tables."""
    v = np.asarray(s, dtype=float)
    return {
        "s_min": float(v.min()),
        "argmin_group": int(np.argmin(v)),
        "ggi2": ggi(v, base=2.0),
        "minmax_ratio": min_max_ratio(v),
        "gini_ratio": gini_or_nan(v),
        "cv": cv(v),
        "D_sumpct": float(v.sum()),
    }


def d_tot(
    segments: Iterable[tuple[int, int]],
    od_raw: np.ndarray,
    total_od: float | None = None,
) -> float:
    """Share of total OD served: sum of od_raw over the set {(i,j),(j,i)} of segments, divided by total OD.

    Interpretation: each segment (i, j) contributes both directed pairs; duplicates are removed by
    using a set, so a pair listed twice (or once per direction) is counted once per direction.
    """
    od = np.asarray(od_raw, dtype=float)
    directed: set[tuple[int, int]] = set()
    for i, j in segments:
        directed.add((int(i), int(j)))
        directed.add((int(j), int(i)))
    if total_od is None:
        total_od = float(od.sum())
    if total_od == 0:
        return 0.0
    served = sum(float(od[i, j]) for i, j in directed)
    return served / total_od


def _dilate_chebyshev(mask: np.ndarray, K: int) -> np.ndarray:
    """Boolean mask of cells within Chebyshev distance K of any True cell."""
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for dy in range(-K, K + 1):
        for dx in range(-K, K + 1):
            ys0, ys1 = max(0, dy), h + min(0, dy)
            xs0, xs1 = max(0, dx), w + min(0, dx)
            src_y0, src_y1 = max(0, -dy), h + min(0, -dy)
            src_x0, src_x1 = max(0, -dx), w + min(0, -dx)
            out[ys0:ys1, xs0:xs1] |= mask[src_y0:src_y1, src_x0:src_x1]
    return out


def _cells_mask(cells: Iterable[tuple[int, int]], grid_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(grid_shape, dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def a_new(
    mu: np.ndarray,
    existing_mask: np.ndarray,
    new_cells_xy: Iterable[tuple[int, int]],
    grid_shape: tuple[int, int],
    K: int = 2,
) -> float:
    """Trip-end-weighted new accessibility: mass newly within K of the new line, outside existing reach E."""
    mu_g = np.asarray(mu, dtype=float).reshape(grid_shape)
    E = _dilate_chebyshev(np.asarray(existing_mask, dtype=bool).reshape(grid_shape), K)
    V = _cells_mask(new_cells_xy, grid_shape)
    reach_new = _dilate_chebyshev(V, K)
    denom = mu_g[~E].sum()
    if denom == 0:
        return 0.0
    numer = mu_g[reach_new & ~E].sum()
    return float(numer / denom)


def cov_uniform(
    new_cells_xy: Iterable[tuple[int, int]],
    grid_shape: tuple[int, int],
    K: int = 2,
) -> float:
    """Legacy uniform coverage: fraction of all grid cells within Chebyshev K of the new line."""
    V = _cells_mask(new_cells_xy, grid_shape)
    return float(_dilate_chebyshev(V, K).mean())


@dataclass
class CityContext:
    """Static per-city quantities needed to score a line."""

    od_raw: np.ndarray
    mu: np.ndarray
    existing_mask: np.ndarray
    group_od_sum: np.ndarray
    total_od: float
    grid_shape: tuple[int, int]


def compute_objectives(
    ctx: CityContext,
    cells: Sequence[tuple[int, int]],
    segments: Iterable[tuple[int, int]],
    s: Sequence[float] | np.ndarray,
    K: int = 2,
) -> dict[str, Any]:
    """Combine demand, fairness and accessibility objectives for one final line state."""
    s_arr = np.asarray(s, dtype=float)
    out: dict[str, Any] = {
        "D_tot": d_tot(segments, ctx.od_raw, ctx.total_od),
        "A_new": a_new(ctx.mu, ctx.existing_mask, cells, ctx.grid_shape, K=K),
        "cov_uniform": cov_uniform(cells, ctx.grid_shape, K=K),
        "a_g": s_arr * np.asarray(ctx.group_od_sum, dtype=float),
    }
    out.update(group_metrics(s_arr))
    return out
