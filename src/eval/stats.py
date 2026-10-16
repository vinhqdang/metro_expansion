"""Unpaired and paired comparison statistics with bootstrap intervals and Holm correction."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sps


def _a12(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney A12 = P(X > Y) + 0.5 P(X = Y), computed from ranks."""
    n1, n2 = x.size, y.size
    ranks = sps.rankdata(np.concatenate([x, y]))
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n2))


def mwu_a12_boot(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Two-sided Mann-Whitney p (exact when small and tie-free), A12, and bootstrap 95% CI of mean(x) - mean(y)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    tie_free = np.unique(np.concatenate([x, y])).size == x.size + y.size
    small = x.size <= 8 and y.size <= 8
    method = "exact" if (small and tie_free) else "auto"
    p = float(sps.mannwhitneyu(x, y, alternative="two-sided", method=method).pvalue)
    rng = np.random.default_rng(seed)
    xb = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    yb = y[rng.integers(0, y.size, size=(n_boot, y.size))].mean(axis=1)
    diffs = xb - yb
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "p_mwu": p,
        "mwu_method": method,
        "a12": _a12(x, y),
        "mean_diff": float(x.mean() - y.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_x": int(x.size),
        "n_y": int(y.size),
    }


def _hodges_lehmann(d: np.ndarray) -> float:
    """Hodges-Lehmann estimator: median of Walsh averages (d_i + d_j)/2 for i <= j."""
    i, j = np.triu_indices(d.size)
    return float(np.median((d[i] + d[j]) / 2.0))


def wilcoxon_hl_boot(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired exact two-sided Wilcoxon p, Hodges-Lehmann shift of x - y, and paired bootstrap 95% CI of that shift."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("paired samples must have equal length")
    d = x - y
    nonzero = d[d != 0]
    if nonzero.size == 0:
        p = 1.0
    else:
        has_ties = np.unique(np.abs(nonzero)).size != nonzero.size
        method = "exact" if (nonzero.size <= 25 and not has_ties) else "approx"
        p = float(sps.wilcoxon(x, y, alternative="two-sided", method=method).pvalue)
    hl = _hodges_lehmann(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = np.array([_hodges_lehmann(d[row]) for row in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "p_wilcoxon": p,
        "hl_shift": hl,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_pairs": int(d.size),
    }


def holm(pvals: np.ndarray | list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the original order (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float).ravel()
    m = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = np.empty(m)
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, (m - rank) * p[k])
        adj_sorted[rank] = min(1.0, running)
    out = np.empty(m)
    out[order] = adj_sorted
    return out.tolist()
