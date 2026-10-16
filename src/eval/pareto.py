"""Pareto dominance, exact 3-D hypervolume, expected utility and IGD+ for maximisation problems."""

from __future__ import annotations

from typing import Callable

import numpy as np


def nondominated(F: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows of F under maximisation."""
    P = np.asarray(F, dtype=float)
    if P.size == 0:
        return np.zeros(0, dtype=bool)
    ge = np.all(P[None, :, :] >= P[:, None, :], axis=-1)  # ge[i, j]: j >= i in all objectives
    gt = np.any(P[None, :, :] > P[:, None, :], axis=-1)
    dominated = np.any(ge & gt, axis=1)
    return ~dominated


def _hv2d(P: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-D hypervolume (maximisation) of points P above ref."""
    keep = np.all(P > ref, axis=1)
    P = P[keep]
    if P.size == 0:
        return 0.0
    order = np.argsort(-P[:, 0], kind="stable")
    x = P[order, 0]
    y_cummax = np.maximum.accumulate(P[order, 1])
    next_x = np.append(x[1:], ref[0])
    return float(np.sum((x - next_x) * (y_cummax - ref[1])))


def hypervolume(F: np.ndarray, ref: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0)) -> float:
    """Exact 3-D hypervolume (maximisation) dominated by F and above the reference point, by z-slicing."""
    r = np.asarray(ref, dtype=float)
    P = np.asarray(F, dtype=float).reshape(-1, 3)
    P = P[np.all(P > r, axis=1)]
    if P.shape[0] == 0:
        return 0.0
    P = P[np.argsort(-P[:, 2], kind="stable")]
    z = P[:, 2]
    z_next = np.append(z[1:], r[2])
    total = 0.0
    for k in range(P.shape[0]):
        total += (z[k] - z_next[k]) * _hv2d(P[: k + 1, :2], r[:2])
    return float(total)


def eum(
    F: np.ndarray,
    W: np.ndarray,
    norm_g: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> float:
    """Expected utility: mean over weights w of max over f in F of -g(f|w); norm_g(F, w) returns shape (n,)."""
    P = np.asarray(F, dtype=float).reshape(-1, 3)
    if P.shape[0] == 0:
        raise ValueError("F must contain at least one objective vector")
    best = [float(np.max(-np.asarray(norm_g(P, w), dtype=float))) for w in np.asarray(W, dtype=float)]
    return float(np.mean(best))


def igd_plus(F: np.ndarray, Ref: np.ndarray) -> float:
    """IGD+ (maximisation): mean over reference points of min over F of sqrt(sum_j max(r_j - f_j, 0)^2)."""
    P = np.asarray(F, dtype=float)
    R = np.asarray(Ref, dtype=float)
    if P.size == 0:
        return float("inf")
    deficit = np.maximum(R[:, None, :] - P[None, :, :], 0.0)
    dist = np.sqrt(np.sum(deficit**2, axis=-1))
    return float(np.mean(np.min(dist, axis=1)))
