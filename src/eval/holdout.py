"""Ground-truth comparison metrics (Chebyshev corridor overlap, distances, orientation) for generated lines."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _as_cells(cells: Sequence[tuple[int, int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(cells, dtype=float).reshape(-1, 2)
    return arr


def _cheb_matrix(L: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Pairwise Chebyshev distances, shape (len(L), len(G))."""
    return np.max(np.abs(L[:, None, :] - G[None, :, :]), axis=-1)


def corridor_scores(
    L: Sequence[tuple[int, int]] | np.ndarray,
    G: Sequence[tuple[int, int]] | np.ndarray,
    r: int = 1,
) -> dict[str, float]:
    """Precision, recall and F1 within Chebyshev tolerance r of the ground-truth line G."""
    Lc, Gc = _as_cells(L), _as_cells(G)
    if Lc.shape[0] == 0 or Gc.shape[0] == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "r": float(r)}
    D = _cheb_matrix(Lc, Gc)
    precision = float(np.mean(D.min(axis=1) <= r))
    recall = float(np.mean(D.min(axis=0) <= r))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "r": float(r)}


def mean_sym_distance(
    L: Sequence[tuple[int, int]] | np.ndarray,
    G: Sequence[tuple[int, int]] | np.ndarray,
) -> float:
    """Mean symmetric Chebyshev distance: half the sum of the two directed mean nearest-distances."""
    Lc, Gc = _as_cells(L), _as_cells(G)
    D = _cheb_matrix(Lc, Gc)
    return float(0.5 * (D.min(axis=1).mean() + D.min(axis=0).mean()))


def hausdorff(
    L: Sequence[tuple[int, int]] | np.ndarray,
    G: Sequence[tuple[int, int]] | np.ndarray,
) -> float:
    """Hausdorff distance under Chebyshev metric."""
    Lc, Gc = _as_cells(L), _as_cells(G)
    D = _cheb_matrix(Lc, Gc)
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


def _first_axis(X: np.ndarray) -> np.ndarray | None:
    centred = X - X.mean(axis=0)
    if not np.any(centred):
        return None
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[0]


def orientation_error(
    L: Sequence[tuple[int, int]] | np.ndarray,
    G: Sequence[tuple[int, int]] | np.ndarray,
) -> float:
    """Angle in degrees in [0, 90] between the first principal axes of L and G; NaN if either is degenerate."""
    u = _first_axis(_as_cells(L))
    v = _first_axis(_as_cells(G))
    if u is None or v is None:
        return float("nan")
    dot = abs(float(np.dot(u, v)))
    cross = abs(float(u[0] * v[1] - u[1] * v[0]))
    return float(np.degrees(np.arctan2(cross, dot)))


def percentile_vs_null(value: float, null_values: Sequence[float] | np.ndarray) -> float:
    """Percentile of value within the null distribution: 100 * (#below + 0.5 #equal) / n, NaNs ignored."""
    null = np.asarray(null_values, dtype=float).ravel()
    null = null[~np.isnan(null)]
    if null.size == 0 or np.isnan(value):
        return float("nan")
    below = np.sum(null < value)
    equal = np.sum(null == value)
    return float(100.0 * (below + 0.5 * equal) / null.size)
