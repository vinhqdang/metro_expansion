"""Archive of every terminal state reached by a search baseline.

ArchiveWrapper wraps the env handed to an upstream agent (GATNDP, QLearningTNDP) without
changing their code. At every terminal step it stores the final covered cells, the
per-group satisfaction s (sum of the env's per-step pct reward), the directed covered
segments and the upstream per-episode returns (max_efficiency and ggi2, summed per step
exactly as calculate_reward does). The upstream returns let the archive recover the
agent's own best episode.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Iterable

import gymnasium as gym
import numpy as np

from src.eval.normalization import Norm
from src.eval.objectives import CityContext, a_new, d_tot, ggi

Record = namedtuple("Record", "cells s segments ret_max_efficiency ret_ggi2")


def upstream_reward(reward: Any, reward_type: str) -> float:
    """Per-step reward exactly as GATNDP.calculate_reward / QLearningTNDP.calculate_reward."""
    r = np.asarray(reward, dtype=float)
    if reward_type == "max_efficiency":
        return float(r.sum())
    if reward_type == "ggi2":
        weights = 1 / (2 ** np.arange(r.shape[0]))
        weights /= weights.sum()
        return float(np.sum(np.sort(r) * weights))
    raise ValueError(f"unsupported upstream reward type {reward_type!r}")


def unique_directed_pairs(segments: Iterable) -> np.ndarray:
    """Set of directed (i, j) pairs from info['segments'] (a list with possible duplicates), sorted."""
    pairs = {(int(a), int(b)) for a, b in segments}
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(sorted(pairs), dtype=np.int32)


def raw_objectives(ctx: CityContext, cells, segments, s) -> np.ndarray:
    """Raw (D_tot, GGI_2, A_new) of one finished line, in the order of objectives.OBJECTIVE_KEYS."""
    cells_t = [(int(c[0]), int(c[1])) for c in cells]
    pairs = [(int(a), int(b)) for a, b in segments]
    return np.array(
        [
            d_tot(pairs, ctx.od_raw, ctx.total_od),
            ggi(s, base=2.0),
            a_new(ctx.mu, ctx.existing_mask, cells_t, ctx.grid_shape, K=2),
        ],
        dtype=float,
    )


def nondominated_mask(F: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Non-dominated rows of F (maximisation), memory-bounded for large archives.

    Same semantics as src.eval.pareto.nondominated (exact duplicates are all kept), but
    compares in blocks so that 10^4 points do not need an n x n x 3 array.
    """
    P = np.asarray(F, dtype=float).reshape(-1, 3)
    n = P.shape[0]
    keep = np.ones(n, dtype=bool)
    for start in range(0, n, chunk):
        Q = P[start : start + chunk]
        ge = np.all(P[None, :, :] >= Q[:, None, :], axis=-1)
        gt = np.any(P[None, :, :] > Q[:, None, :], axis=-1)
        keep[start : start + chunk] = ~np.any(ge & gt, axis=1)
    return keep


def nondominated_points(F: np.ndarray) -> np.ndarray:
    """Unique non-dominated rows of F (maximisation), for archives with 10^5 entries.

    Rows are de-duplicated, then visited in decreasing order of their sum: a point can only be
    dominated by a point with a larger sum, so each row is compared against the current
    non-dominated set alone. Same set as pareto.nondominated on the unique rows.
    """
    P = np.unique(np.asarray(F, dtype=float).reshape(-1, 3), axis=0)
    if P.shape[0] == 0:
        return P
    order = np.argsort(-P.sum(axis=1), kind="stable")
    kept: list[np.ndarray] = []
    S = np.zeros((0, 3))
    for idx in order:
        p = P[idx]
        if S.shape[0]:
            dominated = np.any(np.all(S >= p, axis=1) & np.any(S > p, axis=1))
            if dominated:
                continue
        kept.append(p)
        S = np.vstack([S, p[None, :]])
    return np.asarray(kept)


class Archive:
    """In-memory list of terminal states, one entry per finished episode."""

    def __init__(self) -> None:
        self.records: list[Record] = []
        self._raw_rows: list[np.ndarray] = []
        self._raw_ctx: CityContext | None = None

    def __len__(self) -> int:
        return len(self.records)

    def add(self, cells, s, segments, ret_max_efficiency: float, ret_ggi2: float) -> Record:
        rec = Record(
            cells=np.asarray(cells, dtype=np.int32).reshape(-1, 2).copy(),
            s=np.asarray(s, dtype=float).copy(),
            segments=unique_directed_pairs(segments),
            ret_max_efficiency=float(ret_max_efficiency),
            ret_ggi2=float(ret_ggi2),
        )
        self.records.append(rec)
        return rec

    def raw_matrix(self, ctx: CityContext) -> np.ndarray:
        """Raw objectives of every record (cached; recomputed only for new records or a new context)."""
        if self._raw_ctx is not ctx:
            self._raw_rows, self._raw_ctx = [], ctx
        for rec in self.records[len(self._raw_rows) :]:
            self._raw_rows.append(raw_objectives(ctx, rec.cells, rec.segments, rec.s))
        if not self._raw_rows:
            return np.zeros((0, 3))
        return np.stack(self._raw_rows)

    def objective_matrix(self, ctx: CityContext, norm: Norm) -> tuple[np.ndarray, np.ndarray]:
        """(F_norm, F_raw): normalised (clipped) and raw objective matrices, one row per record."""
        F_raw = self.raw_matrix(ctx)
        F_norm = norm.normalize(F_raw) if F_raw.shape[0] else np.zeros((0, 3))
        return F_norm, F_raw


class ArchiveWrapper(gym.Wrapper):
    """Records (cells, s, segments, upstream returns) at every terminal step of each episode.

    `last_record` holds the record of the most recently finished episode.
    """

    def __init__(self, env: gym.Env, archive: Archive | None = None) -> None:
        super().__init__(env)
        self.archive = Archive() if archive is None else archive
        self.last_record: Record | None = None
        self.n_errors = 0  # steps the env rejected (disallowed action); the episode is not recorded
        self.nr_groups = int(self.unwrapped.city.group_od_mx.shape[0])
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._s = np.zeros(self.nr_groups)
        self._ret_me = 0.0
        self._ret_gg = 0.0

    def reset(self, *, seed=None, options=None):
        self._reset_accumulators()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except Exception:
            self.n_errors += 1
            raise
        r = np.asarray(reward, dtype=float)
        self._s = self._s + r
        self._ret_me += upstream_reward(r, "max_efficiency")
        self._ret_gg += upstream_reward(r, "ggi2")
        if terminated or truncated:
            self.last_record = self.archive.add(
                info["covered_cells_coordinates"],
                self._s,
                info["segments"],
                self._ret_me,
                self._ret_gg,
            )
        return obs, reward, terminated, truncated, info
