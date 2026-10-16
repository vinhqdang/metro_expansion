"""State-dependent candidate features for the RL policy (design section 1.4).

For the current partial line, every feasible neighbour cell c (8 directions) gets

  phi_c = [ a_c | b_c,1..b_c,m | c_c | d_c | e ]          (4 + m entries)

  a_c   log1p(total OD gained by adding c, scaled by a per-city constant)
  b_c,g exact per-group satisfied-demand gain the env would return for the step
        to c, divided by f_ref_1 (the D_tot anchor)
  c_c   increment of trip-end mass newly within Chebyshev K of the line (A_new increment)
  d_c   1 if c lies within Chebyshev 1 of an existing-line cell (transfer opportunity)
  e     remaining steps / nr_stations (same for all candidates)

Gains are computed without stepping or mutating the environment. The pair
construction is the fast env's own machinery (FastMOTNDP._satisfied_pairs), and
the connection set is copied, not updated. `candidate_gains` returns the raw
per-group percentage gains, so tests can compare them with env rewards exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.eval.objectives import _dilate_chebyshev

ACTION_TO_DIRECTION = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])


@dataclass
class FeatureContext:
    """Static, per-city quantities used to build candidate features."""

    grid_shape: tuple[int, int]
    n_groups: int
    od: np.ndarray  # [N, N] raw OD (existing-line pairs already zeroed by City)
    group_od_sum: np.ndarray  # [m]
    mu: np.ndarray  # [X, Y] trip-end mass
    existing_reach: np.ndarray  # [X, Y] bool: E = cells within K of existing lines
    transfer_mask: np.ndarray  # [X, Y] bool: within Chebyshev 1 of an existing-line cell
    mu_denom: float  # sum of mu outside E (A_new denominator)
    nr_stations: int
    K: int = 2
    f_ref1: float = 1.0  # scale for per-group gains
    gain_scale: float = 1.0  # scale for log1p(total gain)


def build_feature_context(city, ctx, nr_stations: int, f_ref1: float, K: int = 2) -> FeatureContext:
    """`ctx` is a src.eval.objectives.CityContext (see src/envs/context.py::build_context)."""
    shape = tuple(ctx.grid_shape)
    existing_full = np.asarray(ctx.existing_mask, dtype=bool)
    E = _dilate_chebyshev(existing_full, K)
    transfer = _dilate_chebyshev(existing_full, 1)
    mu = np.asarray(ctx.mu, dtype=float).reshape(shape)
    denom = float(mu[~E].sum())
    return FeatureContext(
        grid_shape=shape,
        n_groups=int(len(ctx.group_od_sum)),
        od=np.asarray(ctx.od_raw, dtype=float),
        group_od_sum=np.asarray(ctx.group_od_sum, dtype=float),
        mu=mu,
        existing_reach=E,
        transfer_mask=transfer,
        mu_denom=denom if denom > 0 else 1.0,
        nr_stations=int(nr_stations),
        K=K,
        f_ref1=float(f_ref1) if f_ref1 > 0 else 1.0,
    )


def _covered_set(menv) -> set:
    return set(map(tuple, menv.covered_segments))


def _pair_gain(menv, from_idx: int, to_idx: int, cells_to_chain, covered: set):
    """Non-mutating replica of FastMOTNDP._calculate_reward(segment) for one candidate step.

    Returns (pct gain vector [m], unique new pairs [P, 2]).
    """
    city = menv.city
    n_groups = city.group_od_mx.shape[0]
    segment = [from_idx, to_idx]
    if segment in menv.covered_segments:  # upstream early return: an already covered segment earns nothing
        return np.zeros(n_groups), np.empty((0, 2), dtype=np.int64)

    seg = np.asarray(segment)
    if menv.chained_reward:
        connected = city.connections_with_existing_lines(seg)
        # upstream adds `connected` to the persistent set before pairing; here the union is a copy
        conn = set(menv.connections_with_existing_lines)
        conn.update(connected)
        pairs = menv._satisfied_pairs(seg, cells_to_chain, conn)
    else:
        pairs = menv._satisfied_pairs(seg, None, None)

    if covered:
        pairs = np.array([p for p in pairs if tuple(p) not in covered], dtype=np.int64).reshape(-1, 2)
    else:
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.shape[0] == 0:
        return np.zeros(n_groups), pairs
    uniq = np.unique(pairs, axis=0)
    sat = city.group_od_mx[:, uniq[:, 0], uniq[:, 1]].sum(axis=1)
    gsum = city.group_od_sum
    pct = np.divide(sat, gsum, out=np.zeros_like(sat), where=gsum != 0)
    return pct, uniq


def candidate_gains(menv, from_idx: int, cand_idx: np.ndarray, valid: np.ndarray):
    """Raw gains for every candidate (only where valid).

    Returns:
        pct:   [n, m] per-group percentage gain the env would return for stepping to each candidate
        total: [n]    total OD gained (sum of od over the new unique pairs)
    The environment is not mutated.
    """
    city = menv.city
    n_groups = city.group_od_mx.shape[0]
    n = len(cand_idx)
    pct = np.zeros((n, n_groups))
    total = np.zeros(n)
    if menv.od_type != "pct":
        raise ValueError("candidate_gains expects an env with od_type='pct'")
    cells_to_chain = None
    if menv.chained_reward:
        cells_to_chain = np.asarray(city.grid_to_index(np.array(menv.covered_cells_coordinates)))
    covered = _covered_set(menv)
    od = city.od_mx
    for k in range(n):
        if not valid[k]:
            continue
        g, uniq = _pair_gain(menv, int(from_idx), int(cand_idx[k]), cells_to_chain, covered)
        pct[k] = g
        if uniq.shape[0]:
            total[k] = float(od[uniq[:, 0], uniq[:, 1]].sum())
    return pct, total


def _window_sum(arr: np.ndarray, x: int, y: int, K: int) -> float:
    X, Y = arr.shape
    return float(arr[max(0, x - K) : min(X, x + K + 1), max(0, y - K) : min(Y, y + K + 1)].sum())


def covered_reach(fctx: FeatureContext, line_cells_xy) -> np.ndarray:
    """Bool grid: cells already accounted for in A_new (E, plus within K of the current line)."""
    V = np.zeros(fctx.grid_shape, dtype=bool)
    for x, y in line_cells_xy:
        V[int(x), int(y)] = True
    return fctx.existing_reach | _dilate_chebyshev(V, fctx.K)


def candidate_features(menv, fctx: FeatureContext, cand_xy: np.ndarray, cand_idx: np.ndarray,
                       valid: np.ndarray, from_idx: int, line_cells_xy, stations_placed: int,
                       reach: np.ndarray | None = None) -> np.ndarray:
    """Feature matrix [8, 4 + m]. Rows of infeasible candidates are zero."""
    m = fctx.n_groups
    out = np.zeros((len(cand_idx), 4 + m), dtype=np.float64)
    pct, total = candidate_gains(menv, from_idx, cand_idx, valid)
    if reach is None:
        reach = covered_reach(fctx, line_cells_xy)
    M = fctx.mu * (~reach)
    remaining = max(fctx.nr_stations - int(stations_placed), 0) / max(fctx.nr_stations, 1)
    for k in range(len(cand_idx)):
        if not valid[k]:
            continue
        x, y = int(cand_xy[k, 0]), int(cand_xy[k, 1])
        out[k, 0] = np.log1p(total[k] / fctx.gain_scale)
        out[k, 1 : 1 + m] = pct[k] / fctx.f_ref1
        out[k, 1 + m] = _window_sum(M, x, y, fctx.K) / fctx.mu_denom
        out[k, 2 + m] = float(fctx.transfer_mask[x, y])
        out[k, 3 + m] = remaining
    return out


def calibrate_gain_scale(menv, fctx: FeatureContext, starts, rng: np.random.Generator,
                         n_episodes: int = 200, percentile: float = 99.0) -> float:
    """Per-city scale for log1p(total gain): the given percentile of total gains over random feasible rollouts."""
    gains = []
    for ep in range(n_episodes):
        start = starts[int(rng.integers(len(starts)))]
        menv.reset(options={"loc": (int(start[0]), int(start[1]))})
        done = False
        while not done:
            info = menv._get_info()
            loc = info["location_grid_coordinates"]
            nb = loc + ACTION_TO_DIRECTION
            mask = np.asarray(info["action_mask"]).astype(bool)
            idx = np.where(mask, nb[:, 0] * menv.city.grid_y_size + nb[:, 1], 0)
            _, total = candidate_gains(menv, int(info["location_grid_index"]), idx, mask)
            gains.extend(total[mask].tolist())
            choices = np.flatnonzero(mask)
            a = int(choices[rng.integers(len(choices))])
            _, _, term, trunc, _ = menv.step(a)
            done = term or trunc
    g = np.asarray(gains)
    g = g[g > 0]
    return float(np.percentile(g, percentile)) if g.size else 1.0
