"""Candidate features: the gain computed before a step must equal the reward the env returns for that step.

Also checks that computing features leaves the environment state untouched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMP_NUM_THREADS", "1")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.context import build_context  # noqa: E402
from src.envs.fast_motndp import build_city, make_env  # noqa: E402
from src.rl.features import (  # noqa: E402
    ACTION_TO_DIRECTION,
    build_feature_context,
    candidate_features,
    candidate_gains,
    covered_reach,
)

NR_STATIONS = 20
SEEDS = [0, 1, 2, 3, 4]


@pytest.fixture(scope="module")
def xian():
    city = build_city("xian", "full")
    return city, build_context(city)


def _candidates(menv):
    info = menv._get_info()
    loc = info["location_grid_coordinates"]
    nb = loc + ACTION_TO_DIRECTION
    valid = np.asarray(info["action_mask"]).astype(bool)
    cand_idx = np.where(valid, nb[:, 0] * menv.city.grid_y_size + nb[:, 1], 0)
    return info, nb, valid, cand_idx


@pytest.mark.parametrize("seed", SEEDS)
def test_gain_for_taken_action_equals_env_reward(xian, seed):
    city, ctx = xian
    env = make_env(city, nr_stations=NR_STATIONS, chained=True)
    menv = env.unwrapped
    rng = np.random.default_rng(1000 + seed)
    obs, info = env.reset(seed=seed)
    done = False
    steps = 0
    while not done:
        info, nb, valid, cand_idx = _candidates(menv)
        conn_before = set(menv.connections_with_existing_lines)
        n_cov_before = len(menv.covered_segments)
        pct, total = candidate_gains(menv, int(info["location_grid_index"]), cand_idx, valid)
        # features must not mutate the environment
        assert set(menv.connections_with_existing_lines) == conn_before
        assert len(menv.covered_segments) == n_cov_before

        choices = np.flatnonzero(valid)
        a = int(choices[rng.integers(len(choices))])
        _, reward, terminated, truncated, info = env.step(a)
        np.testing.assert_allclose(pct[a], reward, rtol=0, atol=1e-12)
        assert total[a] >= 0.0
        done = terminated or truncated
        steps += 1
    assert steps == NR_STATIONS - 1


def test_feature_matrix_shape_and_masking(xian):
    city, ctx = xian
    env = make_env(city, nr_stations=NR_STATIONS, chained=True)
    menv = env.unwrapped
    env.reset(seed=7)
    m = city.group_od_mx.shape[0]
    fctx = build_feature_context(city, ctx, NR_STATIONS, f_ref1=0.5)
    fctx.gain_scale = 1.0
    info, nb, valid, cand_idx = _candidates(menv)
    feats = candidate_features(
        menv, fctx, nb, cand_idx, valid, int(info["location_grid_index"]),
        info["covered_cells_coordinates"], menv.stations_placed,
    )
    assert feats.shape == (8, 4 + m)
    assert np.all(feats[~valid] == 0.0)
    # group-gain block is the raw env gain divided by f_ref1
    pct, _ = candidate_gains(menv, int(info["location_grid_index"]), cand_idx, valid)
    np.testing.assert_allclose(feats[valid, 1 : 1 + m], pct[valid] / 0.5, atol=1e-12)
    # remaining-steps entry is the same for all valid candidates and equals (nr - placed)/nr
    expected_e = (NR_STATIONS - menv.stations_placed) / NR_STATIONS
    np.testing.assert_allclose(feats[valid, -1], expected_e)
    # transfer flag agrees with the Chebyshev-1 mask
    for k in np.flatnonzero(valid):
        x, y = nb[k]
        assert feats[k, 2 + m] == float(fctx.transfer_mask[x, y])


def test_new_accessibility_increment_matches_a_new(xian):
    """Summing the increment over the candidate chain reproduces A_new of the resulting line."""
    from src.eval.objectives import a_new

    city, ctx = xian
    env = make_env(city, nr_stations=NR_STATIONS, chained=True)
    menv = env.unwrapped
    fctx = build_feature_context(city, ctx, NR_STATIONS, f_ref1=1.0)
    env.reset(seed=11)
    rng = np.random.default_rng(5)
    line = [tuple(int(v) for v in menv._loc_grid_coordinates)]
    # the start cell's own reach is part of A_new but is not produced by any step
    numer = float((fctx.mu * (covered_reach(fctx, line) & ~fctx.existing_reach)).sum())
    done = False
    while not done:
        info, nb, valid, cand_idx = _candidates(menv)
        reach = covered_reach(fctx, line)
        feats = candidate_features(menv, fctx, nb, cand_idx, valid, int(info["location_grid_index"]),
                                   line, menv.stations_placed, reach=reach)
        choices = np.flatnonzero(valid)
        a = int(choices[rng.integers(len(choices))])
        numer += feats[a, 1 + city.groups.shape[0]] * fctx.mu_denom
        _, _, terminated, truncated, info = env.step(a)
        line.append(tuple(int(v) for v in info["location_grid_coordinates"]))
        done = terminated or truncated
    expected = a_new(ctx.mu, ctx.existing_mask, line, ctx.grid_shape, K=2)
    assert numer / fctx.mu_denom == pytest.approx(expected, abs=1e-12)
