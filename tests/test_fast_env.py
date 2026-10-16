"""Equality gate for src/envs/fast_motndp.py against the upstream MOTNDP env.

Both envs are stepped in lock-step with the same reset seed and the same
action sequence (uniform over the valid actions in info["action_mask"]). At
every step the reward vectors must agree within 1e-12 and the covered cells,
covered segments, action masks and termination flags must be identical.

The 2,000-episode version is marked `slow` and runs only with RUN_SLOW=1, e.g.
`RUN_SLOW=1 python3.13 -m pytest -m slow tests/test_fast_env.py`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_ROOT = REPO_ROOT / "external" / "tabular-tndp"
MOTNDP_ROOT = UPSTREAM_ROOT / "envs" / "mo-tndp"
for _p in (str(REPO_ROOT), str(UPSTREAM_ROOT), str(MOTNDP_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OMP_NUM_THREADS", "1")

import envs  # noqa: E402,F401  (registers motndp_xian-v0 and motndp_amsterdam-v0 upstream)
import mo_gymnasium as mo_gym  # noqa: E402
from motndp.constraints import MetroConstraints  # noqa: E402
from motndp.motndp import ACTION_TO_DIRECTION  # noqa: E402

from src.envs.fast_motndp import (  # noqa: E402
    build_city,
    interpolate_path,
    make_env,
    register_fast_envs,
)

NR_STATIONS = 20
TOL = 1e-12
CITY_CONFIGS = [("xian", "full"), ("amsterdam", "full"), ("xian", "line1_only")]
UPSTREAM_ID = {"xian": "motndp_xian-v0", "amsterdam": "motndp_amsterdam-v0"}

_CITY_CACHE: dict = {}


def _city(name: str, variant: str):
    key = (name, variant)
    if key not in _CITY_CACHE:
        _CITY_CACHE[key] = build_city(name, variant)
    return _CITY_CACHE[key]


def _make_pair(name: str, variant: str, chained: bool):
    """Upstream env (registered id) and fast env, both on the same City object."""
    city = _city(name, variant)
    upstream = mo_gym.make(
        UPSTREAM_ID[name],
        city=city,
        constraints=MetroConstraints(city),
        nr_stations=NR_STATIONS,
        od_type="pct",
        chained_reward=chained,
    )
    fast = make_env(city, NR_STATIONS, chained)
    return upstream, fast


def _cells(info) -> list[list[int]]:
    return [[int(v) for v in c] for c in info["covered_cells_coordinates"]]


def _run_lockstep(upstream, fast, n_episodes: int, seed_offset: int) -> int:
    """Step both envs through n_episodes random episodes. Returns total steps."""
    total_steps = 0
    for ep in range(n_episodes):
        seed = seed_offset + ep
        obs_u, info_u = upstream.reset(seed=seed)
        obs_f, info_f = fast.reset(seed=seed)
        assert _cells(info_u) == _cells(info_f)
        assert np.array_equal(info_u["action_mask"], info_f["action_mask"])
        rng = np.random.default_rng(seed + 10_000_000)
        done = False
        step = 0
        while not done:
            valid = np.flatnonzero(info_u["action_mask"])
            assert valid.size > 0, "no valid action before termination"
            assert np.array_equal(info_u["action_mask"], info_f["action_mask"])
            action = int(rng.choice(valid))

            _, rew_u, term_u, trunc_u, info_u = upstream.step(action)
            _, rew_f, term_f, trunc_f, info_f = fast.step(action)
            step += 1
            total_steps += 1

            diff = np.max(np.abs(np.asarray(rew_u) - np.asarray(rew_f)))
            assert diff <= TOL, f"seed {seed} step {step}: reward diff {diff:.3e}\n up={rew_u}\n fast={rew_f}"
            assert _cells(info_u) == _cells(info_f), f"seed {seed} step {step}: cell path differs"
            assert info_u["segments"] == info_f["segments"], f"seed {seed} step {step}: segments differ"
            assert np.array_equal(info_u["action_mask"], info_f["action_mask"]), f"seed {seed} step {step}: action mask differs"
            assert (term_u, trunc_u) == (term_f, trunc_f), f"seed {seed} step {step}: termination differs"
            done = bool(term_u or trunc_u)
    return total_steps


@pytest.mark.parametrize("chained", [True, False], ids=["chained", "unchained"])
@pytest.mark.parametrize("name,variant", CITY_CONFIGS, ids=["xian_full", "amsterdam_full", "xian_line1_only"])
def test_fast_env_matches_upstream_300_episodes(name, variant, chained):
    register_fast_envs()
    upstream, fast = _make_pair(name, variant, chained)
    steps = _run_lockstep(upstream, fast, n_episodes=300, seed_offset=0)
    assert steps > 300  # sanity: episodes actually ran


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="2000-episode gate; set RUN_SLOW=1 to run")
@pytest.mark.parametrize("chained", [True, False], ids=["chained", "unchained"])
@pytest.mark.parametrize("name,variant", CITY_CONFIGS, ids=["xian_full", "amsterdam_full", "xian_line1_only"])
def test_fast_env_matches_upstream_2000_episodes(name, variant, chained):
    register_fast_envs()
    upstream, fast = _make_pair(name, variant, chained)
    steps = _run_lockstep(upstream, fast, n_episodes=2000, seed_offset=50_000)
    assert steps > 2000


def test_registered_id_builds_fast_class():
    register_fast_envs()
    register_fast_envs()  # idempotent
    import gymnasium as gym

    from src.envs.fast_motndp import FastMOTNDP

    city = _city("xian", "full")
    env = gym.make("motndp_fast-v0", city=city, constraints=MetroConstraints(city),
                   nr_stations=5, od_type="pct", chained_reward=True)
    assert isinstance(env.unwrapped, FastMOTNDP)


def test_interpolate_path_fills_gaps_with_chebyshev_steps():
    filled = interpolate_path([[8, 2], [10, 3], [10, 5]])
    assert filled[0] == [8, 2] and filled[-1] == [10, 5]
    steps = np.diff(np.array(filled), axis=0)
    assert np.abs(steps).max() == 1  # every step is a single king move
    assert all(max(abs(a), abs(b)) == 1 for a, b in steps)
    assert [10, 3] in filled and [9, 3] in filled


def test_xian_line2_is_feasible_walk_in_line1_only_city():
    """F10: the actual Line 2 cell path is a legal MetroConstraints walk in the Line-1-only city."""
    full = _city("xian", "full")
    line2 = [list(map(int, c)) for c in full.index_to_grid(full.existing_lines_full[1].flatten())]
    assert len(line2) == 27
    for a, b in zip(line2[:-1], line2[1:]):
        assert max(abs(b[0] - a[0]), abs(b[1] - a[1])) == 1, f"{a}->{b} is not a king step"

    city = _city("xian", "line1_only")
    env = mo_gym.make(UPSTREAM_ID["xian"], city=city, constraints=MetroConstraints(city),
                      nr_stations=len(line2), od_type="pct", chained_reward=True)
    env.reset(seed=0, options={"loc": tuple(line2[0])})
    for k, (a, b) in enumerate(zip(line2[:-1], line2[1:])):
        delta = (b[0] - a[0], b[1] - a[1])
        action = [i for i in range(8) if tuple(ACTION_TO_DIRECTION[i]) == delta]
        assert len(action) == 1
        action = action[0]
        mask = env.unwrapped.action_mask
        assert mask[action] == 1, f"step {k}: {a}->{b} (action {action}) masked out, mask={mask}"
        env.step(action)
    assert env.unwrapped._loc_grid_coordinates.tolist() == line2[-1]


@pytest.mark.parametrize("name,variant", CITY_CONFIGS[:2], ids=["xian_full", "amsterdam_full"])
def test_rollout_timing_report(name, variant, capsys):
    """Prints wall-clock per 20-station random rollout, fast vs upstream, single thread."""
    register_fast_envs()
    n_episodes = 200
    timings = {}
    for label in ("upstream", "fast"):
        upstream, fast = _make_pair(name, variant, True)
        env = upstream if label == "upstream" else fast
        t0 = time.perf_counter()
        for ep in range(n_episodes):
            _, info = env.reset(seed=ep)
            rng = np.random.default_rng(ep)
            for _ in range(NR_STATIONS - 1):
                valid = np.flatnonzero(info["action_mask"])
                if valid.size == 0:
                    break
                _, _, term, trunc, info = env.step(int(rng.choice(valid)))
                if term or trunc:
                    break
        timings[label] = (time.perf_counter() - t0) / n_episodes
    print(f"\n[timing] {name}/{variant} chained=True, 20-station rollout, {n_episodes} episodes, "
          f"upstream {timings['upstream'] * 1e3:.2f} ms/rollout, fast {timings['fast'] * 1e3:.2f} ms/rollout, "
          f"speed-up x{timings['upstream'] / timings['fast']:.1f}")
