"""Tests for the evolutionary, greedy and archive baselines (src/baselines, src/eval/archive.py).

The fast env is built once per module. The tiny end-to-end runs use pop 10 and 3 generations so
that the whole file stays in the order of a minute.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from src.baselines import evo
from src.baselines.run import run_method, run_tabular
from src.envs.context import build_context
from src.envs.fast_motndp import build_city, make_env
from src.eval.archive import ArchiveWrapper, nondominated_mask, nondominated_points, raw_objectives, upstream_reward
from src.eval.normalization import Norm, save_norm
from src.eval.pareto import nondominated
from src.eval.weights import lattice

NR_STATIONS = 20
CITY, VARIANT, GROUPS = "xian", "full", "price_groups_5.txt"
NORM_F_REF = np.array([0.03, 0.015, 0.35])
RECORD_KEYS = {
    "schema", "city", "variant", "groups", "chained", "nr_stations", "method", "transfer", "seed",
    "budget_rollouts", "eval_rollouts", "config", "normalization", "env", "policy_front",
    "archive_nd", "archive_best", "curve", "indicators",
}

_CACHE: dict = {}


def _city():
    if "city" not in _CACHE:
        city = build_city(CITY, VARIANT, GROUPS)
        _CACHE["city"] = city
        _CACHE["ctx"] = build_context(city)
    return _CACHE["city"], _CACHE["ctx"]


@pytest.fixture(scope="module")
def norm_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("norm") / "normalization.json"
    save_norm(Norm(f_ref=NORM_F_REF), path)
    return path


def _wrapped_env():
    city, _ = _city()
    return ArchiveWrapper(make_env(city, NR_STATIONS, True))


def _problem(env=None):
    city, ctx = _city()
    env = _wrapped_env() if env is None else env
    return evo.Problem(env, ctx, Norm(f_ref=NORM_F_REF), NR_STATIONS)


# ----------------------------------------------------------------------------- decode repair


def test_decode_repair_always_yields_feasible_walk():
    p = _problem()
    rng = np.random.default_rng(0)
    base = p.base
    for _ in range(60):
        ind = evo.Individual(start=int(rng.integers(p.N)), acts=rng.integers(0, 8, size=p.n_acts))
        rec = p.decode(ind)
        # independent replay: every executed action must be allowed by the env's own mask
        _, info = base.reset(options={"loc": p.coords(ind.start)})
        executed = []
        for a in ind.acts:
            a = int(a)
            assert info["action_mask"][a] == 1, "decode returned an action the env forbids"
            executed.append(a)
            _, _, terminated, _, info = base.step(a)
            if terminated:
                break
        assert len(rec.cells) == len(executed) + 1  # one new cell per executed step
        cells_env = np.asarray(info["covered_cells_coordinates"])
        assert np.array_equal(cells_env, rec.cells)
        # the repaired genome is written back: its prefix is the executed walk
        assert list(ind.acts[: len(executed)]) == executed


def test_decode_keeps_feasible_genome_unchanged():
    p = _problem()
    rng = np.random.default_rng(1)
    base = p.base
    # build a feasible genome by a random walk, then decode it: nothing should be repaired
    _, info = base.reset(options={"loc": p.coords(p.N // 2)})
    acts = []
    while len(acts) < p.n_acts:
        a = int(rng.choice(np.flatnonzero(info["action_mask"])))
        acts.append(a)
        _, _, terminated, _, info = base.step(a)
        if terminated:
            break
    acts = np.asarray(acts + [0] * (p.n_acts - len(acts)), dtype=np.int64)
    ind = evo.Individual(start=p.N // 2, acts=acts.copy())
    p.decode(ind)
    assert np.array_equal(ind.acts[: len(acts)], acts[: len(acts)])


# ----------------------------------------------------------------------------- NSGA-II sorting


def test_nondominated_sort_on_hand_built_set():
    A = [3, 0, 0]
    B = [0, 3, 0]
    C = [0, 0, 3]
    D = [1, 1, 1]
    E = [0.5, 0.5, 0.5]  # dominated by D
    F = [0, 0, 0]  # dominated by everything
    fronts = evo.fast_nondominated_sort(np.array([A, B, C, D, E, F], dtype=float))
    assert [sorted(f.tolist()) for f in fronts] == [[0, 1, 2, 3], [4], [5]]


def test_nondominated_sort_matches_pareto_module():
    rng = np.random.default_rng(3)
    F = rng.random((60, 3))
    first = evo.fast_nondominated_sort(F)[0]
    expected = np.flatnonzero(nondominated(F))
    assert sorted(first.tolist()) == sorted(expected.tolist())


def test_crowding_distance_on_known_front():
    F = np.array([[0, 3, 0], [1, 2, 0], [2, 1, 0], [3, 0, 0]], dtype=float)
    d = evo.crowding_distance(F)
    assert np.isinf(d[0]) and np.isinf(d[3])
    assert d[1] == pytest.approx(4 / 3) and d[2] == pytest.approx(4 / 3)
    assert np.all(np.isinf(evo.crowding_distance(F[:2])))


def test_select_nsga2_keeps_boundaries_and_fills_by_front():
    F = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3], [1, 1, 1], [0.5, 0.5, 0.5], [0, 0, 0]], dtype=float)
    keep_all_first = sorted(evo.select_nsga2(F, 4).tolist())
    assert keep_all_first == [0, 1, 2, 3]
    # A, B, C are extreme in some objective (crowding inf); D is interior, so n=3 keeps A, B, C
    assert set(evo.select_nsga2(F, 3).tolist()) == {0, 1, 2}


def test_chunked_nondominated_mask_matches_pareto():
    rng = np.random.default_rng(5)
    F = np.round(rng.random((300, 3)), 1)  # ties on purpose
    assert np.array_equal(nondominated_mask(F, chunk=64), nondominated(F))


def test_sweep_nondominated_points_matches_pareto_on_unique_rows():
    rng = np.random.default_rng(6)
    F = np.round(rng.random((400, 3)), 1)  # heavy ties
    expected = np.unique(F[nondominated(F)], axis=0)
    got = nondominated_points(F)
    got_sorted = got[np.lexsort(got.T[::-1])]
    assert got.shape == expected.shape and np.allclose(got_sorted, expected)
    assert nondominated_points(np.zeros((0, 3))).shape == (0, 3)


# ----------------------------------------------------------------------------- archive wrapper


def test_archive_records_exactly_one_entry_per_episode():
    env = _wrapped_env()
    rng = np.random.default_rng(7)
    n_episodes = 12
    for ep in range(n_episodes):
        _, info = env.reset(options={"loc": (int(rng.integers(29)), int(rng.integers(29)))})
        if ep % 3 == 0:  # an unfinished episode that is reset must not be recorded
            env.step(int(rng.choice(np.flatnonzero(info["action_mask"]))))
            _, info = env.reset(options={"loc": (5, 5)})
        s_sum = np.zeros(5)
        terminated = False
        while not terminated:
            a = int(rng.choice(np.flatnonzero(info["action_mask"])))
            _, r, terminated, _, info = env.step(a)
            s_sum += r
        rec = env.last_record
        assert np.array_equal(rec.cells, np.asarray(info["covered_cells_coordinates"]))
        assert np.allclose(rec.s, s_sum, atol=1e-12)
    assert len(env.archive) == n_episodes


def test_upstream_returns_match_upstream_calculate_reward():
    from src.baselines.upstream import PatchedGATNDP

    rng = np.random.default_rng(9)
    r = rng.random(5) * 0.01
    assert upstream_reward(r, "max_efficiency") == pytest.approx(float(r.sum()))
    assert upstream_reward(r, "ggi2") == pytest.approx(float(PatchedGATNDP.ggi_reward(None, r, 2)))


# ----------------------------------------------------------------------------- greedy lookahead


def test_greedy_incremental_state_matches_env_replay():
    city, ctx = _city()
    p = _problem()
    geo = evo._Geometry(city, ctx)
    W = lattice(4, 3)
    for start in (400, 17, 500):
        for w in W[[0, 5, 14]]:
            walk = evo._init_walk(geo, city, 5, p.coords(start))
            while walk.stations < NR_STATIONS:
                mask = p.base.mask_actions(walk.loc, walk.loc + evo.ACTION_TO_DIRECTION, np.asarray(walk.cells))
                allowed = np.flatnonzero(mask)
                if allowed.size == 0:
                    break
                best = None
                for a in allowed:
                    c = evo._candidate(geo, p.base, city, walk, int(a), w, Norm(f_ref=NORM_F_REF))
                    if best is None or c["g"] > best["g"]:
                        best = c
                walk = evo._apply(walk, best)
            rep = evo.replay_walk(p.base, p.coords(start), walk.actions)
            assert np.allclose(rep["s"], walk.s, atol=1e-12)
            assert len(rep["cells"]) == len(walk.cells)
            env_raw = raw_objectives(ctx, rep["cells"], rep["segments"], rep["s"])
            assert np.allclose(env_raw, walk.raw, atol=1e-12)


def test_greedy_by_g_archives_every_walk():
    p = _problem()
    norm = Norm(f_ref=NORM_F_REF)
    starts = np.array([10, 300, 620])
    res = evo.greedy_by_g(p, lattice(4, 3), norm, starts=starts)
    assert len(p.env.archive) == starts.size * 15
    assert res.n_walks == starts.size * 15
    assert all(w is not None for w in res.best_walks)


# ----------------------------------------------------------------------------- determinism and end to end


def _strip_clock(rec: dict) -> dict:
    out = json.loads(json.dumps(rec))
    out["env"].pop("wall_clock_s", None)
    return out


def test_same_seed_gives_identical_record(norm_path):
    kw = dict(city=CITY, variant=VARIANT, groups=GROUPS, nr_stations=NR_STATIONS, chained=True, seed=3,
              norm_path=norm_path, log=lambda *a, **k: None)
    a = run_method("nsga2", pop=10, gens=3, **kw)
    b = run_method("nsga2", pop=10, gens=3, **kw)
    assert json.dumps(_strip_clock(a), sort_keys=True) == json.dumps(_strip_clock(b), sort_keys=True)
    c = run_method("nsga2", pop=10, gens=3, **{**kw, "seed": 4})
    assert json.dumps(_strip_clock(a), sort_keys=True) != json.dumps(_strip_clock(c), sort_keys=True)


@pytest.mark.parametrize(
    "method, extra",
    [
        ("nsga2", dict(pop=10, gens=3)),
        ("ga_scalar", dict(pop=10, gens=3)),
        ("ga_scalar_matched", dict(pop=10, gens=3)),
        ("random", dict(pop=10, gens=3)),
        ("greedy", dict(n_starts=4)),
        ("ga_demand", dict(pop=10, gens=3)),
        ("ga_ggi2", dict(pop=10, gens=3)),
    ],
)
def test_end_to_end_small_run_is_valid(method, extra, norm_path):
    rec = run_method(
        method, city=CITY, variant=VARIANT, groups=GROUPS, nr_stations=NR_STATIONS, chained=True,
        seed=0, norm_path=norm_path, log=lambda *a, **k: None, **extra,
    )
    assert RECORD_KEYS <= set(rec)
    assert rec["method"] == method
    assert rec["budget_rollouts"] > 0
    assert rec["archive_nd"]
    for part in ("policy", "archive"):
        ind = rec["indicators"][part]
        assert {"hv", "eum", "n_nondominated", "igd_plus"} <= set(ind)
        assert np.isfinite(ind["hv"]) and np.isfinite(ind["eum"])
    if method in ("ga_demand", "ga_ggi2"):
        assert len(rec["policy_front"]) == 1 and rec["policy_front"][0]["w"] is None
    elif method == "greedy":
        assert len(rec["policy_front"]) == 15
    else:
        assert len(rec["policy_front"]) == 15
        assert all(p["g_w"] is not None for p in rec["policy_front"])
    for p in rec["policy_front"]:
        assert len(p["cells"]) >= 1 and p["obj"]["D_tot"] >= 0.0


@pytest.mark.parametrize("reward", ["max_efficiency", "ggi2"])
def test_tabular_small_run_is_valid(reward, norm_path, tmp_path):
    rec = run_tabular(
        reward_type=reward, city=CITY, variant=VARIANT, groups=GROUPS, nr_stations=NR_STATIONS, chained=True,
        seed=0, norm_path=norm_path, episodes=30, workdir=tmp_path, log=lambda *a, **k: None,
    )
    assert RECORD_KEYS <= set(rec)
    assert rec["method"] == ("tabular_eff" if reward == "max_efficiency" else "tabular_ggi2")
    assert len(rec["policy_front"]) == 1 and rec["policy_front"][0]["w"] is None
    assert rec["budget_rollouts"] == 30 + 1  # 30 training episodes and one greedy rollout
    assert "hv" in rec["indicators"]["policy"] and "eum" in rec["indicators"]["policy"]
