"""Drivers for the baseline methods. Each returns a mnep.run.v1 record built with src.eval.record."""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines import evo
from src.envs.context import build_context
from src.envs.fast_motndp import build_city, make_env
from src.eval.archive import ArchiveWrapper, nondominated_points
from src.eval.normalization import OBJECTIVE_KEYS, Norm, load_norm, save_norm
from src.eval.objectives import CityContext
from src.eval.record import SCHEMA, front_indicators, make_point, save_record
from src.eval.weights import lattice

REPO_ROOT = Path(__file__).resolve().parents[2]
EVO_METHODS = ("nsga2", "ga_scalar", "ga_scalar_matched", "random", "greedy", "ga_demand", "ga_ggi2")
UPSTREAM_METHODS = {"ga_demand": "max_efficiency", "ga_ggi2": "ggi2"}
DEFAULT_BUDGET = {
    "nsga2": (100, 100),
    "ga_scalar": (100, 100),  # per weight: 100 + 100*100 evaluations
    "ga_scalar_matched": (30, 21),  # per weight: 30 + 21*30 evaluations (15 weights -> 9,900)
    "random": (100, 100),  # n = pop * (gens + 1) = 10,100
    "ga_demand": (100, 50),
    "ga_ggi2": (100, 50),
}


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - git may be unavailable in an exported tree
        return "unknown"


def setup(city: str, variant: str, groups: str, nr_stations: int, chained: bool, norm_path: str | Path):
    """City, context, normalisation and the archive-wrapped fast env."""
    city_obj = build_city(city, variant, groups)
    ctx = build_context(city_obj)
    norm = load_norm(norm_path)
    env = ArchiveWrapper(make_env(city_obj, nr_stations, chained))
    return city_obj, ctx, norm, env


def _best_point(individuals, norm: Norm, ctx: CityContext, w, mode: str) -> dict[str, Any]:
    """Member of `individuals` minimising g(f|w), as a deployable policy point."""
    F = np.stack([ind.f for ind in individuals])
    g = np.asarray(norm.g(F, np.asarray(w, dtype=float)), dtype=float)
    ind = individuals[int(np.argmin(g))]
    return make_point(ctx, norm, w, mode, ind.rec.cells, ind.rec.segments, ind.rec.s)


def _upstream_run(env: ArchiveWrapper, reward_type: str, pop: int, gens: int, nr_stations: int, seed: int):
    """Upstream GATNDP on the archive-wrapped env (crossover/mutation 0.9, as in run_ga.py defaults).

    Uses PatchedGATNDP: the upstream class with one NumPy-2.5 compatibility fix in mutate_path.
    """
    import random

    from src.baselines.upstream import PatchedGATNDP as GATNDP, init_wandb_disabled

    init_wandb_disabled()
    random.seed(seed)
    np.random.seed(seed)
    nr_groups = int(env.unwrapped.city.group_od_mx.shape[0])
    algo = GATNDP(
        env, init_pop_size=pop, mutation_rate=0.9, crossover_rate=0.9, generations=gens,
        nr_stations=nr_stations, nr_groups=nr_groups, seed=seed,
        wandb_project_name="TNDP-GA", wandb_experiment_name="GA-baseline", log=False,
    )
    with contextlib.redirect_stdout(io.StringIO()):  # the upstream loop prints one line per generation
        algo.run(reward_type)


def run_method(
    method: str,
    *,
    city: str,
    variant: str,
    groups: str,
    nr_stations: int,
    chained: bool,
    seed: int,
    norm_path: str | Path,
    pop: int | None = None,
    gens: int | None = None,
    n_starts: int = 0,
    log=print,
) -> dict[str, Any]:
    """Run one evolutionary/search/upstream-GA baseline and return the mnep.run.v1 record (not yet saved)."""
    if method not in EVO_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {EVO_METHODS}")
    d_pop, d_gens = DEFAULT_BUDGET.get(method, (None, None))
    pop = d_pop if pop is None else int(pop)
    gens = d_gens if gens is None else int(gens)

    t0 = time.time()
    city_obj, ctx, norm, env = setup(city, variant, groups, nr_stations, chained, norm_path)
    config: dict[str, Any] = {"algo": method, "pop": pop, "gens": gens}
    weights = lattice(4, 3)

    if method in UPSTREAM_METHODS:
        reward_type = UPSTREAM_METHODS[method]
        config.update({"reward_type": reward_type, "mutation_rate": 0.9, "crossover_rate": 0.9,
                       "evaluations": pop + gens * 2 * pop})
        _upstream_run(env, reward_type, pop, gens, nr_stations, seed)
        arch = env.archive
        if len(arch) == 0:
            raise RuntimeError("upstream GA produced no terminal line")
        key = [r.ret_max_efficiency for r in arch.records] if reward_type == "max_efficiency" else \
            [r.ret_ggi2 for r in arch.records]
        rec = arch.records[int(np.argmax(key))]
        policy = [make_point(ctx, norm, None, "upstream_best_reward", rec.cells, rec.segments, rec.s)]
        config["upstream_best_return"] = float(max(key))
        config["terminal_lines"] = len(arch)
        config["rejected_steps"] = int(env.n_errors)  # children that replayed a disallowed move (dropped upstream)

    elif method == "greedy":
        starts = None
        if n_starts and n_starts < city_obj.grid_size:
            starts = np.sort(np.random.default_rng([seed, 7]).choice(city_obj.grid_size, size=n_starts, replace=False))
        p = evo.Problem(env, ctx, norm, nr_stations)
        res = evo.greedy_by_g(p, weights, norm, starts=starts)
        policy = []
        for k, w in enumerate(weights):
            walk = res.best_walks[k]
            start_xy = p.coords(res.best_starts[k])
            rep = evo.replay_walk(env.unwrapped, start_xy, walk.actions)
            if not (np.allclose(rep["s"], walk.s, atol=1e-12) and len(rep["cells"]) == len(walk.cells)):
                raise RuntimeError("greedy state disagrees with the env replay")
            policy.append(make_point(ctx, norm, w, "greedy_by_g", rep["cells"], rep["segments"], rep["s"]))
        config.update({"n_starts": int(len(starts) if starts is not None else city_obj.grid_size),
                       "starts": "subsample" if starts is not None else "all",
                       "candidate_evals": res.n_candidates, "walks": res.n_walks,
                       "tie_break": "lowest action index"})

    elif method == "random":
        p = evo.Problem(env, ctx, norm, nr_stations)
        rng = np.random.default_rng(seed)
        lines = evo.random_lines(p, rng, pop * (gens + 1))
        policy = [_best_point(lines, norm, ctx, w, "random_min_g") for w in weights]
        config["n_lines"] = len(lines)

    elif method == "nsga2":
        p = evo.Problem(env, ctx, norm, nr_stations)
        rng = np.random.default_rng(seed)
        final = evo.nsga2(p, rng, pop_size=pop, gens=gens)
        policy = [_best_point(final, norm, ctx, w, "final_pop_best") for w in weights]
        config["evaluations"] = pop + gens * pop

    else:  # ga_scalar / ga_scalar_matched: one GA per weight, all sharing the archive
        p = evo.Problem(env, ctx, norm, nr_stations)
        policy = []
        for k, w in enumerate(weights):
            rng = np.random.default_rng([seed, k])
            final = evo.ga_scalar(p, rng, norm, w, pop_size=pop, gens=gens)
            policy.append(_best_point(final, norm, ctx, w, "final_pop_best"))
        config["evaluations_per_weight"] = pop + gens * pop
        config["evaluations_total"] = len(weights) * (pop + gens * pop)

    wall = time.time() - t0
    log(f"[{method}] {len(env.archive)} terminal lines in {wall:.1f}s")
    return build_record(
        method=method, city=city, variant=variant, groups=groups, chained=chained,
        nr_stations=nr_stations, seed=seed, env=env, ctx=ctx, norm=norm, norm_path=norm_path,
        policy=policy, config=config, wall=wall,
    )


def build_record(*, method, city, variant, groups, chained, nr_stations, seed, env, ctx, norm, norm_path,
                 policy, config, wall, curve=None) -> dict[str, Any]:
    """mnep.run.v1 record (the schema tag is added by save_record). Shared by every method."""
    archive = env.archive
    F_norm, F_raw = archive.objective_matrix(ctx, norm)
    nd_unique = nondominated_points(F_norm) if F_norm.shape[0] else np.zeros((0, 3))
    pol_F = np.asarray([p["f_norm"] for p in policy], dtype=float)
    indicators = {
        "policy": {**front_indicators(pol_F, norm), "igd_plus": None},
        "archive": {**front_indicators(nd_unique, norm), "igd_plus": None},
    }
    best: dict[str, Any] = {}
    for col, key in enumerate(OBJECTIVE_KEYS):
        idx = int(np.argmax(F_raw[:, col]))
        rec = archive.records[idx]
        pt = make_point(ctx, norm, None, f"archive_best_{key}", rec.cells, rec.segments, rec.s)
        best[key] = {"cells": pt["cells"], "obj": pt["obj"]}
    return {
        "schema": SCHEMA,
        "city": city,
        "variant": variant,
        "groups": groups,
        "chained": bool(chained),
        "nr_stations": int(nr_stations),
        "method": method,
        "transfer": None,
        "seed": int(seed),
        "budget_rollouts": len(archive),
        "eval_rollouts": 0,
        "config": config,
        "normalization": {"file": str(norm_path), "f_ref": [float(x) for x in norm.f_ref],
                          "z": float(norm.z), "rho": float(norm.rho)},
        "env": {"git_commit": git_commit(), "numpy": np.__version__,
                "omp_threads": os.environ.get("OMP_NUM_THREADS"), "wall_clock_s": round(float(wall), 3)},
        "policy_front": policy,
        "archive_nd": nd_unique.tolist(),
        "archive_best": best,
        "curve": curve or [],
        "indicators": indicators,
    }


def run_tabular(
    *,
    reward_type: str,
    city: str,
    variant: str,
    groups: str,
    nr_stations: int,
    chained: bool,
    seed: int,
    norm_path: str | Path,
    episodes: int = 10000,
    epsilon_decay_steps: int | None = None,
    workdir: str | Path,
    log=print,
) -> dict[str, Any]:
    """Upstream tabular Q-learning (patched test() only), on the archive-wrapped fast env."""
    from src.baselines.upstream import PatchedQLearningTNDP, init_wandb_disabled

    if epsilon_decay_steps is None:
        epsilon_decay_steps = int(round(episodes * 0.8))  # 400/500 ratio of the upstream defaults
    t0 = time.time()
    city_obj, ctx, norm, env = setup(city, variant, groups, nr_stations, chained, norm_path)
    init_wandb_disabled()
    np.random.seed(seed)
    import random

    random.seed(seed)
    cwd = os.getcwd()
    Path(workdir, "carbon_logs").mkdir(parents=True, exist_ok=True)  # codecarbon requires it to exist
    os.chdir(workdir)
    try:
        agent = PatchedQLearningTNDP(
            env, alpha=0.4, gamma=0.8, exploration_type="egreedy", initial_epsilon=1.0, final_epsilon=0.0,
            epsilon_warmup_steps=0, epsilon_decay_steps=epsilon_decay_steps, q_start_initial_value=0.0,
            q_initial_value=0.0, train_episodes=episodes, test_episodes=1, nr_stations=nr_stations,
            nr_groups=int(city_obj.group_od_mx.shape[0]), seed=seed, policy=None,
            wandb_project_name="TNDP-RL", wandb_experiment_name="Q-Learning-baseline", log=False,
            ucb_c_qstart=None, ucb_c_q=None, update_method="td",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            agent.train(reward_type, None)
        start = np.unravel_index(agent.Q_start.argmax(), agent.Q_start.shape)
        roll = agent.greedy_rollout(start, reward_type)
    finally:
        os.chdir(cwd)
    wall = time.time() - t0
    point = make_point(ctx, norm, None, "greedy_learned_start", roll["cells"], roll["segments"], roll["s"])
    log(f"[tabular_{reward_type}] {episodes} episodes, {len(env.archive)} terminal lines in {wall:.1f}s")
    method = "tabular_eff" if reward_type == "max_efficiency" else "tabular_ggi2"
    config = {"algo": method, "reward_type": reward_type, "episodes": episodes,
              "epsilon_decay_steps": epsilon_decay_steps, "alpha": 0.4, "gamma": 0.8,
              "exploration": "egreedy", "update_method": "td", "learned_start": [int(start[0]), int(start[1])],
              "greedy_return": roll["return"]}
    return build_record(
        method=method, city=city, variant=variant, groups=groups, chained=chained,
        nr_stations=nr_stations, seed=seed, env=env, ctx=ctx, norm=norm, norm_path=norm_path,
        policy=[point], config=config, wall=wall,
    )


def compute_anchors(
    *,
    city: str,
    variant: str,
    groups: str,
    nr_stations: int,
    chained: bool,
    seeds,
    budget_scale: float,
    out: str | Path,
    log=print,
) -> Norm:
    """Anchor f_ref_j = max over tuning seeds of the best raw f_j found by a single-objective GA."""
    pop = max(10, int(round(100 * budget_scale)))
    gens = max(2, int(round(100 * budget_scale)))
    city_obj = build_city(city, variant, groups)
    ctx = build_context(city_obj)
    base_env = make_env(city_obj, nr_stations, chained)
    placeholder = Norm(f_ref=np.ones(3))
    t0 = time.time()
    table: dict[str, dict[str, float]] = {}
    f_ref = np.zeros(3)
    for j, key in enumerate(OBJECTIVE_KEYS):
        table[key] = {}
        for seed in seeds:
            env = ArchiveWrapper(base_env)
            p = evo.Problem(env, ctx, placeholder, nr_stations)
            rng = np.random.default_rng([int(seed), j])
            evo.ga_single_objective(p, rng, j, pop_size=pop, gens=gens)
            table[key][str(seed)] = float(p.best_raw[j])
            log(f"[anchor {key} seed {seed}] max={p.best_raw[j]:.6g} evals={p.n_evals}")
        f_ref[j] = max(table[key].values())
    norm = Norm(f_ref=f_ref, z=1.05, rho=0.05)
    save_norm(norm, out)
    # load_norm ignores extra keys, so provenance can be stored next to the anchors
    import json

    payload = json.loads(Path(out).read_text())
    payload["provenance"] = {
        "city": city, "variant": variant, "groups": groups, "nr_stations": nr_stations,
        "chained": bool(chained), "seeds": [int(s) for s in seeds], "budget_scale": budget_scale,
        "pop": pop, "gens": gens, "per_seed_max": table, "nadir": [0.0, 0.0, 0.0],
        "wall_clock_s": round(time.time() - t0, 2), "git_commit": git_commit(),
    }
    Path(out).write_text(json.dumps(payload, indent=2))
    return norm


def write_record(path: str | Path, record: dict[str, Any]) -> None:
    save_record(path, record)
