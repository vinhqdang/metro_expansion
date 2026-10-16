"""Evolutionary, random and greedy baselines on one shared genome (design sections 3.3-3.4).

Genome x = (start, a_1, ..., a_{T-1}): `start` is a grid index and a_t in 0..7 is the action
of step t. decode() walks the genome in the environment and repairs an infeasible action to
the first allowed one in the order x+1, x-1, x+2, x-2, x+3, x-3, x+4 (mod 8). The repaired
action is written back into the genome (Lamarckian). Every evaluation is one terminal rollout
through the ArchiveWrapper, so the archive holds every evaluated line.

Objectives are the normalised (D_tot, GGI_2, A_new), maximised. All comparisons use the
normalisation file and the augmented Tchebycheff g(f|w) from src.eval.normalization.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from src.eval.archive import ArchiveWrapper, raw_objectives, upstream_reward
from src.eval.normalization import Norm
from src.eval.objectives import CityContext, ggi

ACTION_TO_DIRECTION = np.array(
    [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
)
REPAIR_OFFSETS = (1, -1, 2, -2, 3, -3, 4)
N_ACTIONS = 8


@dataclass
class Individual:
    start: int
    acts: np.ndarray
    raw: np.ndarray | None = None  # (D_tot, ggi2, A_new)
    f: np.ndarray | None = None  # normalised, clipped to [0, 1.2]
    rec: object | None = None  # archive Record of the repaired walk
    meta: dict = field(default_factory=dict)


class Problem:
    """Decode/evaluate genomes on one wrapped env. Counts evaluations and tracks the best raw value."""

    def __init__(self, env: ArchiveWrapper, ctx: CityContext, norm: Norm, nr_stations: int):
        if not isinstance(env, ArchiveWrapper):
            raise TypeError("Problem needs the ArchiveWrapper env")
        self.env = env
        self.base = env.unwrapped
        self.city = self.base.city
        self.ctx = ctx
        self.norm = norm
        self.T = int(nr_stations)
        self.n_acts = self.T - 1
        self.gx = int(self.city.grid_x_size)
        self.gy = int(self.city.grid_y_size)
        self.N = self.gx * self.gy
        self.best_raw = np.full(3, -np.inf)
        self.n_evals = 0

    def coords(self, start: int) -> tuple[int, int]:
        return int(start) // self.gy, int(start) % self.gy

    def decode(self, ind: Individual):
        """Walk the genome, repair infeasible actions in place, return the archive record of the walk."""
        env = self.env
        before = len(env.archive)
        _, info = env.reset(options={"loc": self.coords(ind.start)})
        for t in range(self.n_acts):
            mask = info["action_mask"]
            a = int(ind.acts[t])
            if not mask[a]:
                a = next(c for c in ((a + k) % N_ACTIONS for k in REPAIR_OFFSETS) if mask[c])
            ind.acts[t] = a
            _, _, terminated, _, info = env.step(a)
            if terminated:
                break
        if len(env.archive) != before + 1:
            raise RuntimeError("decode did not reach a terminal state")
        return env.last_record

    def evaluate(self, ind: Individual) -> Individual:
        rec = self.decode(ind)
        ind.raw = raw_objectives(self.ctx, rec.cells, rec.segments, rec.s)
        ind.f = self.norm.normalize(ind.raw)
        ind.rec = rec
        self.best_raw = np.maximum(self.best_raw, ind.raw)
        self.n_evals += 1
        return ind


# ----------------------------------------------------------------------------- operators


def random_individual(p: Problem, rng: np.random.Generator) -> Individual:
    return Individual(start=int(rng.integers(p.N)), acts=rng.integers(0, N_ACTIONS, size=p.n_acts))


def crossover(p1: Individual, p2: Individual, rng: np.random.Generator, n_acts: int) -> Individual:
    """One-point crossover on the actions (cut c ~ U{1..T-2}); start from p1, or from p2 with prob 0.5."""
    c = int(rng.integers(1, n_acts))  # 1 .. n_acts-1 = T-2
    acts = np.concatenate([p1.acts[:c], p2.acts[c:]]).astype(np.int64)
    start = p2.start if rng.random() < 0.5 else p1.start
    return Individual(start=int(start), acts=acts)


def mutate(child: Individual, rng: np.random.Generator, p: Problem) -> Individual:
    """Each action is resampled with prob 1/(T-1); the start moves by U{-2..2}^2 (clipped) with prob 0.2."""
    flip = rng.random(p.n_acts) < 1.0 / p.n_acts
    if flip.any():
        child.acts[flip] = rng.integers(0, N_ACTIONS, size=int(flip.sum()))
    if rng.random() < 0.2:
        x, y = p.coords(child.start)
        dx, dy = rng.integers(-2, 3, size=2)
        x = int(np.clip(x + dx, 0, p.gx - 1))
        y = int(np.clip(y + dy, 0, p.gy - 1))
        child.start = x * p.gy + y
    return child


def _init_pop(p: Problem, rng: np.random.Generator, size: int) -> list[Individual]:
    return [p.evaluate(random_individual(p, rng)) for _ in range(size)]


# ----------------------------------------------------------------------------- NSGA-II


def fast_nondominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """Deb's fast non-dominated sort (maximisation). Returns fronts as index arrays, best first."""
    P = np.asarray(F, dtype=float)
    n = P.shape[0]
    if n == 0:
        return []
    ge = np.all(P[:, None, :] >= P[None, :, :], axis=-1)
    gt = np.any(P[:, None, :] > P[None, :, :], axis=-1)
    dom = ge & gt  # dom[i, j]: i dominates j
    count = dom.sum(axis=0)  # number of (unassigned) dominators of each point
    assigned = np.zeros(n, dtype=bool)
    fronts: list[np.ndarray] = []
    while not assigned.all():
        front = np.flatnonzero((count == 0) & ~assigned)
        fronts.append(front)
        assigned[front] = True
        count = count - dom[front].sum(axis=0)
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Crowding distance of the points of one front (boundary points get inf)."""
    P = np.asarray(F, dtype=float)
    k, m = P.shape
    d = np.zeros(k)
    if k <= 2:
        d[:] = np.inf
        return d
    for j in range(m):
        order = np.argsort(P[:, j], kind="stable")
        d[order[0]] = np.inf
        d[order[-1]] = np.inf
        span = P[order[-1], j] - P[order[0], j]
        if span == 0:
            continue
        d[order[1:-1]] += (P[order[2:], j] - P[order[:-2], j]) / span
    return d


def rank_and_crowding(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = F.shape[0]
    rank = np.zeros(n, dtype=int)
    crowd = np.zeros(n)
    for r, front in enumerate(fast_nondominated_sort(F)):
        rank[front] = r
        crowd[front] = crowding_distance(F[front])
    return rank, crowd


def select_nsga2(F: np.ndarray, n: int) -> np.ndarray:
    """(mu + lambda) survival: whole fronts, the last one by descending crowding distance."""
    chosen: list[int] = []
    for front in fast_nondominated_sort(F):
        if len(chosen) + front.size <= n:
            chosen.extend(front.tolist())
        else:
            d = crowding_distance(F[front])
            order = np.argsort(-d, kind="stable")
            chosen.extend(front[order[: n - len(chosen)]].tolist())
            break
    return np.asarray(chosen, dtype=int)


def _tournament_nsga(rank: np.ndarray, crowd: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    cand = rng.integers(0, rank.shape[0], size=(n, 2))
    a, b = cand[:, 0], cand[:, 1]
    a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] >= crowd[b]))
    return np.where(a_wins, a, b)


def nsga2(
    p: Problem,
    rng: np.random.Generator,
    pop_size: int = 100,
    gens: int = 100,
) -> list[Individual]:
    """NSGA-II on (f1, f2, f3) normalised. Evaluations: pop_size + gens * pop_size."""
    pop = _init_pop(p, rng, pop_size)
    for _ in range(gens):
        F = np.stack([ind.f for ind in pop])
        rank, crowd = rank_and_crowding(F)
        picks = _tournament_nsga(rank, crowd, rng, 2 * pop_size)
        children = []
        for k in range(pop_size):
            child = crossover(pop[picks[2 * k]], pop[picks[2 * k + 1]], rng, p.n_acts)
            children.append(p.evaluate(mutate(child, rng, p)))
        union = pop + children
        keep = select_nsga2(np.stack([ind.f for ind in union]), pop_size)
        pop = [union[i] for i in keep]
    return pop


# ----------------------------------------------------------------------------- single-objective GA


def _binary_tournament(score: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    cand = rng.integers(0, score.shape[0], size=(n, 2))
    a, b = cand[:, 0], cand[:, 1]
    return np.where(score[a] >= score[b], a, b)


def generational_ga(
    p: Problem,
    rng: np.random.Generator,
    fitness,
    pop_size: int = 100,
    gens: int = 100,
    elites: int = 2,
) -> list[Individual]:
    """Generational GA, maximising fitness(ind). Each generation evaluates pop_size children.

    The next population is the `elites` best parents plus the best pop_size - elites children.
    Evaluations: pop_size + gens * pop_size.
    """
    pop = _init_pop(p, rng, pop_size)
    fit = np.array([fitness(ind) for ind in pop], dtype=float)
    for _ in range(gens):
        picks1 = _binary_tournament(fit, rng, pop_size)
        picks2 = _binary_tournament(fit, rng, pop_size)
        children = []
        for k in range(pop_size):
            child = crossover(pop[picks1[k]], pop[picks2[k]], rng, p.n_acts)
            children.append(p.evaluate(mutate(child, rng, p)))
        cfit = np.array([fitness(ind) for ind in children], dtype=float)
        best_children = np.argsort(-cfit, kind="stable")[: pop_size - elites]
        elite_idx = np.argsort(-fit, kind="stable")[:elites]
        pop = [pop[i] for i in elite_idx] + [children[i] for i in best_children]
        fit = np.concatenate([fit[elite_idx], cfit[best_children]])
    return pop


def ga_scalar(
    p: Problem,
    rng: np.random.Generator,
    norm: Norm,
    w,
    pop_size: int = 100,
    gens: int = 100,
    elites: int = 2,
) -> list[Individual]:
    """Single-weight GA on fitness -g(f|w). Returns the final population."""
    w_arr = np.asarray(w, dtype=float)

    def fitness(ind: Individual) -> float:
        return -float(norm.g(ind.f, w_arr))

    return generational_ga(p, rng, fitness, pop_size=pop_size, gens=gens, elites=elites)


def ga_single_objective(
    p: Problem,
    rng: np.random.Generator,
    j: int,
    pop_size: int = 100,
    gens: int = 100,
) -> list[Individual]:
    """Anchor GA: maximise the raw objective j (used by compute_normalization)."""
    return generational_ga(p, rng, lambda ind: float(ind.raw[j]), pop_size=pop_size, gens=gens)


# ----------------------------------------------------------------------------- random


def random_lines(p: Problem, rng: np.random.Generator, n: int) -> list[Individual]:
    """n random walks from uniformly random start cells, each to its terminal state."""
    out: list[Individual] = []
    env = p.env
    for _ in range(n):
        start = int(rng.integers(p.N))
        _, info = env.reset(options={"loc": p.coords(start)})
        acts = []
        while True:
            a = int(rng.choice(np.flatnonzero(info["action_mask"])))
            acts.append(a)
            _, _, terminated, _, info = env.step(a)
            if terminated:
                break
        rec = env.last_record
        raw = raw_objectives(p.ctx, rec.cells, rec.segments, rec.s)
        p.n_evals += 1
        p.best_raw = np.maximum(p.best_raw, raw)
        out.append(Individual(start=start, acts=np.asarray(acts, dtype=np.int64), raw=raw,
                              f=p.norm.normalize(raw), rec=rec))
    return out


# ----------------------------------------------------------------------------- greedy-by-g


class _Geometry:
    """Precomputed grid quantities that let greedy evaluate candidate lines exactly like a_new/d_tot."""

    def __init__(self, city, ctx: CityContext, K: int = 2):
        gx, gy = int(city.grid_x_size), int(city.grid_y_size)
        xs = np.repeat(np.arange(gx), gy)
        ys = np.tile(np.arange(gy), gx)
        self.N = gx * gy
        # block[c, j] is True when cell j is within Chebyshev K of cell c
        self.block = (np.abs(xs[:, None] - xs[None, :]) <= K) & (np.abs(ys[:, None] - ys[None, :]) <= K)
        existing = np.flatnonzero(np.asarray(ctx.existing_mask, dtype=bool).reshape(-1))
        E = self.block[existing].any(axis=0) if existing.size else np.zeros(self.N, dtype=bool)
        self.mu_ne = np.where(E, 0.0, np.asarray(ctx.mu, dtype=float).reshape(-1))
        self.denom = float(self.mu_ne.sum())
        self.od = np.asarray(ctx.od_raw, dtype=float)
        self.total_od = float(ctx.total_od)

    def a_new_from_reach(self, reach: np.ndarray) -> float:
        if self.denom == 0:
            return 0.0
        return float(self.mu_ne[reach].sum() / self.denom)


@dataclass
class _Walk:
    loc: np.ndarray
    cells: list
    segs: list
    dset: frozenset
    conns: frozenset
    s: np.ndarray
    served: float
    reach: np.ndarray
    ret_me: float
    ret_gg: float
    stations: int
    actions: list
    raw: np.ndarray


def _init_walk(geo: _Geometry, city, nr_groups: int, start_xy) -> _Walk:
    loc = np.asarray(start_xy, dtype=np.int64)
    idx = int(city.grid_to_index(loc[None, :])[0])
    raw = np.array([0.0, 0.0, geo.a_new_from_reach(geo.block[idx])])
    return _Walk(
        loc=loc, cells=[loc], segs=[], dset=frozenset(), conns=frozenset(),
        s=np.zeros(nr_groups), served=0.0, reach=geo.block[idx].copy(),
        ret_me=0.0, ret_gg=0.0, stations=1, actions=[], raw=raw,
    )


def _candidate(geo, base, city, walk: _Walk, a: int, w: np.ndarray, norm: Norm):
    """Evaluate one feasible move with the env's own reward on a shallow copy (state is not mutated)."""
    new_loc = walk.loc + ACTION_TO_DIRECTION[a]
    from_idx = int(city.grid_to_index(walk.loc[None, :])[0])
    to_idx = int(city.grid_to_index(new_loc[None, :])[0])
    sh = copy.copy(base)
    sh.covered_segments = walk.segs
    sh.covered_cells_coordinates = walk.cells
    sh.connections_with_existing_lines = set(walk.conns)
    reward, pairs = sh._calculate_reward([from_idx, to_idx])
    reward = np.asarray(reward, dtype=float)
    pairs = np.asarray(pairs).reshape(-1, 2)
    new_dir = set()
    for i, j in pairs:
        new_dir.add((int(i), int(j)))
        new_dir.add((int(j), int(i)))
    fresh = new_dir - walk.dset
    served = walk.served + float(sum(geo.od[i, j] for i, j in fresh))
    s_new = walk.s + reward
    reach = walk.reach | geo.block[to_idx]
    raw = np.array([
        served / geo.total_od if geo.total_od > 0 else 0.0,
        ggi(s_new, base=2.0),
        geo.a_new_from_reach(reach),
    ])
    g = float(norm.g(norm.normalize(raw), w))
    return {
        "a": a, "g": g, "loc": new_loc, "reward": reward, "pairs": pairs, "new_dir": new_dir,
        "conns": frozenset(sh.connections_with_existing_lines), "served": served, "s": s_new,
        "reach": reach, "raw": raw,
        "ret_me": walk.ret_me + upstream_reward(reward, "max_efficiency"),
        "ret_gg": walk.ret_gg + upstream_reward(reward, "ggi2"),
    }


def _apply(walk: _Walk, c: dict) -> _Walk:
    segs = list(walk.segs)
    for i, j in c["pairs"]:
        segs.append([int(i), int(j)])
        segs.append([int(j), int(i)])
    return _Walk(
        loc=c["loc"], cells=walk.cells + [c["loc"]], segs=segs,
        dset=walk.dset | frozenset(c["new_dir"]), conns=c["conns"], s=c["s"], served=c["served"],
        reach=c["reach"], ret_me=c["ret_me"], ret_gg=c["ret_gg"], stations=walk.stations + 1,
        actions=walk.actions + [c["a"]], raw=c["raw"],
    )


@dataclass
class GreedyResult:
    best_walks: list  # one _Walk per weight (lowest g_w over all starts)
    best_starts: list  # start index per weight
    n_walks: int
    n_candidates: int


def greedy_by_g(
    p: Problem,
    weights: np.ndarray,
    norm: Norm,
    starts: np.ndarray | None = None,
) -> GreedyResult:
    """For each start and weight, grow the line one step at a time by the feasible move with the best -g.

    Candidate gains come from the env's own _calculate_reward on a shallow copy, so the env is
    never mutated. Every finished walk is added to the archive (state-based record, identical to
    the env's, checked by the caller on the chosen walks). Cost: N starts x len(weights) walks
    x (T-1) steps x <= 8 candidates.
    """
    base, city, ctx = p.base, p.city, p.ctx
    if starts is None:
        starts = np.arange(p.N)
    geo = _Geometry(city, ctx)
    nr_groups = int(city.group_od_mx.shape[0])
    W = np.asarray(weights, dtype=float)
    best_g = np.full(W.shape[0], np.inf)
    best_walks: list = [None] * W.shape[0]
    best_starts = [-1] * W.shape[0]
    n_walks = 0
    n_cand = 0
    for start in starts:
        start_xy = p.coords(int(start))
        for k in range(W.shape[0]):
            walk = _init_walk(geo, city, nr_groups, start_xy)
            while walk.stations < p.T:
                mask = base.mask_actions(walk.loc, walk.loc + ACTION_TO_DIRECTION, np.asarray(walk.cells))
                allowed = np.flatnonzero(mask)
                if allowed.size == 0:
                    break
                best = None
                for a in allowed:
                    c = _candidate(geo, base, city, walk, int(a), W[k], norm)
                    n_cand += 1
                    if best is None or c["g"] > best["g"]:
                        best = c
                walk = _apply(walk, best)
            n_walks += 1
            p.env.archive.add(walk.cells, walk.s, [tuple(x) for x in walk.segs], walk.ret_me, walk.ret_gg)
            g = float(norm.g(norm.normalize(walk.raw), W[k]))
            if g < best_g[k]:
                best_g[k] = g
                best_walks[k] = walk
                best_starts[k] = int(start)
    return GreedyResult(best_walks=best_walks, best_starts=best_starts, n_walks=n_walks, n_candidates=n_cand)


def replay_walk(base, start_xy, actions) -> dict:
    """Replay a walk on the unwrapped env (no archive). Returns the same fields as an archive record."""
    _, info = base.reset(options={"loc": (int(start_xy[0]), int(start_xy[1]))})
    s = np.zeros(int(base.city.group_od_mx.shape[0]))
    for a in actions:
        _, reward, terminated, _, info = base.step(int(a))
        s += np.asarray(reward, dtype=float)
        if terminated:
            break
    return {"cells": info["covered_cells_coordinates"], "segments": info["segments"], "s": s}
