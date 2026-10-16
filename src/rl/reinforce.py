"""Batched, weight-conditioned REINFORCE with a leave-one-out baseline (design section 1.5).

One update:
  1. the GNN encoder runs once on the static graph (h is shared by all episodes of the update);
  2. M = batch_w weight vectors w ~ Dirichlet(1,1,1) are drawn, each rolled out K = batch_k times;
  3. every episode samples its start cell from StartHead and its actions from the policy;
  4. terminal return R = -g(f_norm(final) | w), with g the augmented Tchebycheff of src.eval.normalization;
  5. advantage A = R - mean of the other K-1 returns that share the same w (RLOO);
  6. loss = -(1/B) sum_i A_i sum_t log pi - beta * mean entropy (start step included), Adam, grad clip 1.0.

Evaluation is greedy on the 15-vector simplex lattice (H = 4). Reported metrics always use the
evaluation normalisation file, even for the 'unnormalised' ablation (which changes only the training reward).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from src.envs.context import build_context
from src.envs.fast_motndp import build_city, make_env
from src.eval.normalization import Norm
from src.eval.objectives import compute_objectives, ggi
from src.eval.record import front_indicators, make_point
from src.eval.weights import lattice
from src.gnn.encoder import FlatEncoder, HeteroGNNEncoder
from src.rl.features import ACTION_TO_DIRECTION, build_feature_context, calibrate_gain_scale, candidate_features
from src.rl.graph import build_graph
from src.rl.policy_v2 import AttentivePolicyHeadV2, StartHead

ABLATIONS = (
    "full",
    "single_objective",
    "flat_encoder",
    "no_lookahead",
    "no_start_head",
    "gini_reward",
    "unnormalised",
)
TRANSFER_MODES = ("scratch", "full", "encoder_only", "heads_only", "full_frozen_enc", "random_frozen_enc")
N_FIXED_STARTS = 20


@dataclass
class RunConfig:
    city: str = "xian"
    variant: str = "full"
    groups: str = "price_groups_5.txt"
    chained: bool = True
    nr_stations: int = 20
    episodes: int = 10_000
    batch_w: int = 4
    batch_k: int = 4
    seed: int = 0
    ablation: str = "full"
    hidden_dim: int = 32
    num_layers: int = 2
    lr_start: float = 1e-3
    lr_end: float = 1e-4
    grad_clip: float = 1.0
    entropy0: float = 0.01
    entropy_frac: float = 0.7
    eval_every: int = 50
    best_of: int = 32
    od_topk: int | None = None
    transfer: str = "scratch"
    init_from: str | None = None
    save_ckpt: str | None = None
    calib_episodes: int = 200
    prior_lam: float = 1.0  # weight of the fixed w-scalarised move prior (0 = legacy v2 head)
    prior_lam_s: float = 1.0  # weight of the fixed w-scalarised start prior (0 = legacy StartHead)
    verbose: bool = True


@dataclass
class Setup:
    cfg: RunConfig
    city: Any
    ctx: Any
    env: Any
    menv: Any
    graph: tuple
    fctx: Any
    agent: "Agent"
    eval_norm: Norm
    train_norm: Norm
    use_features: bool
    use_start_head: bool
    fixed_starts: list = field(default_factory=list)
    n_groups: int = 0


class Agent(nn.Module):
    def __init__(self, encoder: nn.Module, weight_embed: nn.Module, policy: nn.Module, start_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.weight_embed = weight_embed
        self.policy = policy
        self.start_head = start_head


@dataclass
class Episode:
    start: tuple
    cells: list
    segments: list
    s: np.ndarray
    logp: torch.Tensor | None
    ents: list


# ----------------------------------------------------------------------------- helpers

def gini_legacy(x: np.ndarray) -> float:
    """Gini as used by the original train_xian.py: 0 when nothing is served (the degenerate behaviour)."""
    x = np.sort(np.abs(np.asarray(x, dtype=float)))
    n = len(x)
    total = x.sum()
    if total == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def rloo(returns: np.ndarray) -> np.ndarray:
    """Leave-one-out advantage within one group of K returns that share a weight vector."""
    r = np.asarray(returns, dtype=float)
    K = r.size
    if K < 2:
        raise ValueError("RLOO needs at least 2 rollouts per weight vector")
    return r - (r.sum() - r) / (K - 1)


def _select(logits: torch.Tensor, mode: str, gen: torch.Generator | None) -> int:
    """Greedy argmax, or inverse-CDF sampling with the given generator. Never picks a masked (-inf) entry."""
    if mode == "greedy":
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits.detach().double(), dim=-1)
    cdf = torch.cumsum(probs, dim=0)
    u = torch.rand(1, generator=gen, dtype=torch.float64).item() * cdf[-1].item()
    idx = int(torch.searchsorted(cdf, torch.tensor([u], dtype=torch.float64)).item())
    idx = min(idx, probs.numel() - 1)
    if probs[idx].item() == 0.0:
        idx = int(torch.nonzero(probs > 0).flatten()[-1].item())
    return idx


def _gridxy(idx: int, gy: int) -> tuple[int, int]:
    return idx // gy, idx % gy


def _node_index(cells_xy: np.ndarray, gy: int) -> np.ndarray:
    return cells_xy[:, 0] * gy + cells_xy[:, 1]


def _standardise(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    v = x[valid]
    if v.size > 1 and v.std() > 1e-9:
        out[valid] = (v - v.mean()) / v.std()
    return out


def move_prior(feats: np.ndarray, s_cum: np.ndarray, valid: np.ndarray, m: int, f_ref1: float) -> np.ndarray:
    """[8, 3] standardised one-step gains: D (log total OD gain), GGI increment, A (newly reached mass)."""
    pct = feats[:, 1:1 + m] * f_ref1
    g0 = ggi(s_cum)
    dg = np.array([ggi(s_cum + pct[k]) - g0 if valid[k] else 0.0 for k in range(len(valid))])
    return np.stack([_standardise(feats[:, 0], valid), _standardise(dg, valid),
                     _standardise(feats[:, 1 + m], valid)], axis=1)


def start_prior(city, fctx) -> np.ndarray:
    """[N, 3] standardised static potentials: log OD degree, GGI of per-group OD degree, uncovered mass within K."""
    od = np.asarray(city.od_mx, dtype=float)
    deg = np.log1p(od.sum(0) + od.sum(1))
    god = np.asarray(city.group_od_mx, dtype=float)
    gs = god.sum(axis=(1, 2))
    gs[gs == 0] = 1.0
    gdeg = (god.sum(2) + god.sum(1)) / gs[:, None]
    bal = np.array([ggi(gdeg[:, i]) for i in range(gdeg.shape[1])])
    M = fctx.mu * (~fctx.existing_reach)
    K = fctx.K
    Mp = np.pad(M, K)
    X, Y = M.shape
    acc = sum(Mp[K + dx:K + dx + X, K + dy:K + dy + Y] for dx in range(-K, K + 1) for dy in range(-K, K + 1))
    P = np.stack([deg, bal, acc.reshape(-1)], axis=1)
    return (P - P.mean(0)) / (P.std(0) + 1e-9)


# ----------------------------------------------------------------------------- episode

def run_episode(S: Setup, h: torch.Tensor, w: np.ndarray, mode: str, gen: torch.Generator | None,
                start: tuple | None = None, grad: bool = True) -> Episode:
    """One episode. start=None means: sampled from StartHead (or env-random for no_start_head)."""
    agent, menv, env, fctx = S.agent, S.menv, S.env, S.fctx
    gy = S.city.grid_y_size
    m = S.n_groups
    w_t = torch.as_tensor(np.asarray(w, dtype=np.float32))
    with torch.set_grad_enabled(grad):
        w_emb = agent.weight_embed(w_t)
        logps: list[torch.Tensor] = []
        ents: list[torch.Tensor] = []
        if start is None and S.use_start_head:
            s_logits, s_logp, s_ent = agent.start_head(h, w_emb, w_t)
            s_idx = _select(s_logits, mode, gen)
            logps.append(s_logp[s_idx])
            ents.append(s_ent)
            start = _gridxy(s_idx, gy)
            obs, info = env.reset(options={"loc": (int(start[0]), int(start[1]))})
        elif start is not None:
            obs, info = env.reset(options={"loc": (int(start[0]), int(start[1]))})
        else:  # env-random start (no_start_head ablation); env RNG seeded once at setup
            obs, info = env.reset()
        start_cell = tuple(int(v) for v in info["location_grid_coordinates"])
        cells = [start_cell]
        sel = [int(info["location_grid_index"])]
        s_cum = np.zeros(m)
        done = False
        while not done:
            loc = info["location_grid_coordinates"]
            nb = loc + ACTION_TO_DIRECTION
            valid = np.asarray(info["action_mask"]).astype(bool)
            cand_idx = np.where(valid, nb[:, 0] * gy + nb[:, 1], 0)
            cand_h = h[torch.as_tensor(cand_idx, dtype=torch.long)]
            pooled = h[torch.as_tensor(sel, dtype=torch.long)].mean(dim=0)
            feat_t = None
            prior_t = None
            if S.use_features:
                feats = candidate_features(menv, fctx, nb, cand_idx, valid, int(info["location_grid_index"]),
                                           cells, menv.stations_placed)
                feat_t = torch.as_tensor(feats, dtype=torch.float32)
                if S.cfg.prior_lam > 0.0:
                    prior_t = torch.as_tensor(move_prior(feats, s_cum, valid, m, fctx.f_ref1), dtype=torch.float32)
            mask_t = torch.as_tensor(valid)
            sbar = torch.as_tensor(s_cum, dtype=torch.float32)
            logits, logp_all, ent = agent.policy(cand_h, feat_t, pooled, w_emb, sbar, mask_t, prior_t, w_t)
            a = _select(logits, mode, gen)
            logps.append(logp_all[a])
            ents.append(ent)
            obs, reward, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            s_cum = s_cum + np.asarray(reward, dtype=float)
            cells.append(tuple(int(v) for v in info["location_grid_coordinates"]))
            sel.append(int(info["location_grid_index"]))
        logp = torch.stack(logps).sum() if (grad and logps) else None
    segments = [list(map(int, p)) for p in info["segments"]]
    return Episode(start=start_cell, cells=cells, segments=segments, s=s_cum, logp=logp, ents=ents)


def _train_objectives(S: Setup, ep: Episode) -> tuple[dict, np.ndarray]:
    obj = compute_objectives(S.ctx, ep.cells, ep.segments, ep.s)
    if S.cfg.ablation == "gini_reward":
        obj = dict(obj)
        obj["ggi2"] = 1.0 - gini_legacy(ep.s)
    f_train = S.train_norm.normalize(obj)
    return obj, f_train


# ----------------------------------------------------------------------------- setup

def build_agent(cfg: RunConfig, in_dim: int, n_groups: int, use_features: bool, flat: bool) -> Agent:
    d = cfg.hidden_dim
    if flat:
        encoder = FlatEncoder(in_dim=in_dim, hidden_dim=d)
    else:
        encoder = HeteroGNNEncoder(in_dim=in_dim, hidden_dim=d, num_layers=cfg.num_layers)
    weight_embed = nn.Linear(3, d, bias=False)
    policy = AttentivePolicyHeadV2(d, n_groups, 4 + n_groups, use_features=use_features, prior_lam=cfg.prior_lam)
    start_head = StartHead(d, prior_lam=cfg.prior_lam_s)
    return Agent(encoder, weight_embed, policy, start_head)


def _check_meta(meta: dict, cfg: RunConfig, n_groups: int, use_features: bool, flat: bool) -> None:
    expect = {"hidden_dim": cfg.hidden_dim, "num_layers": cfg.num_layers, "n_groups": n_groups,
              "use_features": use_features, "flat": flat}
    for k, v in expect.items():
        if meta.get(k) != v:
            raise ValueError(f"checkpoint {k}={meta.get(k)!r} incompatible with run value {v!r}")


def apply_transfer(agent: Agent, cfg: RunConfig, n_groups: int, use_features: bool, flat: bool) -> dict | None:
    """Load / freeze according to cfg.transfer. Returns the transfer record for the run JSON (or None)."""
    mode = cfg.transfer
    if mode not in TRANSFER_MODES:
        raise ValueError(f"unknown transfer mode {mode!r}")
    if mode == "scratch":
        return None
    if mode in ("random_frozen_enc",):
        # control: keep the random encoder initialisation, freeze it, train the heads
        for p in agent.encoder.parameters():
            p.requires_grad_(False)
        return {"condition": mode, "source_city": None, "source_seed": None, "budget": None}
    if not cfg.init_from:
        raise ValueError(f"--transfer {mode} needs --init_from")
    ckpt = torch.load(cfg.init_from, map_location="cpu", weights_only=True)
    _check_meta(ckpt["meta"], cfg, n_groups, use_features, flat)
    load_enc = mode in ("full", "encoder_only", "full_frozen_enc")
    load_heads = mode in ("full", "heads_only", "full_frozen_enc")
    if load_enc:
        agent.encoder.load_state_dict(ckpt["encoder"], strict=True)
    if load_heads:
        agent.weight_embed.load_state_dict(ckpt["weight_embed"], strict=True)
        agent.policy.load_state_dict(ckpt["policy"], strict=True)
        agent.start_head.load_state_dict(ckpt["start_head"], strict=True)
    if mode == "full_frozen_enc":
        for p in agent.encoder.parameters():
            p.requires_grad_(False)
    meta = ckpt["meta"]
    return {"condition": mode, "source_city": meta.get("city"), "source_seed": meta.get("seed"), "budget": None}


def setup(cfg: RunConfig, eval_norm: Norm) -> Setup:
    city = build_city(cfg.city, cfg.variant, cfg.groups)
    ctx = build_context(city)
    env = make_env(city, nr_stations=cfg.nr_stations, chained=cfg.chained)
    menv = env.unwrapped
    if cfg.od_topk is None and cfg.city == "amsterdam":
        cfg.od_topk = 64  # full Amsterdam OD costs 6.3 s per encoder forward+backward; top-64 per node costs 0.08 s
    graph = build_graph(city, od_topk=cfg.od_topk)
    fctx = build_feature_context(city, ctx, cfg.nr_stations, f_ref1=float(eval_norm.f_ref[0]))

    gy, N = city.grid_y_size, city.grid_size
    all_cells = np.column_stack(np.unravel_index(np.arange(N), (city.grid_x_size, city.grid_y_size)))
    fctx.gain_scale = calibrate_gain_scale(menv, fctx, all_cells, np.random.default_rng(cfg.seed + 1),
                                           n_episodes=cfg.calib_episodes)
    # reset env RNG explicitly so env-random starts (no_start_head) depend only on the seed
    env.reset(seed=cfg.seed)

    # 20 highest OD-degree cells (used for the no_start_head evaluation)
    deg = city.od_mx.sum(axis=1) + city.od_mx.sum(axis=0)
    order = np.argsort(-deg, kind="stable")[:N_FIXED_STARTS]
    fixed_starts = [_gridxy(int(i), gy) for i in order]

    use_features = cfg.ablation != "no_lookahead"
    use_start_head = cfg.ablation != "no_start_head"
    flat = cfg.ablation == "flat_encoder"
    n_groups = int(city.group_od_mx.shape[0])
    in_dim = int(graph[0].shape[1])

    train_norm = eval_norm
    if cfg.ablation == "unnormalised":
        train_norm = Norm(f_ref=np.ones(3), z=eval_norm.z, rho=eval_norm.rho)

    torch.manual_seed(cfg.seed)
    agent = build_agent(cfg, in_dim, n_groups, use_features, flat)
    transfer_rec = apply_transfer(agent, cfg, n_groups, use_features, flat)
    if cfg.prior_lam_s > 0.0:
        agent.start_head.psi = torch.as_tensor(start_prior(city, fctx), dtype=torch.float32)

    S = Setup(cfg=cfg, city=city, ctx=ctx, env=env, menv=menv, graph=graph, fctx=fctx, agent=agent,
              eval_norm=eval_norm, train_norm=train_norm, use_features=use_features,
              use_start_head=use_start_head, fixed_starts=fixed_starts, n_groups=n_groups)
    S.transfer_rec = transfer_rec
    S.flat = flat
    return S


# ----------------------------------------------------------------------------- evaluation

def _encode(S: Setup) -> torch.Tensor:
    with torch.no_grad():
        return S.agent.encoder(*S.graph)


def evaluate_lattice(S: Setup, W: np.ndarray, h: torch.Tensor, mode: str = "greedy",
                     gen: torch.Generator | None = None) -> tuple[list[dict], list[np.ndarray], int]:
    """Greedy evaluation on every w. Returns (points, normalised front rows, n_episodes_run)."""
    points, front, n_eps = [], [], 0
    for w in W:
        if S.cfg.ablation == "no_start_head" and mode == "greedy":
            eps = []
            for st in S.fixed_starts:
                eps.append(run_episode(S, h, w, "greedy", None, start=st, grad=False))
            n_eps += len(eps)
            fs, objs = [], []
            for ep in eps:
                obj = compute_objectives(S.ctx, ep.cells, ep.segments, ep.s)
                fs.append(S.eval_norm.normalize(obj))
                objs.append(obj)
            f_mean = np.mean(np.stack(fs), axis=0)
            best_i = int(np.argmin([S.eval_norm.g(f, w) for f in fs]))
            pt = make_point(S.ctx, S.eval_norm, w, "greedy", eps[best_i].cells,
                            eps[best_i].segments, eps[best_i].s)
            pt["start_selection"] = "best_g_of_20_fixed_starts"
            pt["mean_over_fixed_starts"] = {
                "f_norm": f_mean.tolist(),
                "g_w": float(S.eval_norm.g(f_mean, w)),
                "D_tot": float(np.mean([o["D_tot"] for o in objs])),
                "ggi2": float(np.mean([o["ggi2"] for o in objs])),
                "A_new": float(np.mean([o["A_new"] for o in objs])),
                "n_starts": len(eps),
            }
            points.append(pt)
            front.append(f_mean)
            continue
        ep = run_episode(S, h, w, mode, gen, start=None, grad=False)
        n_eps += 1
        pt = make_point(S.ctx, S.eval_norm, w, "greedy" if mode == "greedy" else "best_of_32",
                        ep.cells, ep.segments, ep.s)
        points.append(pt)
        front.append(np.asarray(pt["f_norm"], dtype=float))
    return points, front, n_eps


def topk_starts(S: Setup, W: np.ndarray, h: torch.Tensor, k: int = 8) -> tuple[list[dict], list[np.ndarray], int]:
    """Deployable front with k greedy rollouts per w: greedy moves from each of the k most probable
    start cells under StartHead(.|w); keep the line with the smallest g(.|w). Costs k rollouts per w."""
    points, front = [], []
    gy = S.city.grid_y_size
    for w in W:
        with torch.no_grad():
            s_logits, _, _ = S.agent.start_head(h, S.agent.weight_embed(torch.as_tensor(w, dtype=torch.float32)),
                                                torch.as_tensor(w, dtype=torch.float32))
        top = torch.argsort(s_logits, descending=True)[:k].tolist()
        best = None
        for s_idx in top:
            ep = run_episode(S, h, w, "greedy", None, start=_gridxy(int(s_idx), gy), grad=False)
            g = S.eval_norm.g(S.eval_norm.normalize(compute_objectives(S.ctx, ep.cells, ep.segments, ep.s)), w)
            if best is None or g < best[0]:
                best = (g, ep)
        ep = best[1]
        pt = make_point(S.ctx, S.eval_norm, w, f"greedy_top{k}_starts", ep.cells, ep.segments, ep.s)
        points.append(pt)
        front.append(np.asarray(pt["f_norm"], dtype=float))
    return points, front, k * len(W)


def best_of_k(S: Setup, W: np.ndarray, h: torch.Tensor, k: int, gen: torch.Generator) -> tuple[list[dict], list[np.ndarray], int]:
    points, front = [], []
    for w in W:
        eps = [run_episode(S, h, w, "sample", gen, start=None, grad=False) for _ in range(k)]
        scores = []
        for ep in eps:
            obj = compute_objectives(S.ctx, ep.cells, ep.segments, ep.s)
            scores.append(S.eval_norm.g(S.eval_norm.normalize(obj), w))
        ep = eps[int(np.argmin(scores))]
        pt = make_point(S.ctx, S.eval_norm, w, "best_of_32", ep.cells, ep.segments, ep.s)
        points.append(pt)
        front.append(np.asarray(pt["f_norm"], dtype=float))
    return points, front, k * len(W)


# ----------------------------------------------------------------------------- training

def train(S: Setup, log=print) -> dict:
    cfg = S.cfg
    if cfg.batch_k < 2:
        raise ValueError("batch_k must be >= 2 for the leave-one-out baseline")
    M, K = cfg.batch_w, cfg.batch_k
    B = M * K
    U = cfg.episodes // B
    W_lat = lattice(4, 3)
    agent = S.agent
    trainable = [p for p in agent.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg.lr_start)
    gen = torch.Generator().manual_seed(cfg.seed)
    rng_w = np.random.default_rng(cfg.seed)
    frozen_enc = not any(p.requires_grad for p in agent.encoder.parameters())
    h_frozen = _encode(S) if frozen_enc else None

    curve, eval_total = [], 0
    arch_f: list[np.ndarray] = []
    best_by_obj: dict[str, tuple[float, list, list, np.ndarray, dict]] = {}
    history = []

    def do_eval(rollouts: int) -> None:
        nonlocal eval_total
        h = _encode(S)
        pts, front, n = evaluate_lattice(S, W_lat, h, "greedy")
        eval_total += n
        ind = front_indicators(np.stack(front), S.eval_norm, W_lat)
        curve.append({"rollouts": rollouts, "eum": ind["eum"], "hv": ind["hv"]})
        if cfg.verbose:
            log(f"  eval @ {rollouts} rollouts: eum={ind['eum']:.5f} hv={ind['hv']:.6f}")

    do_eval(0)
    t0 = time.time()
    for u in range(U):
        lr = cfg.lr_end + 0.5 * (cfg.lr_start - cfg.lr_end) * (1 + math.cos(math.pi * u / U))
        for g in opt.param_groups:
            g["lr"] = lr
        beta = cfg.entropy0 * max(0.0, 1.0 - u / (cfg.entropy_frac * U))

        if cfg.ablation == "single_objective":
            ws = np.tile(np.array([1.0, 0.0, 0.0]), (M, 1))
        else:
            ws = rng_w.dirichlet([1.0, 1.0, 1.0], size=M)

        h = h_frozen if frozen_enc else S.agent.encoder(*S.graph)
        eps: list[Episode] = []
        R = np.zeros(B)
        for mi, w in enumerate(ws):
            for ki in range(K):
                ep = run_episode(S, h, w, "sample", gen, start=None, grad=True)
                obj, f_train = _train_objectives(S, ep)
                R[len(eps)] = -S.train_norm.g(f_train, w)
                eps.append(ep)
                f_eval = S.eval_norm.normalize(obj)
                arch_f.append(f_eval)
                for key, val in (("D_tot", obj["D_tot"]), ("ggi2", obj["ggi2"]), ("A_new", obj["A_new"])):
                    if key not in best_by_obj or val > best_by_obj[key][0]:
                        best_by_obj[key] = (val, ep.cells, ep.segments, ep.s, obj)
        A = np.zeros(B)
        for mi in range(M):
            sl = slice(mi * K, (mi + 1) * K)
            A[sl] = rloo(R[sl])

        logp_sum = torch.stack([ep.logp for ep in eps])
        pg = -(torch.as_tensor(A, dtype=torch.float32) * logp_sum).sum() / B
        ent_mean = torch.stack([e for ep in eps for e in ep.ents]).mean()
        loss = pg - beta * ent_mean

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg.grad_clip)
        opt.step()

        history.append((float(R.mean()), float(ent_mean.item()), float(pg.item())))
        if (u + 1) % 10 == 0 or u == U - 1:
            recent = history[-10:]
            log(f"update {u + 1}/{U} rollouts={(u + 1) * B} mean_R={np.mean([x[0] for x in recent]):.4f} "
                f"entropy={np.mean([x[1] for x in recent]):.4f} beta={beta:.4f} lr={lr:.2e}")
        if (u + 1) % cfg.eval_every == 0 or u == U - 1:
            do_eval((u + 1) * B)

    train_s = time.time() - t0
    S.train_wall_s = train_s
    S.eval_total = eval_total
    S.best_by_obj = best_by_obj
    S.arch_f = np.stack(arch_f) if arch_f else np.zeros((0, 3))
    S.curve = curve
    S.history = history
    S.n_updates = U
    return {"train_wall_s": train_s}


def finalize(S: Setup, log=print) -> dict:
    """Final greedy front, best-of-32 front, archive indicators; saves the checkpoint if requested."""
    cfg = S.cfg
    W_lat = lattice(4, 3)
    h = _encode(S)
    pts_g, front_g, n_g = evaluate_lattice(S, W_lat, h, "greedy")
    gen = torch.Generator().manual_seed(cfg.seed + 7919)
    pts_b, front_b, n_b = best_of_k(S, W_lat, h, cfg.best_of, gen)
    S.eval_total += n_g + n_b
    pts_t, front_t, n_t = ([], [np.zeros(3)], 0)
    if S.use_start_head:
        pts_t, front_t, n_t = topk_starts(S, W_lat, h, k=8)
        S.eval_total += n_t

    ind = {
        "policy": front_indicators(np.stack(front_g), S.eval_norm, W_lat),
        "policy_best_of_32": front_indicators(np.stack(front_b), S.eval_norm, W_lat),
    }
    if pts_t:
        ind["policy_top8_starts"] = front_indicators(np.stack(front_t), S.eval_norm, W_lat)
    if S.arch_f.shape[0]:
        ind["archive"] = front_indicators(S.arch_f, S.eval_norm, W_lat)
    else:
        ind["archive"] = {"hv": 0.0, "eum": float("nan"), "n_nondominated": 0}

    archive_best = {}
    for key, (val, cells, segs, s, obj) in S.best_by_obj.items():
        archive_best[key] = make_point(S.ctx, S.eval_norm, None, "archive_best", cells, segs, s)

    if cfg.save_ckpt:
        save_checkpoint(S, cfg.save_ckpt)
        log(f"saved checkpoint to {cfg.save_ckpt}")

    nd = S.arch_f[front_nondom_mask(S.arch_f)] if S.arch_f.shape[0] else S.arch_f
    return {
        "policy_front": pts_g + pts_b + pts_t,
        "archive_nd": nd.tolist(),
        "archive_best": archive_best,
        "curve": S.curve,
        "indicators": ind,
        "budget_rollouts": S.n_updates * (cfg.batch_w * cfg.batch_k),
        "eval_rollouts": S.eval_total,
        "train_wall_s": S.train_wall_s,
        "n_updates": S.n_updates,
        "calibrated_gain_scale": S.fctx.gain_scale,
        "transfer": S.transfer_rec,
    }


def front_nondom_mask(F: np.ndarray) -> np.ndarray:
    from src.eval.pareto import nondominated

    return nondominated(F)


def save_checkpoint(S: Setup, path: str) -> None:
    from pathlib import Path

    meta = {
        "city": S.cfg.city, "variant": S.cfg.variant, "seed": S.cfg.seed, "ablation": S.cfg.ablation,
        "hidden_dim": S.cfg.hidden_dim, "num_layers": S.cfg.num_layers, "n_groups": S.n_groups,
        "use_features": S.use_features, "flat": S.flat, "episodes": S.cfg.episodes,
        "prior_lam": S.cfg.prior_lam, "prior_lam_s": S.cfg.prior_lam_s,
    }
    # every parameter of every module is stored, including encoder.input_proj
    payload = {
        "encoder": S.agent.encoder.state_dict(),
        "weight_embed": S.agent.weight_embed.state_dict(),
        "policy": S.agent.policy.state_dict(),
        "start_head": S.agent.start_head.state_dict(),
        "meta": meta,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def configure_torch(threads: int = 1) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def run(cfg: RunConfig, eval_norm: Norm, log=print) -> dict:
    """Train one run and return the method-specific fields used by scripts/run_rl.py."""
    configure_torch(1)
    if cfg.verbose:
        log(f"setup city={cfg.city}/{cfg.variant} ablation={cfg.ablation} transfer={cfg.transfer} seed={cfg.seed}")
    S = setup(cfg, eval_norm)
    if cfg.verbose:
        log(f"gain_scale={S.fctx.gain_scale:.6g} od_topk={cfg.od_topk} groups={S.n_groups} "
            f"params={sum(p.numel() for p in S.agent.parameters())}")
    train(S, log=log)
    out = finalize(S, log=log)
    out["config"] = {
        "lr": cfg.lr_start, "lr_end": cfg.lr_end, "batch_w": cfg.batch_w, "batch_k": cfg.batch_k,
        "entropy0": cfg.entropy0, "entropy_frac": cfg.entropy_frac, "grad_clip": cfg.grad_clip,
        "hidden_dim": cfg.hidden_dim, "num_layers": cfg.num_layers, "eval_every": cfg.eval_every,
        "best_of": cfg.best_of, "od_topk": cfg.od_topk, "ablation": cfg.ablation,
        "use_features": S.use_features, "use_start_head": S.use_start_head,
        "prior_lam": cfg.prior_lam, "prior_lam_s": cfg.prior_lam_s,
        "train_norm_f_ref": [float(x) for x in S.train_norm.f_ref],
        "f_ref_features": float(S.fctx.f_ref1),
        "optimizer": "adam", "deterministic": True, "threads": 1,
        "eval_lattice_H": 4,
    }
    out["norm_train_f_ref"] = [float(x) for x in S.train_norm.f_ref]
    out["norm_eval_f_ref"] = [float(x) for x in S.eval_norm.f_ref]
    return out
