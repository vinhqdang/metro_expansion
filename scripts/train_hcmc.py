"""Train the full GNN + multi-objective actor-critic method on Ho Chi Minh City.

Second real experimental city (after Xi'an, scripts/train_xian.py), and the
first test of whether the method extends to the irregular, polygon-based
candidate-region graphs described in the manuscript's "Instantiating the
action space per city" section, as opposed to Xi'an's regular grid. Reuses
the real, tested HCMC data pipeline (src/data/hcmc.py) against the real
CSL_HCMC dataset (Section: Case Studies and Data).

Action space (per \\citet{su2024}'s formulation, matching the manuscript):
at each step, the agent may extend the line to any spatially-contiguous
unselected neighbor of the current endpoint (via the queen-contiguity graph
built in src/data/hcmc.py), rather than Xi'an's fixed 8-directional grid
walk. This is a genuinely variable-size action set per step (a zone's
contiguity degree varies), which the existing AttentivePolicyValueHead
already supports without modification, since it scores an arbitrary set of
candidate embeddings rather than a fixed-size action vector.

Objectives, computed incrementally exactly as in train_xian.py (see that
module's docstring for the scalarize-then-difference rationale this reuses
unchanged):
  - demand: cumulative OD flow satisfied between all pairs of zones
    currently connected by the line, as a fraction of total OD flow in the
    disaggregated OD graph (src/data/hcmc.py::build_od_edges).
  - equity: 1 - Gini index over cumulative satisfied demand attributed to
    each of 4 income brackets (a zone's bracket = argmax over its
    res_income_1..4 columns) -- HCMC's data does not ship income-stratified
    OD like Xi'an's price_groups files do, so this is a derived analogue,
    not a literal reuse of Xi'an's group definition. Disclosed as such.
  - coverage: fraction of zones within K contiguity-graph hops of any built
    station -- the polygon-graph analogue of Xi'an's Chebyshev grid distance
    (there is no regular grid to measure Chebyshev distance on here).

Cross-city curriculum support: --init_from_xian loads a checkpoint saved by
scripts/train_xian.py and uses it to initialize the encoder/policy/
weight_embed weights before training on HCMC, per Algorithm 1. Requires
matching hidden_dim (both cities' node feature dimensionality differs, so
only the GNN's hidden-layer and policy weights transfer, not input_proj).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

# Must be set before any CUDA context is created (i.e. before `import torch`)
# for torch.use_deterministic_algorithms(True) below to fully cover cuBLAS
# ops; see the manuscript's reproducibility finding (Section: Experiments).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.use_deterministic_algorithms(True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.hcmc import (  # noqa: E402
    build_bus_stop_density_feature,
    build_contiguity_edges,
    build_node_features,
    build_od_edges,
    load_bus_stops,
    load_od_survey,
    load_zones,
)
from src.gnn.encoder import FlatEncoder, HeteroGNNEncoder  # noqa: E402
from src.rl.policy import AttentivePolicyValueHead, sample_action  # noqa: E402
from src.rl.reward import Objectives, sample_weight_vector, tchebycheff_reward  # noqa: E402

DATA_DIR = REPO_ROOT / "external" / "CSL_HCMC"


def gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(x).astype(np.float64))
    n = len(x)
    total = x.sum()
    if total == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def build_graph_and_env_data(device: torch.device, no_bus_signal: bool = False):
    """Loads real HCMC data and builds everything the training loop needs:
    node features, spatial/OD edges (for the GNN), an adjacency list (for
    the action space), an OD adjacency list (for reward computation), and
    per-zone income-bracket assignment (for equity).

    no_bus_signal: zeroes the bus-stop-density feature column rather than
    dropping it, so the feature vector's dimensionality (and therefore the
    encoder's input layer) is identical between conditions -- isolating the
    marginal value of the bus signal itself, not confounding it with a
    changed input width. Ablation added to test Contribution 3's claim that
    the bus-network auxiliary signal is doing real work (domain reviewer).
    """
    zones = load_zones(DATA_DIR)
    bus_stops = load_bus_stops(DATA_DIR)
    od = load_od_survey(DATA_DIR)

    node_features_np = build_node_features(zones)
    bus_density = build_bus_stop_density_feature(zones, bus_stops)
    if no_bus_signal:
        bus_density = np.zeros_like(bus_density)
    node_features_np = np.concatenate([node_features_np, bus_density[:, None]], axis=1)

    spatial_ei, spatial_ew = build_contiguity_edges(zones)
    od_ei, od_ew = build_od_edges(zones, od)

    n = len(zones)

    # Adjacency list for the action space (contiguity graph).
    contiguity_adj: dict[int, list[int]] = defaultdict(list)
    for s, d in zip(spatial_ei[0], spatial_ei[1]):
        contiguity_adj[int(s)].append(int(d))

    # Adjacency list for reward computation (OD graph), both directions.
    od_adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for s, d, w in zip(od_ei[0], od_ei[1], od_ew):
        od_adj[int(s)].append((int(d), float(w)))
    total_od = float(od_ew.sum()) if od_ew.size > 0 else 1.0

    # Income-bracket assignment (argmax of the 4 income columns).
    income_cols = ["res_income_1", "res_income_2", "res_income_3", "res_income_4"]
    income_vals = zones[income_cols].fillna(0.0).to_numpy(dtype=np.float64)
    income_bracket = income_vals.argmax(axis=1)  # [n], values in {0,1,2,3}

    node_features = torch.tensor(node_features_np, dtype=torch.float32, device=device)
    spatial_edge_index = torch.tensor(spatial_ei, dtype=torch.long, device=device)
    spatial_edge_weight = torch.tensor(spatial_ew, dtype=torch.float32, device=device)
    od_edge_index = torch.tensor(od_ei, dtype=torch.long, device=device)
    od_edge_weight = torch.tensor(od_ew, dtype=torch.float32, device=device)

    graph = (node_features, spatial_edge_index, spatial_edge_weight, od_edge_index, od_edge_weight)
    return graph, contiguity_adj, od_adj, total_od, income_bracket, n


def coverage_metric(contiguity_adj: dict[int, list[int]], n: int, covered_cells: list[int], k: int = 2) -> float:
    """Fraction of zones within K contiguity-graph hops of any built station
    (BFS from each station up to depth K, unioned)."""
    within_k: set[int] = set()
    for start in covered_cells:
        frontier = {start}
        within_k.add(start)
        for _ in range(k):
            next_frontier = set()
            for node in frontier:
                for neighbor in contiguity_adj[node]:
                    if neighbor not in within_k:
                        next_frontier.add(neighbor)
                        within_k.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break
    return len(within_k) / n


class HCMCEnv:
    """Minimal environment matching train_xian.py's step/reset semantics,
    but with a variable-size, contiguity-graph-derived action space instead
    of a fixed 8-directional grid walk."""

    def __init__(self, contiguity_adj, od_adj, total_od, income_bracket, n, nr_stations, rng):
        self.contiguity_adj = contiguity_adj
        self.od_adj = od_adj
        self.total_od = total_od
        self.income_bracket = income_bracket
        self.n = n
        self.nr_stations = nr_stations
        self.rng = rng
        self.nr_groups = 4

    def reset(self):
        self.current = int(self.rng.integers(0, self.n))
        self.selected = {self.current}
        self.covered_cells = [self.current]
        self.step_count = 1
        self.cumulative_group_reward = np.zeros(self.nr_groups)
        return self._candidates()

    def _candidates(self) -> list[int]:
        return [c for c in self.contiguity_adj[self.current] if c not in self.selected]

    def step(self, action_zone: int):
        # marginal OD flow newly satisfied by connecting action_zone to every already-selected zone
        newly_satisfied = 0.0
        for neighbor, w in self.od_adj[action_zone]:
            if neighbor in self.selected:
                newly_satisfied += w
        reward_vector = np.zeros(self.nr_groups)
        reward_vector[self.income_bracket[action_zone]] = newly_satisfied / self.total_od

        self.selected.add(action_zone)
        self.covered_cells.append(action_zone)
        self.current = action_zone
        self.cumulative_group_reward += reward_vector
        self.step_count += 1

        candidates = self._candidates()
        done = self.step_count >= self.nr_stations or len(candidates) == 0
        return reward_vector, candidates, done


def run_episode(env: HCMCEnv, encoder, policy, weight_embed, graph, weights, contiguity_adj, n, device, train: bool):
    node_features, spatial_ei, spatial_ew, od_ei, od_ew = graph
    h = encoder(node_features, spatial_ei, spatial_ew, od_ei, od_ew)

    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
    weight_bias = weight_embed(weight_t)

    candidates = env.reset()
    selected_mask = torch.zeros(n, dtype=torch.bool, device=device)
    selected_mask[env.current] = True

    log_probs, values, demand_rewards = [], [], []
    scalarized = []
    cumulative_demand = 0.0
    prev_equity = 1.0 - gini(env.cumulative_group_reward)
    prev_coverage = coverage_metric(contiguity_adj, n, env.covered_cells, k=2)
    prev_scalarized = tchebycheff_reward(Objectives(cumulative_demand, prev_equity, prev_coverage), weights)

    done = False
    while not done:
        if not candidates:
            break
        candidate_idx = torch.tensor(candidates, dtype=torch.long, device=device)
        candidate_embeddings = h[candidate_idx]
        query = policy.pool_network_state(h, selected_mask) + weight_bias
        action_mask = torch.ones(len(candidates), dtype=torch.bool, device=device)

        logits, value = policy(candidate_embeddings, query, action_mask)
        action_idx, log_prob = sample_action(logits)
        action_zone = candidates[int(action_idx.item())]

        reward_vector, candidates, done = env.step(action_zone)

        selected_mask = selected_mask.clone()
        selected_mask[action_zone] = True

        cumulative_demand += float(reward_vector.sum())
        new_equity = 1.0 - gini(env.cumulative_group_reward)
        new_coverage = coverage_metric(contiguity_adj, n, env.covered_cells, k=2)
        new_scalarized = tchebycheff_reward(Objectives(cumulative_demand, new_equity, new_coverage), weights)

        log_probs.append(log_prob)
        values.append(value)
        demand_rewards.append(float(reward_vector.sum()))
        scalarized.append(new_scalarized - prev_scalarized)
        prev_equity, prev_coverage, prev_scalarized = new_equity, new_coverage, new_scalarized

    if not log_probs:
        return None  # degenerate episode (isolated starting zone with no contiguous neighbors)

    demand_total = float(sum(demand_rewards))
    returns = np.cumsum(scalarized[::-1])[::-1].copy()
    return log_probs, values, returns, demand_total, prev_equity, prev_coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_episodes", type=int, default=500)
    parser.add_argument("--nr_stations", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ablation", choices=["full", "single_objective", "flat_encoder"], default="full"
    )
    parser.add_argument(
        "--init_from_xian", type=str, default=None, help="Path to a Xi'an checkpoint (.pt) to warm-start from"
    )
    parser.add_argument("--save_checkpoint", type=str, default=None)
    parser.add_argument(
        "--no_bus_signal", action="store_true", default=False,
        help="Zero the bus-stop-density auxiliary feature (ablation testing Contribution 3's claim)"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    graph, contiguity_adj, od_adj, total_od, income_bracket, n = build_graph_and_env_data(
        device, no_bus_signal=args.no_bus_signal
    )
    in_dim = graph[0].shape[1]
    print(f"HCMC graph: {n} zones, {graph[1].shape[1]} contiguity edges, {graph[3].shape[1]} OD edges")

    if args.ablation == "flat_encoder":
        encoder = FlatEncoder(in_dim=in_dim, hidden_dim=args.hidden_dim).to(device)
    else:
        encoder = HeteroGNNEncoder(in_dim=in_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    policy = AttentivePolicyValueHead(embed_dim=args.hidden_dim).to(device)
    weight_embed = torch.nn.Linear(3, args.hidden_dim, bias=False).to(device)

    if args.init_from_xian:
        ckpt = torch.load(args.init_from_xian, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt["policy"])
        weight_embed.load_state_dict(ckpt["weight_embed"])
        # encoder.input_proj is city-specific (different feature dims); only
        # the shared message-passing layers transfer.
        if "encoder_layers" in ckpt and hasattr(encoder, "layers"):
            encoder.layers.load_state_dict(ckpt["encoder_layers"])
        print(f"warm-started policy/weight_embed/GNN-layers from {args.init_from_xian}")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(policy.parameters()) + list(weight_embed.parameters()), lr=args.lr
    )

    env = HCMCEnv(contiguity_adj, od_adj, total_od, income_bracket, n, args.nr_stations, rng)

    history = []
    t0 = time.time()
    episodes_run = 0
    for episode in range(args.train_episodes):
        if args.ablation == "single_objective":
            weights = np.array([1.0, 0.0, 0.0])
        else:
            weights = sample_weight_vector(rng, n_objectives=3)

        result = run_episode(env, encoder, policy, weight_embed, graph, weights, contiguity_adj, n, device, train=True)
        if result is None:
            continue  # degenerate start, skip without counting as a gradient step
        episodes_run += 1
        log_probs, values, returns, demand_total, equity, coverage = result

        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        values_t = torch.stack(values)
        advantages = (returns_t - values_t).detach()
        if advantages.numel() > 1 and advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        policy_loss = -torch.stack(log_probs) @ advantages / len(log_probs)
        value_loss = torch.nn.functional.mse_loss(values_t, returns_t)
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(policy.parameters()), max_norm=1.0)
        optimizer.step()

        history.append((demand_total, equity, coverage))
        if episode % 20 == 0 or episode == args.train_episodes - 1:
            recent = history[-20:]
            avg_demand = np.mean([h[0] for h in recent])
            avg_equity = np.mean([h[1] for h in recent])
            avg_coverage = np.mean([h[2] for h in recent])
            print(
                f"episode {episode} (ran {episodes_run}): demand={avg_demand:.5f} equity={avg_equity:.4f} "
                f"coverage={avg_coverage:.4f} loss={loss.item():.4f}"
            )

    elapsed = time.time() - t0
    print(f"\ntraining done in {elapsed:.1f}s ({episodes_run} non-degenerate episodes out of {args.train_episodes})")

    if args.save_checkpoint:
        ckpt = {"policy": policy.state_dict(), "weight_embed": weight_embed.state_dict()}
        if hasattr(encoder, "layers"):
            ckpt["encoder_layers"] = encoder.layers.state_dict()
        # Also save the input projection, in addition to the message-passing
        # layers above: it is only safe to reuse across cities with matching
        # input feature dimensionality (which --init_from_xian's loading path
        # deliberately does not assume and never reads this key for), but a
        # same-city post-hoc evaluation (scripts/eval_hcmc_radiation.py,
        # scripts/make_hcmc_map_data.py) needs it to actually reconstruct the
        # trained model rather than a message-passing-layers-only partial one.
        if hasattr(encoder, "input_proj"):
            ckpt["encoder_input_proj"] = encoder.input_proj.state_dict()
        torch.save(ckpt, args.save_checkpoint)
        print(f"saved checkpoint to {args.save_checkpoint}")

    # Final evaluation: greedy rollout under a fixed starting zone (seeded) for reproducibility.
    def greedy_eval_single_start(eval_weights_np: np.ndarray, eval_seed: int):
        eval_env = HCMCEnv(contiguity_adj, od_adj, total_od, income_bracket, n, args.nr_stations, np.random.default_rng(eval_seed))
        with torch.no_grad():
            node_features, spatial_ei, spatial_ew, od_ei, od_ew = graph
            h = encoder(node_features, spatial_ei, spatial_ew, od_ei, od_ew)
            weight_bias = weight_embed(torch.tensor(eval_weights_np, dtype=torch.float32, device=device))
            candidates = eval_env.reset()
            selected_mask = torch.zeros(n, dtype=torch.bool, device=device)
            selected_mask[eval_env.current] = True
            done = False
            demand_total = 0.0
            while not done and candidates:
                candidate_idx = torch.tensor(candidates, dtype=torch.long, device=device)
                candidate_embeddings = h[candidate_idx]
                query = policy.pool_network_state(h, selected_mask) + weight_bias
                action_mask = torch.ones(len(candidates), dtype=torch.bool, device=device)
                logits, _ = policy(candidate_embeddings, query, action_mask)
                action_zone = candidates[int(torch.argmax(logits).item())]
                reward_vector, candidates, done = eval_env.step(action_zone)
                selected_mask[action_zone] = True
                demand_total += float(reward_vector.sum())
        equity_final = 1.0 - gini(eval_env.cumulative_group_reward)
        coverage_final = coverage_metric(contiguity_adj, n, eval_env.covered_cells, k=2)
        return demand_total, equity_final, coverage_final

    def greedy_eval(eval_weights_np: np.ndarray, base_seed: int, num_starts: int = 5):
        """Average over several random starting zones, not one fixed zone.

        Necessary specifically for HCMC's irregular polygon graph: unlike
        Xi'an's grid, geographic contiguity (what the greedy walk can reach)
        and OD-connectivity (what generates demand reward) are structurally
        different relations here, so a single fixed starting zone can --
        and, discovered empirically, sometimes does -- produce a greedy
        rollout that never intersects any OD-connected zone pair at all
        (demand identically 0.0), which is a property of that one starting
        point, not of the trained policy in general.
        """
        results = [
            greedy_eval_single_start(eval_weights_np, base_seed + 1000 * i) for i in range(num_starts)
        ]
        demand = float(np.mean([r[0] for r in results]))
        equity = float(np.mean([r[1] for r in results]))
        coverage = float(np.mean([r[2] for r in results]))
        return demand, equity, coverage

    for name, w in {
        "uniform (1/3,1/3,1/3)": np.array([1 / 3, 1 / 3, 1 / 3]),
        "demand-only (1,0,0)": np.array([1.0, 0.0, 0.0]),
        "equity-only (0,1,0)": np.array([0.0, 1.0, 0.0]),
        "coverage-only (0,0,1)": np.array([0.0, 0.0, 1.0]),
    }.items():
        demand_total, equity_final, coverage_final = greedy_eval(w, base_seed=args.seed)
        print(f"FINAL (greedy eval over 5 starts, w={name}): demand={demand_total:.5f} equity={equity_final:.4f} coverage={coverage_final:.4f}")

    print(f"wall_clock_s={elapsed:.1f} episodes={args.train_episodes}")


if __name__ == "__main__":
    main()
