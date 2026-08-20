"""Train the full GNN + multi-objective actor-critic method on Xi'an.

First real experimental run for the manuscript (single city, no cross-city
curriculum yet -- that requires multiple trained cities, which don't exist
yet). Reuses the real Xi'an grid/OD/existing-lines/price-group data and the
gym environment mechanics (action masking, episode structure) from
Michailidis et al.'s public mo-tndp environment (MIT licensed,
github.com/sias-uva/mo-tndp, cloned at
external/tabular-tndp/envs/mo-tndp) -- action-space semantics (8-directional
grid walk) are adopted from that environment rather than the arbitrary-node
formulation in the manuscript's general Problem Formulation, since this is
the standard benchmark for this exact city. Our own GNN encoder and
multi-objective policy (src/gnn, src/rl) replace their tabular Q-learning.

Objectives (all tracked as bounded CUMULATIVE values at every step, per
Eq. 1 of the manuscript's Problem Formulation -- r_t^(j) = C^(j)(M_{t+1}) -
C^(j)(M_t) -- then jointly Tchebycheff-scalarized each step, with the
scalarized VALUE's own step-to-step delta used as the RL reward; scalarizing
raw per-objective deltas directly does not work, since Tchebycheff's
ideal-point comparison assumes bounded cumulative values, not near-zero
deltas):
  - demand: cumulative pct of OD demand satisfied so far (natively provided
    by the environment's reward vector, summed over groups).
  - equity: 1 - Gini index over the 5 price-group cumulative satisfied-demand
    vector.
  - coverage: fraction of grid cells within Chebyshev distance K of any
    station built so far. This is a simplified accessibility PROXY, not the
    radiation-model accessibility metric used by zhang2024 -- that remains
    future work, disclosed as such.

The policy is conditioned on the sampled Tchebycheff weight vector w (via a
small learned embedding added to the pooled network-state query) so a
single trained policy actually behaves differently per trade-off, rather
than only having w affect the training reward.

--ablation selects between the full method and two ablations approximating
(not reproducing) the other two reference methods, since neither has public
code we can run directly:
  - full (default): GNN encoder + multi-objective reward, as above.
  - single_objective: GNN encoder, but weights fixed to (1,0,0) (demand
    only) every episode -- approximates MetroGNN (su2024).
  - flat_encoder: FlatEncoder (linear projection, no message passing)
    + multi-objective reward -- approximates Zhang et al. (zhang2024).

Training is on-policy Monte Carlo REINFORCE with a learned baseline
(undiscounted, matching the manuscript's finite-horizon episodic
formulation) -- no prioritized replay or cross-city transfer in this first
single-city run, since those specifically address sample efficiency across
MULTIPLE cities, which this run doesn't yet have.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "tabular-tndp"))
sys.path.insert(0, str(REPO_ROOT / "external" / "tabular-tndp" / "envs" / "mo-tndp"))

import mo_gymnasium as mo_gym  # noqa: E402
from motndp.city import City  # noqa: E402
from motndp.constraints import MetroConstraints  # noqa: E402

import envs  # noqa: F401,E402  (registers motndp_xian-v0 etc.)

from src.gnn.encoder import FlatEncoder, HeteroGNNEncoder  # noqa: E402
from src.rl.policy import AttentivePolicyValueHead, sample_action  # noqa: E402
from src.rl.reward import Objectives, sample_weight_vector, tchebycheff_reward  # noqa: E402

ACTION_TO_DIRECTION = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])


def gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(x).astype(np.float64))
    n = len(x)
    total = x.sum()
    if total == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def build_graph(city: City, device: torch.device):
    """Build the static candidate-region graph (node features, spatial and
    OD edges) once from the real Xi'an data -- see src/gnn/encoder.py's
    RelationalMessagePassingLayer for how these are consumed."""
    grid_x, grid_y, n = city.grid_x_size, city.grid_y_size, city.grid_size

    # Group one-hot feature (price-based equity groups).
    group_ids = city.groups  # sorted unique group values
    grid_groups_flat = city.grid_groups.flatten()
    group_onehot = np.zeros((n, len(group_ids)), dtype=np.float32)
    for gi, gval in enumerate(group_ids):
        group_onehot[grid_groups_flat == gval, gi] = 1.0

    # Existing-line membership flag.
    existing_flag = np.zeros(n, dtype=np.float32)
    for line in city.existing_lines:
        existing_flag[np.asarray(line).flatten().astype(int)] = 1.0

    # OD degree (log1p, min-max normalized).
    od_degree = city.od_mx.sum(axis=1) + city.od_mx.sum(axis=0)
    od_degree = np.log1p(od_degree)
    od_degree = (od_degree - od_degree.min()) / max(od_degree.max() - od_degree.min(), 1e-9)

    node_features = np.concatenate(
        [group_onehot, existing_flag[:, None], od_degree[:, None].astype(np.float32)], axis=1
    )

    # Spatial-contiguity edges: 8-neighbor grid adjacency.
    src, dst = [], []
    for x in range(grid_x):
        for y in range(grid_y):
            idx = x * grid_y + y
            for dx, dy in ACTION_TO_DIRECTION:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_x and 0 <= ny < grid_y:
                    src.append(idx)
                    dst.append(nx * grid_y + ny)
    spatial_edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    spatial_edge_weight = torch.ones(spatial_edge_index.size(1), device=device)

    # OD-flow edges: real Xi'an OD data (nonzero entries of city.od_mx).
    od_src, od_dst = city.od_mx.nonzero()
    od_edge_index = torch.tensor(np.stack([od_src, od_dst]), dtype=torch.long, device=device)
    od_edge_weight = torch.tensor(city.od_mx[od_src, od_dst], dtype=torch.float32, device=device)

    node_features_t = torch.tensor(node_features, dtype=torch.float32, device=device)
    return node_features_t, spatial_edge_index, spatial_edge_weight, od_edge_index, od_edge_weight


def coverage_metric(city: City, covered_cells: list, k: int = 2) -> float:
    grid_x, grid_y = city.grid_x_size, city.grid_y_size
    xs, ys = np.meshgrid(np.arange(grid_x), np.arange(grid_y), indexing="ij")
    all_cells = np.stack([xs.flatten(), ys.flatten()], axis=1)
    stations = np.array(covered_cells)
    dists = np.max(np.abs(all_cells[:, None, :] - stations[None, :, :]), axis=2)  # Chebyshev
    min_dist = dists.min(axis=1)
    return float((min_dist <= k).mean())


def run_episode(
    env,
    city: City,
    encoder: HeteroGNNEncoder,
    policy: AttentivePolicyValueHead,
    weight_embed: torch.nn.Module,
    graph,
    weights: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    train: bool,
):
    node_features, spatial_ei, spatial_ew, od_ei, od_ew = graph
    h = encoder(node_features, spatial_ei, spatial_ew, od_ei, od_ew)  # [n, hidden] -- computed once per episode

    # Condition the policy on the sampled Tchebycheff weight vector so a
    # single trained policy actually learns to behave differently for
    # different objective trade-offs, rather than only having w affect the
    # reward used for the gradient (which would just average over trade-offs).
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
    weight_bias = weight_embed(weight_t)

    obs, info = env.reset()
    n = city.grid_size
    selected_mask = torch.zeros(n, dtype=torch.bool, device=device)
    selected_mask[info["location_grid_index"]] = True

    log_probs, values, scalarized = [], [], []
    demand_rewards = []
    cumulative_group_reward = np.zeros(env.unwrapped.nr_groups)
    covered_cells = [info["location_grid_coordinates"].tolist()]
    cumulative_demand = 0.0
    prev_equity = 1.0 - gini(cumulative_group_reward)
    prev_coverage = coverage_metric(city, covered_cells, k=2)
    prev_scalarized = tchebycheff_reward(
        Objectives(cumulative_demand, prev_equity, prev_coverage), weights
    )

    done = False
    while not done:
        loc = info["location_grid_coordinates"]
        neighbor_coords = loc + ACTION_TO_DIRECTION
        valid = (
            (neighbor_coords[:, 0] >= 0)
            & (neighbor_coords[:, 0] < city.grid_x_size)
            & (neighbor_coords[:, 1] >= 0)
            & (neighbor_coords[:, 1] < city.grid_y_size)
        )
        neighbor_idx = np.where(
            valid, neighbor_coords[:, 0] * city.grid_y_size + neighbor_coords[:, 1], 0
        )
        candidate_embeddings = h[neighbor_idx]  # [8, hidden]
        query = policy.pool_network_state(h, selected_mask) + weight_bias
        action_mask = torch.tensor(info["action_mask"].astype(bool), device=device)

        logits, value = policy(candidate_embeddings, query, action_mask)
        action, log_prob = sample_action(logits)
        action = int(action.item())

        new_obs, reward_vector, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # clone (not in-place mutate) -- selected_mask feeds into the pooled
        # query via boolean indexing on `h`, which autograd tracks; mutating
        # the same tensor object across steps within one episode invalidates
        # the graph built at earlier steps.
        selected_mask = selected_mask.clone()
        selected_mask[info["location_grid_index"]] = True
        covered_cells.append(info["location_grid_coordinates"].tolist())
        cumulative_group_reward += reward_vector

        # Track cumulative (bounded, ideal-point-comparable) objective
        # values at every step -- following Eq. 1 in the manuscript's
        # Problem Formulation, r_t^(j) = C^(j)(M_{t+1}) - C^(j)(M_t) is the
        # marginal change of the SCALARIZED cumulative objective, not of
        # each raw objective independently (Tchebycheff's ideal-point
        # comparison only makes sense against bounded cumulative values,
        # not tiny near-zero per-step deltas).
        cumulative_demand += float(reward_vector.sum())
        new_equity = 1.0 - gini(cumulative_group_reward)
        new_coverage = coverage_metric(city, covered_cells, k=2)
        new_scalarized = tchebycheff_reward(
            Objectives(cumulative_demand, new_equity, new_coverage), weights
        )

        log_probs.append(log_prob)
        values.append(value)
        demand_rewards.append(float(reward_vector.sum()))
        scalarized.append(new_scalarized - prev_scalarized)
        prev_equity, prev_coverage, prev_scalarized = new_equity, new_coverage, new_scalarized

    equity_bonus = prev_equity  # final cumulative value, for logging only
    coverage_bonus = prev_coverage
    demand_total = float(sum(demand_rewards))

    returns = np.cumsum(scalarized[::-1])[::-1].copy()  # undiscounted Monte Carlo return-to-go
    return log_probs, values, returns, demand_total, equity_bonus, coverage_bonus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_episodes", type=int, default=300)
    parser.add_argument("--nr_stations", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nr_groups", type=int, default=5)
    parser.add_argument(
        "--ablation",
        choices=["full", "single_objective", "flat_encoder"],
        default="full",
        help=(
            "full = GNN + multi-objective (our method); "
            "single_objective = GNN + demand-only reward, approximates MetroGNN (su2024); "
            "flat_encoder = FlatEncoder + multi-objective, approximates Zhang et al. (zhang2024)"
        ),
    )
    parser.add_argument("--save_checkpoint", type=str, default=None, help="Path to save trained weights (for cross-city transfer, e.g. scripts/train_hcmc.py --init_from_xian)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    city_path = REPO_ROOT / "external" / "tabular-tndp" / "envs" / "mo-tndp" / "cities" / "xian"
    city = City(city_path, groups_file=f"price_groups_{args.nr_groups}.txt")
    env = mo_gym.make(
        "motndp_xian-v0",
        city=city,
        constraints=MetroConstraints(city),
        nr_stations=args.nr_stations,
        od_type="pct",
        chained_reward=False,
    )

    graph = build_graph(city, device)
    in_dim = graph[0].shape[1]

    if args.ablation == "flat_encoder":
        encoder = FlatEncoder(in_dim=in_dim, hidden_dim=args.hidden_dim).to(device)
    else:
        encoder = HeteroGNNEncoder(in_dim=in_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    policy = AttentivePolicyValueHead(embed_dim=args.hidden_dim).to(device)
    weight_embed = torch.nn.Linear(3, args.hidden_dim, bias=False).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(policy.parameters()) + list(weight_embed.parameters()), lr=args.lr
    )

    history = []
    t0 = time.time()
    for episode in range(args.train_episodes):
        if args.ablation == "single_objective":
            weights = np.array([1.0, 0.0, 0.0])  # demand only, fixed every episode
        else:
            weights = sample_weight_vector(rng, n_objectives=3)
        log_probs, values, returns, demand_total, equity, coverage = run_episode(
            env, city, encoder, policy, weight_embed, graph, weights, device, rng, train=True
        )

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
                f"episode {episode}: demand={avg_demand:.5f} equity={avg_equity:.4f} "
                f"coverage={avg_coverage:.4f} loss={loss.item():.4f}"
            )

    elapsed = time.time() - t0
    print(f"\ntraining done in {elapsed:.1f}s")

    if args.save_checkpoint:
        ckpt = {"policy": policy.state_dict(), "weight_embed": weight_embed.state_dict()}
        if hasattr(encoder, "layers"):
            ckpt["encoder_layers"] = encoder.layers.state_dict()
        torch.save(ckpt, args.save_checkpoint)
        print(f"saved checkpoint to {args.save_checkpoint}")

    # Final evaluation: greedy rollout (argmax action) under several
    # different weight vectors, from the SAME trained model, to directly
    # check whether the policy actually behaves differently per trade-off
    # (the manuscript's "single trained policy supports the full range of
    # trade-offs" claim) rather than converging to one fixed behavior
    # regardless of w.
    def greedy_eval(eval_weights_np: np.ndarray):
        with torch.no_grad():
            node_features, spatial_ei, spatial_ew, od_ei, od_ew = graph
            h = encoder(node_features, spatial_ei, spatial_ew, od_ei, od_ew)
            eval_weights = torch.tensor(eval_weights_np, dtype=torch.float32, device=device)
            weight_bias = weight_embed(eval_weights)
            obs, info = env.reset(seed=args.seed)
            n = city.grid_size
            selected_mask = torch.zeros(n, dtype=torch.bool, device=device)
            selected_mask[info["location_grid_index"]] = True
            cumulative_group_reward = np.zeros(env.unwrapped.nr_groups)
            covered_cells = [info["location_grid_coordinates"].tolist()]
            actions_taken = []
            demand_total = 0.0
            done = False
            while not done:
                loc = info["location_grid_coordinates"]
                neighbor_coords = loc + ACTION_TO_DIRECTION
                valid = (
                    (neighbor_coords[:, 0] >= 0)
                    & (neighbor_coords[:, 0] < city.grid_x_size)
                    & (neighbor_coords[:, 1] >= 0)
                    & (neighbor_coords[:, 1] < city.grid_y_size)
                )
                neighbor_idx = np.where(
                    valid, neighbor_coords[:, 0] * city.grid_y_size + neighbor_coords[:, 1], 0
                )
                candidate_embeddings = h[neighbor_idx]
                query = policy.pool_network_state(h, selected_mask) + weight_bias
                action_mask = torch.tensor(info["action_mask"].astype(bool), device=device)
                logits, _ = policy(candidate_embeddings, query, action_mask)
                action = int(torch.argmax(logits).item())
                actions_taken.append(action)

                obs, reward_vector, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                selected_mask[info["location_grid_index"]] = True
                covered_cells.append(info["location_grid_coordinates"].tolist())
                cumulative_group_reward += reward_vector
                demand_total += float(reward_vector.sum())

        equity_val = 1.0 - gini(cumulative_group_reward)
        coverage_val = coverage_metric(city, covered_cells, k=2)
        return demand_total, equity_val, coverage_val, actions_taken

    eval_settings = {
        "uniform (1/3,1/3,1/3)": np.array([1 / 3, 1 / 3, 1 / 3]),
        "demand-only (1,0,0)": np.array([1.0, 0.0, 0.0]),
        "equity-only (0,1,0)": np.array([0.0, 1.0, 0.0]),
        "coverage-only (0,0,1)": np.array([0.0, 0.0, 1.0]),
    }
    results = {}
    for name, w in eval_settings.items():
        demand_total, equity_val, coverage_val, actions_taken = greedy_eval(w)
        results[name] = (demand_total, equity_val, coverage_val)
        print(
            f"FINAL (greedy eval, w={name}): demand={demand_total:.5f} equity={equity_val:.4f} "
            f"coverage={coverage_val:.4f} actions={actions_taken}"
        )

    all_identical = len({tuple(v) for v in results.values()}) == 1
    print(
        f"\nweight-conditioning sanity check: results {'ARE' if all_identical else 'are NOT'} "
        f"identical across all 4 weight settings "
        f"({'policy is NOT actually responding to w -- needs investigation' if all_identical else 'policy responds to w as designed'})"
    )
    print(f"wall_clock_s={elapsed:.1f} episodes={args.train_episodes}")


if __name__ == "__main__":
    main()
