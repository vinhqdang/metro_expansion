"""Post-hoc evaluation of the true radiation-model accessibility (Simini et al.
2012) for a trained Ho Chi Minh City policy checkpoint (scripts/train_hcmc.py
--save_checkpoint).

This is deliberately a separate, read-only evaluation script rather than a
change to the training loop: computing the radiation model does not (yet)
feed back into the training reward (Section: Discussion/Limitations), only
into this post-hoc check of how the already-reported Chebyshev coverage proxy
(Eq. 4) compares against the design-intent accessibility measure it stands in
for, for the one city (Ho Chi Minh City) where real population and real
geographic coordinates make the true radiation model computable at all.

Reuses build_graph_and_env_data and HCMCEnv from train_hcmc.py unchanged, so
the rollout mechanics (contiguity-graph action space, greedy argmax policy,
5-random-start averaging) are identical to the training script's own
evaluation protocol -- the only new computation here is the radiation
accessibility score itself.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.use_deterministic_algorithms(True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.hcmc import (  # noqa: E402
    build_radiation_probabilities,
    load_zones,
    radiation_accessibility_score,
)
from src.gnn.encoder import FlatEncoder, HeteroGNNEncoder  # noqa: E402
from src.rl.policy import AttentivePolicyValueHead  # noqa: E402
from scripts.train_hcmc import DATA_DIR, HCMCEnv, build_graph_and_env_data  # noqa: E402


def greedy_rollout_stations(env: HCMCEnv, encoder, policy, weight_embed, graph, weights_np: np.ndarray, device) -> list[int]:
    """Argmax rollout identical to train_hcmc.py's greedy_eval_single_start,
    but returning the built station set instead of demand/equity/coverage."""
    with torch.no_grad():
        node_features, spatial_ei, spatial_ew, od_ei, od_ew = graph
        h = encoder(node_features, spatial_ei, spatial_ew, od_ei, od_ew)
        weight_bias = weight_embed(torch.tensor(weights_np, dtype=torch.float32, device=device))
        candidates = env.reset()
        selected_mask = torch.zeros(env.n, dtype=torch.bool, device=device)
        selected_mask[env.current] = True
        done = False
        while not done and candidates:
            candidate_idx = torch.tensor(candidates, dtype=torch.long, device=device)
            candidate_embeddings = h[candidate_idx]
            query = policy.pool_network_state(h, selected_mask) + weight_bias
            action_mask = torch.ones(len(candidates), dtype=torch.bool, device=device)
            logits, _ = policy(candidate_embeddings, query, action_mask)
            action_zone = candidates[int(torch.argmax(logits).item())]
            _, candidates, done = env.step(action_zone)
            selected_mask[action_zone] = True
    return list(env.covered_cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ablation", choices=["full", "single_objective", "flat_encoder"], default="full")
    parser.add_argument("--nr_stations", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="base seed for the 5-random-start eval")
    parser.add_argument("--num_starts", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, contiguity_adj, od_adj, total_od, income_bracket, n = build_graph_and_env_data(device)
    in_dim = graph[0].shape[1]

    zones = load_zones(DATA_DIR)
    assert len(zones) == n, "zones/graph size mismatch"
    p = build_radiation_probabilities(zones)

    if args.ablation == "flat_encoder":
        encoder = FlatEncoder(in_dim=in_dim, hidden_dim=args.hidden_dim).to(device)
    else:
        encoder = HeteroGNNEncoder(in_dim=in_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    policy = AttentivePolicyValueHead(embed_dim=args.hidden_dim).to(device)
    weight_embed = torch.nn.Linear(3, args.hidden_dim, bias=False).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(ckpt["policy"])
    weight_embed.load_state_dict(ckpt["weight_embed"])
    if "encoder_layers" in ckpt and hasattr(encoder, "layers"):
        encoder.layers.load_state_dict(ckpt["encoder_layers"])
    encoder.eval()
    policy.eval()

    weights_np = np.array([1 / 3, 1 / 3, 1 / 3])
    scores = []
    for i in range(args.num_starts):
        rng = np.random.default_rng(args.seed + 1000 * i)
        env = HCMCEnv(contiguity_adj, od_adj, total_od, income_bracket, n, args.nr_stations, rng)
        stations = greedy_rollout_stations(env, encoder, policy, weight_embed, graph, weights_np, device)
        scores.append(radiation_accessibility_score(p, stations))

    scores = np.array(scores)
    print(f"checkpoint={args.checkpoint} ablation={args.ablation}")
    print(f"radiation_accessibility: mean={scores.mean():.5f} std={scores.std(ddof=1) if len(scores) > 1 else 0.0:.5f} per_start={scores.tolist()}")


if __name__ == "__main__":
    main()
