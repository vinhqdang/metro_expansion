"""Reconstruct a real trained Ho Chi Minh City network for the map figure.

Loads a saved full-method checkpoint and does one greedy rollout (reusing
scripts/eval_hcmc_radiation.py's rollout logic unchanged), then writes the
selected zones' real centroid coordinates to a small JSON file for plotting
against the real CSL_HCMC geography. No fabricated station placements.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.data.hcmc import load_zones
from src.gnn.encoder import HeteroGNNEncoder
from src.rl.policy import AttentivePolicyValueHead
from scripts.train_hcmc import DATA_DIR, HCMCEnv, build_graph_and_env_data
from scripts.eval_hcmc_radiation import greedy_rollout_stations


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, contiguity_adj, od_adj, total_od, income_bracket, n = build_graph_and_env_data(device)
    in_dim = graph[0].shape[1]

    encoder = HeteroGNNEncoder(in_dim=in_dim, hidden_dim=32, num_layers=2).to(device)
    policy = AttentivePolicyValueHead(embed_dim=32).to(device)
    weight_embed = torch.nn.Linear(3, 32, bias=False).to(device)

    ckpt_path = Path.home() / "AppData" / "Local" / "Temp" / "hcmc_ckpts" / "full_seed42.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy.load_state_dict(ckpt["policy"])
    weight_embed.load_state_dict(ckpt["weight_embed"])
    encoder.layers.load_state_dict(ckpt["encoder_layers"])
    encoder.eval()
    policy.eval()

    rng = np.random.default_rng(42)
    env = HCMCEnv(contiguity_adj, od_adj, total_od, income_bracket, n, 20, rng)
    weights_np = np.array([1 / 3, 1 / 3, 1 / 3])
    stations = greedy_rollout_stations(env, encoder, policy, weight_embed, graph, weights_np, device)

    zones = load_zones(DATA_DIR)
    station_coords = zones.iloc[stations][["x_centroid", "y_centroid"]].to_numpy().tolist()

    out = {"station_zone_indices": stations, "station_coords_xy": station_coords}
    out_path = REPO_ROOT / "manuscript" / "figures" / "hcmc_map_data.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print(f"stations: {stations}")


if __name__ == "__main__":
    main()
