"""Static candidate-region graph for any rectangular grid city (Xi'an, Amsterdam, ...).

Generalises scripts/train_xian.build_graph: the grid size, the group set and the
existing-line flags are read from the City object instead of being assumed.
With the default settings the output is identical to train_xian.build_graph.

Node features (per cell, in this order):
  one-hot price group (len(city.groups)), existing-line flag, min-max log1p OD degree.
Edges:
  spatial: 8-neighbour contiguity (ACTION_TO_DIRECTION), weight 1.
  OD: nonzero entries of city.od_mx, weight = OD flow. Optionally pruned to the
      top-k largest incoming OD edges per destination node (od_topk).
"""

from __future__ import annotations

import numpy as np
import torch

ACTION_TO_DIRECTION = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])


def _prune_od_topk(src: np.ndarray, dst: np.ndarray, w: np.ndarray, k: int):
    """Keep the k largest-weight edges for every destination node (ties broken by index)."""
    order = np.lexsort((src, -w, dst))  # primary: dst, then weight descending
    src, dst, w = src[order], dst[order], w[order]
    # rank of each edge inside its destination group
    starts = np.r_[0, np.flatnonzero(np.diff(dst)) + 1]
    group_start = np.repeat(starts, np.diff(np.r_[starts, dst.size]))
    rank = np.arange(dst.size) - group_start
    keep = rank < k
    return src[keep], dst[keep], w[keep]


def build_graph(city, device: torch.device | str = "cpu", od_topk: int | None = None):
    """Return (node_features, spatial_ei, spatial_ew, od_ei, od_ew) as tensors on `device`."""
    device = torch.device(device)
    grid_x, grid_y, n = city.grid_x_size, city.grid_y_size, city.grid_size

    group_ids = city.groups  # sorted unique group values
    grid_groups_flat = city.grid_groups.flatten()
    group_onehot = np.zeros((n, len(group_ids)), dtype=np.float32)
    for gi, gval in enumerate(group_ids):
        group_onehot[grid_groups_flat == gval, gi] = 1.0

    existing_flag = np.zeros(n, dtype=np.float32)
    for line in city.existing_lines:
        existing_flag[np.asarray(line).flatten().astype(int)] = 1.0

    od_degree = city.od_mx.sum(axis=1) + city.od_mx.sum(axis=0)
    od_degree = np.log1p(od_degree)
    od_degree = (od_degree - od_degree.min()) / max(od_degree.max() - od_degree.min(), 1e-9)

    node_features = np.concatenate(
        [group_onehot, existing_flag[:, None], od_degree[:, None].astype(np.float32)], axis=1
    )

    # 8-neighbour spatial contiguity; edge order (x, y, direction) matches train_xian.
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

    od_src, od_dst = city.od_mx.nonzero()
    od_w = city.od_mx[od_src, od_dst]
    if od_topk is not None:
        od_src, od_dst, od_w = _prune_od_topk(od_src.astype(np.int64), od_dst.astype(np.int64), od_w, int(od_topk))
    od_edge_index = torch.tensor(np.stack([od_src, od_dst]), dtype=torch.long, device=device)
    od_edge_weight = torch.tensor(od_w, dtype=torch.float32, device=device)

    node_features_t = torch.tensor(node_features, dtype=torch.float32, device=device)
    return node_features_t, spatial_edge_index, spatial_edge_weight, od_edge_index, od_edge_weight
