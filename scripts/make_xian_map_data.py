"""Reconstruct real grid coordinates for the Xi'an network map figure.

Reuses the exact same city/env construction as scripts/train_xian.py and
replays a logged greedy-eval action sequence (compass directions 0-7) from
its actual deterministic starting location (env.reset(seed=42)) to recover
real (x, y) grid coordinates -- no fabricated data, only decoding what the
trained policy already produced and this repository already logged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "tabular-tndp"))
sys.path.insert(0, str(REPO_ROOT / "external" / "tabular-tndp" / "envs" / "mo-tndp"))

import numpy as np
import mo_gymnasium as mo_gym
from motndp.city import City
from motndp.constraints import MetroConstraints

import envs  # noqa: F401  (registers motndp_xian-v0)

ACTION_TO_DIRECTION = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])

# The full model's greedy uniform-weight rollout at seed 42 (Table 3/4's
# checkpoint), from xian_original_seeds_refresh.log.
FULL_SEED42_ACTIONS = [2, 2, 3, 4, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


def main():
    city_path = REPO_ROOT / "external" / "tabular-tndp" / "envs" / "mo-tndp" / "cities" / "xian"
    city = City(city_path, groups_file="price_groups_5.txt")
    env = mo_gym.make(
        "motndp_xian-v0",
        city=city,
        constraints=MetroConstraints(city),
        nr_stations=20,
        od_type="pct",
        chained_reward=False,
    )
    obs, info = env.reset(seed=42)
    start_idx = int(info["location_grid_index"])
    start_xy = city.index_to_grid(np.array([start_idx]))[0]

    path = [start_xy.tolist()]
    x, y = int(start_xy[0]), int(start_xy[1])
    for a in FULL_SEED42_ACTIONS:
        dx, dy = ACTION_TO_DIRECTION[a]
        x, y = x + int(dx), y + int(dy)
        path.append([x, y])

    existing_lines = []
    for line in city.existing_lines:
        idxs = np.asarray(line).flatten().astype(int)
        coords = city.index_to_grid(idxs)
        existing_lines.append(coords.tolist())

    out = {
        "grid_x_size": city.grid_x_size,
        "grid_y_size": city.grid_y_size,
        "existing_lines": existing_lines,
        "trained_path": path,
    }
    out_path = REPO_ROOT / "manuscript" / "figures" / "xian_map_data.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print(f"start_idx={start_idx} start_xy={start_xy.tolist()}")
    print(f"path length: {len(path)}")


if __name__ == "__main__":
    main()
