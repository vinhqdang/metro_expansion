"""Access to the unmodified upstream code in external/tabular-tndp.

The upstream files are imported, never edited. Two upstream issues are handled here:

* GATNDP.run() and QLearningTNDP.train() call wandb unconditionally, so wandb is
  initialised in disabled mode first (documented in docs/baseline_reproduction.md).
* QLearningTNDP.test() reads the state from the raw observation instead of
  info['location_grid_index'] (documented bug). PatchedQLearningTNDP overrides test()
  with the train()-consistent state and adds a greedy rollout that returns the line.
"""

from __future__ import annotations

import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = REPO_ROOT / "external" / "tabular-tndp"

if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

import ga_tndp  # noqa: E402
import wandb  # noqa: E402
from ga_tndp import GATNDP  # noqa: E402
from qlearning_tndp import QLearningTNDP  # noqa: E402

# GATNDP.mutate_path calls int() on a one-element array. NumPy 2.5 raises TypeError there
# (older NumPy only warned), so the GA crashes on its first mutation. PatchedGATNDP takes the
# upstream source of mutate_path, applies this one textual fix (asserted to match exactly once)
# and binds the result. The fix is value-preserving: int(np.ravel(x)[0]) == int(x) for size-1 x.
_MUTATE_BAD = "current_location = int(self.env.unwrapped.city.grid_to_index(np.array(current_location_grid) + direction))"
_MUTATE_FIXED = (
    "current_location = int(np.ravel(self.env.unwrapped.city.grid_to_index("
    "np.array(current_location_grid) + direction))[0])"
)


def init_wandb_disabled() -> None:
    """Fresh no-op wandb session so the upstream wandb calls succeed without network access.

    A new session per upstream run: wandb.config refuses to overwrite a key (reward_type) that a
    previous run in the same process already set.
    """
    if wandb.run is not None:
        wandb.finish()
    wandb.init(mode="disabled")


def _patched_mutate_path():
    src = textwrap.dedent(inspect.getsource(GATNDP.mutate_path))
    if src.count(_MUTATE_BAD) != 1:
        raise RuntimeError("upstream mutate_path changed; re-check the NumPy compatibility patch")
    ns = dict(vars(ga_tndp))
    exec(compile(src.replace(_MUTATE_BAD, _MUTATE_FIXED), "ga_tndp.mutate_path(patched)", "exec"), ns)
    return ns["mutate_path"]


class PatchedGATNDP(GATNDP):
    """GATNDP with the NumPy-2.5 compatibility fix in mutate_path (see module docstring); otherwise upstream."""

    mutate_path = _patched_mutate_path()


class PatchedQLearningTNDP(QLearningTNDP):
    """QLearningTNDP with test() fixed to read info['location_grid_index'] (see module docstring)."""

    def greedy_rollout(self, start, reward_type: str) -> dict:
        """Greedy rollout of the learned Q-table from `start`; returns the finished line."""
        _, info = self.env.reset(options={"loc": (int(start[0]), int(start[1]))})
        state = info["location_grid_index"]
        s = np.zeros(self.nr_groups)
        ret = 0.0
        actions: list[int] = []
        while True:
            action = int(np.argmax(np.where(info["action_mask"], self.Q[state, :], -np.inf)))
            _, reward, done, _, info = self.env.step(action)
            state = info["location_grid_index"]
            s += np.asarray(reward, dtype=float)
            ret += self.calculate_reward(reward, reward_type)
            actions.append(action)
            if done:
                break
        return {
            "start": [int(start[0]), int(start[1])],
            "actions": actions,
            "cells": info["covered_cells_coordinates"],
            "segments": info["segments"],
            "s": s,
            "return": float(ret),
        }

    def test(self, test_episodes, reward_type, starting_loc=None, policy=None):
        """Patched test(): greedy rollouts from the learned best start (no logging, returns the lines)."""
        if starting_loc:
            start = starting_loc
        else:
            start = np.unravel_index(self.Q_start.argmax(), self.Q_start.shape)
        return [self.greedy_rollout(start, reward_type) for _ in range(test_episodes)]
