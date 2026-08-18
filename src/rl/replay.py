"""Prioritized experience replay buffer.

Used by the cross-city curriculum training loop (Algorithm 1 in the
manuscript) as one of the two sample-efficiency mechanisms, alongside
cross-city weight transfer. Priorities follow proportional prioritization
(Schaul et al., 2016): priority = (|TD error| + eps) ** alpha, sampling
probability proportional to priority, with importance-sampling weights to
correct the resulting sampling bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Transition:
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool


class PrioritizedReplayBuffer:
    """Fixed-capacity replay buffer with proportional prioritization.

    Not thread-safe; intended for single-process training loops.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        eps: float = 1e-5,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")

        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps

        self._transitions: list[Transition | None] = [None] * capacity
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._write_index = 0
        self._size = 0
        self._max_priority = 1.0  # new transitions get max priority so they're sampled at least once

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        self._transitions[self._write_index] = transition
        self._priorities[self._write_index] = self._max_priority
        self._write_index = (self._write_index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, beta: float, rng: np.random.Generator
    ) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """Sample a prioritized batch.

        Returns (transitions, indices, importance_sampling_weights). Callers
        must call `update_priorities(indices, td_errors)` after computing TD
        errors for the sampled batch.
        """
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        if batch_size > self._size:
            raise ValueError(f"batch_size {batch_size} exceeds buffer size {self._size}")

        priorities = self._priorities[: self._size] ** self.alpha
        probs = priorities / priorities.sum()

        indices = rng.choice(self._size, size=batch_size, replace=False, p=probs)
        transitions = [self._transitions[i] for i in indices]

        # Importance-sampling weights correct for the non-uniform sampling
        # distribution; normalized by the max weight in the batch for stability.
        weights = (self._size * probs[indices]) ** (-beta)
        weights /= weights.max()

        return transitions, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        new_priorities = np.abs(td_errors) + self.eps
        self._priorities[indices] = new_priorities
        self._max_priority = max(self._max_priority, float(new_priorities.max()))
