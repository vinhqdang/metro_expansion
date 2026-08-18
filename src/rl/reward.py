"""Multi-objective reward scalarization for rail network expansion.

Implements Tchebycheff decomposition over the three MNEP objectives used by
Zhang et al. (Transportation Research Part C, 2024): OD demand satisfaction,
social equity, and radiation accessibility. Tchebycheff scalarization is
preferred over a fixed weighted sum because it can reach non-convex regions
of the Pareto front -- a weighted sum cannot recover a Pareto-optimal point
that lies in a "dent" of the objective space, but Tchebycheff can.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Objectives:
    """One step's (or one episode's) raw objective values, all in [0, 1] after
    normalization by the caller. Higher is better for all three."""

    demand_satisfaction: float
    social_equity: float
    radiation_accessibility: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.demand_satisfaction, self.social_equity, self.radiation_accessibility],
            dtype=np.float64,
        )


def sample_weight_vector(rng: np.random.Generator, n_objectives: int = 3) -> np.ndarray:
    """Sample a weight vector uniformly from the probability simplex.

    Used to draw a fresh Tchebycheff weight vector each training episode
    (Algorithm 1 in the manuscript), so a single policy learns a family of
    trade-offs rather than one fixed weighting.
    """
    # Uniform simplex sampling via normalized exponential (Dirichlet(1,...,1)).
    weights = rng.exponential(scale=1.0, size=n_objectives)
    return weights / weights.sum()


def tchebycheff_scalarize(
    objectives: Objectives,
    weights: np.ndarray,
    ideal_point: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Scalarize a multi-objective reward via (augmented) Tchebycheff decomposition.

    g(x | w, z*) = max_i [ w_i * |f_i(x) - z*_i| ] + epsilon * sum_i [ w_i * |f_i(x) - z*_i| ]

    Since all objectives here are to be *maximized* and are normalized to
    [0, 1], the ideal point z* is typically (1, 1, 1) (or the best value seen
    so far per objective, if tracked adaptively). Minimizing this scalarized
    value is equivalent to maximizing all three objectives jointly; callers
    that want a reward signal to maximize should negate the returned value.

    The small augmentation term (epsilon > 0) avoids weakly-Pareto-optimal
    solutions that a pure Tchebycheff term alone can admit.
    """
    if weights.shape != objectives.as_array().shape:
        raise ValueError(
            f"weights shape {weights.shape} does not match objectives shape "
            f"{objectives.as_array().shape}"
        )
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"weights must sum to 1.0, got {weights.sum()}")

    diffs = np.abs(objectives.as_array() - ideal_point)
    weighted_diffs = weights * diffs
    return float(weighted_diffs.max() + epsilon * weighted_diffs.sum())


def tchebycheff_reward(
    objectives: Objectives,
    weights: np.ndarray,
    ideal_point: np.ndarray | None = None,
    epsilon: float = 1e-6,
) -> float:
    """Reward signal to *maximize*: negative of the Tchebycheff scalarization.

    This is the quantity actually fed to the actor-critic update in
    Algorithm 1 (Section: Cross-city curriculum and sample efficiency).
    """
    if ideal_point is None:
        ideal_point = np.ones(3, dtype=np.float64)
    return -tchebycheff_scalarize(objectives, weights, ideal_point, epsilon)
