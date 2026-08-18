import numpy as np
import pytest

from src.rl.reward import Objectives, sample_weight_vector, tchebycheff_reward, tchebycheff_scalarize


def test_sample_weight_vector_sums_to_one():
    rng = np.random.default_rng(0)
    weights = sample_weight_vector(rng, n_objectives=3)
    assert weights.shape == (3,)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)


def test_sample_weight_vector_covers_simplex_over_many_draws():
    rng = np.random.default_rng(0)
    draws = np.stack([sample_weight_vector(rng, 3) for _ in range(2000)])
    # each objective's weight should sometimes dominate, sometimes be near zero
    assert draws.min(axis=0).max() < 0.05
    assert draws.max(axis=0).min() > 0.5


def test_tchebycheff_scalarize_perfect_solution_is_near_zero():
    objectives = Objectives(1.0, 1.0, 1.0)
    weights = np.array([1 / 3, 1 / 3, 1 / 3])
    ideal = np.ones(3)
    value = tchebycheff_scalarize(objectives, weights, ideal, epsilon=0.0)
    assert value == pytest.approx(0.0, abs=1e-9)


def test_tchebycheff_scalarize_penalizes_worst_objective_most():
    # demand is far from ideal, others are close -- the max term should be
    # driven by demand under equal weighting.
    objectives = Objectives(demand_satisfaction=0.1, social_equity=0.9, radiation_accessibility=0.9)
    weights = np.array([1 / 3, 1 / 3, 1 / 3])
    ideal = np.ones(3)
    value = tchebycheff_scalarize(objectives, weights, ideal, epsilon=0.0)
    expected_max_term = (1 / 3) * 0.9  # |0.1 - 1.0| * weight
    assert value == pytest.approx(expected_max_term, abs=1e-9)


def test_tchebycheff_scalarize_rejects_mismatched_weight_sum():
    objectives = Objectives(0.5, 0.5, 0.5)
    bad_weights = np.array([0.5, 0.5, 0.5])  # sums to 1.5
    with pytest.raises(ValueError, match="sum to 1.0"):
        tchebycheff_scalarize(objectives, bad_weights, np.ones(3))


def test_tchebycheff_reward_is_negative_of_scalarization():
    objectives = Objectives(0.4, 0.6, 0.5)
    weights = np.array([0.2, 0.3, 0.5])
    reward = tchebycheff_reward(objectives, weights, epsilon=0.0)
    scalarized = tchebycheff_scalarize(objectives, weights, np.ones(3), epsilon=0.0)
    assert reward == pytest.approx(-scalarized)


def test_tchebycheff_reward_prefers_more_balanced_solution():
    # Under equal weights, a balanced (0.7, 0.7, 0.7) solution should score
    # higher reward than an unbalanced (1.0, 0.4, 0.7) solution with the same
    # sum, illustrating Tchebycheff's bias against neglecting any objective.
    weights = np.array([1 / 3, 1 / 3, 1 / 3])
    balanced = Objectives(0.7, 0.7, 0.7)
    unbalanced = Objectives(1.0, 0.4, 0.7)
    assert tchebycheff_reward(balanced, weights) > tchebycheff_reward(unbalanced, weights)
