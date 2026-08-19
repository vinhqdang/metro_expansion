"""Unit tests for the pure-logic helpers in scripts/train_xian.py.

Does not exercise the environment/training loop (which needs the real
Xi'an data under external/, gitignored and not guaranteed present) -- only
the standalone functions that don't depend on it.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from train_xian import coverage_metric, gini  # noqa: E402


def test_gini_is_zero_for_perfectly_equal_distribution():
    assert gini(np.array([5.0, 5.0, 5.0, 5.0])) == pytest.approx(0.0, abs=1e-9)


def test_gini_is_zero_for_all_zero_vector():
    # Degenerate case: no demand satisfied for any group yet.
    assert gini(np.zeros(5)) == pytest.approx(0.0)


def test_gini_increases_with_inequality():
    equal = gini(np.array([1.0, 1.0, 1.0, 1.0]))
    unequal = gini(np.array([10.0, 0.0, 0.0, 0.0]))
    assert unequal > equal


def test_gini_is_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.exponential(size=5)
        g = gini(x)
        assert 0.0 <= g <= 1.0


def make_toy_city(grid_x=4, grid_y=4):
    """Minimal stand-in for motndp.city.City exposing only what
    coverage_metric needs (grid_x_size, grid_y_size)."""
    return SimpleNamespace(grid_x_size=grid_x, grid_y_size=grid_y)


def test_coverage_metric_full_coverage_when_k_spans_whole_grid():
    city = make_toy_city(4, 4)
    covered = coverage_metric(city, [[0, 0]], k=10)
    assert covered == pytest.approx(1.0)


def test_coverage_metric_partial_coverage_from_single_station():
    city = make_toy_city(5, 5)
    # A single station at the center with k=1 (Chebyshev) covers a 3x3 block = 9/25 cells.
    covered = coverage_metric(city, [[2, 2]], k=1)
    assert covered == pytest.approx(9 / 25)


def test_coverage_metric_increases_with_more_stations():
    city = make_toy_city(6, 6)
    one_station = coverage_metric(city, [[0, 0]], k=1)
    two_stations = coverage_metric(city, [[0, 0], [5, 5]], k=1)
    assert two_stations >= one_station


def test_coverage_metric_increases_with_larger_k():
    city = make_toy_city(6, 6)
    small_k = coverage_metric(city, [[3, 3]], k=1)
    large_k = coverage_metric(city, [[3, 3]], k=3)
    assert large_k > small_k
