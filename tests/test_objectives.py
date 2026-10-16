import numpy as np
import pytest

from src.eval.objectives import (
    CityContext,
    a_new,
    compute_objectives,
    cov_uniform,
    cv,
    d_tot,
    ggi,
    gini_or_nan,
    group_metrics,
    min_max_ratio,
)


def test_ggi_all_zero_is_zero():
    assert ggi([0, 0, 0, 0, 0]) == 0.0


def test_ggi_equal_values_return_that_value():
    assert ggi([0.3] * 5) == pytest.approx(0.3)


def test_ggi_hand_computed():
    # sorted [0,0,0,1,1]; weights 1, .5, .25, .125, .0625 normalised by 1.9375
    expected = (0.125 + 0.0625) / 1.9375
    assert ggi([1, 0, 1, 0, 0]) == pytest.approx(expected)


def test_ggi_two_groups_base_two():
    # sorted [0,1]; weights [1, 0.5] / 1.5 -> 0.5 / 1.5
    assert ggi([1.0, 0.0]) == pytest.approx(1.0 / 3.0)


def test_gini_nan_at_zero_service():
    assert np.isnan(gini_or_nan([0.0, 0.0, 0.0]))


def test_gini_equal_is_zero():
    assert gini_or_nan([0.4, 0.4, 0.4]) == pytest.approx(0.0)


def test_gini_two_point_extreme():
    assert gini_or_nan([0.0, 1.0]) == pytest.approx(0.5)


def test_gini_four_point_extreme():
    assert gini_or_nan([0.0, 0.0, 0.0, 1.0]) == pytest.approx(0.75)


def test_min_max_ratio_nan_at_zero_and_value_otherwise():
    assert np.isnan(min_max_ratio([0.0, 0.0]))
    assert min_max_ratio([1.0, 2.0, 4.0]) == pytest.approx(0.25)


def test_cv_hand_computed_and_nan_at_zero():
    assert cv([1.0, 3.0]) == pytest.approx(0.5)
    assert np.isnan(cv([0.0, 0.0]))


def test_group_metrics_summary():
    m = group_metrics([0.2, 0.4])
    assert m["s_min"] == pytest.approx(0.2)
    assert m["argmin_group"] == 0
    assert m["D_sumpct"] == pytest.approx(0.6)
    assert m["minmax_ratio"] == pytest.approx(0.5)
    assert m["ggi2"] == pytest.approx(0.2 * (1 / 1.5) + 0.4 * (0.5 / 1.5))


def test_d_tot_counts_both_directions_once():
    od = np.zeros((3, 3))
    od[0, 1], od[1, 0], od[1, 2], od[2, 1] = 2.0, 3.0, 5.0, 7.0
    total = od.sum()
    # segment (0,1) covers directed pairs (0,1) and (1,0)
    assert d_tot([(0, 1)], od) == pytest.approx(5.0 / total)
    # duplicates and both listed orientations collapse to the same set
    assert d_tot([(0, 1), (1, 0), (0, 1)], od) == pytest.approx(5.0 / total)
    # (1,2) covers directed pairs (1,2) and (2,1): 5 + 7 = 12
    assert d_tot([(1, 2)], od) == pytest.approx(12.0 / total)
    # (0,1) and (1,2) together cover every positive entry
    assert d_tot([(0, 1), (1, 2)], od) == pytest.approx(1.0)


def test_d_tot_zero_when_no_segments():
    od = np.ones((2, 2))
    assert d_tot([], od) == 0.0


def test_a_new_hand_computed():
    # 5x5 grid, existing line in column 0, new line in column 4, K=1, uniform trip mass
    grid = (5, 5)
    existing = np.zeros(grid, dtype=bool)
    existing[:, 0] = True
    mu = np.ones(grid)
    new_cells = [(r, 4) for r in range(5)]
    # E = columns 0..1; outside E there are 15 cells (cols 2..4); reach of new line = cols 3..4 -> 10 cells
    assert a_new(mu, existing, new_cells, grid, K=1) == pytest.approx(10.0 / 15.0)


def test_a_new_zero_when_no_mass_outside_existing_reach():
    grid = (3, 3)
    existing = np.ones(grid, dtype=bool)
    assert a_new(np.ones(grid), existing, [(0, 0)], grid, K=1) == 0.0


def test_cov_uniform_hand_computed():
    grid = (5, 5)
    # column 2 with K=1 covers columns 1..3 -> 15 of 25 cells
    assert cov_uniform([(r, 2) for r in range(5)], grid, K=1) == pytest.approx(0.6)


def test_compute_objectives_keys_and_group_service():
    grid = (5, 5)
    od = np.zeros((4, 4))
    od[0, 1] = 1.0
    od[2, 3] = 3.0
    mu = np.ones(grid)
    ctx = CityContext(
        od_raw=od,
        mu=mu,
        existing_mask=np.zeros(grid, dtype=bool),
        group_od_sum=np.array([10.0, 20.0]),
        total_od=float(od.sum()),
        grid_shape=grid,
    )
    out = compute_objectives(ctx, [(0, 0), (0, 1)], [(0, 1)], np.array([0.5, 0.25]))
    assert out["D_tot"] == pytest.approx(1.0 / 4.0)
    assert out["s_min"] == pytest.approx(0.25)
    assert out["argmin_group"] == 1
    np.testing.assert_allclose(out["a_g"], [5.0, 5.0])
    for key in ("ggi2", "gini_ratio", "cv", "D_sumpct", "A_new", "cov_uniform"):
        assert key in out
