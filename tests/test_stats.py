import numpy as np
import pytest

from src.eval.stats import holm, mwu_a12_boot, wilcoxon_hl_boot


def test_holm_hand_computed():
    adj = holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one_and_keeps_order():
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])


def test_wilcoxon_all_negative_exact_p():
    x = np.arange(10, dtype=float)
    y = x + 1.0 + np.linspace(0.0, 0.5, 10)
    res = wilcoxon_hl_boot(x, y, n_boot=500, seed=1)
    assert res["p_wilcoxon"] == pytest.approx(2 / 2**10)
    assert res["ci_low"] < res["hl_shift"] < res["ci_high"]


def test_wilcoxon_hodges_lehmann_hand_computed():
    x = np.array([1.0, 2.0, 3.0])
    y = np.zeros(3)
    res = wilcoxon_hl_boot(x, y, n_boot=200, seed=0)
    # Walsh averages of [1,2,3]: 1,1.5,2,2,2.5,3 -> median 2
    assert res["hl_shift"] == pytest.approx(2.0)


def test_wilcoxon_reproducible_with_seed():
    x = np.array([0.1, 0.4, 0.2, 0.9, 0.5])
    y = np.array([0.2, 0.1, 0.3, 0.5, 0.4])
    a = wilcoxon_hl_boot(x, y, n_boot=300, seed=7)
    b = wilcoxon_hl_boot(x, y, n_boot=300, seed=7)
    assert a == b


def test_a12_extremes_and_ties():
    assert mwu_a12_boot(np.array([1.0, 2, 3]), np.array([4.0, 5, 6]), n_boot=100)["a12"] == pytest.approx(0.0)
    assert mwu_a12_boot(np.array([4.0, 5, 6]), np.array([1.0, 2, 3]), n_boot=100)["a12"] == pytest.approx(1.0)
    assert mwu_a12_boot(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]), n_boot=100)["a12"] == pytest.approx(0.5)


def test_mwu_exact_p_small_samples():
    res = mwu_a12_boot(np.array([1.0, 2, 3, 4]), np.array([5.0, 6, 7, 8]), n_boot=100)
    # exact two-sided p = 2 / C(8,4) = 2/70
    assert res["p_mwu"] == pytest.approx(2 / 70)
    assert res["mwu_method"] == "exact"


def test_mwu_bootstrap_ci_brackets_mean_difference():
    x = np.arange(10, dtype=float)
    y = x + 3.0
    res = mwu_a12_boot(x, y, n_boot=2000, seed=0)
    assert res["mean_diff"] == pytest.approx(-3.0)
    assert res["ci_low"] <= res["mean_diff"] <= res["ci_high"]
