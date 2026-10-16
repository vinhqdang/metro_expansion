import numpy as np
import pytest

from src.eval.holdout import (
    corridor_scores,
    hausdorff,
    mean_sym_distance,
    orientation_error,
    percentile_vs_null,
)

G_LINE = [(0, c) for c in range(5)]


def test_corridor_scores_toy_line_r1():
    L = [(0, 0), (0, 1), (0, 2), (1, 3), (3, 4)]
    s = corridor_scores(L, G_LINE, r=1)
    assert s["precision"] == pytest.approx(0.8)
    assert s["recall"] == pytest.approx(1.0)
    assert s["f1"] == pytest.approx(2 * 0.8 * 1.0 / 1.8)


def test_corridor_scores_toy_line_r0():
    L = [(0, 0), (0, 1), (0, 2), (1, 3), (3, 4)]
    s = corridor_scores(L, G_LINE, r=0)
    assert s["precision"] == pytest.approx(0.6)
    assert s["recall"] == pytest.approx(0.6)
    assert s["f1"] == pytest.approx(0.6)


def test_corridor_scores_empty_generated_line():
    s = corridor_scores([], G_LINE, r=1)
    assert s["f1"] == 0.0


def test_mean_sym_distance_and_hausdorff_hand_computed():
    L = [(0, 0), (0, 1), (0, 2), (1, 3), (3, 4)]
    # L->G nearest: 0,0,0,1,3 (mean 0.8); G->L nearest: 0,0,0,1,1 (mean 0.4)
    assert mean_sym_distance(L, G_LINE) == pytest.approx(0.6)
    assert hausdorff(L, G_LINE) == pytest.approx(3.0)


def test_orientation_error_parallel_lines_zero():
    a = [(0, c) for c in range(5)]
    b = [(7, c) for c in range(3, 9)]
    assert orientation_error(a, b) == pytest.approx(0.0, abs=1e-9)


def test_orientation_error_perpendicular_lines_ninety():
    horizontal = [(0, c) for c in range(5)]
    vertical = [(r, 2) for r in range(5)]
    assert orientation_error(horizontal, vertical) == pytest.approx(90.0)


def test_orientation_error_diagonal_is_45():
    horizontal = [(0, c) for c in range(5)]
    diagonal = [(i, i) for i in range(5)]
    assert orientation_error(horizontal, diagonal) == pytest.approx(45.0)


def test_orientation_error_degenerate_is_nan():
    assert np.isnan(orientation_error([(1, 1)], G_LINE))


def test_percentile_vs_null_hand_computed():
    null = [1.0, 2.0, 3.0, 4.0]
    assert percentile_vs_null(3.0, null) == pytest.approx(62.5)
    assert percentile_vs_null(10.0, null) == pytest.approx(100.0)
    assert percentile_vs_null(-1.0, null) == pytest.approx(0.0)
