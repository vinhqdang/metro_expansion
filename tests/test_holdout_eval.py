"""Synthetic checks for scripts/run_holdout.py: holdout metrics on the real Xi'an Line 2, demand ratio, null generation.

Uses the real city data under external/ (like test_fast_env.py); the module-scoped fixtures load it once.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_holdout", REPO_ROOT / "scripts" / "run_holdout.py")
rh = importlib.util.module_from_spec(_spec)
sys.modules["run_holdout"] = rh  # dataclasses needs the module registered
_spec.loader.exec_module(rh)


@pytest.fixture(scope="module")
def setting():
    return rh.load_setting()


@pytest.fixture(scope="module")
def real():
    return rh.real_line2_cells()


@pytest.fixture(scope="module")
def d_ref(setting, real):
    rp = rh.replay(setting, real)
    assert rp["feasible"], rp.get("reason")
    return rh.demand(setting, rp["segments"])


def test_real_line2_is_feasible_and_serves_about_0292(d_ref, real):
    assert len(real) == 27
    assert d_ref == pytest.approx(0.0292, abs=1e-4)


def test_generated_equal_to_real_gives_f1_one_and_demand_ratio_one(setting, real, d_ref):
    rp = rh.replay(setting, real)
    m = rh.line_metrics(rp["cells"], real, d_gen=rh.demand(setting, rp["segments"]), d_ref=d_ref)
    assert m["f1_r1"] == pytest.approx(1.0)
    assert m["f1_r0"] == pytest.approx(1.0)
    assert m["f1_r2"] == pytest.approx(1.0)
    assert m["demand_ratio"] == pytest.approx(1.0)
    assert m["mean_sym_dist"] == pytest.approx(0.0)
    assert m["hausdorff"] == pytest.approx(0.0)


def test_shift_by_one_cell_gives_f1_at_radius_one_and_less_at_zero(real):
    shifted = [(r, c + 1) for r, c in real]
    m = rh.line_metrics(shifted, real)
    assert m["f1_r1"] == pytest.approx(1.0)
    assert m["f1_r0"] < 1.0
    assert m["hausdorff"] == pytest.approx(1.0)


def test_null_generation_works_for_20_lines(setting, real, d_ref):
    lines = rh.generate_null(20, 0, setting=setting, real_cells=real, d_ref=d_ref)
    assert len(lines) == 20
    for x in lines:
        assert set(rh.LINE_METRICS) <= set(x)
        assert isinstance(x["dead_end"], bool)
        assert 1 <= x["n_cells"] <= 27
        assert (not x["dead_end"]) == (x["n_cells"] == 27)
        assert "cells" not in x
        assert np.isfinite(x["demand_ratio"])
    again = rh.generate_null(20, 0, setting=setting, real_cells=real, d_ref=d_ref)
    assert [x["start"] for x in lines] == [x["start"] for x in again]


def test_cells_to_actions_rejects_non_adjacent_cells():
    with pytest.raises(ValueError):
        rh.cells_to_actions([(0, 0), (0, 2)])
    assert rh.cells_to_actions([(0, 0), (1, 1), (1, 2)]) == [3, 2]
