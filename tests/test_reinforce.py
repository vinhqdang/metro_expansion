"""Smoke tests for the v2 RL trainer: end-to-end JSON, determinism, weights change, RLOO and graph equivalence."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMP_NUM_THREADS", "1")
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.eval.normalization import Norm  # noqa: E402
from src.eval.record import SCHEMA  # noqa: E402
from src.rl.reinforce import RunConfig, rloo, run, setup, train  # noqa: E402

# f_ref from the provisional normalisation (max over random lines); only the scale matters for the smoke test.
NORM = Norm(f_ref=np.array([0.0275, 0.0112, 0.398]), z=1.05, rho=0.05)
SMOKE = dict(city="xian", variant="full", groups="price_groups_5.txt", chained=True, nr_stations=20,
             episodes=32, batch_w=4, batch_k=4, hidden_dim=8, calib_episodes=20, eval_every=1, best_of=2)
REQUIRED_TOP = ["schema", "city", "variant", "groups", "chained", "nr_stations", "method", "transfer", "seed",
                "budget_rollouts", "eval_rollouts", "config", "normalization", "env", "policy_front",
                "archive_nd", "archive_best", "curve", "indicators"]


def test_rloo_advantages_sum_to_zero_within_each_weight_group():
    rng = np.random.default_rng(0)
    M, K = 4, 4
    for _ in range(50):
        R = rng.normal(size=M * K)
        A = np.concatenate([rloo(R[m * K:(m + 1) * K]) for m in range(M)])
        for m in range(M):
            assert abs(A[m * K:(m + 1) * K].sum()) < 1e-12
    # leave-one-out definition: A_i = R_i - mean of the other K-1 returns
    r = np.array([1.0, 2.0, 4.0, 7.0])
    expected = np.array([r[i] - np.delete(r, i).mean() for i in range(4)])
    np.testing.assert_allclose(rloo(r), expected)


def test_graph_matches_train_xian_on_xian():
    from src.envs.fast_motndp import build_city
    from src.rl.graph import build_graph
    import train_xian  # scripts/train_xian.py

    city = build_city("xian")
    g_new = build_graph(city)
    g_old = train_xian.build_graph(city, torch.device("cpu"))
    for a, b in zip(g_new, g_old):
        assert torch.equal(a, b)


def test_smoke_run_writes_schema_and_is_deterministic(tmp_path):
    cfg_a = RunConfig(seed=0, **SMOKE)
    out_a = run(cfg_a, NORM, log=lambda *a, **k: None)
    cfg_b = RunConfig(seed=0, **SMOKE)
    out_b = run(cfg_b, NORM, log=lambda *a, **k: None)
    # determinism: identical policy front (greedy and best-of-k), curve and archive
    assert json.dumps(out_a["policy_front"], sort_keys=True) == json.dumps(out_b["policy_front"], sort_keys=True)
    assert out_a["curve"] == out_b["curve"]
    assert out_a["archive_nd"] == out_b["archive_nd"]
    assert out_a["budget_rollouts"] == 32
    assert len(out_a["policy_front"]) == 45  # 15 greedy + 15 best-of-k + 15 greedy-from-top-8-starts


def test_cli_json_has_schema_fields(tmp_path):
    import run_rl

    norm_path = tmp_path / "norm.json"
    from src.eval.normalization import save_norm

    save_norm(NORM, norm_path)
    out_path = tmp_path / "rl.json"
    rec = run_rl.main([
        "--city", "xian", "--episodes", "32", "--hidden_dim", "8", "--seed", "0", "--norm", str(norm_path),
        "--out", str(out_path), "--best_of", "2", "--eval_every", "1", "--quiet",
    ])
    data = json.loads(out_path.read_text())
    assert data["schema"] == SCHEMA == "mnep.run.v1"
    for key in REQUIRED_TOP:
        assert key in data, key
    assert data["method"] == "rl_full"
    assert data["transfer"] is None
    np.testing.assert_allclose(data["normalization"]["f_ref"], NORM.f_ref, rtol=0, atol=1e-12)
    assert data["env"]["omp_threads"] == 1
    assert "hv" in data["indicators"]["policy"] and "eum" in data["indicators"]["policy"]
    pt = data["policy_front"][0]
    for key in ("w", "mode", "start", "cells", "obj", "f_norm", "g_w"):
        assert key in pt
    for key in ("D_tot", "ggi2", "A_new", "s", "s_min", "argmin_group", "minmax_ratio", "gini_ratio"):
        assert key in pt["obj"]
    assert len(data["curve"]) >= 2 and data["curve"][0]["rollouts"] == 0
    assert rec["method"] == "rl_full"


def test_weights_change_during_training():
    cfg = RunConfig(seed=3, **SMOKE)
    S = setup(cfg, NORM)
    before = {n: p.detach().clone() for n, p in S.agent.named_parameters()}
    train(S, log=lambda *a, **k: None)
    changed = {n for n, p in S.agent.named_parameters() if not torch.equal(before[n], p.detach())}
    # the encoder input projection, the GNN layers, the heads and the w-embedding all receive gradient
    assert any(n.startswith("encoder.input_proj") for n in changed)
    assert any(n.startswith("policy.") for n in changed)
    assert any(n.startswith("start_head.") for n in changed)
    assert any(n.startswith("weight_embed") for n in changed)
    assert len(changed) == len(before)
