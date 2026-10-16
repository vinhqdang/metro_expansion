"""End-to-end checks of scripts/aggregate.py, scripts/aggregate_transfer.py and scripts/make_tables.py
on a small synthetic results tree (schema mnep.run.v1), plus a pdflatex compile of the output."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
for p in (str(SCRIPTS), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import aggregate as agg  # noqa: E402
from src.eval.normalization import Norm  # noqa: E402
from src.eval.pareto import igd_plus, nondominated  # noqa: E402
from src.eval.stats import holm  # noqa: E402
from src.eval.record import front_indicators  # noqa: E402
from src.eval.weights import lattice  # noqa: E402

NORM_F = [0.05, 0.02, 0.5]
W = lattice(4, 3)
CITY_METHODS = {  # method -> (seeds, shift, single_point)
    "rl_full": (range(4), 0.10, False),
    "nsga2": (range(4), 0.05, False),
    "ga_scalar_matched": (range(4), 0.00, False),
    "greedy": (range(4), -0.10, True),
    "tabular_eff": (range(2), 0.00, True),
}
MACRO_RE = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")


def _norm() -> Norm:
    return Norm(f_ref=np.array(NORM_F), z=1.05, rho=0.05)


def _point(rng, norm, w, mode, shift, gini_na=False):
    s = rng.uniform(0.005, 0.05, 5) * (1 + shift)
    obj = {
        "D_tot": float(rng.uniform(0.6, 1.0) * 0.05 * (1 + shift)),
        "ggi2": float(rng.uniform(0.5, 1.0) * 0.02 * (1 + shift)),
        "A_new": float(rng.uniform(0.6, 1.0) * 0.5 * (1 + shift)),
        "s_min": float(s.min()),
        "gini_ratio": None if gini_na else float(rng.uniform(0.1, 0.3)),
        "minmax_ratio": float(rng.uniform(0.2, 0.8)),
        "s": s.tolist(),
    }
    f_norm = norm.normalize(obj)
    return {
        "w": None if w is None else [float(v) for v in w],
        "mode": mode,
        "start": [int(v) for v in rng.integers(0, 29, 2)],
        "cells": rng.integers(0, 29, (20, 2)).tolist(),
        "obj": obj,
        "f_norm": f_norm.tolist(),
        "g_w": None if w is None else float(norm.g(f_norm, w)),
    }


def _run_record(city, variant, method, seed, shift, single, rng, norm, first_gini_na=False):
    pts = []
    if single:
        pts.append(_point(rng, norm, None, "greedy", shift))
    else:
        for i, w in enumerate(W):
            pts.append(_point(rng, norm, w, "greedy", shift, gini_na=(first_gini_na and i == 0)))
        for w in W[:3]:  # non-deployable sampled points must not enter the deployable front
            pts.append(_point(rng, norm, w, "best_of_32", shift + 0.2))
    arc = np.clip(rng.uniform(0.2, 1.1, (80, 3)) * (1 + shift), 0.0, 1.2)
    arc_nd = arc[nondominated(arc)]
    F_pol = np.array([p["f_norm"] for p in pts if p["mode"] == "greedy"])
    ind = {"policy": front_indicators(F_pol, norm), "archive": front_indicators(arc_nd, norm)}
    curve = [
        {"rollouts": r, "eum": float(rng.uniform(-0.5, -0.1)), "hv": float(rng.uniform(0.0, 0.5))}
        for r in (0, 400, 800)
    ]
    return {
        "city": city,
        "variant": variant,
        "groups": "price_groups_5.txt",
        "chained": True,
        "nr_stations": 20,
        "method": method,
        "transfer": None,
        "seed": seed,
        "budget_rollouts": 10000,
        "eval_rollouts": 100,
        "config": {},
        "normalization": {"file": "synthetic", "f_ref": NORM_F, "train_f_ref": NORM_F, "z": 1.05, "rho": 0.05},
        "env": {"git_commit": "synthetic", "torch": "none", "omp_threads": 1,
                "wall_clock_s": float(rng.uniform(100, 200))},
        "policy_front": pts,
        "archive_nd": arc_nd.tolist(),
        "archive_best": {},
        "curve": curve,
        "indicators": ind,
        "schema": "mnep.run.v1",
    }


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic") / "results"
    norm = _norm()
    for mi, (method, (seeds, shift, single)) in enumerate(CITY_METHODS.items()):
        for seed in seeds:
            rng = np.random.default_rng(1000 * mi + seed)
            rec = _run_record("xian", "full", method, seed, shift, single, rng, norm,
                              first_gini_na=(method == "rl_full" and seed == 0))
            _write(root / "xian_full" / method / f"seed{seed}.json", rec)
    # a directory without seed files (as anchor_cache is) must be ignored
    _write(root / "xian_full" / "anchor_cache" / "ggi2_seed100.json", {"anchor": 1.0})

    # transfer tree: scratch and two treatments, both budgets, 4 paired seeds
    for budget in ("full", "reduced"):
        for cond in ("scratch", "full", "encoder_only", "random_frozen_enc"):
            for seed in range(4):
                rng = np.random.default_rng(seed + 7 * (cond == "full") + 13 * (cond == "random_frozen_enc"))
                eum_final = float(rng.uniform(-0.2, -0.1)) + (0.03 if cond == "full" else 0.0)
                curve = [
                    {"rollouts": r, "eum": float(rng.uniform(-0.5, -0.2)), "hv": 0.1}
                    for r in (0, 400, 800)
                ]
                rec = {
                    "city": "amsterdam", "variant": "full", "groups": "g", "chained": True,
                    "nr_stations": 20, "method": "rl_full", "seed": seed,
                    "transfer": None if cond == "scratch" else {
                        "condition": cond, "source_city": "xian", "source_seed": seed, "budget": budget},
                    "budget_rollouts": 800, "eval_rollouts": 10, "config": {},
                    "normalization": {"file": "synthetic", "f_ref": NORM_F, "train_f_ref": NORM_F,
                                      "z": 1.05, "rho": 0.05},
                    "env": {"wall_clock_s": 10.0}, "policy_front": [], "archive_nd": [],
                    "archive_best": {}, "curve": curve,
                    "indicators": {"policy": {"hv": 0.5, "eum": eum_final, "n_nondominated": 3}},
                    "schema": "mnep.run.v1",
                }
                _write(root / "transfer" / "xian_to_amsterdam" / f"{cond}_{budget}" / f"seed{seed}.json", rec)
    return root


def _run(script: str, *args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                          capture_output=True, text=True, cwd=str(REPO), timeout=900)


@pytest.fixture(scope="module")
def summary(tree):
    res = _run("aggregate.py", "--city_variant", "xian_full", "--results", str(tree))
    assert res.returncode == 0, res.stderr
    return json.loads((tree / "summary" / "xian_full_main.json").read_text())


def test_igd_chunked_matches_pareto_igd():
    rng = np.random.default_rng(3)
    P = rng.uniform(0, 1, (30, 3))
    R = rng.uniform(0, 1, (50, 3))
    assert agg.igd_plus_chunked(P, R, chunk=7) == pytest.approx(igd_plus(P, R), rel=1e-12)


def test_summary_schema_and_methods(summary):
    assert summary["schema"] == "mnep.summary.v1"
    assert set(summary["methods"]) == set(CITY_METHODS)
    assert summary["methods"]["rl_full"]["n_seeds"] == 4
    assert summary["methods"]["tabular_eff"]["n_seeds"] == 2
    assert summary["methods"]["greedy"]["single_point"] is True
    assert summary["methods"]["rl_full"]["single_point"] is False
    assert summary["methods"]["rl_full"]["label"] == "RL (ours)"


def test_igd_plus_is_filled_for_every_method(summary):
    for m, s in summary["methods"].items():
        for key in ("policy_igd_plus", "archive_igd_plus"):
            st = s["metrics"][key]
            assert st["n"] == s["n_seeds"], (m, key)
            assert st["mean"] is not None and np.isfinite(st["mean"]), (m, key)
            assert st["ci95"][0] <= st["mean"] <= st["ci95"][1]
    assert summary["reference"]["size"] > 0


def test_policy_eum_uses_greedy_front_only(tree, summary):
    norm = _norm()
    vals = []
    for f in sorted((tree / "xian_full" / "rl_full").glob("seed*.json")):
        rec = json.loads(f.read_text())
        F = np.array([p["f_norm"] for p in rec["policy_front"] if p["mode"] == "greedy"])
        vals.append(front_indicators(F, norm)["eum"])
    assert summary["methods"]["rl_full"]["metrics"]["policy_eum"]["mean"] == pytest.approx(np.mean(vals))


def test_undefined_gini_is_counted_not_zeroed(summary):
    assert summary["methods"]["rl_full"]["n_na"] == 1
    assert summary["methods"]["nsga2"]["n_na"] == 0


def test_mean_over_front_and_vertices_are_available(summary):
    rl = summary["methods"]["rl_full"]["objectives"]
    assert rl["vertex_demand"]["available"] is True
    assert rl["vertex_demand"]["n"] == 4
    assert rl["mean_all"]["available"] is True
    assert rl["mean_all"]["D_tot"] is not None
    assert summary["weight_settings"][3]["on_lattice"] is None
    greedy = summary["methods"]["greedy"]["objectives"]
    assert greedy["vertex_access"]["available"] is True


def test_distinct_lines_per_run(summary):
    # synthetic fronts have 15 greedy points with distinct random cells
    assert summary["methods"]["rl_full"]["distinct_lines_mean"] == pytest.approx(15.0)


def test_tests_holm_and_untested_small_methods(summary):
    fams = summary["tests_vs"]["families"]
    assert set(fams) == {"archive_hv", "policy_eum"}
    for fam, block in fams.items():
        rows = {r["method"]: r for r in block["rows"]}
        assert "rl_full" not in rows
        assert rows["tabular_eff"]["tested"] is False
        assert rows["tabular_eff"]["p"] is None
        tested = [r for r in block["rows"] if r["tested"]]
        assert {r["method"] for r in tested} == {"nsga2", "ga_scalar_matched", "greedy"}
        for r in tested:
            assert r["p_holm"] >= r["p"] - 1e-12
            assert 0.0 <= r["a12"] <= 1.0
            assert r["diff_ci95"][0] <= r["mean_diff"] <= r["diff_ci95"][1]
        adj = holm([r["p"] for r in tested])
        for r, a in zip(tested, adj):
            assert r["p_holm"] == pytest.approx(a)


def test_transfer_summary_and_holm(tree):
    res = _run("aggregate_transfer.py", "--results", str(tree))
    assert res.returncode == 0, res.stderr
    summ = json.loads((tree / "summary" / "transfer_xian_to_amsterdam.json").read_text())
    assert summ["schema"] == "mnep.transfer.v1"
    assert set(summ["budgets"]) == {"full", "reduced"}
    blk = summ["budgets"]["full"]
    assert blk["family_size"] == 6  # 3 treatments x 2 outcomes
    for t in blk["tests"]:
        assert t["n_pairs"] == 4
        assert t["p_holm"] >= t["p"] - 1e-12
    assert "scratch" in blk["conditions"]


def test_make_tables_and_numbers(tree, summary, tmp_path):
    out = tmp_path / "generated"
    res = _run("make_tables.py", "--summary", str(tree / "summary"), "--out", str(out))
    assert res.returncode == 0, res.stderr
    for name in ("tab_main_xian_full.tex", "tab_objectives_xian_full.tex",
                 "tab_tests_xian_full.tex", "tab_transfer_xian_to_amsterdam.tex", "numbers.tex"):
        assert (out / name).is_file(), name

    lines = [ln for ln in (out / "numbers.tex").read_text().splitlines() if ln.startswith("\\newcommand")]
    names = []
    for ln in lines:
        m = MACRO_RE.match(ln)
        assert m, ln
        names.append(m.group(1))
        assert re.fullmatch(r"[A-Za-z]+", m.group(1))
    assert len(names) == len(set(names))
    assert "XianRLEum" in names
    assert "XianRLvsNSGAPolicyEumP" in names
    assert "XianRLvsTabDemandPolicyEumP" not in names  # two seeds: not tested
    assert "XianToAmsterdamFullBudgetFullFinalDelta" in names

    main = (out / "tab_main_xian_full.tex").read_text()
    assert "\\toprule" in main and "\\bottomrule" in main
    assert "|" not in re.sub(r"\\begin\{tabular\}\{[^}]*\}", "", main)  # no vertical rules
    assert "RL (ours)" in main
    tests = (out / "tab_tests_xian_full.tex").read_text()
    assert "not tested" in tests


def test_generated_tex_compiles(tree, summary, tmp_path):
    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex not available")
    out = tmp_path / "generated"
    res = _run("make_tables.py", "--summary", str(tree / "summary"), "--out", str(out))
    assert res.returncode == 0, res.stderr
    build = tmp_path / "build"
    build.mkdir()
    for f in out.iterdir():
        shutil.copy(f, build / f.name)
    doc = (
        "\\documentclass{article}\n\\usepackage{booktabs}\n\\usepackage{amsmath}\n"
        "\\begin{document}\n\\input{numbers.tex}\n"
        "\\input{tab_main_xian_full.tex}\n\\input{tab_objectives_xian_full.tex}\n"
        "\\input{tab_tests_xian_full.tex}\n\\input{tab_transfer_xian_to_amsterdam.tex}\n"
        "\\end{document}\n"
    )
    (build / "check.tex").write_text(doc)
    proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "check.tex"],
                          cwd=build, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert (build / "check.pdf").is_file()
