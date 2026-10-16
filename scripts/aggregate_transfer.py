#!/usr/bin/env python3.13
"""Paired transfer contrasts against scratch, written to results/summary/transfer_<pair>.json.

Input:  results/transfer/<src>_to_<tgt>/<condition>_<budget>/seed<S>.json   (schema mnep.run.v1)
Output: results/summary/transfer_<src>_to_<tgt>.json                          (schema mnep.transfer.v1)

For every budget, each treatment condition is compared with scratch on the same seeds
(paired by seed) with wilcoxon_hl_boot, for two outcomes:
- final: indicators.policy.eum of the run
- auc:   area under the EUM learning curve (curve[].eum over curve[].rollouts, trapezoid rule)
         divided by the rollout span, i.e. the mean EUM over the training budget
Holm correction runs over all tests of one (pair, budget) family (5 conditions x 2 outcomes).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aggregate import ALPHA, BOOT_SEED, N_BOOT, REPO, SEED_RE, summarise, write_json  # noqa: E402
from src.eval.stats import holm, wilcoxon_hl_boot  # noqa: E402

RUN_SCHEMA = "mnep.run.v1"
TRANSFER_SCHEMA = "mnep.transfer.v1"
SCRATCH = "scratch"
CONDITIONS = ("scratch", "full", "encoder_only", "heads_only", "full_frozen_enc", "random_frozen_enc")
TREATMENTS = CONDITIONS[1:]
BUDGETS = ("full", "reduced")
DIR_RE = re.compile(r"^(.+)_(full|reduced)$")
OUTCOMES = ("final_eum", "auc_eum")


def auc_eum(curve: list[dict[str, Any]]) -> float:
    """Mean EUM over the rollout span (trapezoid area divided by the span)."""
    x = np.array([c["rollouts"] for c in curve], dtype=float)
    y = np.array([c["eum"] for c in curve], dtype=float)
    span = x[-1] - x[0]
    if x.size < 2 or span <= 0:
        raise ValueError("learning curve needs at least two points with increasing rollouts")
    area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return area / span


def load_pair(pair_dir: Path) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    """{budget: {condition: {seed: {'final_eum', 'auc_eum'}}}} for one <src>_to_<tgt> directory."""
    data: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for cdir in sorted(p for p in pair_dir.iterdir() if p.is_dir()):
        m = DIR_RE.match(cdir.name)
        if not m:
            continue
        cond, budget = m.group(1), m.group(2)
        if cond not in CONDITIONS:
            raise ValueError(f"{cdir}: unknown condition {cond!r}")
        for f in sorted(cdir.iterdir()):
            sm = SEED_RE.match(f.name)
            if not sm:
                continue
            seed = int(sm.group(1))
            rec = json.loads(f.read_text())
            if rec.get("schema") != RUN_SCHEMA:
                raise ValueError(f"{f}: schema {rec.get('schema')!r}, expected {RUN_SCHEMA}")
            if rec.get("seed") != seed:
                raise ValueError(f"{f}: record seed {rec.get('seed')} does not match file name")
            tr = rec.get("transfer")
            if cond == SCRATCH:
                if tr not in (None, {}) and tr.get("condition") not in (None, SCRATCH):
                    raise ValueError(f"{f}: scratch directory holds condition {tr.get('condition')!r}")
            elif not tr or tr.get("condition") != cond:
                raise ValueError(f"{f}: transfer record {tr!r} does not match directory condition {cond!r}")
            data.setdefault(budget, {}).setdefault(cond, {})[seed] = {
                "final_eum": float(rec["indicators"]["policy"]["eum"]),
                "auc_eum": auc_eum(rec["curve"]),
            }
    return data


def budget_block(cells: dict[str, dict[int, dict[str, float]]]) -> dict[str, Any] | None:
    """Summary and paired tests for one (pair, budget); None if there is no scratch baseline."""
    if SCRATCH not in cells:
        return None
    scratch = cells[SCRATCH]
    conditions: dict[str, Any] = {}
    tests: list[dict[str, Any]] = []
    for cond in CONDITIONS:
        if cond not in cells:
            continue
        seeds = sorted(set(cells[cond]) & set(scratch))
        entry: dict[str, Any] = {
            "n": len(cells[cond]),
            "n_paired": len(seeds),
            "final_eum": summarise(cells[cond][s]["final_eum"] for s in cells[cond]),
            "auc_eum": summarise(cells[cond][s]["auc_eum"] for s in cells[cond]),
        }
        conditions[cond] = entry
        if cond == SCRATCH or not seeds:
            continue
        for outcome in OUTCOMES:
            x = np.array([cells[cond][s][outcome] for s in seeds], dtype=float)
            y = np.array([scratch[s][outcome] for s in seeds], dtype=float)
            res = wilcoxon_hl_boot(x, y, n_boot=N_BOOT, seed=BOOT_SEED)
            tests.append({
                "condition": cond,
                "outcome": outcome,
                "n_pairs": res["n_pairs"],
                "hl_shift": res["hl_shift"],
                "ci95": [res["ci_low"], res["ci_high"]],
                "p": res["p_wilcoxon"],
                "p_holm": None,
            })
    if tests:
        for t, adj in zip(tests, holm([t["p"] for t in tests])):
            t["p_holm"] = adj
            t["significant"] = bool(adj < ALPHA)
    return {"n_scratch": len(scratch), "family_size": len(tests), "conditions": conditions, "tests": tests}


def build_pair_summary(pair: str, pair_dir: Path) -> dict[str, Any]:
    data = load_pair(pair_dir)
    budgets: dict[str, Any] = {}
    for budget in BUDGETS:
        if budget not in data:
            continue
        block = budget_block(data[budget])
        if block is None:
            print(f"warning: {pair}/{budget}: no scratch runs, skipped", file=sys.stderr)
            continue
        budgets[budget] = block
    return {"schema": TRANSFER_SCHEMA, "pair": pair, "alpha": ALPHA,
            "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED}, "budgets": budgets}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=REPO / "results", help="results root")
    args = ap.parse_args(argv)

    tdir = args.results / "transfer"
    if not tdir.is_dir():
        raise FileNotFoundError(f"{tdir} does not exist")
    pairs = sorted(p for p in tdir.iterdir() if p.is_dir())
    if not pairs:
        raise FileNotFoundError(f"no pair directories under {tdir}")
    for pdir in pairs:
        summ = build_pair_summary(pdir.name, pdir)
        out = args.results / "summary" / f"transfer_{pdir.name}.json"
        write_json(out, summ)
        print(f"wrote {out}")
        for b, blk in summ["budgets"].items():
            for t in blk["tests"]:
                print(f"  {b:8s} {t['condition']:18s} {t['outcome']:9s} n={t['n_pairs']} "
                      f"p={t['p']:.4f} p_holm={t['p_holm']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
