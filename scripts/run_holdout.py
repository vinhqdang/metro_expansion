"""Line-1-only holdout: generated Line-2 corridors against the real Xi'an Line 2.

Inputs
  * final run records of the Xi'an line1_only variant: <records_dir>/<method>/seed*.json
  * the real Line 2 = existing_lines_full[1] of the FULL Xi'an city (build_city('xian', 'full'))
  * the line1_only city (Line 1 kept, Line 2 corridor free) with the chained reward (27 stations)

Per record three lines are compared with the real Line 2:
  (a) vertex:   deployable point at w = (1, 0, 0)
  (b) centroid: w = (1/3, 1/3, 1/3) is not on the H = 4 lattice, so the record holds no point for it.
                The centroid line is the deployable lattice line (greedy / search point) that minimises
                g(f | centroid) over the record's own policy front. This is an approximation.
  (c) archive_best_demand: archive_best.D_tot.cells, when the record stores it.

Metrics: corridor precision / recall / F1 at r in {0, 1, 2} (headline r = 1), mean symmetric
distance, Hausdorff, orientation error (degrees), and the demand ratio
D_tot(generated) / D_tot(real Line 2), both evaluated in the line1_only city with the chained reward.
Generated lines are re-stepped through the env so that D_tot comes from the same code path as the
reference.

Null: 10,000 random feasible 27-cell lines in the line1_only city (uniform random start, uniform
choice among the env's allowed actions). Dead-end lines are kept and flagged. The null is cached in
--null_out (per-line metrics only, no cells).

Outputs
  --out   results/summary/xian_line1_holdout.json
  --tex   manuscript/generated/tab_holdout.tex

Example:
  OMP_NUM_THREADS=1 python3.13 scripts/run_holdout.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.envs.context import build_context  # noqa: E402
from src.envs.fast_motndp import build_city, make_env  # noqa: E402
from src.eval.holdout import (  # noqa: E402
    corridor_scores,
    hausdorff,
    mean_sym_distance,
    orientation_error,
    percentile_vs_null,
)
from src.eval.normalization import Norm, load_norm  # noqa: E402
from src.eval.objectives import a_new, d_tot, ggi  # noqa: E402

CITY = "xian"
VARIANT = "line1_only"
NR_STATIONS = 27
CHAINED = True
RADII = (0, 1, 2)
HEADLINE_R = 1
REAL_LINE_INDEX = 1  # existing_lines_full[1] of the full city
W_VERTEX = (1.0, 0.0, 0.0)
W_CENTROID = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
# Modes whose lines are one deterministic deployable policy output. best_of_32 (best of 32 sampled
# rollouts) and greedy_top8_starts (extra start cells) are excluded from the vertex/centroid choice.
DEPLOYABLE_MODES = (
    "greedy", "final_pop_best", "greedy_by_g", "random_min_g", "greedy_learned_start", "upstream_best_reward",
)
METHOD_ORDER = (
    "rl_full", "nsga2", "ga_scalar_matched", "ga_demand", "ga_ggi2",
    "tabular_eff", "tabular_ggi2", "greedy", "random",
)
METHOD_LABEL = {
    "rl_full": "RL (full)", "nsga2": "NSGA-II", "ga_scalar_matched": "Matched scalar GA",
    "ga_demand": "GA (demand)", "ga_ggi2": "GA (GGI2)", "tabular_eff": "Tabular (eff.)",
    "tabular_ggi2": "Tabular (GGI2)", "greedy": "Greedy", "random": "Random",
}
SELECTIONS = ("vertex", "centroid", "archive_best_demand")
SELECTION_LABEL = {"vertex": "(a) $w{=}(1,0,0)$", "centroid": "(b) centroid", "archive_best_demand": "(c) archive best $D$"}
LINE_METRICS = (
    "precision_r0", "recall_r0", "f1_r0", "precision_r1", "recall_r1", "f1_r1",
    "precision_r2", "recall_r2", "f1_r2", "mean_sym_dist", "hausdorff", "orientation_deg",
    "d_tot", "demand_ratio",
)
NULL_SCHEMA = "mnep.holdout_null.v1"
SUMMARY_SCHEMA = "mnep.holdout_summary.v1"
ACTION_TO_DIRECTION = np.array(
    [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
)


# ----------------------------------------------------------------------------- setting


@dataclass
class Setting:
    city: Any  # line1_only City
    env: Any  # unwrapped FastMOTNDP (chained, 27 stations, MetroConstraints)
    ctx: Any  # CityContext of the line1_only city


def load_setting(nr_stations: int = NR_STATIONS) -> Setting:
    city = build_city(CITY, VARIANT)
    ctx = build_context(city)
    env = make_env(city, nr_stations, CHAINED).unwrapped
    return Setting(city=city, env=env, ctx=ctx)


def cells_from_indices(indices: Sequence[int], grid_y: int) -> list[tuple[int, int]]:
    return [divmod(int(i), grid_y) for i in np.asarray(indices).reshape(-1)]


def real_line2_cells() -> list[tuple[int, int]]:
    """Real Line 2 as (row, col) cells: existing_lines_full[1] of the full Xi'an city (City stores flat indices)."""
    full = build_city(CITY, "full")
    return cells_from_indices(full.existing_lines_full[REAL_LINE_INDEX], full.grid_y_size)


def cells_to_actions(cells: Sequence[tuple[int, int]]) -> list[int]:
    acts = []
    for a, b in zip(cells[:-1], cells[1:]):
        d = np.array([b[0] - a[0], b[1] - a[1]])
        hit = np.flatnonzero((ACTION_TO_DIRECTION == d).all(axis=1))
        if hit.size != 1:
            raise ValueError(f"cells {tuple(a)} -> {tuple(b)} are not adjacent")
        acts.append(int(hit[0]))
    return acts


def replay(setting: Setting, cells: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Step the env through a line given as cells. Returns feasibility, covered cells, segments and s."""
    env = setting.env
    cells_t = [(int(r), int(c)) for r, c in cells]
    try:
        acts = cells_to_actions(cells_t)
    except ValueError as exc:
        return {"feasible": False, "reason": str(exc)}
    _, info = env.reset(options={"loc": cells_t[0]})
    s = np.zeros(int(setting.city.group_od_mx.shape[0]))
    for k, a in enumerate(acts):
        if not info["action_mask"][a]:
            return {"feasible": False, "reason": f"action {a} not allowed at step {k + 1}"}
        _, r, terminated, _, info = env.step(a)
        s += np.asarray(r, dtype=float)
        if terminated and k != len(acts) - 1:
            return {"feasible": False, "reason": f"terminated early at step {k + 1}"}
    covered = [tuple(int(v) for v in c) for c in info["covered_cells_coordinates"]]
    return {"feasible": True, "cells": covered, "segments": info["segments"], "s": s}


def demand(setting: Setting, segments) -> float:
    return float(d_tot(segments, setting.ctx.od_raw, setting.ctx.total_od))


# ----------------------------------------------------------------------------- metrics


def line_metrics(L: Sequence[tuple[int, int]], G: Sequence[tuple[int, int]],
                 d_gen: float | None = None, d_ref: float | None = None) -> dict[str, float]:
    out: dict[str, float] = {"n_cells": float(len(L))}
    for r in RADII:
        sc = corridor_scores(L, G, r=r)
        out[f"precision_r{r}"] = sc["precision"]
        out[f"recall_r{r}"] = sc["recall"]
        out[f"f1_r{r}"] = sc["f1"]
    out["mean_sym_dist"] = mean_sym_distance(L, G)
    out["hausdorff"] = hausdorff(L, G)
    out["orientation_deg"] = orientation_error(L, G)
    if d_gen is not None:
        out["d_tot"] = float(d_gen)
        out["demand_ratio"] = float(d_gen / d_ref) if d_ref else float("nan")
    return out


def _json_float(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _clean(x: Any) -> Any:
    """Recursively replace NaN / inf with None so the JSON is strict."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ----------------------------------------------------------------------------- null


def generate_null(n_lines: int, seed: int, *, setting: Setting, real_cells, d_ref: float,
                  nr_stations: int = NR_STATIONS, progress=None) -> list[dict[str, Any]]:
    """Random feasible lines: uniform start cell, then uniform choice among the env's allowed actions.

    A line that stops before nr_stations cells (no allowed action left) is kept with dead_end=True.
    Returns per-line metrics only (no cells).
    """
    rng = np.random.default_rng(seed)
    env, city = setting.env, setting.city
    gy, N = city.grid_y_size, city.grid_size
    lines: list[dict[str, Any]] = []
    for i in range(n_lines):
        start = int(rng.integers(N))
        _, info = env.reset(options={"loc": divmod(start, gy)})
        terminated = False
        while not terminated:
            choices = np.flatnonzero(info["action_mask"])
            if choices.size == 0:
                break
            _, _, terminated, _, info = env.step(int(rng.choice(choices)))
        cells = [tuple(int(v) for v in c) for c in info["covered_cells_coordinates"]]
        m = line_metrics(cells, real_cells, d_gen=demand(setting, info["segments"]), d_ref=d_ref)
        r0, c0 = divmod(start, gy)
        rec: dict[str, Any] = {"start": [r0, c0], "dead_end": bool(len(cells) < nr_stations)}
        rec.update(m)
        lines.append(rec)
        if progress and (i + 1) % 2000 == 0:
            progress(f"null {i + 1}/{n_lines}")
    return lines


def build_null_payload(lines: list[dict[str, Any]], *, n_lines: int, seed: int, nr_stations: int,
                       d_ref: float) -> dict[str, Any]:
    return {
        "schema": NULL_SCHEMA,
        "city": CITY,
        "variant": VARIANT,
        "nr_stations": nr_stations,
        "chained": CHAINED,
        "seed": seed,
        "n_lines": n_lines,
        "n_dead_end": int(sum(1 for x in lines if x["dead_end"])),
        "reference": {"name": "real_line2", "n_cells": nr_stations, "d_tot": d_ref},
        "generator": "uniform random start; uniform choice among env action_mask",
        "lines": lines,
        "git_commit": git_commit(),
    }


def load_or_make_null(path: Path, n_lines: int, seed: int, setting: Setting, real_cells, d_ref: float,
                      regen: bool, log) -> dict[str, Any]:
    if path.exists() and not regen:
        payload = json.loads(path.read_text())
        if (payload.get("schema") == NULL_SCHEMA and payload.get("n_lines") == n_lines
                and payload.get("seed") == seed and payload.get("nr_stations") == NR_STATIONS):
            log(f"null: using cached {path}")
            return payload
        log("null: cached file does not match the requested parameters, regenerating")
    t0 = time.time()
    lines = generate_null(n_lines, seed, setting=setting, real_cells=real_cells, d_ref=d_ref,
                          progress=log)
    payload = build_null_payload(lines, n_lines=n_lines, seed=seed, nr_stations=NR_STATIONS, d_ref=d_ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), separators=(",", ":")))
    log(f"null: wrote {path} ({n_lines} lines, {payload['n_dead_end']} dead ends) in {time.time() - t0:.1f}s")
    return payload


def null_values(null_lines: list[dict[str, Any]], key: str, full_length_only: bool = False) -> np.ndarray:
    vals = [x.get(key, float("nan")) for x in null_lines
            if not (full_length_only and x["dead_end"])]
    return np.asarray([float("nan") if v is None else float(v) for v in vals], dtype=float)


# ----------------------------------------------------------------------------- records


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_records(records_dir: Path, log) -> dict[str, dict[int, dict[str, Any]]]:
    """{method: {seed: record}} for <records_dir>/<method>/seed<S>.json with matching city/variant/stations."""
    out: dict[str, dict[int, dict[str, Any]]] = {}
    if not records_dir.exists():
        log(f"records: {records_dir} does not exist")
        return out
    for path in sorted(records_dir.glob("*/seed*.json")):
        m = re.fullmatch(r"seed(\d+)\.json", path.name)
        if not m:
            continue
        rec = json.loads(path.read_text())
        if rec.get("city") != CITY or rec.get("variant") != VARIANT or rec.get("nr_stations") != NR_STATIONS:
            log(f"records: skip {path} (city/variant/nr_stations mismatch)")
            continue
        out.setdefault(path.parent.name, {})[int(m.group(1))] = rec
    return out


def select_lines(rec: dict[str, Any], norm_fallback: Norm) -> dict[str, dict[str, Any]]:
    """Candidate lines (a), (b), (c) from one record. Each entry carries its cells and the point mode."""
    pf = rec.get("policy_front") or []
    deployable = [p for p in pf if p.get("mode") in DEPLOYABLE_MODES]
    if not deployable:
        deployable = [p for p in pf if p.get("w") is not None]
    out: dict[str, dict[str, Any]] = {}

    hits = [p for p in deployable if p.get("w") is not None and np.allclose(p["w"], W_VERTEX, atol=1e-9)]
    if hits:
        out["vertex"] = {"cells": hits[0]["cells"], "mode": hits[0]["mode"], "w": hits[0]["w"], "source": "policy_front"}
    elif len(deployable) == 1 and deployable[0].get("w") is None:
        out["vertex"] = {"cells": deployable[0]["cells"], "mode": deployable[0]["mode"], "w": None,
                         "source": "single_point"}

    weighted = [p for p in deployable if p.get("w") is not None and p.get("f_norm") is not None]
    if weighted:
        g = [float(norm_fallback.g(np.asarray(p["f_norm"], dtype=float), W_CENTROID)) for p in weighted]
        best = weighted[int(np.argmin(g))]
        out["centroid"] = {"cells": best["cells"], "mode": best["mode"], "w": best["w"],
                           "source": "policy_front_argmin_g_centroid"}
    elif len(deployable) == 1 and deployable[0].get("w") is None:
        out["centroid"] = {"cells": deployable[0]["cells"], "mode": deployable[0]["mode"], "w": None,
                           "source": "single_point"}

    ab = (rec.get("archive_best") or {}).get("D_tot") or {}
    if ab.get("cells"):
        out["archive_best_demand"] = {"cells": ab["cells"], "mode": "archive_best_D_tot", "w": None,
                                      "source": "archive_best",
                                      "stored_d_tot": (ab.get("obj") or {}).get("D_tot")}
    return out


def evaluate_record(setting: Setting, rec: dict[str, Any], real_cells, d_ref: float,
                    norm_fallback: Norm) -> dict[str, dict[str, Any]]:
    """Metrics for each selection of one record. Infeasible replays give NaN metrics and a flag."""
    res: dict[str, dict[str, Any]] = {}
    for key, cand in select_lines(rec, norm_fallback).items():
        rp = replay(setting, cand["cells"])
        entry: dict[str, Any] = {k: v for k, v in cand.items() if k != "cells"}
        entry["n_cells"] = len(cand["cells"])
        if not rp["feasible"]:
            entry.update({"feasible": False, "reason": rp["reason"]})
            entry.update({k: float("nan") for k in LINE_METRICS})
            res[key] = entry
            continue
        d_gen = demand(setting, rp["segments"])
        entry["feasible"] = True
        entry.update(line_metrics(rp["cells"], real_cells, d_gen=d_gen, d_ref=d_ref))
        if cand.get("stored_d_tot") is not None:
            entry["d_tot_stored_abs_diff"] = abs(float(cand["stored_d_tot"]) - d_gen)
        res[key] = entry
    return res


def aggregate(per_seed: dict[int, dict[str, Any]], null_lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and sd over seeds of each metric, and the percentile of the mean within the null."""
    seeds = sorted(per_seed)
    mean, sd, pct, pct_full = {}, {}, {}, {}
    for key in LINE_METRICS:
        vals = np.asarray([per_seed[s].get(key, float("nan")) for s in seeds], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            mean[key] = sd[key] = pct[key] = pct_full[key] = float("nan")
            continue
        mean[key] = float(vals.mean())
        sd[key] = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
        pct[key] = percentile_vs_null(mean[key], null_values(null_lines, key))
        pct_full[key] = percentile_vs_null(mean[key], null_values(null_lines, key, full_length_only=True))
    return {"n_seeds": len(seeds), "per_seed": {str(s): per_seed[s] for s in seeds}, "mean": mean, "sd": sd,
            "percentile_of_mean": pct, "percentile_of_mean_full_length_null": pct_full}


def reference_block(setting: Setting, real_cells, d_ref: float, norm: Norm | None) -> dict[str, Any]:
    ctx = setting.ctx
    s_ref = replay(setting, real_cells)["s"]
    ggi2 = ggi(s_ref, base=2.0)
    a_ref = a_new(ctx.mu, ctx.existing_mask, real_cells, ctx.grid_shape, K=2)
    raw = np.array([d_ref, ggi2, a_ref])
    f_norm = norm.normalize(raw).tolist() if norm is not None else None
    self_metrics = line_metrics(real_cells, real_cells, d_gen=d_ref, d_ref=d_ref)
    return {
        "name": "real_line2",
        "description": "existing_lines_full[1] of the full Xi'an city, scored in the line1_only city",
        "n_cells": len(real_cells),
        "d_tot": d_ref,
        "s": [float(x) for x in s_ref],
        "ggi2": float(ggi2),
        "A_new": float(a_ref),
        "f_norm": f_norm,
        "metrics_vs_itself": {k: _json_float(v) for k, v in self_metrics.items()},
    }


# ----------------------------------------------------------------------------- output


def _fmt(x: float, nd: int = 2) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def write_tex(path: Path, summary: dict[str, Any]) -> None:
    ref = summary["reference"]
    cols = ["P@1", "R@1", "F1@1", "F1@0", "F1@2", "MSD", "Haus.", "Orient.", "$D$ ratio", "Pct F1@1"]
    lines = [
        "% Generated by scripts/run_holdout.py. Do not edit by hand.",
        "% MSD: mean symmetric distance (cells); Haus.: Hausdorff (cells); Orient.: degrees.",
        "% Pct F1@1: percentile of the mean over seeds within the random null (higher F1 is better;",
        "% for MSD, Haus. and Orient. lower is better, and the null percentile is printed in the JSON).",
        "\\begin{tabular}{llr" + "r" * len(cols) + "}",
        "\\toprule",
        "Method & Point & Seeds & " + " & ".join(cols) + " \\\\",
        "\\midrule",
    ]
    ref_row = (
        ["Real Line 2", "reference", "--", "1.00", "1.00", "1.00", "1.00", "1.00",
         "0.00", "0.00", "0.00", "1.00", "--"]
    )
    lines.append(" & ".join(ref_row) + " \\\\")
    lines.append(f"\\multicolumn{{{len(cols) + 3}}}{{l}}{{Real Line 2: demand $D_{{\\mathrm{{tot}}}}={_fmt(ref['d_tot'], 4)}$"
                 + (f", $f_{{\\mathrm{{norm}}}}=({', '.join(_fmt(v, 3) for v in ref['f_norm'])})$" if ref.get("f_norm") else "")
                 + "} \\\\")
    for m in summary["method_order"]:
        block = summary["methods"].get(m)
        if not block:
            continue
        lines.append("\\midrule")
        label = METHOD_LABEL.get(m, m.replace("_", "\\_"))
        first = True
        for sel in SELECTIONS:
            agg = block["selections"].get(sel)
            if not agg:
                continue
            mu, pc = agg["mean"], agg["percentile_of_mean"]
            vals = [
                _fmt(mu["precision_r1"]), _fmt(mu["recall_r1"]), _fmt(mu["f1_r1"]), _fmt(mu["f1_r0"]),
                _fmt(mu["f1_r2"]), _fmt(mu["mean_sym_dist"]), _fmt(mu["hausdorff"], 1),
                _fmt(mu["orientation_deg"], 1), _fmt(mu["demand_ratio"]), _fmt(pc["f1_r1"], 1),
            ]
            lines.append(" & ".join([label if first else "", SELECTION_LABEL[sel], str(agg["n_seeds"])] + vals) + " \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records_dir", default="results/xian_line1_only")
    ap.add_argument("--norm", default="results/xian_line1_only/normalization.json")
    ap.add_argument("--null_out", default="results/xian_line1_only/null_lines.json")
    ap.add_argument("--n_null", type=int, default=10_000)
    ap.add_argument("--null_seed", type=int, default=2026)
    ap.add_argument("--regen_null", action="store_true")
    ap.add_argument("--out", default="results/summary/xian_line1_holdout.json")
    ap.add_argument("--tex", default="manuscript/generated/tab_holdout.tex")
    args = ap.parse_args(argv)

    def P(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else REPO_ROOT / p

    def log(msg: str) -> None:
        print(msg, flush=True)

    t0 = time.time()
    setting = load_setting()
    real = real_line2_cells()
    rp_real = replay(setting, real)
    if not rp_real["feasible"]:
        raise RuntimeError(f"real Line 2 is not feasible in the line1_only city: {rp_real['reason']}")
    d_ref = demand(setting, rp_real["segments"])
    log(f"real Line 2: {len(real)} cells, D_tot (line1_only, chained) = {d_ref:.6f}")

    norm_path = P(args.norm)
    norm = load_norm(norm_path) if norm_path.exists() else None
    if norm is None:
        log(f"normalisation file {norm_path} not found: f_norm of the reference is omitted")

    null_payload = load_or_make_null(P(args.null_out), args.n_null, args.null_seed, setting, real, d_ref,
                                     args.regen_null, log)
    null_lines = null_payload["lines"]

    records = load_records(P(args.records_dir), log)
    norm_fallback = norm or Norm()
    methods: dict[str, Any] = {}
    for method in sorted(records, key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, m)):
        per_sel: dict[str, dict[int, dict[str, Any]]] = {sel: {} for sel in SELECTIONS}
        modes: dict[str, dict[str, Any]] = {}
        for seed, rec in sorted(records[method].items()):
            ev = evaluate_record(setting, rec, real, d_ref, norm_fallback)
            for sel, entry in ev.items():
                per_sel[sel][seed] = entry
                modes.setdefault(sel, {"source": entry.get("source"), "mode": entry.get("mode"),
                                       "w": entry.get("w")})
        selections = {}
        for sel in SELECTIONS:
            if per_sel[sel]:
                agg = aggregate(per_sel[sel], null_lines)
                agg["point"] = modes.get(sel)
                agg["n_infeasible"] = int(sum(1 for e in per_sel[sel].values() if not e.get("feasible")))
                selections[sel] = agg
        methods[method] = {"seeds": sorted(records[method]), "selections": selections}
        log(f"{method}: seeds {sorted(records[method])}, selections {sorted(selections)}")
    if not methods:
        log("no run records found under the records directory; the summary holds the reference and the null only")

    summary = {
        "schema": SUMMARY_SCHEMA,
        "city": CITY,
        "variant": VARIANT,
        "nr_stations": NR_STATIONS,
        "chained": CHAINED,
        "headline_r": HEADLINE_R,
        "radii": list(RADII),
        "null": {
            "file": str(P(args.null_out).relative_to(REPO_ROOT)) if P(args.null_out).is_relative_to(REPO_ROOT) else str(P(args.null_out)),
            "n_lines": int(len(null_lines)),
            "seed": int(null_payload["seed"]),
            "n_dead_end": int(null_payload["n_dead_end"]),
            "quantiles": {
                k: {q: float(np.nanquantile(null_values(null_lines, k), p)) for q, p in
                    (("p05", 0.05), ("p50", 0.5), ("p95", 0.95))}
                for k in LINE_METRICS
            },
        },
        "reference": reference_block(setting, real, d_ref, norm),
        "method_order": [m for m in METHOD_ORDER if m in methods] + [m for m in methods if m not in METHOD_ORDER],
        "methods": methods,
        "notes": [
            "Real Line 2 is scored in the line1_only city (chained reward, 27 stations).",
            "Centroid weight (1/3,1/3,1/3) is not on the H=4 lattice; the centroid line is the deployable "
            "lattice line minimising g(f|centroid) over the record's policy front.",
            "Vertex and centroid use deployable modes only (best_of_32 and greedy_top8_starts are excluded).",
            "Percentiles use the null of all lines (dead ends kept); the *_full_length_null variant excludes dead ends.",
            "Percentile of MSD, Hausdorff and orientation: lower values are better.",
        ],
        "git_commit": git_commit(),
        "wall_clock_s": round(time.time() - t0, 2),
    }
    out_path = P(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_clean(summary), indent=1))
    write_tex(P(args.tex), summary)
    log(f"wrote {out_path}")
    log(f"wrote {P(args.tex)}")
    return summary


if __name__ == "__main__":
    main()
