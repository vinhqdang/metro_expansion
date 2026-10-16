#!/usr/bin/env python3.13
"""Aggregate the per-seed run records of one city variant into results/summary/<city_variant>_main.json.

Input:  results/<city_variant>/<method>/seed<S>.json            (schema mnep.run.v1)
Output: results/summary/<city_variant>_main.json                 (schema mnep.summary.v1)

Conventions:
- Deployable front of a run = its policy_front points with mode 'greedy' (this is what
  indicators.policy is computed on). Runs without greedy-mode points (search baselines)
  use all of their policy_front points. Runs whose points carry no weight vector
  (w is None) are weight-free: single_point=true and the same point is used at every
  weight setting.
- policy hv/eum are recomputed from the deployable front with the run's normalisation and
  checked against the stored indicators.policy. archive hv/eum come from indicators.archive.
- IGD+ (policy front and archive) is measured against the non-dominated union of all
  deployable fronts and archives of the city variant. The values are written into the
  in-memory copy of each record only; the record files are not changed.
- Undefined gini_ratio values (None in the record) are excluded from every statistic and
  counted in n_na.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.archive import nondominated_points  # noqa: E402
from src.eval.normalization import Norm  # noqa: E402
from src.eval.record import front_indicators  # noqa: E402
from src.eval.stats import holm, mwu_a12_boot  # noqa: E402
from src.eval.weights import lattice  # noqa: E402

RUN_SCHEMA = "mnep.run.v1"
SUMMARY_SCHEMA = "mnep.summary.v1"
N_BOOT = 10_000
BOOT_SEED = 0
ALPHA = 0.05
MIN_SEEDS_FOR_TEST = 3
PRIMARY_REF = "rl_full"
# Ablations use five seeds and are compared with the full method on matched seeds in their own table.
ABLATION_METHODS = {"rl_noprior", "rl_single_objective", "rl_flat_encoder"}
PRIMARY_FAMILIES = {"archive_hv": "Archive hypervolume", "policy_eum": "Policy EUM"}

WEIGHT_SETTINGS: tuple[tuple[str, tuple[float, float, float] | None], ...] = (
    ("vertex_demand", (1.0, 0.0, 0.0)),
    ("vertex_fairness", (0.0, 1.0, 0.0)),
    ("vertex_access", (0.0, 0.0, 1.0)),
    ("mean_all", None),  # mean over all deployable points (the centroid is not on the H=4 lattice)
)
W_MATCH_ATOL = 1e-6
RAW_KEYS = ("D_tot", "ggi2", "s_min", "A_new")

METHOD_ORDER = (
    "rl_full",
    "rl_single_objective",
    "rl_flat_encoder",
    "rl_noprior",
    "nsga2",
    "ga_scalar_matched",
    "ga_scalar",
    "ga_demand",
    "ga_ggi2",
    "tabular_eff",
    "tabular_ggi2",
    "greedy",
    "random",
)
METHOD_LABELS = {
    "rl_full": "RL (ours)",
    "rl_single_objective": "RL, demand-only reward",
    "rl_flat_encoder": "RL, no message passing",
    "rl_noprior": "RL, no scalarised prior",
    "nsga2": "NSGA-II",
    "ga_scalar_matched": "Scalarised GA (matched)",
    "ga_scalar": "Scalarised GA (full budget)",
    "ga_demand": "GA of Michailidis et al. (demand)",
    "ga_ggi2": "GA of Michailidis et al. (GGI)",
    "tabular_eff": "Tabular Q-learning (demand)",
    "tabular_ggi2": "Tabular Q-learning (GGI)",
    "greedy": "Greedy",
    "random": "Random search",
}

SEED_RE = re.compile(r"^seed(\d+)\.json$")


# ---------------------------------------------------------------- JSON helpers
def json_clean(x: Any) -> Any:
    """Recursively convert numpy types and non-finite floats (written as null)."""
    if isinstance(x, dict):
        return {str(k): json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_clean(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_clean(obj), indent=1, allow_nan=False) + "\n")


# ---------------------------------------------------------------- statistics
def bootstrap_ci_mean(x: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> list[float]:
    """Percentile 95% bootstrap interval of the mean (fixed seed)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return [float(lo), float(hi)]


def summarise(values: Any) -> dict[str, Any]:
    """n, mean, sd (ddof=1, null for n<2) and 95% bootstrap CI of the mean."""
    x = np.asarray(list(values), dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95": [None, None]}
    mean = float(x.mean())
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "ci95": [None, None]}
    return {
        "n": n,
        "mean": mean,
        "sd": float(x.std(ddof=1)),
        "ci95": bootstrap_ci_mean(x),
    }


def igd_plus_chunked(P: np.ndarray, R: np.ndarray, chunk: int = 256) -> float:
    """IGD+ as in src.eval.pareto.igd_plus, evaluated in blocks of reference points to bound memory."""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    R = np.asarray(R, dtype=float).reshape(-1, 3)
    if P.shape[0] == 0:
        return float("inf")
    mins = []
    for s in range(0, R.shape[0], chunk):
        Rc = R[s : s + chunk]
        deficit = np.maximum(Rc[:, None, :] - P[None, :, :], 0.0)
        mins.append(np.sqrt(np.sum(deficit**2, axis=-1)).min(axis=1))
    return float(np.concatenate(mins).mean())


# ---------------------------------------------------------------- loading
def load_city_runs(city_dir: Path, city_variant: str) -> dict[str, list[dict[str, Any]]]:
    """{method: [run, ...]} sorted by seed. A run is {'seed', 'path', 'record'}."""
    runs: dict[str, list[dict[str, Any]]] = {}
    for mdir in sorted(p for p in Path(city_dir).iterdir() if p.is_dir()):
        found = []
        for f in mdir.iterdir():
            m = SEED_RE.match(f.name)
            if m:
                found.append((int(m.group(1)), f))
        if not found:
            continue
        recs = []
        for seed_file, f in sorted(found):
            rec = json.loads(f.read_text())
            if rec.get("schema") != RUN_SCHEMA:
                raise ValueError(f"{f}: schema {rec.get('schema')!r}, expected {RUN_SCHEMA}")
            if rec.get("seed") != seed_file:
                raise ValueError(f"{f}: record seed {rec.get('seed')} does not match file name")
            if f"{rec.get('city')}_{rec.get('variant')}" != city_variant:
                raise ValueError(f"{f}: record is {rec.get('city')}_{rec.get('variant')}, not {city_variant}")
            recs.append({"seed": seed_file, "path": str(f), "record": rec})
        runs[mdir.name] = recs
    return runs


def common_norm(all_runs: list[dict[str, Any]]) -> Norm:
    """All runs of a city variant must share one normalisation, otherwise fronts are not comparable."""
    ref = all_runs[0]["record"]["normalization"]
    for r in all_runs[1:]:
        nz = r["record"]["normalization"]
        same = (
            np.allclose(nz["f_ref"], ref["f_ref"], rtol=0.0, atol=1e-12)
            and float(nz["z"]) == float(ref["z"])
            and float(nz["rho"]) == float(ref["rho"])
        )
        if not same:
            raise ValueError(f"{r['path']}: normalisation differs from {all_runs[0]['path']}")
    return Norm(f_ref=np.asarray(ref["f_ref"], dtype=float), z=float(ref["z"]), rho=float(ref["rho"]))


def deployable_points(rec: dict[str, Any]) -> list[dict[str, Any]]:
    pts = rec["policy_front"]
    greedy = [p for p in pts if p.get("mode") == "greedy"]
    return greedy if greedy else list(pts)


def score_run(run: dict[str, Any], norm: Norm) -> None:
    """Attach deployable-front arrays, recomputed policy indicators and per-setting objectives to a run."""
    rec = run["record"]
    pts = deployable_points(rec)
    if not pts:
        raise ValueError(f"{run['path']}: empty deployable front")
    F_pol = np.array([p["f_norm"] for p in pts], dtype=float).reshape(-1, 3)
    F_arc = np.asarray(rec["archive_nd"], dtype=float).reshape(-1, 3)
    if F_arc.shape[0] == 0:
        raise ValueError(f"{run['path']}: empty archive_nd")

    pol = front_indicators(F_pol, norm)
    stored = rec.get("indicators", {}).get("policy")
    if stored is not None and (
        abs(stored["hv"] - pol["hv"]) > 1e-6 or abs(stored["eum"] - pol["eum"]) > 1e-6
    ):
        print(f"warning: {run['path']}: recomputed policy indicators differ from stored values", file=sys.stderr)

    stored_arc = rec.get("indicators", {}).get("archive")
    if stored_arc is not None:
        arc = {"hv": float(stored_arc["hv"]), "eum": float(stored_arc["eum"]),
               "n_nondominated": stored_arc.get("n_nondominated")}
    else:
        arc = front_indicators(F_arc, norm)

    single = all(p.get("w") is None for p in pts)
    if single and len(pts) > 1:
        print(f"warning: {run['path']}: {len(pts)} weight-free points; using the first", file=sys.stderr)

    rec["indicators"] = {
        "policy": {"hv": pol["hv"], "eum": pol["eum"], "n_nondominated": pol["n_nondominated"], "igd_plus": None},
        "archive": {"hv": arc["hv"], "eum": arc["eum"], "n_nondominated": arc.get("n_nondominated"), "igd_plus": None},
    }
    run["points"] = pts
    run["F_pol"] = F_pol
    run["F_arc"] = F_arc
    run["single_point"] = single
    run["wall_clock_s"] = float(rec["env"]["wall_clock_s"])
    run["n_na"] = int(sum(p["obj"].get("gini_ratio") is None for p in pts))
    run["n_points"] = len(pts)
    run["n_lines"] = len({tuple(tuple(int(v) for v in c) for c in p["cells"]) for p in pts})
    run["objectives"] = {}
    for name, w in WEIGHT_SETTINGS:
        if w is None:
            run["objectives"][name] = {k: float(np.mean([p["obj"][k] for p in pts])) for k in RAW_KEYS}
            continue
        p = _objective_point(run, w)
        run["objectives"][name] = None if p is None else {k: float(p["obj"][k]) for k in RAW_KEYS}


def _objective_point(run: dict[str, Any], w_target: tuple[float, float, float]) -> dict[str, Any] | None:
    """Deployable point whose weight equals w_target (weight-free runs: their single point).

    If several points match (repeated weight vector), the one with the lowest g_w is used.
    """
    pts = run["points"]
    if run["single_point"]:
        return pts[0]
    matches = [
        p for p in pts
        if p.get("w") is not None and np.allclose(p["w"], w_target, rtol=0.0, atol=W_MATCH_ATOL)
    ]
    if not matches:
        return None
    if len(matches) > 1 and all(p.get("g_w") is not None for p in matches):
        return min(matches, key=lambda p: p["g_w"])
    return matches[0]


def reference_front(all_runs: list[dict[str, Any]]) -> np.ndarray:
    """Non-dominated union of every run's deployable front and archive (normalised space)."""
    pts = np.vstack([r["F_pol"] for r in all_runs] + [r["F_arc"] for r in all_runs])
    return nondominated_points(pts)


# ---------------------------------------------------------------- per-method summary
def method_summary(method: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    ind = [r["record"]["indicators"] for r in runs]
    metrics = {
        "policy_hv": [i["policy"]["hv"] for i in ind],
        "policy_eum": [i["policy"]["eum"] for i in ind],
        "archive_hv": [i["archive"]["hv"] for i in ind],
        "archive_eum": [i["archive"]["eum"] for i in ind],
        "policy_igd_plus": [i["policy"]["igd_plus"] for i in ind],
        "archive_igd_plus": [i["archive"]["igd_plus"] for i in ind],
        "wall_clock_s": [r["wall_clock_s"] for r in runs],
    }
    objectives: dict[str, Any] = {}
    for name, w in WEIGHT_SETTINGS:
        got = [r["objectives"][name] for r in runs if r["objectives"][name] is not None]
        entry: dict[str, Any] = {"w": None if w is None else list(w), "available": bool(got), "n": len(got)}
        for k in RAW_KEYS:
            entry[k] = summarise(v[k] for v in got) if got else None
        objectives[name] = entry
    return {
        "label": METHOD_LABELS.get(method, method),
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "single_point": bool(all(r["single_point"] for r in runs)),
        "metrics": {k: summarise(v) for k, v in metrics.items()},
        "objectives": objectives,
        "n_na": int(sum(r["n_na"] for r in runs)),
        "n_points": int(sum(r["n_points"] for r in runs)),
        "distinct_lines_mean": float(np.mean([r["n_lines"] for r in runs])),
    }


def _family_values(runs: list[dict[str, Any]], fam: str) -> np.ndarray:
    """Values of a test family: 'archive_hv' -> archive hv, 'policy_eum' -> policy EUM."""
    if fam == "archive_hv":
        return np.array([r["record"]["indicators"]["archive"]["hv"] for r in runs], dtype=float)
    if fam == "policy_eum":
        return np.array([r["record"]["indicators"]["policy"]["eum"] for r in runs], dtype=float)
    raise KeyError(fam)


def tests_vs_reference(runs_by_method: dict[str, list[dict[str, Any]]], ref: str = PRIMARY_REF) -> dict[str, Any]:
    """Unpaired two-sided Mann-Whitney with A12 and bootstrap CI of the mean difference (ref - other).

    Holm correction is applied within each metric family over the tested comparisons.
    Methods with fewer than MIN_SEEDS_FOR_TEST seeds (or a reference with fewer) are listed, not tested.
    """
    out: dict[str, Any] = {"reference": ref, "alpha": ALPHA, "min_seeds": MIN_SEEDS_FOR_TEST,
                           "families": {}}
    if ref not in runs_by_method:
        return out
    others = [m for m in ordered_methods(runs_by_method) if m != ref and m not in ABLATION_METHODS]
    for fam in PRIMARY_FAMILIES:
        x = _family_values(runs_by_method[ref], fam)
        rows: list[dict[str, Any]] = []
        for m in others:
            ys = _family_values(runs_by_method[m], fam)
            row: dict[str, Any] = {"method": m, "label": METHOD_LABELS.get(m, m), "n_seeds": int(ys.size),
                                   "tested": False, "p": None, "p_holm": None}
            if x.size >= MIN_SEEDS_FOR_TEST and ys.size >= MIN_SEEDS_FOR_TEST:
                res = mwu_a12_boot(x, ys, n_boot=N_BOOT, seed=BOOT_SEED)
                row.update({
                    "tested": True,
                    "mean_diff": res["mean_diff"],
                    "diff_ci95": [res["ci_low"], res["ci_high"]],
                    "a12": res["a12"],
                    "p": res["p_mwu"],
                    "mwu_method": res["mwu_method"],
                })
            rows.append(row)
        idx = [i for i, r in enumerate(rows) if r["tested"]]
        if idx:
            for i, adj in zip(idx, holm([rows[i]["p"] for i in idx])):
                rows[i]["p_holm"] = adj
        out["families"][fam] = {"label": PRIMARY_FAMILIES[fam], "n_ref_seeds": int(x.size), "rows": rows}
    return out


def ordered_methods(runs_by_method: dict[str, Any]) -> list[str]:
    known = [m for m in METHOD_ORDER if m in runs_by_method]
    return known + sorted(m for m in runs_by_method if m not in METHOD_ORDER)


def build_summary(city_variant: str, results_root: Path) -> dict[str, Any]:
    city_dir = Path(results_root) / city_variant
    if not city_dir.is_dir():
        raise FileNotFoundError(f"{city_dir} does not exist")
    runs_by_method = load_city_runs(city_dir, city_variant)
    if not runs_by_method:
        raise FileNotFoundError(f"no seed<S>.json files under {city_dir}/<method>/")
    all_runs = [r for m in ordered_methods(runs_by_method) for r in runs_by_method[m]]
    norm = common_norm(all_runs)
    for r in all_runs:
        score_run(r, norm)
    ref = reference_front(all_runs)
    for r in all_runs:
        ind = r["record"]["indicators"]
        ind["policy"]["igd_plus"] = igd_plus_chunked(r["F_pol"], ref)
        ind["archive"]["igd_plus"] = igd_plus_chunked(r["F_arc"], ref)

    lattice_w = lattice(4, 3)
    weight_settings = [
        {"name": name, "w": None if w is None else list(w),
         "on_lattice": None if w is None else bool(any(np.allclose(w, L, atol=1e-9) for L in lattice_w))}
        for name, w in WEIGHT_SETTINGS
    ]
    methods = {m: method_summary(m, runs_by_method[m]) for m in ordered_methods(runs_by_method)}
    return {
        "schema": SUMMARY_SCHEMA,
        "city_variant": city_variant,
        "n_runs": len(all_runs),
        "alpha": ALPHA,
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile", "ci": 0.95},
        "reference": {
            "size": int(ref.shape[0]),
            "definition": "non-dominated union of all deployable policy fronts and archives of the city variant",
        },
        "normalization": {
            "f_ref": [float(v) for v in norm.f_ref],
            "z": float(norm.z),
            "rho": float(norm.rho),
        },
        "weight_settings": weight_settings,
        "methods": methods,
        "tests_vs": tests_vs_reference(runs_by_method),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--city_variant", required=True, help="e.g. xian_full")
    ap.add_argument("--results", type=Path, default=REPO / "results", help="results root")
    args = ap.parse_args(argv)

    summary = build_summary(args.city_variant, args.results)
    out = args.results / "summary" / f"{args.city_variant}_main.json"
    write_json(out, summary)
    print(f"wrote {out}")
    for m, s in summary["methods"].items():
        e = s["metrics"]["policy_eum"]
        print(f"  {m:22s} n={s['n_seeds']:2d} policy_eum={e['mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
