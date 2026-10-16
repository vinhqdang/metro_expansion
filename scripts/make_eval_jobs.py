"""Generate the evaluation-phase job files under results/jobs/ (one shell command per line, each with --out).

Run them with scripts/run_jobs.py, in the file order below. Block 6 needs results/xian_full47/normalization.json
from jobs_norm47.txt first, and block 5 needs the rl_full checkpoints written by blocks 1 and 3.

Blocks (priority order):
  jobs_1_xian_main.txt   (a) xian full, seeds 0-9, all nine methods; rl_full saves results/ckpt/xian_full_seed<S>.pt
  jobs_2_holdout.txt     (c) xian line1_only, 27 stations, seeds 0-9, all nine methods
  jobs_3_amsterdam.txt   (d) amsterdam full, seeds 0-9, all nine methods; rl_full saves results/ckpt/amsterdam_full_seed<S>.pt
  jobs_4_ablation.txt    (b) xian full ablations, seeds 0-4: rl_single_objective, rl_flat_encoder, rl_noprior
  jobs_5_transfer.txt    (f) xian->amsterdam and amsterdam->xian, six conditions, 2,000 episodes, seeds 0-4
  jobs_6_xian47.txt      (e) xian full at 47 stations, seeds 0-4: rl_full, nsga2, ga_demand
  jobs_norm47.txt        normalisation for block 6 (compute_normalization.py, 47 stations), run separately

Usage:
  python3.13 scripts/make_eval_jobs.py --rl_args "--entropy0 0.05 --prior_lam 1" --nsga_pop 100 --nsga_gens 100

Core-minute estimates use the unit costs of the plan (rl 24 min xian / 12 min amsterdam, nsga2 3, ga_scalar_matched 3.5,
ga_demand 2, ga_ggi2 2, tabular 4, greedy 4.5, random 1), at 20 stations, 10,000 episodes and the default
budgets. They are scaled by episodes / 10,000 (transfer uses 2,000), by (stations - 1) / 19 (a rough
episode-length scaling) and, for nsga2, by evaluations P(G+1) / 10,100. These are rough planning numbers.
"""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
redundant_greedy_min = 0.0
SEEDS_MAIN = list(range(10))
SEEDS_ABL = list(range(5))
SEEDS_47 = list(range(5))
SEEDS_TRANSFER = list(range(5))
TRANSFER_CONDITIONS = ("scratch", "full", "encoder_only", "heads_only", "full_frozen_enc", "random_frozen_enc")
TRANSFER_EPISODES = 2000
DEFAULT_EPISODES = 10_000
REF_STATIONS = 20
NSGA_REF_EVALS = 100 * 101  # pop 100 + 100 generations x 100
ALL_METHODS = ("rl_full", "nsga2", "ga_scalar_matched", "ga_demand", "ga_ggi2",
               "tabular_eff", "tabular_ggi2", "greedy", "random")
UNIT_MIN = {
    "rl_xian": 24.0, "rl_amsterdam": 12.0, "nsga2": 3.0, "ga_scalar_matched": 3.5,
    "ga_demand": 2.0, "ga_ggi2": 2.0, "tabular": 4.0, "greedy": 4.5, "random": 1.0,
}
NORM_XIAN_FULL = "results/xian_full/normalization.json"
NORM_AMS_FULL = "results/amsterdam_full/normalization.json"
NORM_LINE1 = "results/xian_line1_only/normalization.json"
NORM_47 = "results/xian_full47/normalization.json"
# Measured in results/xian_line1_only/normalization.json provenance (27 stations, 3 seeds x 3 objectives x 10,100 evals).
NORM_MEASURED_MIN_27 = 2254.77 / 60.0
PY = "python3.13"


@dataclass
class Job:
    cmd: str
    method: str
    est_min: float


@dataclass
class Context:
    rl_args: str
    nsga_pop: int
    nsga_gens: int


def _station_scale(nr: int) -> float:
    return (nr - 1) / (REF_STATIONS - 1)


def _rl(ctx: Context, *, city: str, variant: str, nr: int, seed: int, norm: str, out: str,
        episodes: int = DEFAULT_EPISODES, extra: str = "", tail: str = "") -> Job:
    """run_rl.py line. rl_args come before the job-specific flags, so argparse's last-value rule lets those win."""
    parts = [PY, "scripts/run_rl.py", f"--city {city}", f"--variant {variant}", f"--nr_stations {nr}",
             f"--episodes {episodes}", f"--seed {seed}", f"--norm {norm}", "--quiet"]
    if ctx.rl_args:
        parts.append(ctx.rl_args)
    if extra:
        parts.append(extra)
    if tail:
        parts.append(tail)
    parts.append(f"--out {out}")
    unit = UNIT_MIN["rl_amsterdam" if city == "amsterdam" else "rl_xian"]
    est = unit * (episodes / DEFAULT_EPISODES) * _station_scale(nr)
    return Job(" ".join(parts), "rl", est)


def _evo(ctx: Context, algo: str, *, city: str, variant: str, nr: int, seed: int, norm: str, out: str) -> Job:
    parts = [PY, "scripts/run_evo.py", f"--city {city}", f"--variant {variant}", f"--nr_stations {nr}",
             f"--algo {algo}", f"--seed {seed}", f"--norm {norm}"]
    unit_key = algo
    est_scale = _station_scale(nr)
    if algo == "nsga2":
        parts += [f"--pop {ctx.nsga_pop}", f"--gens {ctx.nsga_gens}"]
        est_scale *= ctx.nsga_pop * (ctx.nsga_gens + 1) / NSGA_REF_EVALS
    parts.append(f"--out {out}")
    return Job(" ".join(parts), algo, UNIT_MIN[unit_key] * est_scale)


def _tab(ctx: Context, reward: str, *, city: str, variant: str, nr: int, seed: int, norm: str, out: str) -> Job:
    parts = [PY, "scripts/run_tabular.py", f"--reward {reward}", f"--city {city}", f"--variant {variant}",
             f"--nr_stations {nr}", f"--episodes {DEFAULT_EPISODES}", f"--seed {seed}", f"--norm {norm}",
             f"--out {out}"]
    return Job(" ".join(parts), "tabular", UNIT_MIN["tabular"] * _station_scale(nr))


def method_job(ctx: Context, method: str, *, city: str, variant: str, nr: int, seed: int, norm: str,
               out_dir: str, save_ckpt: str | None = None) -> Job:
    out = f"{out_dir}/{method}/seed{seed}.json"
    common = dict(city=city, variant=variant, nr=nr, seed=seed, norm=norm, out=out)
    if method == "rl_full":
        tail = f"--save_ckpt {save_ckpt}" if save_ckpt else ""
        return _rl(ctx, tail=tail, **common)
    if method == "tabular_eff":
        return _tab(ctx, "max_efficiency", **common)
    if method == "tabular_ggi2":
        return _tab(ctx, "ggi2", **common)
    return _evo(ctx, method, **common)


def ablation_job(ctx: Context, name: str, seed: int) -> Job:
    out = f"results/xian_full/{name}/seed{seed}.json"
    common = dict(city="xian", variant="full", nr=REF_STATIONS, seed=seed, norm=NORM_XIAN_FULL, out=out)
    if name == "rl_single_objective":
        return _rl(ctx, extra="--ablation single_objective", **common)
    if name == "rl_flat_encoder":
        return _rl(ctx, extra="--ablation flat_encoder", **common)
    if name == "rl_noprior":
        return _rl(ctx, extra="--prior_lam 0 --prior_lam_s 0", **common)
    raise ValueError(name)


def transfer_job(ctx: Context, src: str, tgt: str, cond: str, seed: int) -> Job:
    out = f"results/transfer/{src}_to_{tgt}/{cond}_reduced/seed{seed}.json"
    norm = NORM_XIAN_FULL if tgt == "xian" else NORM_AMS_FULL
    init = f"--init_from results/ckpt/{src}_full_seed{seed}.pt --transfer {cond}"
    return _rl(ctx, city=tgt, variant="full", nr=REF_STATIONS, seed=seed, norm=norm, out=out,
               episodes=TRANSFER_EPISODES, tail=init)


def norm47_command() -> str:
    return " ".join([
        PY, "scripts/compute_normalization.py", "--city xian", "--variant full", "--groups price_groups_5.txt",
        "--seeds 100 101 102", "--nr_stations 47", "--chained 1", f"--out {NORM_47}",
    ])


def build_blocks(ctx: Context) -> list[tuple[str, list[Job]]]:
    blocks: list[tuple[str, list[Job]]] = []

    jobs = [method_job(ctx, m, city="xian", variant="full", nr=REF_STATIONS, seed=s, norm=NORM_XIAN_FULL,
                       out_dir="results/xian_full",
                       save_ckpt=f"results/ckpt/xian_full_seed{s}.pt" if m == "rl_full" else None)
            for s in SEEDS_MAIN for m in ALL_METHODS]
    blocks.append(("jobs_1_xian_main.txt", jobs))

    jobs = [method_job(ctx, m, city="xian", variant="line1_only", nr=27, seed=s, norm=NORM_LINE1,
                       out_dir="results/xian_line1_only")
            for s in SEEDS_MAIN for m in ALL_METHODS]
    blocks.append(("jobs_2_holdout.txt", jobs))

    jobs = [method_job(ctx, m, city="amsterdam", variant="full", nr=REF_STATIONS, seed=s, norm=NORM_AMS_FULL,
                       out_dir="results/amsterdam_full",
                       save_ckpt=f"results/ckpt/amsterdam_full_seed{s}.pt" if m == "rl_full" else None)
            for s in SEEDS_MAIN for m in ALL_METHODS]
    blocks.append(("jobs_3_amsterdam.txt", jobs))

    jobs = [ablation_job(ctx, name, s) for s in SEEDS_ABL
            for name in ("rl_single_objective", "rl_flat_encoder", "rl_noprior")]
    blocks.append(("jobs_4_ablation.txt", jobs))

    jobs = [transfer_job(ctx, src, tgt, cond, s)
            for src, tgt in (("xian", "amsterdam"), ("amsterdam", "xian"))
            for cond in TRANSFER_CONDITIONS for s in SEEDS_TRANSFER]
    blocks.append(("jobs_5_transfer.txt", jobs))

    jobs = [method_job(ctx, m, city="xian", variant="full", nr=47, seed=s, norm=NORM_47, out_dir="results/xian_full47")
            for s in SEEDS_47 for m in ("rl_full", "nsga2", "ga_demand")]
    blocks.append(("jobs_6_xian47.txt", jobs))
    return blocks


def write_jobs(out_dir: Path, ctx: Context) -> list[tuple[str, int, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    global redundant_greedy_min
    redundant_greedy_min = 0.0
    for name, jobs in build_blocks(ctx):
        for j in jobs:
            if "--out " not in j.cmd:
                raise AssertionError(f"job without --out: {j.cmd}")
            shlex.split(j.cmd)  # must parse as a plain argv for run_jobs.py
        # greedy is deterministic: one run per instance is enough
        jobs = [j for j in jobs if not (j.method == "greedy" and "--seed 0 " not in j.cmd + " ")]
        (out_dir / name).write_text("\n".join(j.cmd for j in jobs) + "\n")
        summary.append((name, len(jobs), sum(j.est_min for j in jobs)))
    norm_file = out_dir / "jobs_norm47.txt"
    norm_file.write_text(norm47_command() + "\n")
    summary.append((norm_file.name, 1, NORM_MEASURED_MIN_27 * 46 / 26))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rl_args", default="", help='extra arguments for run_rl.py, e.g. "--entropy0 0.05 --prior_lam 1"')
    ap.add_argument("--nsga_pop", type=int, required=True)
    ap.add_argument("--nsga_gens", type=int, required=True)
    ap.add_argument("--out_dir", default="results/jobs")
    args = ap.parse_args(argv)
    if "--out" in args.rl_args or "--seed" in args.rl_args or "--norm" in args.rl_args:
        ap.error("--rl_args must not contain --out, --seed or --norm (set per job)")
    shlex.split(args.rl_args)

    ctx = Context(rl_args=args.rl_args.strip(), nsga_pop=args.nsga_pop, nsga_gens=args.nsga_gens)
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else REPO_ROOT / args.out_dir
    rows = write_jobs(out_dir, ctx)
    total_jobs = total_min = 0.0
    print(f"{'file':<24}{'jobs':>6}{'core-min (est.)':>18}")
    for name, n, minutes in rows:
        total_jobs += n
        total_min += minutes
        print(f"{name:<24}{n:>6}{minutes:>18.0f}")
    print(f"{'total':<24}{int(total_jobs):>6}{total_min:>18.0f}")
    print(f"note: greedy is deterministic with the default n_starts=0, so greedy seeds 1-9 repeat seed 0; "
          f"{redundant_greedy_min:.0f} of the {total_min:.0f} core-min are that repetition (kept as requested).")


if __name__ == "__main__":
    main()
