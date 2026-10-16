"""Train one weight-conditioned RL policy (src/rl/reinforce.py) and write a mnep.run.v1 record.

Example:
  OMP_NUM_THREADS=1 python3.13 scripts/run_rl.py --city xian --episodes 10000 --seed 0 \
      --norm results/xian_full/normalization.json --out results/xian_full/rl_full/seed0.json

Ablations: full | single_objective | flat_encoder | no_lookahead | no_start_head | gini_reward | unnormalised.
Transfer:  --init_from CKPT --transfer {scratch,full,encoder_only,heads_only,full_frozen_enc,random_frozen_enc}.
Amsterdam uses a top-64 OD graph by default (see src/rl/graph.py); the value is recorded in config.od_topk.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.eval.normalization import load_norm  # noqa: E402
from src.eval.record import save_record  # noqa: E402
from src.rl.reinforce import ABLATIONS, TRANSFER_MODES, RunConfig, run  # noqa: E402


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--city", choices=["xian", "amsterdam"], default="xian")
    p.add_argument("--variant", choices=["full", "line1_only", "line2_only"], default="full")
    p.add_argument("--groups", default="price_groups_5.txt")
    p.add_argument("--ablation", choices=list(ABLATIONS), default="full")
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument("--batch_w", type=int, default=4)
    p.add_argument("--batch_k", type=int, default=4)
    p.add_argument("--nr_stations", type=int, default=20)
    p.add_argument("--chained", type=int, choices=[0, 1], default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm", required=True, help="normalisation JSON (src.eval.normalization.save_norm)")
    p.add_argument("--init_from", default=None, help="checkpoint written by --save_ckpt")
    p.add_argument("--transfer", choices=list(TRANSFER_MODES), default="scratch")
    p.add_argument("--save_ckpt", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--best_of", type=int, default=32)
    p.add_argument("--od_topk", type=int, default=None)
    p.add_argument("--lr_start", type=float, default=1e-3)
    p.add_argument("--lr_end", type=float, default=1e-4)
    p.add_argument("--entropy0", type=float, default=0.01)
    p.add_argument("--entropy_frac", type=float, default=0.7)
    p.add_argument("--prior_lam", type=float, default=1.0, help="fixed w-scalarised move prior (0 = legacy)")
    p.add_argument("--prior_lam_s", type=float, default=1.0, help="fixed w-scalarised start prior (0 = legacy)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    norm_path = Path(args.norm)
    eval_norm = load_norm(norm_path)
    cfg = RunConfig(
        city=args.city, variant=args.variant, groups=args.groups, chained=bool(args.chained),
        nr_stations=args.nr_stations, episodes=args.episodes, batch_w=args.batch_w, batch_k=args.batch_k,
        seed=args.seed, ablation=args.ablation, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        eval_every=args.eval_every, best_of=args.best_of, od_topk=args.od_topk, transfer=args.transfer,
        init_from=args.init_from, save_ckpt=args.save_ckpt, verbose=not args.quiet,
        lr_start=args.lr_start, lr_end=args.lr_end, entropy0=args.entropy0, entropy_frac=args.entropy_frac,
        prior_lam=args.prior_lam, prior_lam_s=args.prior_lam_s,
    )

    t0 = time.time()
    out = run(cfg, eval_norm, log=(lambda *a, **k: print(*a, flush=True)))
    total = time.time() - t0

    transfer = out["transfer"]
    if transfer is not None:
        transfer = dict(transfer)
        transfer["budget"] = "full" if args.episodes >= 4000 else "reduced"
    config = dict(out["config"])
    config.update({"episodes": args.episodes, "n_updates": out["n_updates"], "init_from": args.init_from,
                   "save_ckpt": args.save_ckpt, "batch_size": args.batch_w * args.batch_k,
                   "calibrated_gain_scale": out["calibrated_gain_scale"]})
    record = {
        "city": args.city,
        "variant": args.variant,
        "groups": args.groups,
        "chained": bool(args.chained),
        "nr_stations": args.nr_stations,
        "method": f"rl_{args.ablation}",
        "transfer": transfer,
        "seed": args.seed,
        "budget_rollouts": out["budget_rollouts"],
        "eval_rollouts": out["eval_rollouts"],
        "config": config,
        "normalization": {
            "file": str(norm_path),
            "f_ref": [float(x) for x in eval_norm.f_ref],
            "train_f_ref": out["norm_train_f_ref"],
            "z": float(eval_norm.z),
            "rho": float(eval_norm.rho),
        },
        "env": {
            "git_commit": _git_commit(),
            "torch": torch.__version__,
            "omp_threads": 1,
            "wall_clock_s": round(total, 2),
            "train_wall_s": round(out["train_wall_s"], 2),
        },
        "policy_front": out["policy_front"],
        "archive_nd": out["archive_nd"],
        "archive_best": out["archive_best"],
        "curve": out["curve"],
        "indicators": out["indicators"],
    }
    save_record(args.out, record)
    ind = out["indicators"]
    print(f"wrote {args.out}")
    print(f"wall_clock_s={total:.1f} train_wall_s={out['train_wall_s']:.1f} updates={out['n_updates']} "
          f"budget_rollouts={out['budget_rollouts']} eval_rollouts={out['eval_rollouts']}")
    print(f"policy: hv={ind['policy']['hv']:.6f} eum={ind['policy']['eum']:.5f}  "
          f"best_of_32: hv={ind['policy_best_of_32']['hv']:.6f} eum={ind['policy_best_of_32']['eum']:.5f}")
    if "policy_top8_starts" in ind:
        print(f"top8_starts: hv={ind['policy_top8_starts']['hv']:.6f} eum={ind['policy_top8_starts']['eum']:.5f}")
    return record


if __name__ == "__main__":
    main()
