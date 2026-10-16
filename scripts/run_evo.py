"""Run one evolutionary, search or upstream-GA baseline and write an mnep.run.v1 JSON.

Examples (Xi'an, full network, 20 stations, chained reward, price groups with 5 classes):
    python3.13 scripts/run_evo.py --city xian --variant full --algo nsga2 --seed 0 \
        --norm results/xian_full/normalization.json --out results/xian_full/nsga2/seed0.json
    python3.13 scripts/run_evo.py --algo ga_demand --seed 0 --norm ... --out ...   # upstream GATNDP, 50 gens
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.run import run_method, write_record  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", required=True, choices=["xian", "amsterdam"])
    ap.add_argument("--variant", default="full", choices=["full", "line1_only", "line2_only"])
    ap.add_argument("--groups", default="price_groups_5.txt")
    ap.add_argument(
        "--algo", required=True,
        choices=["nsga2", "ga_scalar", "ga_scalar_matched", "random", "greedy", "ga_demand", "ga_ggi2"],
    )
    ap.add_argument("--nr_stations", type=int, default=20)
    ap.add_argument("--chained", type=int, default=1, choices=[0, 1])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--norm", required=True, help="normalization.json written by compute_normalization.py")
    ap.add_argument("--pop", type=int, default=None, help="population size (default: per-algorithm budget)")
    ap.add_argument("--gens", type=int, default=None, help="generations (default: per-algorithm budget)")
    ap.add_argument("--n_starts", type=int, default=0, help="greedy: subsample this many start cells (0 = all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    record = run_method(
        args.algo,
        city=args.city, variant=args.variant, groups=args.groups, nr_stations=args.nr_stations,
        chained=bool(args.chained), seed=args.seed, norm_path=args.norm,
        pop=args.pop, gens=args.gens, n_starts=args.n_starts,
    )
    write_record(args.out, record)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
