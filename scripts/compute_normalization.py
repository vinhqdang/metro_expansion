"""Anchor normalisation for (D_tot, GGI_2, A_new): f_ref_j = max over tuning seeds of the best raw f_j
found by a single-objective GA with the same operators, pop 100, 100 generations.

Writes the file with src.eval.normalization.save_norm (z = 1.05, rho = 0.05, nadir 0), plus a
"provenance" block with the per-seed maxima. --budget_scale scales pop and generations for quick tests.

Real run (the lead runs this, not the pilot):
    python3.13 scripts/compute_normalization.py --city xian --variant full --groups price_groups_5.txt \
        --seeds 100 101 102 --nr_stations 20 --chained 1 --out results/xian_full/normalization.json
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

from src.baselines.run import compute_anchors  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", required=True, choices=["xian", "amsterdam"])
    ap.add_argument("--variant", default="full", choices=["full", "line1_only", "line2_only"])
    ap.add_argument("--groups", default="price_groups_5.txt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    ap.add_argument("--nr_stations", type=int, default=20)
    ap.add_argument("--chained", type=int, default=1, choices=[0, 1])
    ap.add_argument("--budget_scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    compute_anchors(
        city=args.city, variant=args.variant, groups=args.groups, nr_stations=args.nr_stations,
        chained=bool(args.chained), seeds=args.seeds, budget_scale=args.budget_scale, out=args.out,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
