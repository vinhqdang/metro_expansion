"""Upstream tabular Q-learning (QLearningTNDP, unmodified apart from the patched test() in src/baselines/upstream.py).

Writes an mnep.run.v1 JSON with method tabular_eff (max_efficiency) or tabular_ggi2 (ggi2).
The policy front is one point (w = null): the greedy rollout of the learned Q-table from its best learned start.
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

from src.baselines.run import run_tabular, write_record  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reward", required=True, choices=["max_efficiency", "ggi2"])
    ap.add_argument("--city", required=True, choices=["xian", "amsterdam"])
    ap.add_argument("--variant", default="full", choices=["full", "line1_only", "line2_only"])
    ap.add_argument("--groups", default="price_groups_5.txt")
    ap.add_argument("--nr_stations", type=int, default=20)
    ap.add_argument("--chained", type=int, default=1, choices=[0, 1])
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--epsilon_decay_steps", type=int, default=None,
                    help="default: 0.8 x episodes (the upstream 400/500 ratio)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--workdir", default=None,
                    help="working directory for codecarbon output (default: the directory of --out)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    record = run_tabular(
        reward_type=args.reward, city=args.city, variant=args.variant, groups=args.groups,
        nr_stations=args.nr_stations, chained=bool(args.chained), seed=args.seed, norm_path=args.norm,
        episodes=args.episodes, epsilon_decay_steps=args.epsilon_decay_steps,
        workdir=args.workdir or out.parent,
    )
    write_record(out, record)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
