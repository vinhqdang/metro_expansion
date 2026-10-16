"""Run a file of shell commands in parallel, skipping jobs whose --out file already holds a finished record.

Usage: python3.13 scripts/run_jobs.py jobs.txt --parallel 4
Each line is one command containing `--out PATH`; the job is skipped if PATH exists and contains "schema".
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", WANDB_MODE="disabled")


def finished(cmd: str) -> bool:
    m = re.search(r"--out\s+(\S+)", cmd)
    if not m:
        return False
    p = Path(m.group(1))
    return p.exists() and "schema" in p.read_text(errors="ignore")[:2000]


def run(cmd: str, log_dir: Path) -> tuple[str, int, float]:
    out = re.search(r"--out\s+(\S+)", cmd)
    name = Path(out.group(1)).with_suffix("").as_posix().replace("/", "__") if out else str(abs(hash(cmd)))
    t0 = time.time()
    with open(log_dir / f"{name}.log", "w") as f:
        rc = subprocess.call(shlex.split(cmd), stdout=f, stderr=subprocess.STDOUT, env=ENV)
    return cmd, rc, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--log_dir", default="results/logs/jobs")
    args = ap.parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    cmds = [c.strip() for c in Path(args.jobs).read_text().splitlines() if c.strip() and not c.startswith("#")]
    todo = [c for c in cmds if not finished(c)]
    print(f"{len(cmds)} jobs, {len(cmds) - len(todo)} already done, running {len(todo)}", flush=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for cmd, rc, dt in ex.map(lambda c: run(c, log_dir), todo):
            failed += rc != 0
            print(f"[{'ok' if rc == 0 else 'FAIL'} {dt:6.0f}s] {cmd}", flush=True)
    print(f"done, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
