"""Figures for the manuscript: Pareto projections (Xi'an) and the Line-1-only holdout map."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.envs.fast_motndp import build_city  # noqa: E402
from src.eval.normalization import load_norm  # noqa: E402

INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e4e4e4"
SERIES = [  # fixed order; marker shape is the secondary encoding
    ("rl_full", "RL (ours)", "#0072B2", "o"),
    ("nsga2", "NSGA-II", "#D55E00", "s"),
    ("ga_scalar_matched", "Scalarised GA (matched)", "#009E73", "^"),
    ("random", "Random", "#CC79A7", "D"),
]
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
OUT = ROOT / "manuscript/figures"


DEPLOYABLE = {"greedy", "final_pop_best", "random_min_g"}


def deployable(rec: dict) -> list:
    return [p for p in rec["policy_front"] if p["mode"] in DEPLOYABLE]


def load(city_variant: str, method: str, seed: int = 0) -> dict:
    return json.loads((ROOT / f"results/{city_variant}/{method}/seed{seed}.json").read_text())


def pareto_fig() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    for ax, (i, j, xl, yl) in zip(axes, [(0, 2, "Demand (scaled)", "Accessibility (scaled)"), (0, 1, "Demand (scaled)", "Fairness (scaled)")]):
        for m, label, col, mk in SERIES:
            rec = load("xian_full", m)
            arc = np.array(rec["archive_nd"])
            ax.scatter(arc[:, i], arc[:, j], s=6, color=col, alpha=0.18, linewidths=0)
            dep = np.array([p["f_norm"] for p in deployable(rec)])
            ax.scatter(dep[:, i], dep[:, j], s=22, marker=mk, color=col, edgecolor="white", linewidths=0.5, label=label, zorder=3)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, 1.2)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower left", fontsize=7.5)
    fig.savefig(OUT / "fig_pareto_xian.pdf")
    plt.close(fig)


def balanced_point(rec: dict, norm) -> dict:
    pts = deployable(rec)
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    return min(pts, key=lambda p: norm.g(np.array(p["f_norm"]), w))


def holdout_fig() -> None:
    full = build_city("xian", "full")
    gy = full.grid_y_size
    to_rc = lambda idx: (int(idx) // gy, int(idx) % gy)
    line1 = [to_rc(i) for i in np.asarray(full.existing_lines_full[0]).reshape(-1)]
    line2 = [to_rc(i) for i in np.asarray(full.existing_lines_full[1]).reshape(-1)]
    norm = load_norm(ROOT / "results/xian_line1_only/normalization.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True, sharex=True, sharey=True)
    for ax, (m, label, col, mk) in zip(axes, SERIES[:3]):
        for cells, style, lab in [(line1, dict(color=INK, lw=2.2), "Line 1 (kept)"),
                                  (line2, dict(color=MUTED, lw=2.2, ls=(0, (4, 2))), "Line 2 (real, removed)")]:
            ax.plot([c[1] for c in cells], [c[0] for c in cells], label=lab, **style)
        rec = load("xian_line1_only", m)
        p = balanced_point(rec, norm)
        cells = p["cells"]
        ax.plot([c[1] for c in cells], [c[0] for c in cells], color=col, lw=1.6, marker=mk, ms=3.5, mfc="white", label="Generated, seed 0")
        ax.set_title(label, fontsize=8.5, color=INK)
        ax.set_xlim(-0.5, 28.5)
        ax.set_ylim(28.5, -0.5)
        ax.set_aspect("equal")
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Grid row")
    for ax in axes:
        ax.set_xlabel("Grid column")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.savefig(OUT / "fig_holdout_map.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pareto_fig()
    holdout_fig()
    print("wrote figures to", OUT)
