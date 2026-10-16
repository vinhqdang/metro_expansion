"""Maps of the two benchmark cities, generated lines per weight vector, and the algorithm schematic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.envs.context import build_context  # noqa: E402
from src.envs.fast_motndp import build_city  # noqa: E402
from src.eval.normalization import load_norm  # noqa: E402

INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e4e4e4"
BLUE, VERM, GREEN, PINK = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
OUT = ROOT / "manuscript/figures"
DEPLOYABLE = {"greedy", "final_pop_best"}


def line_cells(city, k: int) -> list[tuple[int, int]]:
    gy = city.grid_y_size
    return [(int(i) // gy, int(i) % gy) for i in np.asarray(city.existing_lines_full[k]).reshape(-1)]


def draw_lines(ax, city, lw: float = 2.4) -> None:
    for k in range(len(city.existing_lines_full)):
        c = line_cells(city, k)
        ax.plot([p[1] for p in c], [p[0] for p in c], color="white", lw=lw + 2.2, solid_capstyle="round", zorder=3)
        ax.plot([p[1] for p in c], [p[0] for p in c], color=INK, lw=lw, solid_capstyle="round", zorder=4)


def demand_map(ax, mu: np.ndarray, city, title: str) -> None:
    m = np.where(mu > 0, mu, np.nan)
    im = ax.imshow(m, cmap="Blues", norm=LogNorm(vmin=np.nanpercentile(m, 2), vmax=np.nanmax(m)), interpolation="nearest")
    draw_lines(ax, city)
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    return im


def group_map(ax, city, title: str) -> None:
    g = np.ma.masked_invalid(city.grid_groups)
    cm = plt.get_cmap("Oranges", 5)
    ax.imshow(np.where(np.isnan(city.grid_groups), 0, 1), cmap="Greys", vmin=0, vmax=4, interpolation="nearest")
    ax.imshow(g, cmap=cm, vmin=0.5, vmax=5.5, interpolation="nearest")
    draw_lines(ax, city)
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    return cm


def instances_fig() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1]})
    for row, (name, label) in enumerate([("xian", "Xi'an, 29 × 29 cells"), ("amsterdam", "Amsterdam, 35 × 47 cells")]):
        city = build_city(name)
        mu = build_context(city).mu
        im = demand_map(axes[row, 0], mu, city, f"{label}: trip-end demand and existing lines")
        cm = group_map(axes[row, 1], city, f"{label}: population groups")
        cb = fig.colorbar(im, ax=axes[row, 0], shrink=0.75, pad=0.02)
        cb.set_label("Trip-end mass per cell (log scale)", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0.5, 5.5))
        cb2 = fig.colorbar(sm, ax=axes[row, 1], shrink=0.75, pad=0.02, ticks=[1, 2, 3, 4, 5])
        cb2.set_label("Price band (1 = lowest, 5 = highest)", fontsize=7)
        cb2.ax.tick_params(labelsize=6.5)
        axes[row, 1].text(0.99, 0.01, "grey: no group", transform=axes[row, 1].transAxes, ha="right", va="bottom",
                          fontsize=6.5, color=MUTED, bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.8))
    axes[0, 0].plot([], [], color=INK, lw=2.4, label="Existing metro lines")
    axes[0, 0].legend(frameon=True, fontsize=6.5, loc="upper left", facecolor="white", edgecolor="none", framealpha=0.85)
    fig.savefig(OUT / "fig_instances.pdf")
    plt.close(fig)


def vertex_lines_fig() -> None:
    city = build_city("xian")
    mu = build_context(city).mu
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True)
    verts = [((1.0, 0.0, 0.0), "Demand weight"), ((0.0, 1.0, 0.0), "Fairness weight"), ((0.0, 0.0, 1.0), "Accessibility weight")]
    for r, (m, label, col, mk) in enumerate([("rl_full", "RL policy", BLUE, "o"), ("nsga2", "NSGA-II", VERM, "s")]):
        rec = json.loads((ROOT / f"results/xian_full/{m}/seed0.json").read_text())
        pts = [p for p in rec["policy_front"] if p["mode"] in DEPLOYABLE]
        for c, (w, wl) in enumerate(verts):
            ax = axes[r, c]
            ax.imshow(np.where(mu > 0, mu, np.nan), cmap="Greys", norm=LogNorm(vmin=np.nanpercentile(mu[mu > 0], 2), vmax=mu.max()),
                      interpolation="nearest", alpha=0.45)
            draw_lines(ax, city, lw=1.8)
            p = next(p for p in pts if p["w"] is not None and np.allclose(p["w"], w))
            cells = p["cells"]
            ax.plot([q[1] for q in cells], [q[0] for q in cells], color="white", lw=4.0, zorder=5)
            ax.plot([q[1] for q in cells], [q[0] for q in cells], color=col, lw=2.0, marker=mk, ms=3.2, mfc="white", zorder=6)
            f = p["obj"]
            ax.set_title(f"{label}, {wl.lower()}", fontsize=8, loc="left")
            ax.text(0.02, 0.02, f"D {f['D_tot']:.3f}   G {f['ggi2']:.4f}   A {f['A_new']:.2f}", transform=ax.transAxes,
                    fontsize=6.3, color=INK, bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.85), va="bottom")
            ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(OUT / "fig_vertex_lines.pdf")
    plt.close(fig)


def holdout_fig() -> None:
    full = build_city("xian", "full")
    l1 = build_city("xian", "line1_only")
    mu = build_context(l1).mu
    norm = load_norm(ROOT / "results/xian_line1_only/normalization.json")
    series = [("rl_full", "RL policy", BLUE, "o"), ("nsga2", "NSGA-II", VERM, "s"), ("ga_scalar_matched", "Scalarised GA (matched)", GREEN, "^")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), constrained_layout=True)
    real2 = line_cells(full, 1)
    kept = line_cells(full, 0)
    for ax, (m, label, col, mk) in zip(axes, series):
        ax.imshow(np.where(mu > 0, mu, np.nan), cmap="Greys", norm=LogNorm(vmin=np.nanpercentile(mu[mu > 0], 2), vmax=mu.max()),
                  interpolation="nearest", alpha=0.45)
        ax.plot([p[1] for p in kept], [p[0] for p in kept], color="white", lw=4.4, zorder=3)
        ax.plot([p[1] for p in kept], [p[0] for p in kept], color=INK, lw=2.2, zorder=4, label="Line 1 (kept)")
        ax.plot([p[1] for p in real2], [p[0] for p in real2], color="white", lw=4.4, zorder=3)
        ax.plot([p[1] for p in real2], [p[0] for p in real2], color=INK, lw=2.2, ls=(0, (4, 2)), zorder=4, label="Line 2 (real, removed)")
        rec = json.loads((ROOT / f"results/xian_line1_only/{m}/seed0.json").read_text())
        pts = [p for p in rec["policy_front"] if p["mode"] in DEPLOYABLE]
        p = min(pts, key=lambda q: norm.g(np.array(q["f_norm"]), np.array([1 / 3] * 3)))
        c = p["cells"]
        ax.plot([q[1] for q in c], [q[0] for q in c], color="white", lw=4.0, zorder=5)
        ax.plot([q[1] for q in c], [q[0] for q in c], color=col, lw=2.0, marker=mk, ms=3.2, mfc="white", zorder=6, label="Generated, seed 0")
        ax.set_title(label, fontsize=8.5, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(frameon=True, fontsize=6.2, loc="lower left", facecolor="white", edgecolor="none", framealpha=0.85)
    fig.savefig(OUT / "fig_holdout_map.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ algorithm schematic
def box(ax, x, y, w, h, text, fc, ec, fs=6.8, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.1", fc=fc, ec=ec, lw=0.9, zorder=2))
    lines = text.split("\n")
    ax.text(x + w / 2, y + h - 1.6, lines[0], ha="center", va="top", fontsize=fs + 0.5, fontweight="bold" if bold_first else "normal", color=INK, zorder=3)
    if len(lines) > 1:
        ax.text(x + w / 2, y + h - 4.4, "\n".join(lines[1:]), ha="center", va="top", fontsize=fs, color=INK, linespacing=1.25, zorder=3)


def arrow(ax, p, q, color=MUTED, rad=0.0, lw=1.1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=0, shrinkB=0))


def algorithm_fig() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 106); ax.axis("off")
    B_FC, B_EC = "#dcebf6", BLUE
    V_FC, V_EC = "#f7e1d3", VERM
    G_FC, G_EC = "#d6efe6", GREEN
    N_FC, N_EC = "#f1f1f1", MUTED
    X = [0.5, 34.5, 68.5]; W = 31
    R1, R2, R3, R4 = 82, 55, 30, 3          # lower edge of each row
    H1, H2, H3, H4 = 20, 22, 20, 21

    def title(x, y, h, t):
        ax.text(x + 1.2, y + h - 1.2, t, ha="left", va="top", fontsize=7.2, fontweight="bold", color=INK, zorder=4)

    def body(x, y, h, t):
        ax.text(x + 1.2, y + h - 5.2, t, ha="left", va="top", fontsize=6.3, color=INK, linespacing=1.3, zorder=4)

    def panel(x, y, w, h, fc, ec, t, b):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", fc=fc, ec=ec, lw=0.9, zorder=2))
        title(x, y, h, t); body(x, y, h, b)

    # row 1
    panel(X[0], R1, W, H1, N_FC, N_EC, "1  City graph", "cells with price group,\nexisting-line flag and OD degree;\nspatial edges (8 neighbours)\nand OD-flow edges")
    panel(X[1], R1, W, H1, B_FC, B_EC, "2  Graph encoder", "two message-passing layers\nproduce an embedding h for\nevery cell; computed once per\nupdate for all 16 episodes")
    panel(X[2], R1, W, H1, G_FC, G_EC, "3  Weight vectors", "4 vectors w per update from\nDirichlet(1,1,1); 4 rollouts\nfor each. w tells the policy\nwhich trade-off to build for")
    arrow(ax, (X[0] + W + 0.4, R1 + 10), (X[1] - 0.4, R1 + 10))
    # row 2
    panel(X[0], R2, W, H2, B_FC, B_EC, "4a  Start head", "scores every cell as the first\nstation; query uses the mean\nembedding and w; plus the\nfixed scalarised prior")
    panel(X[1], R2, W, H2, B_FC, B_EC, "4b  Move policy", "scores the 8 neighbours.\nkey: embedding h_c and look-\nahead features (one-step gain\nper group, new reach, line\nadjacency, stations left);\nquery: line pool, w, group shares")
    panel(X[2], R2, W, H2, V_FC, V_EC, "5  Rollout", "start, then up to 19 moves\nunder the build rules\n(inside grid, no revisit,\nnear-straight line)")
    gx0, gy0 = X[2] + 20.2, R2 + 3.0
    for a in range(5):
        for b in range(5):
            ax.add_patch(plt.Rectangle((gx0 + b * 2.0, gy0 + a * 2.0), 2.0, 2.0, fc="white", ec=GRID, lw=0.4, zorder=3))
    path = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3)]
    ax.plot([gx0 + (c + 0.5) * 2.0 for r, c in path], [gy0 + (r + 0.5) * 2.0 for r, c in path], color=VERM, lw=1.3, marker="o", ms=2.0, zorder=5)
    arrow(ax, (X[1] + W / 2, R1 - 0.4), (X[1] + W / 2, R2 + H2 + 0.4))
    arrow(ax, (X[1] + 6, R1 - 0.4), (X[0] + W - 6, R2 + H2 + 0.4), rad=0.0)
    arrow(ax, (X[2] + 6, R1 - 0.4), (X[1] + W - 6, R2 + H2 + 0.4), color=GREEN)
    arrow(ax, (X[0] + W + 0.4, R2 + 11), (X[1] - 0.4, R2 + 11))
    arrow(ax, (X[1] + W + 0.4, R2 + 11), (X[2] - 0.4, R2 + 11))
    # row 3 (right to left)
    panel(X[2], R3, W, H3, V_FC, V_EC, "6  Objectives of the line", "demand D, fairness G and\naccessibility A, each divided\nby the best single-objective\nvalue found by a genetic search")
    panel(X[1], R3, W, H3, G_FC, G_EC, "7  Return and advantage", "R = -g(f | w), augmented\nTchebycheff. Advantage = R\nminus the mean R of the other\nthree rollouts with the same w")
    panel(X[0], R3, W, H3, G_FC, G_EC, "8  Update", "advantage-weighted log-\nlikelihood of every decision,\nentropy bonus, one Adam step;\nno replay buffer, no critic")
    arrow(ax, (X[2] + W / 2, R2 - 0.4), (X[2] + W / 2, R3 + H3 + 0.4), color=VERM)
    arrow(ax, (X[2] - 0.4, R3 + 10), (X[1] + W + 0.4, R3 + 10))
    arrow(ax, (X[1] - 0.4, R3 + 10), (X[0] + W + 0.4, R3 + 10))
    arrow(ax, (X[0] + 4, R3 + H3 + 0.4), (X[0] + 4, R2 - 0.4), color=GREEN)
    ax.text(X[0] + 5.5, (R2 + R3 + H3) / 2, "new parameters", fontsize=6.0, color=GREEN, va="center")
    # evaluation row
    ax.add_patch(FancyBboxPatch((X[0], R4), 99, H4 - 2, boxstyle="round,pad=0.2,rounding_size=1.0", fc="white", ec=MUTED, lw=0.9, ls=(0, (4, 2)), zorder=2))
    ax.text(27, R4 + H4 - 4.2, "Evaluation (after training, same for every method)", fontsize=7.2, fontweight="bold", va="top", zorder=4)
    tri = np.array([[4.5, R4 + 2.2], [22.5, R4 + 2.2], [13.5, R4 + 13.8]])
    ax.add_patch(plt.Polygon(tri, fc="#f7f7f7", ec=MUTED, lw=0.7, zorder=3))
    for i in range(5):
        for j in range(5 - i):
            k = 4 - i - j
            pt = (i * tri[0] + j * tri[1] + k * tri[2]) / 4
            ax.plot(*pt, "o", ms=2.6, color=GREEN, zorder=5)
    ax.text(4.0, R4 + 1.2, "demand", fontsize=5.5, color=MUTED, ha="left", va="top")
    ax.text(23.0, R4 + 1.2, "fairness", fontsize=5.5, color=MUTED, ha="right", va="top")
    ax.text(13.5, R4 + 14.6, "accessibility", fontsize=5.5, color=MUTED, ha="center", va="bottom")
    ax.text(27, R4 + H4 - 8.6,
            "15 weight vectors on a simplex lattice. For each one the method hands over one line\n"
            "(greedy rollout, or best member of the final population): the deployable set.\n"
            "All lines evaluated during the run form the archive. Both sets are scored by\n"
            "hypervolume and expected utility over the 15 weights, on the scaled objectives.",
            fontsize=6.3, va="top", ha="left", linespacing=1.3, zorder=4)
    fig.savefig(OUT / "fig_algorithm.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    instances_fig()
    vertex_lines_fig()
    holdout_fig()
    algorithm_fig()
    print("wrote map and algorithm figures to", OUT)
