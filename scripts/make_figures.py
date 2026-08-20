"""Generate all manuscript figures from real logged/computed data.

No fabricated numbers: every figure either replots a table already in the
manuscript (bar/line charts) or, for the two network maps, decodes a real
logged action sequence into real grid/geographic coordinates
(scripts/make_xian_map_data.py for Xi'an; direct geopandas plotting of the
real CSL_HCMC shapefiles for Ho Chi Minh City).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "manuscript" / "figures"
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLOR_FULL = "#1b6ca8"
COLOR_SO = "#e07b39"
COLOR_FE = "#7a5195"
COLOR_TAB = "#999999"


def fig_xian_map():
    data = json.loads((FIG_DIR / "xian_map_data.json").read_text())
    gx, gy = data["grid_x_size"], data["grid_y_size"]

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.set_xlim(-0.5, gx - 0.5)
    ax.set_ylim(-0.5, gy - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, gx, 4))
    ax.set_yticks(range(0, gy, 4))
    ax.grid(True, color="#e5e5e5", linewidth=0.5, zorder=0)

    for line in data["existing_lines"]:
        xs = [c[0] for c in line]
        ys = [c[1] for c in line]
        ax.plot(xs, ys, color="#333333", linewidth=2.5, zorder=2, solid_capstyle="round")
    ax.plot([], [], color="#333333", linewidth=2.5, label="Existing metro lines")

    path = data["trained_path"]
    xs = [c[0] for c in path]
    ys = [c[1] for c in path]
    ax.plot(xs, ys, color=COLOR_FULL, linewidth=2.5, zorder=3, solid_capstyle="round", label="Trained extension (full, seed 42)")
    ax.scatter([xs[0]], [ys[0]], color=COLOR_FULL, marker="o", s=70, zorder=4, edgecolor="white", linewidth=1)
    ax.scatter(xs[1:], ys[1:], color=COLOR_FULL, marker=".", s=40, zorder=4)

    ax.set_xlabel("Grid $x$")
    ax.set_ylabel("Grid $y$")
    ax.set_title("Xi'an: existing network and the trained policy's extension")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.savefig(FIG_DIR / "fig_xian_map.pdf")
    plt.close(fig)
    print("wrote fig_xian_map.pdf")


def fig_hcmc_map():
    import geopandas as gpd

    data_dir = REPO_ROOT / "external" / "CSL_HCMC"
    zones = gpd.read_file(data_dir / "outputs" / "zones.geojson")
    bus_stops = gpd.read_file(data_dir / "Data" / "GIS" / "Bus" / "Bus_shapefile" / "BusStop.shp").to_crs(zones.crs)

    # Reproject once to a projected (UTM) CRS so both the polygon plot and the
    # centroid-based connecting line use consistent, metrically-correct geometry.
    utm_crs = zones.estimate_utm_crs()
    zones_proj = zones.to_crs(utm_crs)
    bus_stops_proj = bus_stops.to_crs(utm_crs)

    map_data = json.loads((FIG_DIR / "hcmc_map_data.json").read_text())
    station_idx = map_data["station_zone_indices"]
    station_zones_proj = zones_proj.iloc[station_idx]

    fig, ax = plt.subplots(figsize=(6.4, 7.0))
    zones_proj.plot(ax=ax, color="#f2f2f2", edgecolor="#cfcfcf", linewidth=0.4, zorder=1)
    bus_stops_proj.plot(ax=ax, color="#cfcfcf", markersize=2, zorder=2)
    station_zones_proj.plot(ax=ax, color=COLOR_FULL, edgecolor="#0d3a56", linewidth=0.8, alpha=0.85, zorder=3)

    centroids = station_zones_proj.geometry.centroid
    ax.plot(centroids.x, centroids.y, color="#0d3a56", linewidth=1.2, zorder=4, alpha=0.8)

    ax.set_axis_off()
    ax.set_title("Ho Chi Minh City: 322 candidate zones,\nexisting bus network, and the trained policy's line")
    handles = [
        plt.Line2D([0], [0], color="#333333", marker="s", linestyle="None", markerfacecolor="#f2f2f2", markeredgecolor="#cfcfcf", label="Candidate zones"),
        plt.Line2D([0], [0], color="#cfcfcf", marker="o", linestyle="None", markersize=4, label="Existing bus stops"),
        plt.Line2D([0], [0], color=COLOR_FULL, marker="s", linestyle="-", markerfacecolor=COLOR_FULL, label="Trained line (20 zones)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True)
    fig.savefig(FIG_DIR / "fig_hcmc_map.pdf")
    plt.close(fig)
    print("wrote fig_hcmc_map.pdf")


def _grouped_bar(ax, labels, series, series_labels, colors, ylabel, title, ylim=None):
    n_groups = len(labels)
    n_series = len(series)
    width = 0.8 / n_series
    x = np.arange(n_groups)
    for i, (vals, errs) in enumerate(series):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, vals, width * 0.92, yerr=errs, capsize=3, color=colors[i], label=series_labels[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_ablation_bars():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    metrics = ["Demand (%)", "Equity", "Coverage"]
    xian_full = ([0.18, 0.30, 0.10], [0.37, 0.19, 0.04])
    xian_so = ([0.48, 0.70, 0.15], [0.49, 0.26, 0.06])
    xian_fe = ([0.39, 0.38, 0.11], [0.46, 0.18, 0.04])
    xian_tab = ([1.36, np.nan, np.nan], [0.56, 0, 0])

    hcmc_full = ([2.83, 0.84, 0.247], [3.51, 0.15, 0.032])
    hcmc_so = ([4.07, 0.72, 0.230], [3.52, 0.10, 0.014])
    hcmc_fe = ([0.19, 0.82, 0.222], [0.29, 0.16, 0.038])

    ax = axes[0]
    series = [
        ([xian_tab[0][i] for i in range(3)], [xian_tab[1][i] for i in range(3)]),
        (xian_full[0], xian_full[1]),
        (xian_so[0], xian_so[1]),
        (xian_fe[0], xian_fe[1]),
    ]
    _grouped_bar(
        ax, metrics, series,
        ["Tabular RL", "full (ours)", "single_objective", "flat_encoder"],
        [COLOR_TAB, COLOR_FULL, COLOR_SO, COLOR_FE],
        "Value", "Xi'an (5 seeds)",
    )

    ax = axes[1]
    series = [
        (hcmc_full[0], hcmc_full[1]),
        (hcmc_so[0], hcmc_so[1]),
        (hcmc_fe[0], hcmc_fe[1]),
    ]
    _grouped_bar(
        ax, metrics, series,
        ["full (ours)", "single_objective", "flat_encoder"],
        [COLOR_FULL, COLOR_SO, COLOR_FE],
        "Value", "Ho Chi Minh City (5 seeds)",
    )

    fig.suptitle("Ablation comparison: demand, equity, and coverage across both cities", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ablation_bars.pdf")
    plt.close(fig)
    print("wrote fig_ablation_bars.pdf")


def fig_training_curve():
    episodes = [0, 20, 40, 499]
    demand = [0.00, 0.38, 0.34, 0.43]
    equity = [1.000, 0.554, 0.532, 0.454]
    coverage = [0.042, 0.128, 0.133, 0.132]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
    ax2 = ax1.twinx()

    l1, = ax1.plot(episodes, demand, marker="o", color=COLOR_FULL, label="Demand (%)")
    l2, = ax2.plot(episodes, equity, marker="s", color=COLOR_SO, label="Equity")
    l3, = ax2.plot(episodes, coverage, marker="^", color=COLOR_FE, label="Coverage")

    ax1.set_xlabel("Training episode")
    ax1.set_ylabel("Demand (%)", color=COLOR_FULL)
    ax2.set_ylabel("Equity / Coverage", color="#333333")
    ax1.tick_params(axis="y", labelcolor=COLOR_FULL)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.set_title("Xi'an \\texttt{full} (seed 42): training dynamics, not monotonic")
    ax1.legend(handles=[l1, l2, l3], loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_training_curve.pdf")
    plt.close(fig)
    print("wrote fig_training_curve.pdf")


def fig_curriculum():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    labels = ["Demand (%)", "Equity", "Coverage"]

    matched_scratch = ([2.57, 0.75, 0.267], [2.42, 0.25, 0.016])
    matched_curr = ([3.45, 0.72, 0.265], [3.77, 0.17, 0.036])
    reduced_scratch = ([0.03, 0.90, 0.253], [0.04, 0.17, 0.010])
    reduced_curr = ([1.34, 0.91, 0.291], [1.19, 0.01, 0.014])

    for i, metric in enumerate(labels):
        ax = axes[i]
        x = np.arange(2)
        width = 0.35
        vals_scratch = [matched_scratch[0][i], reduced_scratch[0][i]]
        err_scratch = [matched_scratch[1][i], reduced_scratch[1][i]]
        vals_curr = [matched_curr[0][i], reduced_curr[0][i]]
        err_curr = [matched_curr[1][i], reduced_curr[1][i]]
        ax.bar(x - width / 2, vals_scratch, width, yerr=err_scratch, capsize=3, color="#b0b0b0", label="scratch")
        ax.bar(x + width / 2, vals_curr, width, yerr=err_curr, capsize=3, color=COLOR_FULL, label="curriculum")
        ax.set_xticks(x)
        ax.set_xticklabels(["500 ep.\n(matched)", "100 ep.\n(reduced)"])
        ax.set_title(metric)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Ho Chi Minh City: cross-city curriculum, matched vs. reduced training budget", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_curriculum.pdf")
    plt.close(fig)
    print("wrote fig_curriculum.pdf")


def fig_weight_sensitivity():
    weights = ["Uniform", "Demand-only", "Equity-only", "Coverage-only"]
    demand = [0.00, 0.00, 0.00, 0.06]
    equity = [0.200, 0.200, 0.200, 0.317]
    coverage = [0.050, 0.048, 0.048, 0.117]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(weights))
    width = 0.25
    ax.bar(x - width, demand, width, color=COLOR_FULL, label="Demand (%)")
    ax.bar(x, equity, width, color=COLOR_SO, label="Equity")
    ax.bar(x + width, coverage, width, color=COLOR_FE, label="Coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(weights, rotation=15)
    ax.set_title("Xi'an \\texttt{full} (seed 42): weight-conditioning sanity check")
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_weight_sensitivity.pdf")
    plt.close(fig)
    print("wrote fig_weight_sensitivity.pdf")


def fig_radiation_vs_coverage():
    labels = ["single_objective", "flat_encoder", "full"]
    coverage = [0.230, 0.222, 0.247]
    coverage_err = [0.014, 0.038, 0.032]
    radiation = [0.0376, 0.0367, 0.0371]
    radiation_err = [0.0016, 0.0008, 0.0022]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    ax = axes[0]
    ax.bar(labels, coverage, yerr=coverage_err, capsize=3, color=[COLOR_SO, COLOR_FE, COLOR_FULL])
    ax.set_title("Coverage proxy (Eq. 4)")
    ax.set_ylabel("Coverage")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    ax.bar(labels, radiation, yerr=radiation_err, capsize=3, color=[COLOR_SO, COLOR_FE, COLOR_FULL])
    ax.set_title("True radiation-model accessibility")
    ax.set_ylabel("Accessibility")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Ho Chi Minh City: the coverage proxy's architectural sensitivity does not\ncarry over to the true radiation model", y=1.06, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_radiation_vs_coverage.pdf")
    plt.close(fig)
    print("wrote fig_radiation_vs_coverage.pdf")


if __name__ == "__main__":
    fig_xian_map()
    fig_hcmc_map()
    fig_ablation_bars()
    fig_training_curve()
    fig_curriculum()
    fig_weight_sensitivity()
    fig_radiation_vs_coverage()
