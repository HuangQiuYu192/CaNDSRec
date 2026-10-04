"""Generate manuscript figures from completed score-geometry experiments.

Figure contract
---------------
Core conclusion: final-score normalization improves trained ranking through
more than a global logit scale; its sequence-side and item-side effects are
geometrically distinct.
Archetype: quantitative grid (one result figure, three mechanism figures).
Evidence hierarchy: Fig. 2 is held-out three-seed endpoint evidence; Fig. 3
tests the frozen-score theorem; Fig. 4 is a single-seed training trace; Fig. 5
is a fixed-temperature development ablation; Fig. A records validation-only
temperature selection.  The captions and manuscript label the last three as
mechanism/development evidence, not multi-seed endpoint claims.
Exports: vector PDF/SVG plus 600-dpi PNG for paper preview; all input CSVs are
copied under ``results/geometry_evidence``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Keep editable SVG text and a consistent compact conference-paper style.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "geometry_evidence"
FIGURES = ROOT / "figures"

COLORS = {
    "Dot": "#484878",
    "Scaled-Dot": "#7884B4",
    "Sequence": "#42949E",
    "Joint": "#B64342",
    "sequence-side": "#0F4D92",
    "item-side": "#B64342",
}


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.04, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (("pdf", 600), ("svg", 600), ("png", 600)):
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def mean_sd(frame: pd.DataFrame, key: str) -> tuple[float, float]:
    values = frame[key].astype(float).to_numpy()
    return float(values.mean()), float(values.std(ddof=1))


def plot_endpoint_ladder() -> None:
    """Fig. 2: held-out three-seed four-way endpoint comparison."""
    summary = pd.read_csv(DATA / "scaled_dot_confirmation.csv")
    metric_specs = [("ndcg@10", "NDCG@10"), ("recall@10", "Recall@10")]
    methods = [("dot", "Dot"), ("scaled_dot", "Scaled-Dot"), ("sequence", "Sequence"), ("joint", "Joint")]
    datasets = ["Sports", "Toys"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), sharex=True)
    width = 0.18
    positions = np.arange(len(datasets))
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        subset = summary[summary["metric"] == metric]
        for idx, (column, label) in enumerate(methods):
            means, stds = [], []
            for dataset in datasets:
                group = subset[subset["dataset"] == dataset]
                mean, std = mean_sd(group, column)
                means.append(mean); stds.append(std)
            offset = (idx - 1.5) * width
            ax.bar(positions + offset, means, width=width, color=COLORS[label],
                   edgecolor="white", linewidth=0.4, yerr=stds, capsize=2,
                   error_kw={"elinewidth": 0.8, "capthick": 0.8}, label=label)
        ax.set_ylabel(ylabel)
        ax.set_xticks(positions, datasets)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_title("NDCG@10")
    axes[1].set_title("Recall@10")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08), columnspacing=1.35)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.subplots_adjust(wspace=0.34, top=0.73, bottom=0.24, left=0.09, right=0.99)
    save(fig, "Fig2_score_geometry_ladder")


def plot_frozen_asymmetry() -> None:
    """Fig. 3: empirical verification of frozen ranking asymmetry."""
    sweep = pd.read_csv(DATA / "frozen_radial_power_sweep.csv")
    seq = sweep[sweep["tag"].str.startswith("sequence")].sort_values("sequence_norm_power")
    item = sweep[sweep["tag"].str.startswith("item")].sort_values("item_norm_power")
    p = seq["sequence_norm_power"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.42))
    for frame, label, color, power_col in ((seq, "sequence-side ($p_e=0$)", COLORS["sequence-side"], "sequence_norm_power"),
                                            (item, "item-side ($p_h=0$)", COLORS["item-side"], "item_norm_power")):
        x = frame[power_col].to_numpy()
        axes[0].plot(x, frame["topk_jaccard_vs_dot"], marker="o", ms=4.5, lw=1.8, color=color, label=label)
        axes[1].plot(x, frame["mean_abs_target_rank_change"], marker="o", ms=4.5, lw=1.8, color=color, label=label)
    axes[0].set_ylim(0.62, 1.02)
    axes[0].set_ylabel("Top-10 Jaccard vs. dot")
    axes[1].set_ylabel("Mean |target-rank change|")
    for ax in axes:
        ax.set_xlabel("Radial-removal power")
        ax.set_xticks(p)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.55)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower left")
    axes[0].set_title("Candidate set under frozen scoring")
    axes[1].set_title("Target-rank displacement")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.subplots_adjust(wspace=0.35, top=0.85, bottom=0.25, left=0.09, right=0.99)
    save(fig, "Fig3_frozen_radial_asymmetry")


def plot_training_trajectory() -> None:
    """Fig. 4: single-seed diagnostic trace, not an endpoint aggregate."""
    dot = pd.read_csv(DATA / "trajectory_dot_seed2025.csv")
    joint = pd.read_csv(DATA / "trajectory_joint_seed2025.csv")
    curves = [(dot, "Dot", COLORS["Dot"]), (joint, "Joint", COLORS["Joint"])]
    specs = [
        ("valid_ndcg_at_10", "Validation NDCG@10"),
        ("radial_gradient_fraction", "Sequence radial-gradient fraction"),
        ("sequence_norm_median", "Median raw sequence norm"),
        ("softmax_entropy", "Training-batch softmax entropy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.55))
    for index, (column, ylabel) in enumerate(specs):
        ax = axes.flat[index]
        for frame, label, color in curves:
            ax.plot(frame["epoch"], frame[column], lw=1.45, color=color, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.55)
        ax.set_axisbelow(True)
        panel_label(ax, chr(ord("a") + index))
    axes[0, 1].set_yscale("log")
    axes[0, 0].legend(loc="lower right")
    fig.text(0.5, 0.995, "Beauty mechanism trace (seed 2025; identical architecture and optimiser)",
             ha="center", va="top", fontsize=9)
    fig.subplots_adjust(wspace=0.38, hspace=0.43, top=0.91, bottom=0.12, left=0.11, right=0.99)
    save(fig, "Fig4_training_geometry_trajectory")


def plot_partial_screen() -> None:
    """Fig. 5: fixed-temperature partial-normalisation mechanism screen."""
    screen = pd.read_csv(DATA / "partial_normalization_screen.csv")
    powers = [0.0, 0.5, 1.0]
    matrix = np.full((3, 3), np.nan)
    for _, row in screen.iterrows():
        y = powers.index(float(row["sequence_norm_power"]))
        x = powers.index(float(row["item_norm_power"]))
        matrix[y, x] = float(row["test_ndcg@10"])
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#F2F2F2")
    fig, ax = plt.subplots(figsize=(3.55, 3.0))
    image = ax.imshow(matrix, cmap=cmap, vmin=np.nanmin(matrix), vmax=np.nanmax(matrix), origin="lower")
    for y in range(3):
        for x in range(3):
            if np.isfinite(matrix[y, x]):
                ax.text(x, y, f"{matrix[y, x]:.4f}", ha="center", va="center", fontsize=8,
                        color="white" if matrix[y, x] < 0.043 else "black")
            else:
                ax.text(x, y, "—", ha="center", va="center", color="#777777", fontsize=9)
    ax.set_xticks(range(3), ["0", "0.5", "1"])
    ax.set_yticks(range(3), ["0", "0.5", "1"])
    ax.set_xlabel("Item-side power $p_e$")
    ax.set_ylabel("Sequence-side power $p_h$")
    ax.set_title("Beauty fixed-$\\tau$ partial-normalisation screen")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Test NDCG@10")
    panel_label(ax, "a",)
    fig.subplots_adjust(top=0.86, bottom=0.19, left=0.22, right=0.92)
    save(fig, "Fig5_partial_normalization_screen")


def plot_temperature_screen() -> None:
    """Supplementary Fig. A: validation-only Scaled-Dot temperature selection."""
    screen = pd.read_csv(DATA / "scaled_dot_temperature_screen.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35), sharey=False)
    for ax, dataset in zip(axes, ["Sports", "Toys"]):
        group = screen[screen["dataset"] == dataset].sort_values("temperature")
        x = group["temperature"].to_numpy()
        y = group["valid_ndcg@10"].to_numpy()
        selected = group["selected_by_valid_ndcg@10"].astype(bool).to_numpy()
        ax.plot(x, y, color=COLORS["Scaled-Dot"], lw=1.6, marker="o", ms=4)
        ax.scatter(x[selected], y[selected], s=45, zorder=3, color=COLORS["Joint"], edgecolor="white", linewidth=0.7,
                   label="validation-selected")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:g}" for value in x])
        ax.set_xlabel("Scaled-Dot temperature $\\tau$")
        ax.set_ylabel("Validation NDCG@10")
        ax.set_title(dataset)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.55)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower left")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.subplots_adjust(wspace=0.32, top=0.86, bottom=0.25, left=0.09, right=0.99)
    save(fig, "FigA_scaled_dot_temperature_selection")


def main() -> None:
    plot_endpoint_ladder()
    plot_frozen_asymmetry()
    plot_training_trajectory()
    plot_partial_screen()
    plot_temperature_screen()


if __name__ == "__main__":
    main()
