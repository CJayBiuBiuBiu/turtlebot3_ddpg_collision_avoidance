#!/usr/bin/env python3
"""Generate the paper's evidence figures from formal evaluation CSV files."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def by_method(path):
    return {row["method"]: row for row in read_rows(path)}


def value(row, field):
    return float(row[field])


def bar_panel(ax, labels, means, errors, ylabel, title, colors):
    positions = np.arange(len(labels))
    ax.bar(positions, means, yerr=errors, capsize=4, color=colors, width=0.68)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def plot_primary(root, output_dir):
    rows = by_method(
        root / "outputs/evaluation/hu_final_pair_combined_n8_20260804/summary.csv"
    )
    methods = ["geometric", "mappo_tail"]
    labels = ["Geometric\nVoronoi", "Final\nhybrid"]
    colors = ["#7a7a7a", "#2468a2"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    success = [value(rows[m], "success_rate_97") * 100 for m in methods]
    bar_panel(axes[0], labels, success, None, "Successful trials (%)", "97% completion", colors)
    axes[0].set_ylim(0, 110)
    axes[0].text(0, success[0] + 3, "7/8", ha="center", fontsize=9)
    axes[0].text(1, success[1] + 3, "8/8", ha="center", fontsize=9)

    bar_panel(
        axes[1], labels,
        [value(rows[m], "elapsed_s_mean") for m in methods],
        [value(rows[m], "elapsed_s_std") for m in methods],
        "Capped duration (s)", "Mission duration", colors,
    )
    bar_panel(
        axes[2], labels,
        [value(rows[m], "path_length_total_m_mean") for m in methods],
        [value(rows[m], "path_length_total_m_std") for m in methods],
        "Team path (m)", "Path length", colors,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "hu_aligned_seed_matched_n8.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_stack(root, output_dir):
    rows = by_method(
        root / "outputs/evaluation/stack_ablation_final_checkpoint_20260804_v1/summary.csv"
    )
    methods = ["mappo_vision", "mappo_hybrid", "mappo_tail"]
    labels = ["MAPPO", "MAPPO +\nfallback", "Complete\nstack"]
    colors = ["#7a7a7a", "#3f8f5f", "#2468a2"]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.25))
    success = [value(rows[m], "success_rate_97") * 100 for m in methods]
    bar_panel(axes[0], labels, success, None, "Successful trials (%)", "97% completion", colors)
    axes[0].set_ylim(0, 110)
    for index, text_value in enumerate(["2/3", "3/3", "3/3"]):
        axes[0].text(index, success[index] + 3, text_value, ha="center", fontsize=9)

    bar_panel(
        axes[1], labels,
        [value(rows[m], "elapsed_s_mean") for m in methods],
        [value(rows[m], "elapsed_s_std") for m in methods],
        "Capped duration (s)", "Mission duration", colors,
    )
    bar_panel(
        axes[2], labels,
        [value(rows[m], "near_collision_sample_ratio_mean") for m in methods],
        [value(rows[m], "near_collision_sample_ratio_std") for m in methods],
        "Near-collision exposure ratio", "Duration-normalised safety", colors,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "control_stack_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training(root, output_dir):
    rows = read_rows(
        root
        / "outputs/evaluation/independent_training_seed_study/independent_seed_summary.csv"
    )
    labels = [row["training_seed"] for row in rows]
    colors = ["#2468a2", "#b2473e", "#3f8f5f"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    bar_panel(
        axes[0], labels,
        [value(row, "coverage_pct_mean") for row in rows],
        [value(row, "coverage_pct_std") for row in rows],
        "Coverage after 90 s (%)", "Independent evaluation", colors,
    )
    axes[0].set_xlabel("Training seed")
    bar_panel(
        axes[1], labels,
        [value(row, "last_mean_reward") for row in rows],
        None,
        "Last rollout mean reward", "Six-update training endpoint", colors,
    )
    axes[1].set_xlabel("Training seed")
    fig.tight_layout()
    fig.savefig(output_dir / "independent_seed_training.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    plot_primary(args.project_root, args.output_dir)
    plot_stack(args.project_root, args.output_dir)
    plot_training(args.project_root, args.output_dir)


if __name__ == "__main__":
    main()
