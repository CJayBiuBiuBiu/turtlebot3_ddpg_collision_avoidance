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
        root / "outputs/evaluation/hu_final_pair_state_mask_n5_20260805/summary.csv"
    )
    methods = ["geometric", "mappo_tail"]
    labels = ["Geometric\nVoronoi", "Final\nhybrid"]
    colors = ["#7a7a7a", "#2468a2"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    success = [value(rows[m], "success_rate_97") * 100 for m in methods]
    bar_panel(axes[0], labels, success, None, "Successful trials (%)", "97% completion", colors)
    axes[0].set_ylim(0, 110)
    for index, method in enumerate(methods):
        completed = round(success[index] * int(rows[method]["runs"]) / 100.0)
        axes[0].text(index, success[index] + 3,
                     "%d/%s" % (completed, rows[method]["runs"]),
                     ha="center", fontsize=9)

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
    fig.savefig(output_dir / "hu_aligned_state_mask_n5.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_stack(root, output_dir):
    rows = by_method(
        root / "outputs/evaluation/stack_ablation_combined_n5_20260805/summary.csv"
    )
    methods = ["mappo_vision", "mappo_hybrid", "mappo_tail"]
    labels = ["MAPPO", "MAPPO +\nfallback", "Complete\nstack"]
    colors = ["#7a7a7a", "#3f8f5f", "#2468a2"]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.25))
    success = [value(rows[m], "success_rate_97") * 100 for m in methods]
    bar_panel(axes[0], labels, success, None, "Successful trials (%)", "97% completion", colors)
    axes[0].set_ylim(0, 110)
    for index, method in enumerate(methods):
        completed = round(success[index] * int(rows[method]["runs"]) / 100.0)
        axes[0].text(index, success[index] + 3,
                     "%d/%s" % (completed, rows[method]["runs"]),
                     ha="center", fontsize=9)

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
    conditions = [
        ("6 updates\nno state mask", "independent_training_seed_study"),
        ("6 updates\nstate mask", "independent_training_seed_study_short_masked"),
        ("18 updates\nstate mask", "independent_training_seed_study_long_masked"),
    ]
    condition_rows = []
    for _, directory in conditions:
        condition_rows.append(read_rows(
            root / "outputs/evaluation" / directory / "independent_seed_summary.csv"))
    seeds = [row["training_seed"] for row in condition_rows[0]]
    colors = ["#7a7a7a", "#3f8f5f", "#2468a2"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    x = np.arange(len(seeds))
    width = 0.24
    for index, ((label, _), rows, color) in enumerate(
            zip(conditions, condition_rows, colors)):
        positions = x + (index - 1) * width
        axes[0].bar(
            positions, [value(row, "coverage_pct_mean") for row in rows],
            width=width,
            yerr=[value(row, "coverage_pct_std") for row in rows],
            capsize=3, color=color, label=label)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(seeds)
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Coverage after 90 s (%)")
    axes[0].set_title("Independent policy evaluation")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(axis="y", alpha=0.25)

    aggregate_values = []
    aggregate_errors = []
    for _, directory in conditions:
        aggregate = read_rows(
            root / "outputs/evaluation" / directory /
            "independent_seed_aggregate.csv")[0]
        aggregate_values.append(value(aggregate, "coverage_pct_mean_across_seeds"))
        aggregate_errors.append(value(
            aggregate, "coverage_pct_mean_between_seed_std"))
    bar_panel(
        axes[1], [label for label, _ in conditions], aggregate_values,
        aggregate_errors, "Coverage after 90 s (%)",
        "Across-seed mean and variability", colors)
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
