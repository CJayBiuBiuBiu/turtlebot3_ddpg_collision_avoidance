#!/usr/bin/env python3
"""Plot the Hu-aligned Gazebo completion-time experiment."""

import argparse
import csv
import math
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHOD_ORDER = [
    'geometric', 'execution_aware', 'mappo_legacy', 'mappo_ttl',
    'mappo_hybrid', 'mappo_vision']
METHOD_LABELS = {
    'geometric': 'Geometric\nVoronoi',
    'execution_aware': 'Execution-\naware',
    'mappo_legacy': 'MAPPO\npermanent recovery',
    'mappo_ttl': 'MAPPO\ntimed recovery',
    'mappo_hybrid': 'MAPPO\ntimed + fallback',
    'mappo_vision': 'MAPPO\n+ vision',
}
COLORS = ['#4C78A8', '#F58518', '#8E6C8A', '#54A24B', '#2E8B57', '#72B7B2']
THRESHOLDS = (80, 90, 95, 97)


def load_summary(path):
    with open(path, newline='') as handle:
        rows = {row['method']: row for row in csv.DictReader(handle)}
    return rows


def values(rows, methods, field):
    return [float(rows[method][field]) for method in methods]


def annotate_bars(ax, bars, labels, offset):
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset, label,
                ha='center', va='bottom', fontsize=8)


def main():
    parser = argparse.ArgumentParser(
        description='Plot the Hu-aligned Gazebo benchmark.')
    parser.add_argument('summary_csv')
    parser.add_argument('-o', '--output', default='hu_aligned_comparison.png')
    args = parser.parse_args()

    rows = load_summary(args.summary_csv)
    methods = [method for method in METHOD_ORDER if method in rows]
    if not methods:
        raise ValueError('summary does not contain a supported method')
    labels = [METHOD_LABELS[method] for method in methods]
    colors = COLORS[:len(methods)]
    x = list(range(len(methods)))
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.6))

    ax = axes[0, 0]
    width = 0.19
    for index, threshold in enumerate(THRESHOLDS):
        rates = values(rows, methods, 'success_rate_%d' % threshold)
        offset = (index - 1.5) * width
        ax.bar([position + offset for position in x], rates,
               width=width, label='%d%%' % threshold,
               color=plt.cm.Blues(0.35 + index * 0.16),
               edgecolor='#333333', linewidth=0.5)
    ax.set_title('Coverage-threshold success rate')
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Fraction of 5 runs')
    ax.legend(ncol=4, fontsize=7, loc='upper right')

    ax = axes[0, 1]
    means = values(rows, methods, 'time_to_97_s_mean')
    stds = values(rows, methods, 'time_to_97_s_std')
    finite_stds = [value if math.isfinite(value) else 0.0 for value in stds]
    bars = ax.bar(x, means, yerr=finite_stds, capsize=4, color=colors,
                  edgecolor='#333333', linewidth=0.6)
    annotate_bars(ax, bars, ['%.0f' % value for value in means], 12)
    ax.set_title('Time to 97% among successful runs')
    ax.set_ylabel('Simulation time (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(means) * 1.24)

    ax = axes[1, 0]
    means = values(rows, methods, 'coverage_pct_mean')
    stds = values(rows, methods, 'coverage_pct_std')
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  edgecolor='#333333', linewidth=0.6)
    annotate_bars(ax, bars, ['%.2f' % value for value in means], 1.2)
    ax.set_title('Final coverage at completion or 900 s')
    ax.set_ylabel('Coverage (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 110)

    ax = axes[1, 1]
    means = values(rows, methods, 'path_length_total_m_mean')
    stds = values(rows, methods, 'path_length_total_m_std')
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  edgecolor='#333333', linewidth=0.6)
    annotate_bars(ax, bars, ['%.1f' % value for value in means], 8)
    ax.set_title('Total team path length')
    ax.set_ylabel('Path length (m)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(mean + std for mean, std in zip(means, stds)) * 1.2)

    for ax in axes.flat:
        ax.grid(axis='y', color='#D9D9D9', linewidth=0.7)
        ax.set_axisbelow(True)

    fig.suptitle('Hu-Aligned Gazebo Multi-Robot Exploration (5 runs per method)',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
