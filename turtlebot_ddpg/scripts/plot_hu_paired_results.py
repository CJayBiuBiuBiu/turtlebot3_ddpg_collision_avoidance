#!/usr/bin/env python3
"""Plot the seed-matched Hu-reproduction comparison."""

import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    'geometric': 'Hu reproduction baseline',
    'mappo_hybrid': 'Full proposed method',
}
COLORS = {
    'geometric': '#4C78A8',
    'mappo_hybrid': '#E45756',
}
THRESHOLDS = (80, 90, 95, 97)


def read_runs(directory, methods):
    runs = {method: {} for method in methods}
    pattern = re.compile(r'repeat_(\d+)_(.+)\.csv$')
    for path in glob.glob(os.path.join(directory, 'repeat_*.csv')):
        match = pattern.search(os.path.basename(path))
        if not match or match.group(2) not in runs:
            continue
        with open(path, newline='') as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            runs[match.group(2)][int(match.group(1))] = rows
    return runs


def read_summary(directory):
    path = os.path.join(directory, 'summary.csv')
    with open(path, newline='') as stream:
        return {row['method']: row for row in csv.DictReader(stream)}


def threshold_time(rows, threshold):
    for row in rows:
        if float(row.get('coverage_pct', 0.0) or 0.0) >= threshold:
            return float(row['elapsed_s'])
    return np.nan


def coverage_series(rows, timeline):
    elapsed = np.asarray([float(row['elapsed_s']) for row in rows])
    coverage = np.asarray([float(row['coverage_pct']) for row in rows])
    return np.interp(timeline, elapsed, coverage,
                     left=coverage[0], right=coverage[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation_dir')
    parser.add_argument('-o', '--output', default='hu_paired_comparison.png')
    parser.add_argument('--baseline', default='geometric')
    parser.add_argument('--proposed', default='mappo_tail')
    parser.add_argument('--baseline-label', default='Hu-style geometric baseline')
    parser.add_argument('--proposed-label', default='Final proposed method')
    args = parser.parse_args()

    methods = (args.baseline, args.proposed)
    labels = {
        args.baseline: args.baseline_label,
        args.proposed: args.proposed_label,
    }
    colors = {
        args.baseline: COLORS['geometric'],
        args.proposed: COLORS['mappo_hybrid'],
    }

    runs = read_runs(args.evaluation_dir, methods)
    summary = read_summary(args.evaluation_dir)
    repeats = sorted(set(runs[methods[0]]) & set(runs[methods[1]]))
    if not repeats:
        raise ValueError('no seed-matched Hu comparison runs found')

    horizon = max(float(rows[-1]['elapsed_s'])
                  for method in methods for rows in runs[method].values())
    timeline = np.arange(0.0, np.ceil(horizon) + 1.0, 2.0)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.7))

    ax = axes[0, 0]
    for method in methods:
        curves = np.vstack([
            coverage_series(runs[method][repeat], timeline)
            for repeat in repeats
        ])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1) if len(repeats) > 1 else np.zeros_like(mean)
        ax.plot(timeline, mean, color=colors[method], linewidth=2.0,
                label=labels[method])
        ax.fill_between(timeline, mean - std, mean + std,
                        color=colors[method], alpha=0.16, linewidth=0)
    ax.axhline(97, color='#666666', linestyle='--', linewidth=1.0)
    ax.set_title('Coverage over simulation time')
    ax.set_xlabel('Simulation time (s)')
    ax.set_ylabel('Mapped free-space coverage (%)')
    ax.set_xlim(0, np.ceil(horizon / 100.0) * 100.0)
    ax.set_ylim(0, 102)
    ax.legend(frameon=False, fontsize=8, loc='lower right')

    ax = axes[0, 1]
    x = np.arange(len(THRESHOLDS))
    width = 0.36
    for index, method in enumerate(methods):
        means = [float(summary[method]['time_to_%d_s_mean' % threshold])
                 for threshold in THRESHOLDS]
        stds = [float(summary[method]['time_to_%d_s_std' % threshold])
                for threshold in THRESHOLDS]
        positions = x + (index - 0.5) * width
        ax.bar(positions, means, width=width, yerr=stds, capsize=3,
               color=colors[method], edgecolor='#333333', linewidth=0.5,
               label=labels[method])
    ax.set_title('Time to coverage thresholds')
    ax.set_ylabel('Simulation time (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(['%d%%' % threshold for threshold in THRESHOLDS])
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    path_means = [float(summary[method]['path_length_total_m_mean'])
                  for method in methods]
    path_stds = [float(summary[method]['path_length_total_m_std'])
                 for method in methods]
    bars = ax.bar(range(len(methods)), path_means, yerr=path_stds,
                  capsize=4, color=[colors[method] for method in methods],
                  edgecolor='#333333', linewidth=0.5)
    for bar, value in zip(bars, path_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                '%.1f m' % value, ha='center', va='bottom', fontsize=8)
    ax.set_title('Total team path length')
    ax.set_ylabel('Path length (m)')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([labels[method] for method in methods], fontsize=8)
    ax.set_ylim(0, max(mean + std for mean, std in zip(path_means, path_stds)) * 1.2)

    ax = axes[1, 1]
    right_labels = []
    for repeat in repeats:
        actual = [threshold_time(runs[method][repeat], 97) for method in methods]
        values = [value if np.isfinite(value) else horizon for value in actual]
        ax.plot((0, 1), values, color='#A0A0A0', linewidth=1.0, alpha=0.8)
        for index, (value, actual_value, method) in enumerate(
                zip(values, actual, methods)):
            marker = 'o' if np.isfinite(actual_value) else 'x'
            ax.scatter(index, value, color=colors[method], marker=marker,
                       s=32, zorder=3)
        right_labels.append((
            values[1],
            'seed %d' % int(
                runs[methods[1]][repeat][-1]['experiment_seed'])))
    adjusted_labels = []
    for actual_y, label in sorted(right_labels):
        label_y = actual_y
        if adjusted_labels:
            label_y = max(label_y, adjusted_labels[-1][1] + 11.0)
        adjusted_labels.append((actual_y, label_y, label))
    for actual_y, label_y, label in adjusted_labels:
        ax.plot((1.0, 1.04), (actual_y, label_y),
                color='#777777', linewidth=0.6)
        ax.text(1.055, label_y, label, fontsize=7, va='center')
    ax.set_title('Seed-matched time to 97% coverage')
    ax.set_ylabel('Simulation time (s)')
    ax.set_xlim(-0.2, 1.42)
    current_low, current_high = ax.get_ylim()
    ax.set_ylim(current_low, max(current_high, adjusted_labels[-1][1] + 15.0))
    ax.set_xticks((0, 1))
    ax.set_xticklabels([labels[method] for method in methods], fontsize=8)

    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Hu-Aligned Gazebo Evaluation (%d seed-matched runs)' %
                 len(repeats), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches='tight')
    if os.path.splitext(output)[1].lower() != '.pdf':
        fig.savefig(os.path.splitext(output)[0] + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
