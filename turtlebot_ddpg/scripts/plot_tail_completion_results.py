#!/usr/bin/env python3
"""Plot the seed-matched MAPPO tail-completion experiment."""

import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS = ('mappo_hybrid', 'mappo_tail')
LABELS = {
    'mappo_hybrid': 'Previous MAPPO hybrid',
    'mappo_tail': 'MAPPO with tail completion',
}
COLORS = {
    'mappo_hybrid': '#4C78A8',
    'mappo_tail': '#E45756',
}


def read_runs(directory):
    runs = {method: {} for method in METHODS}
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


def tail_duration(rows):
    return threshold_time(rows, 97) - threshold_time(rows, 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation_dir')
    parser.add_argument('-o', '--output', default='tail_completion_comparison.png')
    args = parser.parse_args()

    runs = read_runs(args.evaluation_dir)
    repeats = sorted(set(runs[METHODS[0]]) & set(runs[METHODS[1]]))
    if not repeats:
        raise ValueError('no seed-matched tail-completion runs found')

    max_time = max(float(rows[-1]['elapsed_s'])
                   for method in METHODS for rows in runs[method].values())
    timeline = np.arange(0.0, max_time + 2.0, 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    ax = axes[0]
    for method in METHODS:
        curves = np.vstack([
            coverage_series(runs[method][repeat], timeline)
            for repeat in repeats
        ])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1) if len(repeats) > 1 else np.zeros_like(mean)
        ax.plot(timeline, mean, color=COLORS[method], linewidth=2.0,
                label=LABELS[method])
        ax.fill_between(timeline, mean - std, mean + std,
                        color=COLORS[method], alpha=0.15, linewidth=0)
    ax.axhline(90, color='#777777', linestyle=':', linewidth=1.0)
    ax.axhline(97, color='#555555', linestyle='--', linewidth=1.0)
    ax.set_title('(a) Coverage over simulation time')
    ax.set_xlabel('Simulation time (s)')
    ax.set_ylabel('Mapped free-space coverage (%)')
    ax.set_ylim(0, 102)
    ax.legend(frameon=False, fontsize=8, loc='lower right')

    ax = axes[1]
    method_values = {method: [] for method in METHODS}
    for repeat in repeats:
        values = [tail_duration(runs[method][repeat]) for method in METHODS]
        for method, value in zip(METHODS, values):
            method_values[method].append(value)
        ax.plot((0, 1), values, color='#A0A0A0', linewidth=1.0, alpha=0.8)
        ax.scatter((0, 1), values,
                   color=[COLORS[method] for method in METHODS],
                   s=30, zorder=3)
    means = [np.mean(method_values[method]) for method in METHODS]
    ax.scatter((0, 1), means, marker='D', s=65, facecolor='white',
               edgecolor=[COLORS[method] for method in METHODS],
               linewidth=2.0, zorder=4, label='Mean')
    ax.set_title('(b) Seed-matched 90%-97% duration')
    ax.set_ylabel('Tail-completion time (s)')
    ax.set_xticks((0, 1))
    ax.set_xticklabels([LABELS[method] for method in METHODS], fontsize=8)
    ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Hu-Aligned Gazebo Tail-Completion Evaluation '
                 '(%d matched seeds)' % len(repeats), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches='tight')
    fig.savefig(os.path.splitext(output)[0] + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
