#!/usr/bin/env python3
"""Plot the corrected seed-matched final-method versus Hu-style study."""

import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS = ('geometric', 'mappo_tail')
LABELS = {
    'geometric': 'Hu-style geometric Voronoi',
    'mappo_tail': 'Vision-MAPPO hybrid',
}
COLORS = {'geometric': '#4C78A8', 'mappo_tail': '#E45756'}


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
    if any(runs.values()):
        return runs

    provenance_path = os.path.join(directory, 'provenance.csv')
    if not os.path.exists(provenance_path):
        return runs
    by_seed = {}
    with open(provenance_path, newline='') as stream:
        for record in csv.DictReader(stream):
            method = record['method']
            if method not in runs:
                continue
            path = os.path.join(
                record['source_directory'], record['source_file'])
            with open(path, newline='') as source:
                rows = list(csv.DictReader(source))
            if rows:
                by_seed.setdefault(int(record['experiment_seed']), {})[method] = rows
    for repeat, seed in enumerate(sorted(by_seed), 1):
        if all(method in by_seed[seed] for method in METHODS):
            for method in METHODS:
                runs[method][repeat] = by_seed[seed][method]
    return runs


def read_summary(directory):
    with open(os.path.join(directory, 'summary.csv'), newline='') as stream:
        return {row['method']: row for row in csv.DictReader(stream)}


def number(row, key, default=0.0):
    raw = row.get(key, '')
    return float(raw) if raw not in ('', None) else default


def coverage_series(rows, timeline):
    elapsed = np.asarray([
        number(row, 'elapsed_sim_s', number(row, 'elapsed_s'))
        for row in rows
    ])
    coverage = np.asarray([number(row, 'coverage_pct') for row in rows])
    return np.interp(timeline, elapsed, coverage,
                     left=coverage[0], right=coverage[-1])


def final_metric(rows, key):
    return number(rows[-1], key)


def final_elapsed(rows):
    return number(rows[-1], 'elapsed_sim_s', number(rows[-1], 'elapsed_s'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation_dir')
    parser.add_argument('-o', '--output', default='final_vs_hu_corrected.png')
    parser.add_argument('--horizon', type=float, default=900.0)
    parser.add_argument('--title', default='Corrected Hu-Aligned Gazebo Evaluation')
    args = parser.parse_args()

    runs = read_runs(args.evaluation_dir)
    summary = read_summary(args.evaluation_dir)
    repeats = sorted(set(runs[METHODS[0]]) & set(runs[METHODS[1]]))
    if not repeats:
        raise ValueError('no seed-matched final Hu comparison runs found')

    timeline = np.arange(0.0, args.horizon + 0.01, 2.0)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.8))

    ax = axes[0, 0]
    for method in METHODS:
        curves = np.vstack([
            coverage_series(runs[method][repeat], timeline)
            for repeat in repeats
        ])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1) if len(repeats) > 1 else np.zeros_like(mean)
        ax.plot(timeline, mean, color=COLORS[method], linewidth=2.0,
                label=LABELS[method])
        ax.fill_between(timeline, np.maximum(mean - std, 0.0),
                        np.minimum(mean + std, 100.0),
                        color=COLORS[method], alpha=0.15, linewidth=0)
    ax.axhline(97.0, color='#666666', linestyle='--', linewidth=1.0)
    ax.set_title('Coverage over simulation time')
    ax.set_xlabel('Gazebo simulation time (s)')
    ax.set_ylabel('Mapped free-space coverage (%)')
    ax.set_xlim(0, args.horizon)
    ax.set_ylim(0, 101)
    ax.legend(frameon=False, fontsize=8, loc='lower right')

    ax = axes[0, 1]
    right_labels = []
    any_failure = False
    for repeat in repeats:
        durations = [final_elapsed(runs[method][repeat])
                     for method in METHODS]
        completed = [final_metric(runs[method][repeat], 'mission_done') >= 0.5
                     for method in METHODS]
        any_failure = any_failure or not all(completed)
        ax.plot((0, 1), durations, color='#A0A0A0', linewidth=1.0, alpha=0.8)
        for index, method in enumerate(METHODS):
            marker = 'o' if completed[index] else 'X'
            ax.scatter(index, durations[index], color=COLORS[method], marker=marker,
                       s=44 if completed[index] else 76, zorder=3,
                       edgecolors='white' if not completed[index] else 'none',
                       linewidths=0.8)
        seed = int(number(runs[METHODS[0]][repeat][-1], 'experiment_seed'))
        right_labels.append((durations[1], str(seed)))
    adjusted_labels = []
    for actual_y, label in sorted(right_labels):
        label_y = actual_y
        if adjusted_labels:
            label_y = max(label_y, adjusted_labels[-1][1] + 19.0)
        adjusted_labels.append((actual_y, label_y, label))
    for actual_y, label_y, label in adjusted_labels:
        ax.plot((1.0, 1.035), (actual_y, label_y),
                color='#777777', linewidth=0.6)
        ax.text(1.045, label_y, label, fontsize=7, va='center')
    ax.axhline(args.horizon, color='#777777', linestyle=':', linewidth=1.0)
    ax.set_title('Seed-matched capped mission duration')
    ax.set_ylabel('Simulation time (s)')
    ax.set_xlim(-0.2, 1.24)
    ax.set_ylim(0, args.horizon * 1.06)
    ax.set_xticks((0, 1))
    ax.set_xticklabels([LABELS[method] for method in METHODS], fontsize=8)
    if any_failure:
        ax.text(0.98, 0.97, 'X = 97% not reached by 900 s',
                transform=ax.transAxes, fontsize=7, va='top', ha='right')

    ax = axes[1, 0]
    success = [100.0 * float(summary[method]['success_rate_97'])
               for method in METHODS]
    bars = ax.bar(range(2), success, color=[COLORS[method] for method in METHODS],
                  edgecolor='#333333', linewidth=0.5)
    for bar, value in zip(bars, success):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2.0,
                '%d/%d' % (round(value * len(repeats) / 100.0), len(repeats)),
                ha='center', fontsize=9)
    ax.set_title('97% coverage completion reliability')
    ax.set_ylabel('Successful runs (%)')
    ax.set_ylim(0, 112)
    ax.set_xticks(range(2))
    ax.set_xticklabels([LABELS[method] for method in METHODS], fontsize=8)

    ax = axes[1, 1]
    for repeat in repeats:
        values = [
            final_metric(runs[method][repeat], 'coverage_pct') /
            max(final_metric(runs[method][repeat], 'path_length_total_m'), 1e-9)
            for method in METHODS
        ]
        ax.plot((0, 1), values, color='#A0A0A0', linewidth=1.0, alpha=0.8)
        ax.scatter((0, 1), values, color=[COLORS[method] for method in METHODS],
                   s=34, zorder=3)
    means = [float(summary[method]['coverage_per_meter_mean']) for method in METHODS]
    ax.plot((0, 1), means, color='#222222', linewidth=2.2, marker='D',
            markersize=5, label='%d-run mean' % len(repeats))
    ax.set_title('Coverage per metre travelled')
    ax.set_ylabel('Coverage (% points/m)')
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks((0, 1))
    ax.set_xticklabels([LABELS[method] for method in METHODS], fontsize=8)
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('%s (%d matched seeds)' % (args.title, len(repeats)),
                 fontsize=13)
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
