#!/usr/bin/env python3
"""Summarize final rows from a 3/5/7-robot Gazebo scaling study."""

import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_runs(root):
    grouped = {}
    for path in sorted(glob.glob(os.path.join(root, 'robots_*_repeat_*.csv'))):
        match = re.search(r'robots_(\d+)_repeat_', os.path.basename(path))
        if not match:
            continue
        with open(path, newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        row = rows[-1]
        count = int(row.get('robot_count') or match.group(1))
        elapsed = float(row.get('elapsed_sim_s') or row.get('elapsed_s') or 0.0)
        grouped.setdefault(count, []).append({
            'coverage': float(row['coverage_pct']),
            'path_per_robot': float(row['path_length_total_m']) / count,
            'near_collisions': float(row.get('near_collision_events') or 0.0),
            'real_time_factor': float(
                row.get('observed_real_time_factor') or 0.0),
            'elapsed': elapsed,
        })
    return grouped


def mean_std(runs, key):
    values = np.asarray([run[key] for run in runs], dtype=np.float64)
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return float(values.mean()), float(std)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('study_root')
    parser.add_argument('-o', '--output', default='scaling_summary.csv')
    parser.add_argument('--figure', default='scaling_summary.png')
    args = parser.parse_args()
    grouped = read_runs(args.study_root)
    if not grouped:
        raise ValueError('no scaling metrics found')

    rows = []
    for count, runs in sorted(grouped.items()):
        coverage = mean_std(runs, 'coverage')
        path = mean_std(runs, 'path_per_robot')
        collisions = mean_std(runs, 'near_collisions')
        factor = mean_std(runs, 'real_time_factor')
        rows.append({
            'robot_count': count, 'runs': len(runs),
            'coverage_mean': coverage[0], 'coverage_std': coverage[1],
            'path_per_robot_mean': path[0], 'path_per_robot_std': path[1],
            'near_collisions_mean': collisions[0],
            'near_collisions_std': collisions[1],
            'real_time_factor_mean': factor[0],
            'real_time_factor_std': factor[1],
        })
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = np.asarray([row['robot_count'] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7))
    plots = (
        ('coverage', 'Coverage at horizon (%)', '#4C78A8'),
        ('path_per_robot', 'Path per robot (m)', '#59A14F'),
        ('real_time_factor', 'Gazebo real-time factor', '#F28E2B'),
    )
    for axis, (key, label, color) in zip(axes, plots):
        means = [row[key + '_mean'] for row in rows]
        errors = [row[key + '_std'] for row in rows]
        axis.errorbar(counts, means, yerr=errors, marker='o', color=color,
                      linewidth=1.8, capsize=4)
        axis.set_xlabel('Number of robots')
        axis.set_ylabel(label)
        axis.set_xticks(counts)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    fig.tight_layout()
    figure = os.path.abspath(args.figure)
    fig.savefig(figure, dpi=240, bbox_inches='tight')
    fig.savefig(os.path.splitext(figure)[0] + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
