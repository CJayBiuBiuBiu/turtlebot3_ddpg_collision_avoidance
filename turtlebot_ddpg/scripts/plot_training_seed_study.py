#!/usr/bin/env python3
"""Plot short MAPPO continuation curves for independent training seeds."""

import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_logs(root):
    runs = {}
    pattern = os.path.join(root, 'hu_aligned_*', 'mappo_seed_*_training.csv')
    for path in sorted(glob.glob(pattern)):
        match = re.search(r'seed_(\d+)', os.path.basename(path))
        if not match:
            continue
        with open(path, newline='') as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            runs[int(match.group(1))] = rows
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('training_root')
    parser.add_argument('-o', '--output', default='continuation_training_curves.png')
    args = parser.parse_args()
    runs = read_logs(args.training_root)
    if not runs:
        raise ValueError('no continuation-seed training logs found')

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    colors = plt.get_cmap('tab10')
    coverage_curves = []
    reward_curves = []
    for index, (seed, rows) in enumerate(sorted(runs.items())):
        steps = np.arange(1, len(rows) + 1)
        coverage = np.asarray([float(row['coverage_pct']) for row in rows])
        reward = np.asarray([float(row['mean_reward']) for row in rows])
        coverage_curves.append(coverage)
        reward_curves.append(reward)
        axes[0].plot(steps, coverage, color=colors(index), alpha=0.35,
                     linewidth=1.0, label='Seed %d' % seed)
        axes[1].plot(steps, reward, color=colors(index), alpha=0.35,
                     linewidth=1.0, label='Seed %d' % seed)

    if len(set(len(curve) for curve in coverage_curves)) == 1:
        steps = np.arange(1, len(coverage_curves[0]) + 1)
        coverage_values = np.asarray(coverage_curves)
        reward_values = np.asarray(reward_curves)
        for axis, values in ((axes[0], coverage_values),
                             (axes[1], reward_values)):
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            axis.plot(steps, mean, color='#222222', linewidth=2.4,
                      label='Mean')
            axis.fill_between(steps, mean - std, mean + std,
                              color='#777777', alpha=0.14, linewidth=0)

    axes[0].set_title('(a) Coverage at each PPO update')
    axes[0].set_ylabel('Coverage (%)')
    axes[1].set_title('(b) Mean rollout reward')
    axes[1].set_ylabel('Mean reward')
    for ax in axes:
        ax.set_xlabel('Continuation update')
        ax.set_xticks(np.arange(1, max(len(rows) for rows in runs.values()) + 1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('MAPPO Continuation Training Across Three Seeds', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches='tight')
    fig.savefig(os.path.splitext(output)[0] + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
