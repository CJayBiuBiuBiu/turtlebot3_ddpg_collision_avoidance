#!/usr/bin/env python3
"""Plot matched multi-seed MAPPO/IPPO learning curves without chart grids."""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = {'ippo': '#59A14F', 'mappo': '#4C78A8'}
LABELS = {'ippo': 'IPPO', 'mappo': 'MAPPO'}


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size < window:
        return values
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    smooth = (prefix[window:] - prefix[:-window]) / float(window)
    return np.concatenate([values[:window - 1], smooth])


def read_runs(root, algorithm):
    pattern = os.path.join(root, algorithm, 'seed_*', '*_training.csv')
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline='') as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            runs.append({
                'coverage': np.asarray(
                    [float(row['coverage_pct']) for row in rows]),
                'reward': np.asarray(
                    [float(row['mean_reward']) for row in rows]),
            })
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('study_root')
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('-o', '--output', default='learning_comparison.png')
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    found = False
    for algorithm in ('ippo', 'mappo'):
        runs = read_runs(args.study_root, algorithm)
        if not runs:
            continue
        found = True
        length = min(len(run['coverage']) for run in runs)
        steps = np.arange(1, length + 1)
        for axis, key in zip(axes, ('coverage', 'reward')):
            curves = np.stack([
                moving_average(run[key][:length], args.window)
                for run in runs])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            axis.plot(steps, mean, color=COLORS[algorithm], linewidth=2.0,
                      label=LABELS[algorithm])
            axis.fill_between(steps, mean - std, mean + std,
                              color=COLORS[algorithm], alpha=0.16,
                              linewidth=0)
    if not found:
        raise ValueError('no learning logs found under %s' % args.study_root)

    axes[0].set_title('(a) Coverage during training')
    axes[0].set_ylabel('Coverage (%)')
    axes[1].set_title('(b) Mean rollout reward')
    axes[1].set_ylabel('Mean reward')
    for axis in axes:
        axis.set_xlabel('PPO update')
        axis.legend(frameon=False)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    fig.tight_layout()
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches='tight')
    fig.savefig(os.path.splitext(output)[0] + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
