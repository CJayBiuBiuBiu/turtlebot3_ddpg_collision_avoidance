#!/usr/bin/env python3
"""Aggregate matched IPPO/MAPPO evaluations across training lineages."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS = ('ippo', 'mappo')
METRICS = (
    ('coverage_pct', 'Final coverage (%)'),
    ('path_length_total_m', 'Team path length (m)'),
    ('goals_reached', 'Reached frontier goals'),
)


def final_row(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('empty metrics file: {}'.format(path))
    return rows[-1]


def load_runs(root):
    runs = []
    for lineage in sorted(root.glob('training_seed_*')):
        training_seed = int(lineage.name.rsplit('_', 1)[-1])
        for method in METHODS:
            for path in sorted(lineage.glob('repeat_*_{}.csv'.format(method))):
                repeat = int(path.stem.split('_')[1])
                row = final_row(path)
                run = {
                    'training_seed': training_seed,
                    'evaluation_repeat': repeat,
                    'method': method,
                }
                for key, _ in METRICS:
                    run[key] = float(row.get(key, 0.0) or 0.0)
                runs.append(run)
    if not runs:
        raise ValueError('no policy-baseline runs found under {}'.format(root))
    return runs


def lineage_means(runs):
    rows = []
    training_seeds = sorted({run['training_seed'] for run in runs})
    for training_seed in training_seeds:
        for method in METHODS:
            selected = [run for run in runs
                        if run['training_seed'] == training_seed
                        and run['method'] == method]
            if not selected:
                raise ValueError(
                    'missing {} evaluations for training seed {}'.format(
                        method, training_seed))
            row = {
                'training_seed': training_seed,
                'method': method,
                'evaluation_runs': len(selected),
            }
            for key, _ in METRICS:
                row[key] = float(np.mean([run[key] for run in selected]))
            rows.append(row)
    return rows


def write_runs(path, runs):
    fields = ['training_seed', 'evaluation_repeat', 'method'] + [m[0] for m in METRICS]
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(runs)


def summarize(lineage_rows):
    summary_rows = []
    for method in METHODS:
        selected = [row for row in lineage_rows if row['method'] == method]
        if not selected:
            raise ValueError('no {} lineage means found'.format(method))
        row = {'method': method, 'independent_training_seeds': len(selected)}
        for key, _ in METRICS:
            values = np.asarray([run[key] for run in selected], dtype=float)
            row[key + '_mean'] = float(values.mean())
            row[key + '_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    return summary_rows


def write_summary(path, rows):
    fields = ['method', 'independent_training_seeds']
    for key, _ in METRICS:
        fields.extend([key + '_mean', key + '_std'])
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(path, rows):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11.2, 3.6))
    colors = {'ippo': '#6B7280', 'mappo': '#2563EB'}
    positions = np.arange(len(METHODS))
    for axis, (key, label) in zip(axes, METRICS):
        for training_seed in sorted({row['training_seed'] for row in rows}):
            means = []
            for method in METHODS:
                values = [row[key] for row in rows
                          if row['method'] == method
                          and row['training_seed'] == training_seed]
                means.append(values[0])
            axis.plot(positions, means, color='#B8B8B8', linewidth=1.0, zorder=1)
        for index, method in enumerate(METHODS):
            values = np.asarray([row[key] for row in rows if row['method'] == method])
            jitter = np.linspace(-0.035, 0.035, len(values))
            axis.scatter(np.full(len(values), index) + jitter, values,
                         color=colors[method], s=24, alpha=0.8, zorder=2)
            axis.errorbar(index, values.mean(),
                          yerr=values.std(ddof=1) if len(values) > 1 else 0.0,
                          color='#111827', marker='D', capsize=4, zorder=3)
        axis.set_xticks(positions)
        axis.set_xticklabels(['IPPO', 'MAPPO'])
        axis.set_ylabel(label)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs-output', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.root)
    lineages = lineage_means(runs)
    summary = summarize(lineages)
    for path in (args.output, args.runs_output, args.figure):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_runs(args.runs_output, runs)
    write_summary(args.output, summary)
    plot(args.figure, lineages)
    print(args.output)


if __name__ == '__main__':
    main()
