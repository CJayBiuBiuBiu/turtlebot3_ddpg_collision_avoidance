#!/usr/bin/env python3
"""Summarize nested vision-mask evaluations by independent training seed."""

import argparse
import csv
import math
import os
import statistics

from summarize_evaluation import t_critical_95


METRICS = (
    'mean_coverage_0_90_pct_diff',
    'coverage_pct_diff',
    'path_length_total_m_diff',
)


def summarize_nested_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row['training_seed']), []).append(row)

    seed_rows = []
    for seed, group in sorted(grouped.items(), key=lambda item: int(item[0])):
        summary = {'training_seed': seed, 'evaluation_pairs': len(group)}
        for metric in METRICS:
            summary[metric + '_mean'] = statistics.mean(
                float(row[metric]) for row in group)
        seed_rows.append(summary)

    aggregate_rows = []
    for metric in METRICS:
        values = [float(row[metric + '_mean']) for row in seed_rows]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        half_width = (t_critical_95(len(values)) * std / math.sqrt(len(values))
                      if len(values) > 1 else 0.0)
        aggregate_rows.append({
            'metric': metric,
            'training_seeds': len(values),
            'evaluation_pairs': len(rows),
            'mean_of_training_seed_means': mean,
            'between_training_seed_std': std,
            'ci95_low': mean - half_width,
            'ci95_high': mean + half_width,
            'positive_training_seeds': sum(value > 0.0 for value in values),
            'negative_training_seeds': sum(value < 0.0 for value in values),
            'per_training_seed_means': ';'.join(str(value) for value in values),
        })
    return seed_rows, aggregate_rows


def write_rows(path, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('--output-dir', default='')
    args = parser.parse_args()

    with open(args.input_csv, newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('vision seed study contains no rows')
    seed_rows, aggregate_rows = summarize_nested_rows(rows)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input_csv))
    os.makedirs(output_dir, exist_ok=True)
    seed_path = os.path.join(output_dir, 'vision_effect_by_training_seed.csv')
    aggregate_path = os.path.join(output_dir, 'vision_effect_hierarchical_summary.csv')
    write_rows(seed_path, seed_rows)
    write_rows(aggregate_path, aggregate_rows)
    for row in aggregate_rows:
        print('%s: %.3f [%.3f, %.3f], positive seeds=%d/%d' % (
            row['metric'], row['mean_of_training_seed_means'],
            row['ci95_low'], row['ci95_high'],
            row['positive_training_seeds'], row['training_seeds']))
    print(aggregate_path)


if __name__ == '__main__':
    main()
