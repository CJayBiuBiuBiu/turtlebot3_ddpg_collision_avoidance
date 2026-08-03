#!/usr/bin/env python3
"""Audit finite MAPPO updates across continuation-training seeds."""

import argparse
import csv
import glob
import math
import os
import re
import statistics


NUMERIC_FIELDS = (
    'coverage_pct', 'mean_reward', 'actor_loss', 'value_loss', 'entropy',
    'approx_kl', 'clip_fraction', 'actor_grad_norm', 'critic_grad_norm',
)


def summarize_training_file(path):
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('%s contains no training updates' % path)
    values = dict((field, [float(row[field]) for row in rows])
                  for field in NUMERIC_FIELDS)
    finite = all(math.isfinite(value) for field in NUMERIC_FIELDS
                 for value in values[field])
    match = re.search(r'seed_(\d+)', os.path.basename(path))
    seed = int(match.group(1)) if match else -1
    return {
        'training_seed': seed,
        'updates': len(rows),
        'first_update': int(rows[0]['update']),
        'last_update': int(rows[-1]['update']),
        'all_finite': int(finite),
        'coverage_mean': statistics.mean(values['coverage_pct']),
        'coverage_last': values['coverage_pct'][-1],
        'reward_mean': statistics.mean(values['mean_reward']),
        'max_abs_kl': max(abs(value) for value in values['approx_kl']),
        'clip_fraction_mean': statistics.mean(values['clip_fraction']),
        'max_actor_grad_norm': max(values['actor_grad_norm']),
        'max_critic_grad_norm': max(values['critic_grad_norm']),
        'source_file': os.path.abspath(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('training_root')
    parser.add_argument('-o', '--output', default='')
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(
        args.training_root, 'hu_aligned_*', 'mappo_seed_*_training.csv')))
    if not paths:
        raise ValueError('no continuation-seed training logs found')
    rows = [summarize_training_file(path) for path in paths]
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.training_root)),
        'evaluation', 'continuation_training_stability.csv')
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print('training seeds: %d' % len(rows))
    print('finite update logs: %d/%d' % (
        sum(row['all_finite'] for row in rows), len(rows)))
    print('max |KL|: %.6f' % max(row['max_abs_kl'] for row in rows))
    print(output)


if __name__ == '__main__':
    main()
