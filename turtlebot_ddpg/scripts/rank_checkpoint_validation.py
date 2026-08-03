#!/usr/bin/env python3
"""Rank frozen MAPPO checkpoints from validation-only Gazebo runs."""

import argparse
import csv
import glob
import math
import os


def read_manifest(path):
    with open(path, newline='') as stream:
        return dict(csv.reader(stream))


def read_candidate(directory, expected_world_profile='research'):
    manifest_path = os.path.join(directory, 'experiment_manifest.csv')
    summary_path = os.path.join(directory, 'summary.csv')
    if not os.path.isfile(manifest_path) or not os.path.isfile(summary_path):
        return None
    manifest = read_manifest(manifest_path)
    if manifest.get('world_profile') != expected_world_profile:
        raise ValueError(
            '%s uses %s rather than validation profile %s' %
            (directory, manifest.get('world_profile'), expected_world_profile))
    with open(summary_path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    policy_rows = [row for row in rows if row.get('method') == 'mappo_vision']
    if len(policy_rows) != 1:
        raise ValueError('%s must contain one mappo_vision row' % directory)
    row = policy_rows[0]

    def number(name, default=math.nan):
        value = row.get(name, '')
        return float(value) if value not in ('', None) else default

    return {
        'directory': directory,
        'checkpoint': manifest.get('checkpoint', ''),
        'checkpoint_sha256': manifest.get('checkpoint_sha256', ''),
        'checkpoint_update': manifest.get('checkpoint_update', ''),
        'runs': int(row.get('runs', 0) or 0),
        'success_rate_97': number('success_rate_97', 0.0),
        'mean_coverage_0_90_pct': number(
            'mean_coverage_0_90_pct_mean', 0.0),
        'mean_coverage_0_300_pct': number(
            'mean_coverage_0_300_pct_mean', 0.0),
        'time_to_97_s_mean': number('time_to_97_s_mean'),
        'path_length_total_m_mean': number('path_length_total_m_mean'),
    }


def rank_candidates(candidates):
    def finite_or(value, fallback):
        return value if math.isfinite(value) else fallback

    return sorted(candidates, key=lambda item: (
        -item['mean_coverage_0_90_pct'],
        finite_or(item['time_to_97_s_mean'], math.inf),
        finite_or(item['path_length_total_m_mean'], math.inf),
        item['checkpoint_sha256']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('validation_root')
    parser.add_argument('--world-profile', default='research')
    parser.add_argument('--output', default='checkpoint_ranking.csv')
    args = parser.parse_args()

    candidates = []
    for directory in sorted(glob.glob(os.path.join(
            args.validation_root, 'candidate_*'))):
        candidate = read_candidate(directory, args.world_profile)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise SystemExit('no completed checkpoint validation directories found')

    ranked = rank_candidates(candidates)
    output_path = os.path.join(args.validation_root, args.output)
    fieldnames = ['rank'] + list(ranked[0].keys())
    with open(output_path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(ranked, 1):
            writer.writerow(dict(candidate, rank=rank))
    selected_path = os.path.join(
        args.validation_root, 'selected_checkpoint.txt')
    with open(selected_path, 'w') as stream:
        stream.write(ranked[0]['checkpoint'] + '\n')
    print('selected checkpoint: %s' % ranked[0]['checkpoint'])
    print('ranking: %s' % output_path)


if __name__ == '__main__':
    main()
