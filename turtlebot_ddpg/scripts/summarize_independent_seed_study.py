#!/usr/bin/env python3
import argparse
import csv
import hashlib
from pathlib import Path
from statistics import mean, stdev


def read_key_value_csv(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return {row['field']: row['value'] for row in rows}


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def sample_std(values):
    return stdev(values) if len(values) > 1 else 0.0


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_summary(project_root):
    training_dirs = sorted((project_root / 'outputs' / 'training').glob('hu_aligned_*'))
    eval_root = project_root / 'outputs' / 'evaluation' / 'independent_training_seed_study'
    checkpoint_root = project_root / 'outputs' / 'checkpoints' / 'independent_seed_study'

    training_by_seed = {}
    for directory in training_dirs:
        manifest_path = directory / 'training_manifest.csv'
        if not manifest_path.exists():
            continue
        manifest = read_key_value_csv(manifest_path)
        if manifest.get('checkpoint_initial_state') != 'absent_random_initialisation':
            continue
        training_by_seed[int(manifest['training_seed'])] = (directory, manifest)

    records = []
    for eval_dir in sorted(eval_root.glob('training_seed_*')):
        training_seed = int(eval_dir.name.rsplit('_', 1)[-1])
        if training_seed not in training_by_seed:
            raise RuntimeError('No from-scratch training manifest for seed %d' % training_seed)
        training_dir, manifest = training_by_seed[training_seed]
        training_logs = list(training_dir.glob('*_training.csv'))
        if len(training_logs) != 1:
            raise RuntimeError('Expected one training log for seed %d' % training_seed)
        updates = read_rows(training_logs[0])
        if not updates:
            raise RuntimeError('Training log is empty for seed %d' % training_seed)

        eval_rows = read_rows(eval_dir / 'summary.csv')
        eval_row = next(row for row in eval_rows if row['method'] == 'mappo_vision')
        checkpoint = checkpoint_root / ('mappo_from_scratch_seed_%d.pt' % training_seed)
        rewards = [float(row['mean_reward']) for row in updates]
        records.append({
            'training_seed': training_seed,
            'updates': len(updates),
            'first_mean_reward': rewards[0],
            'last_mean_reward': rewards[-1],
            'best_mean_reward': max(rewards),
            'mean_training_reward': mean(rewards),
            'training_reward_std': sample_std(rewards),
            'evaluation_runs': int(eval_row['runs']),
            'coverage_pct_mean': float(eval_row['coverage_pct_mean']),
            'coverage_pct_std': float(eval_row['coverage_pct_std']),
            'path_length_total_m_mean': float(eval_row['path_length_total_m_mean']),
            'path_length_total_m_std': float(eval_row['path_length_total_m_std']),
            'goals_reached_mean': float(eval_row['goals_reached_mean']),
            'goals_reached_std': float(eval_row['goals_reached_std']),
            'checkpoint_sha256': sha256(checkpoint),
            'render_backend': manifest['render_backend'],
            'checkpoint_initial_state': manifest['checkpoint_initial_state'],
        })
    if not records:
        raise RuntimeError('No independent-seed evaluation directories found')
    return eval_root, records


def write_outputs(eval_root, records):
    per_seed_path = eval_root / 'independent_seed_summary.csv'
    with per_seed_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    aggregate_fields = [
        'coverage_pct_mean', 'path_length_total_m_mean', 'goals_reached_mean',
        'last_mean_reward', 'best_mean_reward',
    ]
    aggregate = {
        'training_seeds': len(records),
        'evaluation_runs_per_seed': records[0]['evaluation_runs'],
    }
    for field in aggregate_fields:
        values = [float(row[field]) for row in records]
        aggregate[field + '_across_seeds'] = mean(values)
        aggregate[field + '_between_seed_std'] = sample_std(values)
        aggregate[field + '_min'] = min(values)
        aggregate[field + '_max'] = max(values)

    aggregate_path = eval_root / 'independent_seed_aggregate.csv'
    with aggregate_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate))
        writer.writeheader()
        writer.writerow(aggregate)
    return per_seed_path, aggregate_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project-root', type=Path,
        default=Path('/home/ros/catkin_ws/src/turtlebot3_ddpg_collision_avoidance'))
    args = parser.parse_args()
    eval_root, records = build_summary(args.project_root)
    per_seed_path, aggregate_path = write_outputs(eval_root, records)
    print('per-seed summary:', per_seed_path)
    print('aggregate summary:', aggregate_path)


if __name__ == '__main__':
    main()
