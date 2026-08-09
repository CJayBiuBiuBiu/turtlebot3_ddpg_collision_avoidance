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


def build_summary(project_root, eval_root=None, checkpoint_root=None):
    training_dirs = sorted((project_root / 'outputs' / 'training').glob('hu_aligned_*'))
    if eval_root is None:
        eval_root = project_root / 'outputs' / 'evaluation' / 'independent_training_seed_study'
    if checkpoint_root is None:
        checkpoint_root = project_root / 'outputs' / 'checkpoints' / 'independent_seed_study'
    eval_root = eval_root.resolve()
    checkpoint_root = checkpoint_root.resolve()

    training_by_seed = {}
    for directory in training_dirs:
        manifest_path = directory / 'training_manifest.csv'
        if not manifest_path.exists():
            continue
        manifest = read_key_value_csv(manifest_path)
        seed = int(manifest['training_seed'])
        checkpoint = Path(manifest['checkpoint']).resolve()
        is_initial = manifest.get('checkpoint_initial_state') == 'absent_random_initialisation'
        is_target_resume = checkpoint.parent == checkpoint_root
        if is_initial or is_target_resume:
            training_by_seed.setdefault(seed, []).append((directory, manifest))

    records = []
    for eval_dir in sorted(eval_root.glob('training_seed_*')):
        training_seed = int(eval_dir.name.rsplit('_', 1)[-1])
        if training_seed not in training_by_seed:
            raise RuntimeError('No from-scratch training manifest for seed %d' % training_seed)
        training_entries = training_by_seed[training_seed]
        initial_entries = [entry for entry in training_entries
                           if entry[1].get('checkpoint_initial_state') ==
                           'absent_random_initialisation']
        if len(initial_entries) != 1:
            raise RuntimeError('Expected one from-scratch run for seed %d' % training_seed)
        manifest = initial_entries[0][1]
        updates_by_number = {}
        for training_dir, training_manifest in training_entries:
            training_logs = list(training_dir.glob('*_training.csv'))
            if len(training_logs) != 1:
                raise RuntimeError('Expected one training log in %s' % training_dir)
            for row in read_rows(training_logs[0]):
                update_number = int(row.get('update', len(updates_by_number) + 1))
                updates_by_number[update_number] = row
        updates = [updates_by_number[key] for key in sorted(updates_by_number)]
        if not updates:
            raise RuntimeError('Training log is empty for seed %d' % training_seed)

        eval_rows = read_rows(eval_dir / 'summary.csv')
        eval_row = next(row for row in eval_rows if row['method'] == 'mappo_vision')
        checkpoint = checkpoint_root / ('mappo_from_scratch_seed_%d.pt' % training_seed)
        rewards = [float(row['mean_reward']) for row in updates]
        records.append({
            'training_seed': training_seed,
            'updates': max(updates_by_number),
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
            'training_segments': len(training_entries),
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
        default=Path(__file__).resolve().parents[2])
    parser.add_argument('--evaluation-root', type=Path)
    parser.add_argument('--checkpoint-root', type=Path)
    args = parser.parse_args()
    eval_root, records = build_summary(
        args.project_root, args.evaluation_root, args.checkpoint_root)
    per_seed_path, aggregate_path = write_outputs(eval_root, records)
    print('per-seed summary:', per_seed_path)
    print('aggregate summary:', aggregate_path)


if __name__ == '__main__':
    main()
