#!/usr/bin/env python3
"""Combine compatible Gazebo evaluation directories by experiment seed."""

import argparse
import csv
import os

from summarize_evaluation import (
    paired_differences,
    read_final_rows,
    summarize,
    write_paired_summary,
    write_summary,
)


COMPATIBILITY_FIELDS = (
    'checkpoint_sha256',
    'world_profile',
    'episode_seconds',
    'coverage_threshold',
)


def read_manifest(directory):
    path = os.path.join(directory, 'experiment_manifest.csv')
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    return {row['field']: row['value'] for row in rows}


def validate_manifests(directories):
    manifests = [read_manifest(directory) for directory in directories]
    reference = manifests[0]
    for directory, manifest in zip(directories[1:], manifests[1:]):
        for field in COMPATIBILITY_FIELDS:
            if manifest.get(field) != reference.get(field):
                raise ValueError(
                    '%s differs for %s: %r != %r' % (
                        field, directory, manifest.get(field),
                        reference.get(field)))
    return reference


def combine_rows(directories):
    rows_by_seed_method = {}
    provenance = []
    for directory in directories:
        for row in read_final_rows(directory):
            seed = str(row.get('experiment_seed', '')).strip()
            if not seed:
                raise ValueError('%s contains a row without experiment_seed' % directory)
            method = row['selection_mode']
            key = (seed, method)
            if key in rows_by_seed_method:
                raise ValueError('duplicate seed/method pair: %s/%s' % key)
            rows_by_seed_method[key] = row
            provenance.append({
                'experiment_seed': seed,
                'method': method,
                'source_directory': os.path.abspath(directory),
                'source_file': row['_file'],
            })

    seed_order = {
        seed: index + 1
        for index, seed in enumerate(sorted(
            {seed for seed, _ in rows_by_seed_method}, key=int))
    }
    combined = []
    for (seed, method), row in sorted(
            rows_by_seed_method.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        row = dict(row)
        row['_file'] = 'repeat_%02d_%s.csv' % (seed_order[seed], method)
        combined.append(row)
    return combined, provenance


def write_rows(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_directory')
    parser.add_argument('directories', nargs='+')
    parser.add_argument(
        '--pair', action='append', nargs=3,
        metavar=('POSITIVE', 'NEGATIVE', 'OUTPUT'), default=[])
    args = parser.parse_args()

    manifest = validate_manifests(args.directories)
    rows, provenance = combine_rows(args.directories)
    os.makedirs(args.output_directory, exist_ok=True)
    write_summary(
        os.path.join(args.output_directory, 'summary.csv'), summarize(rows))
    write_rows(
        os.path.join(args.output_directory, 'provenance.csv'), provenance)
    with open(os.path.join(args.output_directory, 'combined_manifest.csv'),
              'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('field', 'value'))
        for field in COMPATIBILITY_FIELDS:
            writer.writerow((field, manifest.get(field, '')))
        writer.writerow(('source_directories', ';'.join(
            os.path.abspath(directory) for directory in args.directories)))

    for positive, negative, filename in args.pair:
        write_paired_summary(
            os.path.join(args.output_directory, filename),
            paired_differences(rows, positive, negative))

    print('combined %d runs from %d directories' % (
        len(rows), len(args.directories)))
    print('output: %s' % args.output_directory)


if __name__ == '__main__':
    main()
