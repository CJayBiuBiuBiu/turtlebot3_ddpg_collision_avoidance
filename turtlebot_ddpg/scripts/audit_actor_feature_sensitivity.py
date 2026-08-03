#!/usr/bin/env python3
"""Audit how strongly a frozen MAPPO actor reacts to each frontier feature."""

import argparse
import csv
import hashlib
import os
import sys

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'marl'))

from networks import FrontierActor


FEATURE_NAMES = (
    'dx', 'dy', 'analytic_cost', 'distance', 'path_cost', 'obstacle_risk',
    'collision_risk', 'overlap_risk', 'communication_risk',
    'information_gain', 'vision_openness')


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(checkpoint, samples=32, seed=123, max_frontiers=48):
    torch.manual_seed(int(seed))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    actor = FrontierActor(max_frontiers=max_frontiers)
    actor.load_state_dict(payload['actor'])
    actor.eval()

    bev = torch.zeros(samples, 3, 64, 64)
    bev[:, 0] = 1.0
    tokens = torch.rand(samples, max_frontiers, len(FEATURE_NAMES)) * 0.5
    robot_state = torch.zeros(samples, 5)
    teammates = torch.zeros(samples, 2, 7)
    teammate_mask = torch.zeros(samples, 2, dtype=torch.bool)
    action_mask = torch.ones(samples, max_frontiers + 2, dtype=torch.bool)

    rows = []
    with torch.no_grad():
        for index, name in enumerate(FEATURE_NAMES):
            low = tokens.clone()
            high = tokens.clone()
            low[:, :, index] = 0.0
            high[:, :, index] = 1.0
            low_logits = actor(
                bev, low, robot_state, teammates, teammate_mask,
                action_mask)[:, :max_frontiers]
            high_logits = actor(
                bev, high, robot_state, teammates, teammate_mask,
                action_mask)[:, :max_frontiers]
            difference = high_logits - low_logits
            rows.append({
                'feature': name,
                'mean_absolute_logit_change': float(difference.abs().mean()),
                'mean_signed_logit_change': float(difference.mean()),
            })

    metadata = payload.get('metadata', {})
    return rows, {
        'checkpoint': os.path.abspath(checkpoint),
        'checkpoint_sha256': checkpoint_sha256(checkpoint),
        'checkpoint_update': metadata.get('updates', 'unknown'),
        'samples': int(samples),
        'seed': int(seed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--output')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error('--samples must be positive')

    rows, metadata = audit(
        args.checkpoint, samples=args.samples, seed=args.seed)
    fieldnames = (
        'checkpoint', 'checkpoint_sha256', 'checkpoint_update', 'samples',
        'seed', 'feature', 'mean_absolute_logit_change',
        'mean_signed_logit_change')
    output_rows = []
    for row in rows:
        combined = dict(metadata)
        combined.update(row)
        output_rows.append(combined)

    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

    for row in sorted(
            output_rows,
            key=lambda item: item['mean_absolute_logit_change'],
            reverse=True):
        print('%-20s %.6f' % (
            row['feature'], row['mean_absolute_logit_change']))


if __name__ == '__main__':
    main()
