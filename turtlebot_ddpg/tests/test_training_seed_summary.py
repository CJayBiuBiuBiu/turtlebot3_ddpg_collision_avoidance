#!/usr/bin/env python3

import csv
import os
import sys
import tempfile
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, SCRIPTS_DIR)

from summarize_training_seed_study import summarize_training_file


class TrainingSeedSummaryTests(unittest.TestCase):
    def test_finite_training_log_is_audited(self):
        fields = [
            'update', 'coverage_pct', 'mean_reward', 'actor_loss',
            'value_loss', 'entropy', 'approx_kl', 'clip_fraction',
            'actor_grad_norm', 'critic_grad_norm']
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'mappo_seed_101_training.csv')
            with open(path, 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                row = dict((field, 0.1) for field in fields)
                row['update'] = 23
                writer.writerow(row)
            result = summarize_training_file(path)
        self.assertEqual(result['training_seed'], 101)
        self.assertEqual(result['updates'], 1)
        self.assertEqual(result['all_finite'], 1)


if __name__ == '__main__':
    unittest.main()
