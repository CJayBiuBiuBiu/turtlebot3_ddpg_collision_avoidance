#!/usr/bin/env python3

import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, SCRIPTS_DIR)

from summarize_vision_seed_study import summarize_nested_rows


class VisionSeedSummaryTests(unittest.TestCase):
    def test_evaluation_repeats_are_nested_within_training_seed(self):
        rows = []
        for seed, values in ((101, (1.0, 3.0)), (202, (5.0, 7.0))):
            for value in values:
                rows.append({
                    'training_seed': str(seed),
                    'mean_coverage_0_90_pct_diff': str(value),
                    'coverage_pct_diff': str(value + 1.0),
                    'path_length_total_m_diff': str(value - 1.0),
                })
        seed_rows, aggregate = summarize_nested_rows(rows)
        self.assertEqual(len(seed_rows), 2)
        early = next(row for row in aggregate
                     if row['metric'] == 'mean_coverage_0_90_pct_diff')
        self.assertEqual(early['training_seeds'], 2)
        self.assertEqual(early['evaluation_pairs'], 4)
        self.assertAlmostEqual(early['mean_of_training_seed_means'], 4.0)
        self.assertEqual(early['positive_training_seeds'], 2)


if __name__ == '__main__':
    unittest.main()
