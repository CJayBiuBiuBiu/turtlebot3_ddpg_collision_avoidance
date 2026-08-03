#!/usr/bin/env python3

import math
import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, SCRIPTS_DIR)

from rank_checkpoint_validation import rank_candidates


class CheckpointRankingTests(unittest.TestCase):
    def test_fixed_horizon_coverage_precedes_completion_status(self):
        complete = {
            'success_rate_97': 1.0,
            'mean_coverage_0_90_pct': 30.0,
            'mean_coverage_0_300_pct': 40.0,
            'time_to_97_s_mean': 250.0,
            'path_length_total_m_mean': 100.0,
            'checkpoint_sha256': 'b',
        }
        incomplete = {
            'success_rate_97': 0.0,
            'mean_coverage_0_90_pct': 35.0,
            'mean_coverage_0_300_pct': 80.0,
            'time_to_97_s_mean': math.nan,
            'path_length_total_m_mean': 80.0,
            'checkpoint_sha256': 'a',
        }
        self.assertIs(rank_candidates([incomplete, complete])[0], incomplete)

    def test_coverage_breaks_equal_success_tie(self):
        lower = {
            'success_rate_97': 0.0,
            'mean_coverage_0_90_pct': 30.0,
            'mean_coverage_0_300_pct': 30.0,
            'time_to_97_s_mean': math.nan,
            'path_length_total_m_mean': 80.0,
            'checkpoint_sha256': 'a',
        }
        higher = dict(lower, mean_coverage_0_90_pct=35.0,
                      checkpoint_sha256='b')
        self.assertIs(rank_candidates([lower, higher])[0], higher)


if __name__ == '__main__':
    unittest.main()
