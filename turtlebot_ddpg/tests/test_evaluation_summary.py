#!/usr/bin/env python3

import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, SCRIPTS_DIR)

from summarize_evaluation import (
    analysis_elapsed, exact_two_sided_sign_test, paired_differences, summarize,
    t_critical_95, time_weighted_coverage)


class EvaluationSummaryTests(unittest.TestCase):
    def test_small_sample_t_critical_values(self):
        self.assertAlmostEqual(t_critical_95(3), 4.303)
        self.assertAlmostEqual(t_critical_95(5), 2.776)
        self.assertAlmostEqual(exact_two_sided_sign_test(4, 1), 0.375)

    def test_named_ablation_pair_difference(self):
        rows = []
        for repeat, vision, no_vision in ((1, 12.0, 10.0), (2, 13.0, 10.0)):
            for method, coverage in (
                    ('mappo_vision', vision), ('mappo_no_vision', no_vision)):
                rows.append({
                    '_file': 'repeat_%02d_%s.csv' % (repeat, method),
                    'selection_mode': method,
                    'coverage_pct': str(coverage),
                    'path_length_total_m': '10',
                    'goals_reached': '8',
                    'goal_assignments': '10',
                    'stuck_events': '0',
                    'near_collision_events': '0',
                })
        result = paired_differences(
            rows, positive_method='mappo_vision',
            negative_method='mappo_no_vision')
        coverage = next(row for row in result if row['metric'] == 'coverage_pct')
        self.assertAlmostEqual(coverage['positive_minus_negative_mean'], 2.5)

    def test_threshold_time_uses_successful_runs_only(self):
        rows = [
            {'selection_mode': 'method', 'time_to_80_s': '120',
             'time_90_to_97_s': '40',
             'mean_coverage_0_90_pct': '20',
             'mean_coverage_0_300_pct': '50'},
            {'selection_mode': 'method', 'time_to_80_s': '',
             'time_90_to_97_s': '',
             'mean_coverage_0_90_pct': '10',
             'mean_coverage_0_300_pct': '40'},
        ]
        result = summarize(rows)[0]
        self.assertAlmostEqual(result['success_rate_80'], 0.5)
        self.assertAlmostEqual(result['time_to_80_s_mean'], 120.0)
        self.assertAlmostEqual(result['time_90_to_97_s_mean'], 40.0)

    def test_tail_completion_is_in_paired_differences(self):
        rows = []
        for repeat, tail, baseline in ((1, 20.0, 50.0), (2, 30.0, 60.0)):
            for method, tail_time in (
                    ('mappo_tail', tail), ('mappo_hybrid', baseline)):
                rows.append({
                    '_file': 'repeat_%02d_%s.csv' % (repeat, method),
                    'selection_mode': method,
                    'time_90_to_97_s': str(tail_time),
                    'coverage_pct': '97',
                    'path_length_total_m': '100',
                    'goals_reached': '8',
                    'goal_assignments': '10',
                    'stuck_events': '1',
                    'near_collision_events': '0',
                })
        result = paired_differences(
            rows, positive_method='mappo_tail',
            negative_method='mappo_hybrid')
        tail = next(row for row in result
                    if row['metric'] == 'time_90_to_97_s')
        self.assertAlmostEqual(tail['positive_minus_negative_mean'], -30.0)
        self.assertEqual(tail['preferred_direction'], 'lower')
        self.assertEqual(tail['positive_better_pairs'], 2)
        self.assertAlmostEqual(tail['sign_test_p'], 0.5)

    def test_fixed_horizon_coverage_auc(self):
        records = [
            {'elapsed_s': '100', 'coverage_pct': '50'},
            {'elapsed_s': '200', 'coverage_pct': '100'},
        ]
        # Linear growth to 100% by 200 s, then completed coverage is held.
        self.assertAlmostEqual(time_weighted_coverage(records, 300), 66.6666667)

    def test_gazebo_time_is_preferred_over_wall_time(self):
        record = {
            'elapsed_s': '100',
            'elapsed_wall_s': '100',
            'elapsed_sim_s': '75',
        }
        self.assertAlmostEqual(analysis_elapsed(record), 75.0)

    def test_legacy_elapsed_time_remains_supported(self):
        self.assertAlmostEqual(analysis_elapsed({'elapsed_s': '42'}), 42.0)

    def test_paired_difference_rejects_mismatched_seeds(self):
        rows = [
            {
                '_file': 'repeat_01_mappo.csv',
                'selection_mode': 'mappo',
                'experiment_seed': '1000',
                'coverage_pct': '12',
                'path_length_total_m': '10',
                'goals_reached': '8',
                'goal_assignments': '10',
                'stuck_events': '0',
                'near_collision_events': '0',
            },
            {
                '_file': 'repeat_01_execution_aware.csv',
                'selection_mode': 'execution_aware',
                'experiment_seed': '1001',
                'coverage_pct': '10',
                'path_length_total_m': '10',
                'goals_reached': '8',
                'goal_assignments': '10',
                'stuck_events': '0',
                'near_collision_events': '0',
            },
        ]
        with self.assertRaises(ValueError):
            paired_differences(rows)


if __name__ == '__main__':
    unittest.main()
