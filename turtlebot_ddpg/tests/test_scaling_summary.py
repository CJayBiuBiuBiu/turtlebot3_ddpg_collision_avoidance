#!/usr/bin/env python3

import csv
import os
import sys
import tempfile
import unittest


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, SCRIPT_DIR)

from summarize_scaling_study import mean_std, read_runs


class ScalingSummaryTests(unittest.TestCase):
    def test_reads_final_row_and_normalizes_path_by_team_size(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'robots_5_repeat_01.csv')
            fields = [
                'robot_count', 'coverage_pct', 'path_length_total_m',
                'near_collision_events', 'observed_real_time_factor',
                'elapsed_sim_s']
            with open(path, 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    'robot_count': 5, 'coverage_pct': 20,
                    'path_length_total_m': 10, 'near_collision_events': 1,
                    'observed_real_time_factor': 0.8, 'elapsed_sim_s': 10})
                writer.writerow({
                    'robot_count': 5, 'coverage_pct': 60,
                    'path_length_total_m': 50, 'near_collision_events': 3,
                    'observed_real_time_factor': 0.7, 'elapsed_sim_s': 20})
            grouped = read_runs(root)
            self.assertEqual(len(grouped[5]), 1)
            self.assertEqual(grouped[5][0]['coverage'], 60.0)
            self.assertEqual(grouped[5][0]['path_per_robot'], 10.0)

    def test_mean_std_returns_population_statistics(self):
        mean, std = mean_std([
            {'coverage': 80.0}, {'coverage': 100.0}], 'coverage')
        self.assertEqual(mean, 90.0)
        self.assertEqual(std, 10.0)


if __name__ == '__main__':
    unittest.main()
