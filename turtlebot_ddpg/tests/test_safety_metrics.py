#!/usr/bin/env python3

import os
import sys
import unittest


SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'scripts',
    'voronoi_exploration')
sys.path.insert(0, SCRIPT_DIR)

from safety_metrics import ClearanceTelemetry


class ClearanceTelemetryTests(unittest.TestCase):
    def test_counts_samples_and_integrates_only_previous_low_state(self):
        telemetry = ClearanceTelemetry(event_threshold=0.18)
        telemetry.observe('robot1', 0.15, 1.0)
        telemetry.observe('robot1', 0.16, 1.1)
        telemetry.observe('robot1', 0.30, 1.2)
        result = telemetry.snapshot()
        self.assertEqual(result['sample_count'], 3)
        self.assertEqual(result['below_threshold_samples'], 2)
        self.assertAlmostEqual(result['sample_ratio'], 2.0 / 3.0)
        self.assertAlmostEqual(result['active_seconds'], 0.2)
        self.assertAlmostEqual(result['minimum_clearance'], 0.15)

    def test_caps_missing_sample_gap_and_ignores_time_reset(self):
        telemetry = ClearanceTelemetry(event_threshold=0.18, max_sample_gap=0.5)
        telemetry.observe('robot1', 0.10, 2.0)
        telemetry.observe('robot1', 0.10, 4.0)
        telemetry.observe('robot1', 0.10, 1.0)
        self.assertAlmostEqual(telemetry.snapshot()['active_seconds'], 0.5)

    def test_invalid_clearance_is_ignored(self):
        telemetry = ClearanceTelemetry()
        telemetry.observe('robot1', float('nan'), 1.0)
        telemetry.observe('robot1', -1.0, 2.0)
        self.assertEqual(telemetry.snapshot()['sample_count'], 0)


if __name__ == '__main__':
    unittest.main()
