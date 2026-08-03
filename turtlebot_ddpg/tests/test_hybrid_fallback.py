#!/usr/bin/env python3

import os
import sys
import unittest


SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'scripts', 'voronoi_exploration')
sys.path.insert(0, SCRIPT_DIR)

from hybrid_fallback import CoverageStallFallback


class CoverageStallFallbackTests(unittest.TestCase):
    def test_progress_resets_stall_timer(self):
        fallback = CoverageStallFallback(
            stall_seconds=60, active_seconds=30, progress_epsilon=0.2,
            minimum_coverage=20)
        self.assertEqual(fallback.update(25.0, 0.0), (False, False))
        self.assertEqual(fallback.update(25.3, 50.0), (False, False))
        self.assertEqual(fallback.update(25.3, 100.0), (False, False))
        self.assertEqual(fallback.update(25.3, 111.0), (True, True))

    def test_activation_is_bounded_and_can_repeat(self):
        fallback = CoverageStallFallback(
            stall_seconds=60, active_seconds=30, progress_epsilon=0.2,
            minimum_coverage=20)
        fallback.update(80.0, 0.0)
        self.assertEqual(fallback.update(80.1, 60.0), (True, True))
        self.assertEqual(fallback.update(80.1, 89.0), (False, True))
        self.assertEqual(fallback.update(81.0, 90.0), (False, False))
        self.assertEqual(fallback.update(81.0, 150.0), (True, True))
        self.assertEqual(fallback.activations, 2)

    def test_minimum_coverage_prevents_startup_fallback(self):
        fallback = CoverageStallFallback(
            stall_seconds=60, active_seconds=30, progress_epsilon=0.2,
            minimum_coverage=20)
        fallback.update(0.0, 0.0)
        self.assertEqual(fallback.update(10.0, 120.0), (False, False))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            CoverageStallFallback(stall_seconds=0)
        with self.assertRaises(ValueError):
            CoverageStallFallback(progress_epsilon=0)


if __name__ == '__main__':
    unittest.main()
