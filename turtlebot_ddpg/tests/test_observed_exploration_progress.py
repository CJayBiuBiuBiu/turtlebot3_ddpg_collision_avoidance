import os
import sys
import unittest

import numpy as np


sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'scripts', 'voronoi_exploration'))

from hybrid_fallback import (
    CoverageStallFallback, OBSERVED_FREE_AREA, REFERENCE_COVERAGE,
    validate_progress_mode)
from scan_mapping import clear_robot_footprints, update_grid_from_scan


class ObservedExplorationProgressTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.zeros((10, 10), dtype=np.uint8)
        self.observed_free = np.zeros_like(self.grid, dtype=bool)
        self.arena = (-5.0, -5.0, 5.0, 5.0)

    def observe(self, rays, yaw=0.0):
        return update_grid_from_scan(
            self.grid, (0.0, 0.0), yaw, rays, 0.0, 1.0, 0.1, 4.0,
            self.arena, 1.0, inflation_cells=0, observed_free=self.observed_free)

    def test_counts_unique_ray_cells_and_excludes_hit_endpoint(self):
        self.assertEqual(self.observe([2.0]), 1)
        self.assertEqual(int(self.observed_free.sum()), 2)
        self.assertFalse(self.observed_free[5, 7])
        self.observe([2.0])
        self.assertEqual(int(self.observed_free.sum()), 2)
        self.observe([2.0], yaw=np.pi / 2)
        self.assertEqual(int(self.observed_free.sum()), 3)

    def test_bad_scan_and_forced_footprint_do_not_add_progress(self):
        self.assertEqual(self.observe([float('nan'), -1.0]), 0)
        clear_robot_footprints(self.grid, [(1.0, 1.0)], self.arena, 1.0)
        self.assertGreater(int(self.grid.sum()), 0)
        self.assertEqual(int(self.observed_free.sum()), 0)

    def test_accepts_only_frozen_laser_audit_with_positive_horizon(self):
        validate_progress_mode(OBSERVED_FREE_AREA, 'laser_scan', 300, False, False)
        validate_progress_mode(REFERENCE_COVERAGE, 'ground_truth_radius', 0, True, True)
        bad_options = (
            ('ground_truth_radius', 300, False, False),
            ('laser_scan', 0, False, False),
            ('laser_scan', float('inf'), False, False),
            ('laser_scan', 300, True, False),
            ('laser_scan', 300, False, True),
        )
        for options in bad_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_progress_mode(OBSERVED_FREE_AREA, *options)

    def test_area_progress_resets_stall_and_opens_bounded_fallback(self):
        fallback = CoverageStallFallback(
            stall_seconds=60, active_seconds=30, progress_epsilon=0.5,
            minimum_coverage=0.0)
        self.observe([2.0])
        self.assertEqual(fallback.update(self.observed_free.sum(), 0), (False, False))
        self.observe([2.0], yaw=np.pi / 2)
        self.assertEqual(fallback.update(self.observed_free.sum(), 59), (False, False))
        self.assertEqual(fallback.update(self.observed_free.sum(), 118), (False, False))
        self.assertEqual(fallback.update(self.observed_free.sum(), 119), (True, True))
        self.assertEqual(fallback.update(self.observed_free.sum(), 149), (False, False))


if __name__ == '__main__':
    unittest.main()
