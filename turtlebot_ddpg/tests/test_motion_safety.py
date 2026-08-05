#!/usr/bin/env python3

import math
import os
import sys
import unittest

import numpy as np


MODULE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts', 'voronoi_exploration'))
sys.path.insert(0, MODULE_DIR)

from motion_safety import clearance_speed_scale, frontal_clearance


class MotionSafetyTests(unittest.TestCase):
    def test_speed_scale_matches_predeclared_boundaries(self):
        self.assertEqual(clearance_speed_scale(0.18), 0.0)
        self.assertEqual(clearance_speed_scale(0.20), 0.0)
        self.assertAlmostEqual(clearance_speed_scale(0.275), 0.5)
        self.assertEqual(clearance_speed_scale(0.35), 1.0)
        self.assertEqual(clearance_speed_scale(math.inf), 1.0)

    def test_frontal_clearance_ignores_rear_obstacle(self):
        values = np.full(24, 2.0, dtype=np.float32)
        angle_increment = 2.0 * math.pi / 24.0
        values[12] = 0.10
        values[0] = 0.42
        self.assertAlmostEqual(
            frontal_clearance(values, 0.0, angle_increment), 0.42, places=5)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            clearance_speed_scale(0.3, stop_range=0.4, slow_range=0.3)


if __name__ == '__main__':
    unittest.main()
