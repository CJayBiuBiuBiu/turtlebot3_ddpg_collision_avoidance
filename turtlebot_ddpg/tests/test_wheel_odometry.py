#!/usr/bin/env python3

import math
import os
import sys
import unittest


SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'scripts', 'voronoi_exploration')
sys.path.insert(0, SCRIPT_DIR)

from wheel_odometry import integrate_wheel_positions


class WheelOdometryTests(unittest.TestCase):
    def test_equal_wheel_motion_moves_forward(self):
        pose = integrate_wheel_positions(
            (1.0, 2.0, 0.0), (0.0, 0.0), (1.0, 1.0), 0.1, 0.5)
        self.assertAlmostEqual(pose[0], 1.1)
        self.assertAlmostEqual(pose[1], 2.0)
        self.assertAlmostEqual(pose[2], 0.0)

    def test_opposite_wheel_motion_rotates_in_place(self):
        pose = integrate_wheel_positions(
            (0.0, 0.0, 0.0), (0.0, 0.0), (-1.0, 1.0), 0.1, 0.5)
        self.assertAlmostEqual(pose[0], 0.0)
        self.assertAlmostEqual(pose[1], 0.0)
        self.assertAlmostEqual(pose[2], 0.4)

    def test_motion_respects_current_heading(self):
        pose = integrate_wheel_positions(
            (0.0, 0.0, math.pi / 2.0), (0.0, 0.0),
            (1.0, 1.0), 0.1, 0.5)
        self.assertAlmostEqual(pose[0], 0.0, places=7)
        self.assertAlmostEqual(pose[1], 0.1)

    def test_reset_jump_is_ignored(self):
        pose = integrate_wheel_positions(
            (1.0, 2.0, 0.5), (0.0, 0.0), (5.0, 5.0), 0.1, 0.5)
        self.assertEqual(pose, (1.0, 2.0, 0.5))

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            integrate_wheel_positions(
                (0.0, 0.0, 0.0), (0.0, 0.0), (1.0, 1.0), 0.0, 0.5)


if __name__ == '__main__':
    unittest.main()
