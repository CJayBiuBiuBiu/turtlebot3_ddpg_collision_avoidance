#!/usr/bin/env python3

import math
import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'scripts', 'voronoi_exploration')
sys.path.insert(0, SCRIPT_DIR)

from scan_mapping import (
    FREE, OCCUPIED, clear_robot_footprints, update_grid_from_scan)


class ScanMappingTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.zeros((10, 10), dtype=np.uint8)
        self.arena = (-5.0, -5.0, 5.0, 5.0)

    def update(self, ranges, yaw=0.0, angle_min=0.0,
               angle_increment=1.0, inflation_cells=0):
        return update_grid_from_scan(
            self.grid, (0.0, 0.0), yaw, ranges, angle_min,
            angle_increment, 0.1, 4.0, self.arena, 1.0,
            inflation_cells=inflation_cells)

    def test_finite_hit_marks_free_ray_and_occupied_endpoint(self):
        self.assertEqual(self.update([2.0]), 1)
        self.assertEqual(self.grid[5, 5], FREE)
        self.assertEqual(self.grid[5, 6], FREE)
        self.assertEqual(self.grid[5, 7], OCCUPIED)

    def test_infinite_return_clears_without_creating_obstacle(self):
        self.update([float('inf')])
        self.assertTrue(np.all(self.grid[5, 5:10] == FREE))
        self.assertFalse(np.any(self.grid == OCCUPIED))

    def test_free_ray_does_not_erase_previous_obstacle(self):
        self.update([2.0])
        self.update([float('inf')])
        self.assertEqual(self.grid[5, 7], OCCUPIED)

    def test_robot_yaw_rotates_scan_into_world_frame(self):
        self.update([2.0], yaw=math.pi / 2.0)
        self.assertEqual(self.grid[7, 5], OCCUPIED)

    def test_close_inflated_hit_cannot_mark_robot_cell_occupied(self):
        self.update([0.2], inflation_cells=1)
        self.assertEqual(self.grid[5, 5], FREE)

    def test_robot_footprints_clear_dynamic_robot_returns(self):
        self.grid[5, 7] = OCCUPIED
        clear_robot_footprints(
            self.grid, [(2.0, 0.0)], self.arena, 1.0, radius_cells=1)
        self.assertEqual(self.grid[5, 7], FREE)
        self.assertEqual(self.grid[5, 6], FREE)


if __name__ == '__main__':
    unittest.main()
