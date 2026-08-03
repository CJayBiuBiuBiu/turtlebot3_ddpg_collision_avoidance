#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'scripts', 'voronoi_exploration')
sys.path.insert(0, SCRIPT_DIR)

import exploration_coordinator as coordinator_module
from exploration_coordinator import Coordinator, astar_grid


class GridPlanningTests(unittest.TestCase):
    def tearDown(self):
        coordinator_module.configure_world('research')

    def test_known_free_planner_rejects_disconnected_unknown_space(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 0] = 1
        self.assertIsNone(astar_grid(grid, (0, 2), (4, 2)))

    def test_unknown_mode_connects_disjoint_observed_regions(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 0] = 1
        path = astar_grid(
            grid, (0, 2), (4, 2), allow_unknown=True, unknown_cost=2.0)
        self.assertEqual(path[0], (0, 2))
        self.assertEqual(path[-1], (4, 2))

    def test_discovered_obstacle_wall_remains_impassable(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 0] = 1
        grid[:, 2] = 2
        self.assertIsNone(astar_grid(
            grid, (0, 2), (4, 2), allow_unknown=True, unknown_cost=2.0))

    def test_coordinator_plans_from_robot_pose_snapshot(self):
        coordinator = Coordinator.__new__(Coordinator)
        coordinator.grid = np.zeros((5, 5), dtype=np.uint8)
        coordinator.grid[2, 0] = 2
        coordinator.mapping_mode = 'laser_scan'
        path = coordinator._plan_path((0, 2), (4, 2))
        self.assertEqual(path[0], (0, 2))
        self.assertEqual(path[-1], (4, 2))
        self.assertEqual(coordinator.grid[2, 0], 2)

    def test_hu_aligned_world_profile_uses_paper_boundary(self):
        coordinator_module.configure_world('hu_aligned')
        self.assertEqual(coordinator_module.ARENA, (-10.0, -5.0, 10.0, 5.0))
        self.assertEqual(coordinator_module.NX, 100)
        self.assertEqual(coordinator_module.NY, 50)
        self.assertEqual(len(coordinator_module.OBSTACLES), 3)

    def test_layout_transfer_profile_preserves_benchmark_scale(self):
        coordinator_module.configure_world('hu_layout_b')
        self.assertEqual(coordinator_module.ARENA, (-10.0, -5.0, 10.0, 5.0))
        self.assertEqual(coordinator_module.NX, 100)
        self.assertEqual(coordinator_module.NY, 50)
        self.assertEqual(len(coordinator_module.OBSTACLES), 3)
        self.assertNotEqual(
            coordinator_module.WORLD_PROFILES['hu_layout_b']['obstacles'],
            coordinator_module.WORLD_PROFILES['hu_aligned']['obstacles'])

    def test_second_transfer_profile_is_distinct_and_scale_matched(self):
        coordinator_module.configure_world('hu_layout_c')
        self.assertEqual(coordinator_module.ARENA, (-10.0, -5.0, 10.0, 5.0))
        self.assertEqual(coordinator_module.NX, 100)
        self.assertEqual(coordinator_module.NY, 50)
        self.assertEqual(len(coordinator_module.OBSTACLES), 3)
        self.assertNotEqual(
            coordinator_module.WORLD_PROFILES['hu_layout_c']['obstacles'],
            coordinator_module.WORLD_PROFILES['hu_layout_b']['obstacles'])

    def test_unknown_world_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            coordinator_module.configure_world('not_a_world')


if __name__ == '__main__':
    unittest.main()
