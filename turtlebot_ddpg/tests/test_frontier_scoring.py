#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'scripts', 'voronoi_exploration')
sys.path.insert(0, SCRIPT_DIR)

from frontier_scoring import (
    assign_by_cost, blend_phase_costs, combine_assignment_cost_terms,
    combine_cost_terms, coverage_phase_weight, line_risk_path_cost,
    normalize_global, normalize_rows)


class FrontierScoringTests(unittest.TestCase):
    def test_line_risk_cost_orders_free_unknown_and_blocked_rays(self):
        free = np.ones((5, 5), dtype=np.uint8)
        unknown = free.copy()
        unknown[2, 2:4] = 0
        blocked = free.copy()
        blocked[2, 2] = 2

        free_cost = line_risk_path_cost(free, (0, 2), (4, 2), 0.2)
        unknown_cost = line_risk_path_cost(unknown, (0, 2), (4, 2), 0.2)
        blocked_cost = line_risk_path_cost(blocked, (0, 2), (4, 2), 0.2)

        self.assertLess(free_cost, unknown_cost)
        self.assertLess(unknown_cost, blocked_cost)

    def test_line_risk_cost_rejects_invalid_grid_resolution(self):
        with self.assertRaises(ValueError):
            line_risk_path_cost(np.ones((2, 2)), (0, 0), (1, 1), 0.0)

    def test_normalize_rows_replaces_non_finite_values(self):
        values = np.array([[1.0, np.inf, 3.0], [np.inf, np.inf, np.inf]])
        result = normalize_rows(values)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[0], [0.0, 1.0, 1.0])
        np.testing.assert_allclose(result[1], [1.0, 1.0, 1.0])

    def test_execution_cost_never_masks_all_frontiers(self):
        distance = np.array([[1.0, 2.0], [2.0, 1.0]])
        path = np.array([[np.inf, 3.0], [4.0, np.inf]])
        costs = combine_cost_terms('execution_aware', distance, path=path)
        self.assertTrue(np.isfinite(costs).all())
        self.assertEqual(costs.shape, (2, 2))

    def test_assignment_uses_lowest_robot_cost_per_frontier(self):
        costs = np.array([[0.1, 0.9, 0.4], [0.8, 0.2, 0.5]])
        np.testing.assert_array_equal(assign_by_cost(costs), [0, 1, 0])

    def test_information_gain_reduces_execution_cost(self):
        distance = np.ones((1, 2))
        information = np.array([[0.0, 1.0]])
        costs = combine_cost_terms(
            'execution_aware', distance, path=distance,
            information=information)
        self.assertLess(costs[0, 1], costs[0, 0])

    def test_assignment_costs_preserve_cross_robot_distance_scale(self):
        distance = np.array([[1.0, 100.0], [10.0, 30.0]])
        row_costs = combine_cost_terms(
            'execution_aware', distance, path=distance)
        assignment_costs = combine_assignment_cost_terms(
            distance, path=distance)
        self.assertEqual(assign_by_cost(row_costs)[1], 0)
        self.assertEqual(assign_by_cost(assignment_costs)[1], 1)

    def test_phase_weight_is_bounded_and_reaches_endpoints(self):
        self.assertEqual(coverage_phase_weight(20.0, 40.0, 90.0), 0.0)
        self.assertAlmostEqual(coverage_phase_weight(65.0, 40.0, 90.0), 0.5)
        self.assertEqual(coverage_phase_weight(95.0, 40.0, 90.0), 1.0)
        with self.assertRaises(ValueError):
            coverage_phase_weight(50.0, 90.0, 40.0)

    def test_adaptive_cost_uses_geometry_early_and_execution_late(self):
        geometric = np.array([[1.0, 5.0], [4.0, 2.0]])
        execution = np.array([[0.9, 0.1], [0.2, 0.8]])
        early, early_phase = blend_phase_costs(
            geometric, execution, 20.0, cross_robot=True)
        late, late_phase = blend_phase_costs(
            geometric, execution, 95.0, cross_robot=True)
        np.testing.assert_allclose(early, normalize_global(geometric))
        np.testing.assert_allclose(late, normalize_global(execution))
        self.assertEqual(early_phase, 0.0)
        self.assertEqual(late_phase, 1.0)


if __name__ == '__main__':
    unittest.main()
