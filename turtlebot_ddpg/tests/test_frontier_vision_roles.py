import os
import sys
import unittest

import numpy as np


sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'scripts', 'voronoi_exploration'))

from frontier_scoring import (
    assign_by_cost, combine_assignment_cost_terms, combine_cost_terms,
    score_frontier_roles, vision_role_flags)


class FrontierVisionRoleTests(unittest.TestCase):
    def setUp(self):
        self.distance = np.array([[2.0, 3.0], [2.0, 4.0]])
        self.vision = np.array([[0.0, 0.6], [1.0, 0.0]])

    def score(self, role, enabled=True):
        actor, rules = vision_role_flags(role, enabled)
        return score_frontier_roles(
            'execution_aware', self.vision, actor, rules, distance=self.distance)

    def test_both_matches_legacy_scores(self):
        rule_cost, owner_cost, actor_cost, actor_vision = self.score('both')
        expected = combine_cost_terms(
            'execution_aware', distance=self.distance, vision=self.vision)
        np.testing.assert_array_equal(rule_cost, expected)
        np.testing.assert_array_equal(actor_cost, expected)
        np.testing.assert_array_equal(owner_cost, combine_assignment_cost_terms(
            self.distance, vision=self.vision))
        np.testing.assert_array_equal(actor_vision, self.vision)

    def test_actor_only_preserves_the_visual_free_owner_mask_and_rule_cost(self):
        rule_cost, owner_cost, actor_cost, actor_vision = self.score('actor_only')
        no_rule, no_owner, no_actor, _ = self.score('off')
        np.testing.assert_array_equal(rule_cost, no_rule)
        np.testing.assert_array_equal(owner_cost, no_owner)
        self.assertFalse(np.array_equal(actor_cost, no_actor))
        np.testing.assert_array_equal(actor_vision, self.vision)

    def test_rules_only_changes_ownership_without_visual_leak_in_actor_cost(self):
        rule_cost, owner_cost, actor_cost, actor_vision = self.score('rules_only')
        no_rule, no_owner, no_actor, _ = self.score('off')
        self.assertFalse(np.array_equal(rule_cost, no_rule))
        self.assertEqual(assign_by_cost(no_owner)[0], 0)
        self.assertEqual(assign_by_cost(owner_cost)[0], 1)
        np.testing.assert_array_equal(actor_cost, no_actor)
        np.testing.assert_array_equal(actor_vision, np.zeros_like(self.vision))

    def test_disabled_camera_matches_off_and_invalid_role_is_rejected(self):
        for role in ('both', 'actor_only', 'rules_only'):
            for actual, expected in zip(self.score(role, enabled=False), self.score('off')):
                np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(ValueError):
            self.score('invalid')


if __name__ == '__main__':
    unittest.main()
