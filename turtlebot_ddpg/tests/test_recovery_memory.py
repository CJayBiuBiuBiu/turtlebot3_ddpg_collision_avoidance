#!/usr/bin/env python3

import os
import sys
import unittest


SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'scripts', 'voronoi_exploration')
sys.path.insert(0, SCRIPT_DIR)

from recovery_memory import TimedSpatialBlacklist


class TimedSpatialBlacklistTests(unittest.TestCase):
    def test_nearby_reports_merge_and_refresh_expiry(self):
        memory = TimedSpatialBlacklist(ttl_seconds=90, merge_radius=1.0)
        self.assertTrue(memory.add(0.0, 0.0, now=0.0))
        self.assertFalse(memory.add(0.5, 0.0, now=60.0))
        self.assertEqual(memory.count(now=100.0), 1)
        self.assertEqual(memory.active_points(now=100.0), [(0.5, 0.0)])

    def test_expired_region_can_be_explored_again(self):
        memory = TimedSpatialBlacklist(ttl_seconds=90, merge_radius=1.0)
        memory.add(0.0, 0.0, now=0.0)
        self.assertEqual(memory.count(now=90.0), 0)
        self.assertTrue(memory.add(0.0, 0.0, now=91.0))

    def test_separate_regions_remain_separate(self):
        memory = TimedSpatialBlacklist(ttl_seconds=90, merge_radius=1.0)
        memory.add(0.0, 0.0, now=0.0)
        memory.add(2.0, 0.0, now=0.0)
        self.assertEqual(memory.count(now=1.0), 2)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            TimedSpatialBlacklist(ttl_seconds=0)
        with self.assertRaises(ValueError):
            TimedSpatialBlacklist(merge_radius=-1)


if __name__ == '__main__':
    unittest.main()
