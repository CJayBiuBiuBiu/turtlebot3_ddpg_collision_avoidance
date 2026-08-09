#!/usr/bin/env python3

import os
import sys
import unittest


MODULE_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'scripts', 'voronoi_exploration')
sys.path.insert(0, os.path.abspath(MODULE_DIR))

from communication_model import CommunicationModel


class CommunicationModelTests(unittest.TestCase):
    def test_default_model_delivers_current_message(self):
        model = CommunicationModel(seed=1)
        payload = {'target': 7}
        self.assertEqual(
            model.exchange('r1', 'r2', 0.0, True, payload), payload)

    def test_delay_and_staleness_are_applied(self):
        model = CommunicationModel(delay_s=1.0, stale_after_s=0.5, seed=1)
        payload = {'target': 3}
        self.assertIsNone(model.exchange('r1', 'r2', 0.0, True, payload))
        self.assertEqual(
            model.exchange('r1', 'r2', 1.0, False, None), payload)
        self.assertIsNone(model.exchange('r1', 'r2', 1.6, False, None))

    def test_full_packet_loss_blocks_delivery(self):
        model = CommunicationModel(drop_probability=1.0, seed=1)
        self.assertIsNone(
            model.exchange('r1', 'r2', 0.0, True, {'target': 2}))
        self.assertEqual(model.diagnostics()['communication_messages_dropped'], 1)

    def test_bandwidth_limits_repeated_messages(self):
        model = CommunicationModel(
            bandwidth_bytes_per_s=24.0, message_bytes=48, seed=1)
        self.assertIsNotNone(
            model.exchange('r1', 'r2', 0.0, True, {'target': 1}))
        model.exchange('r1', 'r2', 0.1, True, {'target': 2})
        self.assertEqual(
            model.diagnostics()['communication_bandwidth_blocked'], 1)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            CommunicationModel(drop_probability=1.1)
        with self.assertRaises(ValueError):
            CommunicationModel(delay_s=-1.0)


if __name__ == '__main__':
    unittest.main()

