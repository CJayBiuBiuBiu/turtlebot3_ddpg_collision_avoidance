#!/usr/bin/env python3

import os
import sys
from types import SimpleNamespace
import unittest

import numpy as np


MARL_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts', 'marl'))
sys.path.insert(0, MARL_DIR)

try:
    import torch
    from mappo import MAPPO, RolloutBuffer
    from observation_adapter import ObservationSpec, adapt_observation
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed in this Python environment')
class MAPPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)
        self.spec = ObservationSpec(max_frontiers=8)

    def make_team_observation(self, robots=3):
        return {
            'bev': np.random.rand(robots, 3, 64, 64).astype(np.float32),
            'tokens': np.random.rand(robots, 8, 11).astype(np.float32),
            'robot_state': np.random.rand(robots, 5).astype(np.float32),
            'teammates': np.random.rand(robots, 2, 7).astype(np.float32),
            'teammate_mask': np.ones((robots, 2), dtype=np.bool_),
            'action_mask': np.ones((robots, 10), dtype=np.bool_),
            'frontier_ids': np.tile(np.arange(8), (robots, 1)),
        }

    def test_masked_actor_never_selects_invalid_frontier(self):
        policy = MAPPO(max_frontiers=8)
        obs = self.make_team_observation()
        obs['action_mask'][:, :8] = False
        actions, _ = policy.act(obs)
        self.assertTrue(np.all(actions >= 8))

    def test_commitment_mask_blocks_non_selectable_frontier(self):
        occupancy = SimpleNamespace(
            data=[-1] * (64 * 64),
            info=SimpleNamespace(width=64, height=64))

        def token(frontier_id, selectable):
            return SimpleNamespace(
                id=frontier_id,
                position=SimpleNamespace(x=float(frontier_id), y=0.0),
                analytic_cost=0.0,
                distance=1.0,
                path_cost=1.0,
                obstacle_risk=0.0,
                collision_risk=0.0,
                overlap_risk=0.0,
                communication_risk=0.0,
                information_gain=1.0,
                vision_openness=1.0,
                assigned=True,
                reachable=True,
                selectable=selectable)

        message = SimpleNamespace(
            local_bev=occupancy,
            robot_pose=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
            coverage_pct=10.0,
            frontiers=[token(1, True), token(2, False)],
            teammate_intents=[])
        observation = adapt_observation(message, self.spec)
        self.assertTrue(observation['action_mask'][0])
        self.assertFalse(observation['action_mask'][1])
        self.assertTrue(observation['action_mask'][-2])
        self.assertTrue(observation['action_mask'][-1])

    def test_synthetic_rollout_update_is_finite(self):
        policy = MAPPO(max_frontiers=8)
        buffer = RolloutBuffer()
        for step in range(4):
            obs = self.make_team_observation()
            actions, log_probs = policy.act(obs)
            value = float(policy.team_value({
                key: value[None, ...] for key, value in obs.items()
            }).detach().cpu()[0])
            buffer.add(obs, actions, log_probs, reward=0.1 * step,
                       value=value, done=False)
        batch = buffer.build_batch(last_value=0.0)
        diagnostics = policy.update(batch, epochs=1, minibatch_size=12)
        for value in diagnostics.values():
            self.assertTrue(np.isfinite(value))
        self.assertIn('actor_grad_norm', diagnostics)
        self.assertIn('critic_grad_norm', diagnostics)


if __name__ == '__main__':
    unittest.main()
