#!/usr/bin/env python3
"""Convert ROS frontier observations into fixed-size MAPPO tensors."""

from __future__ import division

import numpy as np


BEV_CHANNELS = 3
TOKEN_FEATURES = 11
ROBOT_FEATURES = 5
TEAMMATE_FEATURES = 7


class ObservationSpec(object):
    def __init__(self, bev_size=64, max_frontiers=48, max_teammates=2,
                 arena_width=30.0, arena_height=15.0):
        self.bev_size = int(bev_size)
        self.max_frontiers = int(max_frontiers)
        self.max_teammates = int(max_teammates)
        self.arena_width = float(arena_width)
        self.arena_height = float(arena_height)
        self.map_diagonal = float(np.hypot(arena_width, arena_height))

    @property
    def action_dim(self):
        return self.max_frontiers + 2


def occupancy_to_channels(occupancy, size):
    data = np.asarray(occupancy.data, dtype=np.int16)
    expected = int(occupancy.info.width) * int(occupancy.info.height)
    if data.size != expected:
        raise ValueError('occupancy data length does not match width*height')
    if occupancy.info.width != size or occupancy.info.height != size:
        raise ValueError('expected %dx%d local BEV, got %dx%d' % (
            size, size, occupancy.info.width, occupancy.info.height))
    grid = data.reshape(size, size)
    return np.stack([
        (grid < 0).astype(np.float32),
        (grid == 0).astype(np.float32),
        (grid >= 50).astype(np.float32),
    ], axis=0)


def adapt_observation(observation, spec=None):
    """Return fixed arrays for one decentralized actor observation."""
    spec = spec or ObservationSpec()
    bev = occupancy_to_channels(observation.local_bev, spec.bev_size)

    tokens = np.zeros((spec.max_frontiers, TOKEN_FEATURES), dtype=np.float32)
    frontier_ids = np.full((spec.max_frontiers,), -1, dtype=np.int64)
    action_mask = np.zeros((spec.action_dim,), dtype=np.bool_)
    px, py = observation.robot_pose.x, observation.robot_pose.y
    for index, token in enumerate(observation.frontiers[:spec.max_frontiers]):
        tokens[index] = np.asarray([
            (token.position.x - px) / spec.arena_width,
            (token.position.y - py) / spec.arena_height,
            np.clip(token.analytic_cost, -2.0, 2.0),
            token.distance / spec.map_diagonal,
            token.path_cost / spec.map_diagonal,
            token.obstacle_risk,
            token.collision_risk,
            token.overlap_risk,
            token.communication_risk,
            token.information_gain,
            token.vision_openness,
        ], dtype=np.float32)
        frontier_ids[index] = token.id
        # The coordinator applies target commitment here: while a robot has
        # a valid goal, other frontier actions remain visible as features but
        # are not selectable. This prevents a policy decision from replacing
        # a healthy low-level navigation goal every decision interval.
        action_mask[index] = bool(token.selectable)

    has_active_goal = bool(getattr(observation, 'has_active_goal', False))
    has_selectable_frontier = bool(action_mask[:spec.max_frontiers].any())
    # CONTINUE is meaningful only while a low-level target is active. WAIT can
    # abandon an active target, or keep the policy finite when no frontier is
    # currently selectable. With an idle robot and valid frontiers, both extra
    # actions are masked so the policy cannot collapse into doing nothing.
    action_mask[spec.max_frontiers] = has_active_goal
    action_mask[spec.max_frontiers + 1] = (
        has_active_goal or not has_selectable_frontier)

    theta = observation.robot_pose.theta
    robot_state = np.asarray([
        px / (0.5 * spec.arena_width),
        py / (0.5 * spec.arena_height),
        np.sin(theta),
        np.cos(theta),
        observation.coverage_pct / 100.0,
    ], dtype=np.float32)

    teammates = np.zeros(
        (spec.max_teammates, TEAMMATE_FEATURES), dtype=np.float32)
    teammate_mask = np.zeros((spec.max_teammates,), dtype=np.bool_)
    for index, intent in enumerate(
            observation.teammate_intents[:spec.max_teammates]):
        if not intent.received:
            continue
        teammate_mask[index] = True
        target_dx = intent.target_position.x - px
        target_dy = intent.target_position.y - py
        teammates[index] = np.asarray([
            (intent.pose.x - px) / spec.arena_width,
            (intent.pose.y - py) / spec.arena_height,
            np.sin(intent.pose.theta),
            np.cos(intent.pose.theta),
            intent.link_quality,
            target_dx / spec.arena_width,
            target_dy / spec.arena_height,
        ], dtype=np.float32)

    return {
        'bev': bev,
        'tokens': tokens,
        'robot_state': robot_state,
        'teammates': teammates,
        'teammate_mask': teammate_mask,
        'action_mask': action_mask,
        'frontier_ids': frontier_ids,
    }


def stack_observations(observations):
    if not observations:
        raise ValueError('at least one observation is required')
    keys = observations[0].keys()
    return dict((key, np.stack([obs[key] for obs in observations], axis=0))
                for key in keys)
