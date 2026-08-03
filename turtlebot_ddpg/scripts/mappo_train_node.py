#!/usr/bin/env python3
"""Minimal online MAPPO trainer for one Gazebo exploration episode."""

import csv
import os
import random
import sys
import threading

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Point
from std_msgs.msg import Bool


SCRIPT_DIR = os.path.dirname(__file__)
MARL_DIR = os.path.join(SCRIPT_DIR, 'marl')
sys.path.insert(0, MARL_DIR)

from mappo import MAPPO, RolloutBuffer
from observation_adapter import ObservationSpec, adapt_observation, stack_observations
from turtlebot_ddpg.msg import FrontierAction, FrontierObservation


DEFAULT_ROBOTS = ['robot1', 'robot2', 'robot3']


class MAPPOTrainerNode(object):
    def __init__(self):
        self.training_seed = int(rospy.get_param('~training_seed', 0))
        random.seed(self.training_seed)
        np.random.seed(self.training_seed)
        torch.manual_seed(self.training_seed)
        self.robot_names = rospy.get_param('~robot_names', DEFAULT_ROBOTS)
        self.spec = ObservationSpec(
            bev_size=int(rospy.get_param('~bev_size', 64)),
            max_frontiers=int(rospy.get_param('~max_frontiers', 48)),
            max_teammates=max(0, len(self.robot_names) - 1))
        self.rollout_length = int(rospy.get_param('~rollout_length', 32))
        self.decision_interval = float(rospy.get_param('~decision_interval', 2.5))
        self.coverage_reward = float(rospy.get_param('~coverage_reward', 1.0))
        self.step_penalty = float(rospy.get_param('~step_penalty', 0.01))
        self.near_collision_penalty = float(
            rospy.get_param('~near_collision_penalty', 0.02))
        self.stuck_penalty = float(rospy.get_param('~stuck_penalty', 1.0))
        self.target_switch_penalty = float(
            rospy.get_param('~target_switch_penalty', 0.05))
        self.done_coverage = float(rospy.get_param('~done_coverage', 97.0))
        self.update_epochs = int(rospy.get_param('~update_epochs', 5))
        self.minibatch_size = int(rospy.get_param('~minibatch_size', 96))
        self.checkpoint = os.path.expanduser(rospy.get_param(
            '~checkpoint', '~/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/'
                           'outputs/checkpoints/mappo_latest.pt'))
        self.snapshot_interval = max(
            0, int(rospy.get_param('~snapshot_interval', 5)))
        default_snapshot_dir = os.path.join(
            os.path.dirname(self.checkpoint), 'snapshots')
        self.snapshot_dir = os.path.expanduser(rospy.get_param(
            '~snapshot_dir', default_snapshot_dir))
        self.training_log = os.path.expanduser(rospy.get_param(
            '~training_log', '~/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/'
                             'outputs/mappo_training.csv'))
        self.diagnostics_log = os.path.expanduser(rospy.get_param(
            '~diagnostics_log', '~/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/'
                                'outputs/mappo_diagnostics.csv'))

        self.policy = MAPPO(max_frontiers=self.spec.max_frontiers)
        self.update_count = 0
        if os.path.isfile(self.checkpoint):
            metadata = self.policy.load(self.checkpoint, load_optimizers=True)
            self.update_count = int(metadata.get('updates', 0))
            rospy.loginfo('[mappo_train] resumed %s at update %d',
                          self.checkpoint, self.update_count)
        self.buffer = RolloutBuffer()
        self.pending = None
        self.last_decision_time = None
        self.messages_by_stamp = {}
        self.near_collision_events = 0
        self.stuck_events = 0
        self.last_selected_frontier = dict((name, None) for name in self.robot_names)
        self._reset_rollout_diagnostics()
        self.lock = threading.Lock()

        self.publishers = {
            name: rospy.Publisher('/%s/frontier_action' % name,
                                  FrontierAction, queue_size=1)
            for name in self.robot_names
        }
        for name in self.robot_names:
            rospy.Subscriber('/%s/frontier_observation' % name,
                             FrontierObservation, self._observation_cb,
                             callback_args=name, queue_size=1)
            rospy.Subscriber('/%s/near_collision_event' % name, Bool,
                             self._near_collision_cb, queue_size=10)
            rospy.Subscriber('/%s/stuck_event' % name, Point,
                             self._stuck_cb, queue_size=10)
        self._prepare_log()
        rospy.loginfo('[mappo_train] seed=%d, rollout=%d, decision_interval=%.2fs',
                      self.training_seed, self.rollout_length, self.decision_interval)

    def _prepare_log(self):
        directory = os.path.dirname(self.training_log)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        if not os.path.exists(self.training_log):
            with open(self.training_log, 'w', newline='') as stream:
                csv.writer(stream).writerow([
                    'update', 'coverage_pct', 'mean_reward', 'actor_loss',
                    'value_loss', 'entropy', 'approx_kl', 'clip_fraction',
                    'actor_grad_norm', 'critic_grad_norm'])
        directory = os.path.dirname(self.diagnostics_log)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        if not os.path.exists(self.diagnostics_log):
            with open(self.diagnostics_log, 'w', newline='') as stream:
                csv.writer(stream).writerow([
                    'update', 'coverage_pct', 'transitions',
                    'coverage_gain_sum', 'coverage_reward_sum',
                    'step_penalty_sum', 'near_collision_events',
                    'near_collision_penalty_sum', 'stuck_events',
                    'stuck_penalty_sum', 'target_switches',
                    'target_switch_penalty_sum', 'mean_valid_frontiers',
                    'select_actions', 'continue_actions', 'wait_actions'])

    def _reset_rollout_diagnostics(self):
        self.rollout_diagnostics = {
            'transitions': 0,
            'coverage_gain': 0.0,
            'near_collisions': 0,
            'stuck_events': 0,
            'target_switches': 0,
            'valid_frontiers': 0,
            'robot_decisions': 0,
            'select_actions': 0,
            'continue_actions': 0,
            'wait_actions': 0,
        }

    def _near_collision_cb(self, message):
        if message.data:
            with self.lock:
                self.near_collision_events += 1

    def _stuck_cb(self, _message):
        with self.lock:
            self.stuck_events += 1

    def _observation_cb(self, message, name):
        stamp = message.header.stamp.to_nsec()
        with self.lock:
            group = self.messages_by_stamp.setdefault(stamp, {})
            group[name] = message
            if len(group) < len(self.robot_names):
                return
            messages = self.messages_by_stamp.pop(stamp)
            self.messages_by_stamp = dict(
                (key, value) for key, value in self.messages_by_stamp.items()
                if key > stamp)
        self._process_team_observation(messages)

    def _process_team_observation(self, messages):
        stamp_s = messages[self.robot_names[0]].header.stamp.to_sec()
        if (self.last_decision_time is not None and
                stamp_s - self.last_decision_time < self.decision_interval):
            return
        self.last_decision_time = stamp_s
        adapted = [adapt_observation(messages[name], self.spec)
                   for name in self.robot_names]
        team_obs = stack_observations(adapted)
        coverage = float(messages[self.robot_names[0]].coverage_pct)

        if self.pending is not None:
            with self.lock:
                near_collisions = self.near_collision_events
                stuck_events = self.stuck_events
                self.near_collision_events = 0
                self.stuck_events = 0
            coverage_gain = coverage - self.pending['coverage']
            reward = (self.coverage_reward * coverage_gain - self.step_penalty
                      - self.near_collision_penalty * near_collisions
                      - self.stuck_penalty * stuck_events
                      - self.target_switch_penalty * self.pending['switches'])
            self.rollout_diagnostics['transitions'] += 1
            self.rollout_diagnostics['coverage_gain'] += coverage_gain
            self.rollout_diagnostics['near_collisions'] += near_collisions
            self.rollout_diagnostics['stuck_events'] += stuck_events
            self.rollout_diagnostics['target_switches'] += self.pending['switches']
            done = coverage >= self.done_coverage
            self.buffer.add(
                self.pending['observation'], self.pending['actions'],
                self.pending['log_probs'], reward, self.pending['value'], done)
            if len(self.buffer) >= self.rollout_length or done:
                self._update(team_obs, coverage, done)
            if done:
                self._publish_wait()
                self.pending = None
                rospy.loginfo('[mappo_train] episode complete at %.1f%% coverage', coverage)
                return

        actions, log_probs = self.policy.act(team_obs, deterministic=False)
        team_batch = dict((key, value[None, ...]) for key, value in team_obs.items())
        value = float(self.policy.team_value(team_batch).detach().cpu()[0])
        self._record_action_diagnostics(actions, adapted)
        switches = self._publish_actions(actions, adapted)
        self.pending = {
            'observation': team_obs,
            'actions': actions,
            'log_probs': log_probs,
            'value': value,
            'coverage': coverage,
            'switches': switches,
        }

    def _update(self, current_obs, coverage, done):
        if done:
            last_value = 0.0
        else:
            team_batch = dict((key, value[None, ...])
                              for key, value in current_obs.items())
            last_value = float(self.policy.team_value(team_batch).detach().cpu()[0])
        rewards = list(self.buffer.rewards)
        batch = self.buffer.build_batch(last_value=last_value)
        diagnostics = self.policy.update(
            batch, epochs=self.update_epochs, minibatch_size=self.minibatch_size)
        self.update_count += 1
        checkpoint_metadata = {
            'updates': self.update_count,
            'training_seed': self.training_seed,
            'selection_rule': 'training_endpoint_not_validation_selected',
        }
        self.policy.save(self.checkpoint, metadata=checkpoint_metadata)
        if (self.snapshot_interval > 0 and
                self.update_count % self.snapshot_interval == 0):
            if not os.path.isdir(self.snapshot_dir):
                os.makedirs(self.snapshot_dir)
            snapshot_path = os.path.join(
                self.snapshot_dir,
                'mappo_update_%04d.pt' % self.update_count)
            if os.path.exists(snapshot_path):
                rospy.logwarn(
                    '[mappo_train] immutable snapshot already exists: %s',
                    snapshot_path)
            else:
                snapshot_metadata = dict(checkpoint_metadata)
                snapshot_metadata['selection_rule'] = (
                    'periodic_training_snapshot_not_validation_selected')
                self.policy.save(snapshot_path, metadata=snapshot_metadata)
                rospy.loginfo('[mappo_train] saved snapshot %s', snapshot_path)
        with open(self.training_log, 'a', newline='') as stream:
            csv.writer(stream).writerow([
                self.update_count, '%.4f' % coverage,
                '%.6f' % (sum(rewards) / max(1, len(rewards))),
                diagnostics['actor_loss'], diagnostics['value_loss'],
                diagnostics['entropy'], diagnostics['approx_kl'],
                diagnostics['clip_fraction'], diagnostics['actor_grad_norm'],
                diagnostics['critic_grad_norm']])
        rollout = self.rollout_diagnostics
        transitions = max(1, rollout['transitions'])
        robot_decisions = max(1, rollout['robot_decisions'])
        with open(self.diagnostics_log, 'a', newline='') as stream:
            csv.writer(stream).writerow([
                self.update_count, '%.4f' % coverage, rollout['transitions'],
                rollout['coverage_gain'],
                self.coverage_reward * rollout['coverage_gain'],
                self.step_penalty * transitions,
                rollout['near_collisions'],
                self.near_collision_penalty * rollout['near_collisions'],
                rollout['stuck_events'],
                self.stuck_penalty * rollout['stuck_events'],
                rollout['target_switches'],
                self.target_switch_penalty * rollout['target_switches'],
                float(rollout['valid_frontiers']) / robot_decisions,
                rollout['select_actions'], rollout['continue_actions'],
                rollout['wait_actions']])
        rospy.loginfo(
            '[mappo_train] update=%d reward=%.3f switches=%d actor=%.4f value=%.4f kl=%.5f',
            self.update_count, sum(rewards) / max(1, len(rewards)),
            rollout['target_switches'],
            diagnostics['actor_loss'], diagnostics['value_loss'],
            diagnostics['approx_kl'])
        self._reset_rollout_diagnostics()
        self.buffer.clear()

    def _record_action_diagnostics(self, actions, observations):
        for action_index, observation in zip(actions, observations):
            action_index = int(action_index)
            self.rollout_diagnostics['valid_frontiers'] += int(sum(
                observation['action_mask'][:self.spec.max_frontiers]))
            self.rollout_diagnostics['robot_decisions'] += 1
            if action_index < self.spec.max_frontiers:
                self.rollout_diagnostics['select_actions'] += 1
            elif action_index == self.spec.max_frontiers:
                self.rollout_diagnostics['continue_actions'] += 1
            else:
                self.rollout_diagnostics['wait_actions'] += 1

    def _publish_actions(self, actions, observations):
        switches = 0
        for name, action_index, observation in zip(
                self.robot_names, actions, observations):
            message = FrontierAction()
            message.header.stamp = rospy.Time.now()
            message.header.frame_id = 'map'
            message.robot_name = name
            action_index = int(action_index)
            if action_index < self.spec.max_frontiers:
                frontier_id = int(observation['frontier_ids'][action_index])
                if frontier_id >= 0 and observation['action_mask'][action_index]:
                    message.action_type = FrontierAction.SELECT
                    message.frontier_id = frontier_id
                    previous = self.last_selected_frontier[name]
                    if previous is not None and previous != frontier_id:
                        switches += 1
                    self.last_selected_frontier[name] = frontier_id
                else:
                    message.action_type = FrontierAction.WAIT
                    message.frontier_id = -1
            elif action_index == self.spec.max_frontiers:
                message.action_type = FrontierAction.CONTINUE
                message.frontier_id = -1
            else:
                message.action_type = FrontierAction.WAIT
                message.frontier_id = -1
            self.publishers[name].publish(message)
        return switches

    def _publish_wait(self):
        empty = [self.spec.max_frontiers + 1] * len(self.robot_names)
        observations = [
            {'frontier_ids': [-1] * self.spec.max_frontiers,
             'action_mask': [False] * self.spec.max_frontiers + [True, True]}
            for _ in self.robot_names]
        self._publish_actions(empty, observations)


def main():
    rospy.init_node('mappo_train', anonymous=False)
    MAPPOTrainerNode()
    rospy.spin()


if __name__ == '__main__':
    main()
