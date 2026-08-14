#!/usr/bin/env python3
"""Minimal decentralized ROS inference node for frontier-level MAPPO."""

import os
import sys

import rospy


SCRIPT_DIR = os.path.dirname(__file__)
MARL_DIR = os.path.join(SCRIPT_DIR, 'marl')
sys.path.insert(0, MARL_DIR)

from mappo import MAPPO
from observation_adapter import ObservationSpec, adapt_observation, stack_observations
from turtlebot_ddpg.msg import FrontierAction, FrontierObservation


DEFAULT_ROBOTS = ['robot1', 'robot2', 'robot3']


class MAPPOPolicyNode(object):
    def __init__(self):
        self.team_robot_names = rospy.get_param('~robot_names', DEFAULT_ROBOTS)
        self.robot_name = rospy.get_param('~robot_name', '')
        if self.robot_name and self.robot_name not in self.team_robot_names:
            raise ValueError('robot_name must be one of %s' %
                             self.team_robot_names)
        self.robot_names = ([self.robot_name] if self.robot_name else
                            list(self.team_robot_names))
        self.deterministic = bool(rospy.get_param('~deterministic', True))
        self.allow_untrained = bool(rospy.get_param('~allow_untrained', False))
        self.decision_interval = float(rospy.get_param('~decision_interval', 2.5))
        self.checkpoint = os.path.expanduser(rospy.get_param('~checkpoint', ''))
        self.spec = ObservationSpec(
            bev_size=int(rospy.get_param('~bev_size', 64)),
            max_frontiers=int(rospy.get_param('~max_frontiers', 48)),
            max_teammates=max(0, len(self.team_robot_names) - 1))
        self.policy = MAPPO(max_frontiers=self.spec.max_frontiers)
        self.last_decision_time = dict((name, None) for name in self.robot_names)
        if self.checkpoint:
            metadata = self.policy.load_actor(self.checkpoint)
            rospy.loginfo('[mappo_policy] loaded %s actor checkpoint %s',
                          metadata.get('algorithm', 'mappo'), self.checkpoint)
        elif not self.allow_untrained:
            raise RuntimeError(
                'checkpoint is required unless ~allow_untrained:=true is used for a smoke test')
        else:
            rospy.logwarn('[mappo_policy] using untrained weights for interface testing only')

        self.publishers = {
            name: rospy.Publisher('/%s/frontier_action' % name,
                                  FrontierAction, queue_size=1)
            for name in self.robot_names
        }
        for name in self.robot_names:
            rospy.Subscriber('/%s/frontier_observation' % name,
                             FrontierObservation, self._observation_cb,
                             callback_args=name, queue_size=1)
        rospy.loginfo('[mappo_policy] actors=%s decision_interval=%.2fs',
                      ','.join(self.robot_names), self.decision_interval)

    def _observation_cb(self, message, name):
        stamp_s = message.header.stamp.to_sec()
        previous = self.last_decision_time[name]
        if previous is not None and stamp_s - previous < self.decision_interval:
            return
        self.last_decision_time[name] = stamp_s
        try:
            observation = adapt_observation(message, self.spec)
            batched = stack_observations([observation])
            action_index, _ = self.policy.act(
                batched, deterministic=self.deterministic)
            action_index = int(action_index[0])
        except (ValueError, RuntimeError) as error:
            rospy.logerr_throttle(2.0, '[mappo_policy] observation failed: %s', error)
            return

        action = FrontierAction()
        action.header.stamp = rospy.Time.now()
        action.header.frame_id = 'map'
        action.robot_name = name
        if action_index < self.spec.max_frontiers:
            frontier_id = int(observation['frontier_ids'][action_index])
            if frontier_id >= 0 and observation['action_mask'][action_index]:
                action.action_type = FrontierAction.SELECT
                action.frontier_id = frontier_id
            else:
                action.action_type = FrontierAction.WAIT
                action.frontier_id = -1
        elif action_index == self.spec.max_frontiers:
            action.action_type = FrontierAction.CONTINUE
            action.frontier_id = -1
        else:
            action.action_type = FrontierAction.WAIT
            action.frontier_id = -1
        self.publishers[name].publish(action)


def main():
    rospy.init_node('mappo_policy', anonymous=False)
    MAPPOPolicyNode()
    rospy.spin()


if __name__ == '__main__':
    main()
