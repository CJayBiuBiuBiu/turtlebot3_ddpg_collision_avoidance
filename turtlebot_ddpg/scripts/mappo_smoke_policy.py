#!/usr/bin/env python3
"""Deterministic frontier policy for validating the ROS MAPPO boundary.

This is deliberately not a learning algorithm. It consumes exactly the
messages that a future decentralized MAPPO actor will consume, applies the
published action mask, and returns a FrontierAction. A successful Gazebo run
therefore proves that observations and selected actions affect real robots
before expensive reinforcement-learning training begins.
"""

import time

import rospy

from turtlebot_ddpg.msg import FrontierAction, FrontierObservation


DEFAULT_ROBOTS = ['robot1', 'robot2', 'robot3']


class SmokePolicy(object):
    def __init__(self):
        self.robot_names = rospy.get_param('~robot_names', DEFAULT_ROBOTS)
        self.publish_period = float(rospy.get_param('~publish_period', 0.1))
        self.last_publish = dict((name, 0.0) for name in self.robot_names)
        self.publishers = {
            name: rospy.Publisher('/%s/frontier_action' % name,
                                  FrontierAction, queue_size=1)
            for name in self.robot_names
        }
        for name in self.robot_names:
            rospy.Subscriber('/%s/frontier_observation' % name,
                             FrontierObservation, self._observation_cb,
                             callback_args=name, queue_size=1)
        rospy.loginfo('[mappo_smoke_policy] listening for %s', self.robot_names)

    def _observation_cb(self, observation, name):
        now = time.time()
        if now - self.last_publish[name] < self.publish_period:
            return
        self.last_publish[name] = now

        candidates = [token for token in observation.frontiers
                      if token.assigned and token.reachable]
        action = FrontierAction()
        action.header.stamp = rospy.Time.now()
        action.header.frame_id = 'map'
        action.robot_name = name
        if candidates:
            selected = min(candidates, key=lambda token: token.analytic_cost)
            action.action_type = FrontierAction.SELECT
            action.frontier_id = selected.id
        else:
            action.action_type = FrontierAction.WAIT
            action.frontier_id = -1
        self.publishers[name].publish(action)


def main():
    rospy.init_node('mappo_smoke_policy', anonymous=False)
    SmokePolicy()
    rospy.spin()


if __name__ == '__main__':
    main()
