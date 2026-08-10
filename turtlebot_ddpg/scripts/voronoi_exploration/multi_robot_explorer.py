#!/usr/bin/env python
# Deploys ONE robot's trained DDPG collision-avoidance actor network in a
# continuous navigate-to-target loop. Run one instance of this script per
# robot (namespace given as the first CLI arg), driven by target waypoints
# published by the Voronoi exploration coordinator node on
# /<robot_name>/exploration_target.
#
# Position is read from /gazebo/model_states (ground truth) rather than TF,
# to sidestep multi-robot TF-prefix plumbing -- this mirrors the simplified
# "known position" exploration-map assumption used by the coordinator.

import sys
import threading
import math
import time
import random
import os

import rospy
import numpy as np
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Bool, Float32

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.compat.v1.keras.backend as K

from motion_safety import (
    clearance_guided_turn_direction, clearance_speed_scale,
    frontal_clearance)

STATE_DIM = 28
WEIGHTS_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'fd_replay', 'play_human_data',
    'actormodel-450-500.h5'))
NEAR_COLLISION_RANGE = 0.18
NEAR_COLLISION_CLEAR_RANGE = 0.25

# The trained DDPG policy has no concept of "I'm wedged in a concave corner"
# -- training only ever saw a straight corridor. When that happens it can
# keep commanding forward motion that the physics engine just absorbs as
# wheel-slip against the obstacle. Detect near-zero displacement despite
# ongoing commands and run a short open-loop back-up-and-turn to break out.
STUCK_CHECK_WINDOW = 3.0    # seconds
STUCK_DIST_THRESHOLD = 0.15  # meters
RECOVERY_DURATION = 1.5     # seconds
USE_SIM_TIME = False


def control_time():
    """Use Gazebo simulation time for controller state-machine timers."""
    return rospy.get_time() if USE_SIM_TIME else time.time()


class InfoGetter(object):
    def __init__(self):
        self._event = threading.Event()
        self._msg = None

    def __call__(self, msg):
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=None):
        self._event.wait(timeout)
        return self._msg


def build_actor_model():
    state_input = Input(shape=(STATE_DIM,))
    h1 = Dense(500, activation='relu')(state_input)
    h2 = Dense(500, activation='relu')(h1)
    h3 = Dense(500, activation='relu')(h2)
    delta_theta = Dense(1, activation='tanh')(h3)
    speed = Dense(1, activation='sigmoid')(h3)
    output = Concatenate()([delta_theta, speed])
    return Model(inputs=state_input, outputs=output)


class RobotAgent(object):
    def __init__(self, robot_name, experiment_seed=0,
                 near_collision_range=NEAR_COLLISION_RANGE,
                 near_collision_clear_range=NEAR_COLLISION_CLEAR_RANGE,
                 safety_shield_enabled=False,
                 safety_stop_range=0.20,
                 safety_slow_range=0.35):
        self.robot_name = robot_name
        robot_index = int(robot_name[-1]) if robot_name[-1:].isdigit() else 0
        self.experiment_seed = int(experiment_seed) + robot_index
        self.rng = random.Random(self.experiment_seed)
        self.pub = rospy.Publisher('/%s/cmd_vel' % robot_name, Twist, queue_size=1)

        self.laser_ig = InfoGetter()
        rospy.Subscriber('/%s/laserscan_filtered' % robot_name, LaserScan, self.laser_ig)

        self.model_states_ig = InfoGetter()
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_ig)

        self.target_x = 0.0
        self.target_y = 0.0
        self.have_target = False
        rospy.Subscriber('/%s/exploration_target' % robot_name, Point, self._target_cb)

        self.mission_done = False
        rospy.Subscriber('/%s/exploration_done' % robot_name, Bool, self._done_cb)

        self.stuck_pub = rospy.Publisher('/%s/stuck_event' % robot_name, Point, queue_size=1)
        self.recovery_displacement_pub = rospy.Publisher(
            '/%s/recovery_displacement_m' % robot_name,
            Float32, queue_size=1)
        self.near_collision_pub = rospy.Publisher(
            '/%s/near_collision_event' % robot_name, Bool, queue_size=1)
        self.min_clearance_pub = rospy.Publisher(
            '/%s/min_clearance_m' % robot_name, Float32, queue_size=1)
        self.near_collision_range = float(near_collision_range)
        self.near_collision_clear_range = float(near_collision_clear_range)
        if self.near_collision_range <= 0.0:
            raise ValueError('near_collision_range must be positive')
        if self.near_collision_clear_range <= self.near_collision_range:
            raise ValueError(
                'near_collision_clear_range must exceed near_collision_range')
        self.near_collision_active = False
        self.safety_shield_enabled = bool(safety_shield_enabled)
        self.safety_stop_range = float(safety_stop_range)
        self.safety_slow_range = float(safety_slow_range)
        clearance_speed_scale(
            math.inf, self.safety_stop_range, self.safety_slow_range)

        self.prev_linear = 0.0
        self.prev_angular = 0.0

    def _target_cb(self, msg):
        self.target_x = msg.x
        self.target_y = msg.y
        self.have_target = True

    def _done_cb(self, msg):
        if msg.data and not self.mission_done:
            rospy.loginfo("[%s] exploration complete, stopping", self.robot_name)
        self.mission_done = bool(msg.data)

    def get_pose(self):
        ms = self.model_states_ig.get_msg(timeout=2.0)
        idx = ms.name.index(self.robot_name)
        pos = ms.pose[idx].position
        ori = ms.pose[idx].orientation
        # yaw from quaternion (z-rotation only, matches planar robot assumption)
        siny_cosp = 2 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1 - 2 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return pos.x, pos.y, yaw

    def build_state(self):
        x, y, yaw = self.get_pose()
        laser_msg = self.laser_ig.get_msg(timeout=2.0)
        if laser_msg is None:
            raise RuntimeError('laser scan unavailable for %s' % self.robot_name)
        range_max = laser_msg.range_max if laser_msg.range_max > 0 else 3.5
        ranges = np.asarray(laser_msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=range_max, posinf=range_max, neginf=0.0)
        ranges = np.clip(ranges, 0.0, range_max)
        if ranges.size != 24:
            raise RuntimeError(
                'expected 24 filtered laser beams for %s, got %d' %
                (self.robot_name, ranges.size))
        min_range = float(np.min(ranges))
        self.min_clearance_pub.publish(Float32(data=min_range))
        if (not self.near_collision_active and
                min_range <= self.near_collision_range):
            self.near_collision_pub.publish(Bool(data=True))
            self.near_collision_active = True
        elif (self.near_collision_active and
              min_range >= self.near_collision_clear_range):
            self.near_collision_active = False
        normalized_laser = (ranges / range_max).tolist()

        dx, dy = self.target_x - x, self.target_y - y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        angle_to_target = math.atan2(dy, dx)

        yaw_pos = yaw + 2 * math.pi if yaw < 0 else yaw
        angle_to_target_pos = angle_to_target + 2 * math.pi if angle_to_target < 0 else angle_to_target
        angle_diff = angle_to_target_pos - yaw_pos
        if angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi

        front_clearance = frontal_clearance(
            ranges, laser_msg.angle_min, laser_msg.angle_increment)
        state = np.array(normalized_laser + [distance, angle_diff, self.prev_linear, self.prev_angular],
                          dtype=np.float32)
        return state.reshape(1, STATE_DIM), (x, y), distance, front_clearance

    def step(self, actor_model):
        state, pos, distance, front_clearance = self.build_state()
        action = actor_model.predict(state, verbose=0)
        angular = float(action[0][0])
        linear = float(action[0][1]) * 0.26
        if self.safety_shield_enabled:
            linear *= clearance_speed_scale(
                front_clearance, self.safety_stop_range,
                self.safety_slow_range)

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.pub.publish(cmd)

        self.prev_linear = linear
        self.prev_angular = angular
        return pos, distance


def main():
    global USE_SIM_TIME
    if len(sys.argv) < 2:
        print("usage: multi_robot_explorer.py <robot_name>")
        sys.exit(1)
    robot_name = sys.argv[1]

    rospy.init_node('%s_explorer' % robot_name, anonymous=True)
    USE_SIM_TIME = bool(rospy.get_param('/use_sim_time', False))

    sess = tf.Session()
    K.set_session(sess)
    actor_model = build_actor_model()
    weights_path = rospy.get_param('~weights_path', WEIGHTS_PATH)
    if not os.path.isfile(weights_path):
        raise IOError('DDPG actor weights not found: %s' % weights_path)
    actor_model.load_weights(weights_path)

    experiment_seed = int(rospy.get_param('~experiment_seed', 0))
    near_collision_range = float(rospy.get_param(
        '~near_collision_range', NEAR_COLLISION_RANGE))
    near_collision_clear_range = float(rospy.get_param(
        '~near_collision_clear_range', NEAR_COLLISION_CLEAR_RANGE))
    safety_shield_enabled = bool(rospy.get_param(
        '~safety_shield_enabled', False))
    safety_stop_range = float(rospy.get_param(
        '~safety_stop_range', 0.20))
    safety_slow_range = float(rospy.get_param(
        '~safety_slow_range', 0.35))
    recovery_direction_mode = rospy.get_param(
        '~recovery_direction_mode', 'random')
    if recovery_direction_mode not in ('random', 'clearance'):
        raise ValueError(
            'recovery_direction_mode must be random or clearance')
    agent = RobotAgent(
        robot_name, experiment_seed=experiment_seed,
        near_collision_range=near_collision_range,
        near_collision_clear_range=near_collision_clear_range,
        safety_shield_enabled=safety_shield_enabled,
        safety_stop_range=safety_stop_range,
        safety_slow_range=safety_slow_range)
    rospy.loginfo('[%s] experiment_seed=%d (robot seed=%d)',
                  robot_name, experiment_seed, agent.experiment_seed)
    rospy.loginfo('[%s] near-collision hysteresis: event<=%.3fm, clear>=%.3fm',
                  robot_name, near_collision_range,
                  near_collision_clear_range)
    rospy.loginfo('[%s] safety shield: enabled=%s, stop<=%.3fm, full-speed>=%.3fm',
                  robot_name, safety_shield_enabled,
                  safety_stop_range, safety_slow_range)
    rospy.loginfo('[%s] recovery direction: mode=%s',
                  robot_name, recovery_direction_mode)

    rospy.loginfo("[%s] waiting for first laser/model_states message...", robot_name)
    agent.laser_ig.get_msg(timeout=10.0)
    agent.model_states_ig.get_msg(timeout=10.0)
    rospy.loginfo("[%s] ready, waiting for first exploration target...", robot_name)
    while not agent.have_target and not rospy.is_shutdown():
        try:
            rospy.sleep(0.2)
        except rospy.exceptions.ROSInterruptException:
            return
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            pass
    rospy.loginfo("[%s] starting navigation loop", robot_name)

    rate = rospy.Rate(10)
    stuck_check_pos = None
    stuck_check_time = control_time()
    recovering_until = 0.0
    recovery_turn_dir = 1.0
    recovery_start_pos = None

    while not rospy.is_shutdown():
        now = control_time()

        if recovery_start_pos is not None and now >= recovering_until:
            x, y, _ = agent.get_pose()
            displacement = math.hypot(
                x - recovery_start_pos[0], y - recovery_start_pos[1])
            agent.recovery_displacement_pub.publish(
                Float32(data=displacement))
            rospy.loginfo(
                '[%s] recovery displacement: %.3fm',
                robot_name, displacement)
            recovery_start_pos = None
            stuck_check_pos = (x, y)
            stuck_check_time = now

        if agent.mission_done:
            agent.pub.publish(Twist())  # all-zero: full stop
        elif now < recovering_until:
            cmd = Twist()
            cmd.linear.x = -0.15
            cmd.angular.z = recovery_turn_dir * 0.6
            agent.pub.publish(cmd)
            x, y, _ = agent.get_pose()
            pos = (x, y)
        else:
            pos, distance = agent.step(actor_model)

            if stuck_check_pos is None:
                stuck_check_pos = pos
                stuck_check_time = now
            elif now - stuck_check_time > STUCK_CHECK_WINDOW:
                moved = math.hypot(pos[0] - stuck_check_pos[0], pos[1] - stuck_check_pos[1])
                if moved < STUCK_DIST_THRESHOLD:
                    rospy.logwarn("[%s] stuck detected (moved %.2fm in %.1fs), backing off",
                                   robot_name, moved, STUCK_CHECK_WINDOW)
                    recovering_until = now + RECOVERY_DURATION
                    recovery_start_pos = pos
                    random_direction = agent.rng.choice([-1.0, 1.0])
                    if recovery_direction_mode == 'clearance':
                        scan = agent.laser_ig.get_msg(timeout=0.1)
                        recovery_turn_dir = clearance_guided_turn_direction(
                            scan.ranges, scan.angle_min, scan.angle_increment,
                            fallback_direction=random_direction)
                    else:
                        recovery_turn_dir = random_direction
                    # Tell the coordinator to stop immediately retrying this
                    # spot. The exploration grid remains unchanged so the
                    # coverage metric is not inflated by recovery logic.
                    agent.stuck_pub.publish(Point(x=pos[0], y=pos[1], z=0.0))
                stuck_check_pos = pos
                stuck_check_time = now

        try:
            rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            break
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            rospy.logwarn("[%s] ROS time moved backwards, resyncing rate", robot_name)
            rate = rospy.Rate(10)


if __name__ == "__main__":
    main()
