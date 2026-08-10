#!/usr/bin/env python
# Voronoi-based cooperative exploration coordinator (Algorithm 1 in Hu, Niu,
# Carrasco, Lennox, Arvin, IEEE TVT 2020) for N=3 robots in Gazebo.
#
# The thesis mode keeps Gazebo ground-truth robot poses to avoid a full SLAM
# stack, but starts the occupancy grid unknown and discovers free/occupied
# cells from each robot's laser scan. A legacy ground-truth-radius mode is
# retained only as an experimental baseline. The coordinator then performs
# frontier detection + dynamic Voronoi
# partitioning + vision-guided, utility-based frontier selection (eq. 3 plus
# a camera-derived openness bonus -- see openness_from_image/vision_bias).
# To reach the chosen frontier (Algorithm 1 line 9: move via shortest path
# through sensed space rather than a beeline), it runs grid-based A* over
# the same occupancy grid -- tried the paper's sparser information-node-
# graph version of this too; empirically the two performed about the same
# (both bottlenecked by the DDPG policy's own limits in tight corners, not
# the path planner), so kept the simpler, more precise grid A*. Robot
# motion and low-level collision avoidance are handled separately by each
# robot's own trained (laser-based) DDPG policy (multi_robot_explorer.py) --
# the camera informs *where to explore next*, not the reactive avoidance.

import time
import math
import heapq
import os
import csv
import hashlib
import threading
from pathlib import Path

import rospy
import numpy as np
import cv2
from geometry_msgs.msg import Point, Pose2D
from std_msgs.msg import Bool, Float32, Header
from sensor_msgs.msg import Image, JointState, LaserScan
from nav_msgs.msg import OccupancyGrid
from gazebo_msgs.msg import ModelStates
from turtlebot_ddpg.msg import (
    FrontierAction, FrontierObservation, FrontierToken, TeammateIntent)
from scipy.spatial import Voronoi

from frontier_scoring import (
    assign_by_cost, blend_phase_costs, combine_assignment_cost_terms,
    combine_cost_terms, line_risk_path_cost, normalize_rows, SELECTION_MODES)
from hybrid_fallback import CoverageStallFallback
from communication_model import CommunicationModel
from motion_safety import recovery_succeeded
from recovery_memory import TimedSpatialBlacklist
from safety_metrics import ClearanceTelemetry
from scan_mapping import (
    bresenham_cells, clear_robot_footprints, update_grid_from_scan)
from vision_scoring import (
    directional_openness, openness_from_image, VISION_PROJECTION_MODES,
    VISION_SCORING_MODES)
from wheel_odometry import integrate_wheel_positions

ROBOT_NAMES = ['robot1', 'robot2', 'robot3']
PROJECT_ROOT = Path(os.environ.get(
    'TURTLEBOT_DDPG_PROJECT_ROOT', Path(__file__).resolve().parents[3]))
ROBOT_COLORS = ['red', 'blue', 'green']
ADAPTIVE_SELECTION_MODES = ('adaptive_execution_aware',
                            'mappo_adaptive_hybrid')
POLICY_SELECTION_MODES = ('mappo', 'mappo_hybrid',
                          'mappo_adaptive_hybrid')
EXECUTION_SELECTION_MODES = ('execution_aware',) + ADAPTIVE_SELECTION_MODES + \
                            ('mappo', 'mappo_hybrid')
HYBRID_SELECTION_MODES = ('mappo_hybrid', 'mappo_adaptive_hybrid')
COORDINATOR_SELECTION_MODES = SELECTION_MODES + \
                              ('adaptive_execution_aware',) + \
                              POLICY_SELECTION_MODES
MAPPING_MODES = ('ground_truth_radius', 'laser_scan')
POSE_SOURCES = ('gazebo_truth', 'wheel_odometry')
WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.287

GRID_RES = 0.2
WORLD_PROFILES = {
    'research': {
        'arena': (-15.0, -7.5, 15.0, 7.5),
        'obstacles': [
            (-10.5, 1.5, -7.5, 4.5),
            (-7.25, -5.25, -4.75, -2.75),
            (-2.0, 4.0, 2.0, 6.0),
            (2.0, -5.0, 4.0, -1.0),
            (7.5, 0.5, 10.5, 3.5),
            (8.75, -6.25, 11.25, -3.75),
        ],
    },
    # Figure 10 in Hu et al. specifies the 20 x 10 m boundary but not exact
    # obstacle dimensions. These bounds are a documented visual approximation:
    # one tall obstacle on the left and two near-circular obstacles on the right.
    'hu_aligned': {
        'arena': (-10.0, -5.0, 10.0, 5.0),
        'obstacles': [
            (-6.1, -3.2, -3.9, 3.4),
            (2.4, -2.6, 5.6, 0.6),
            (5.6, 0.6, 8.4, 3.4),
        ],
    },
    # Same 20 x 10 m boundary and three-obstacle count as the Hu-aligned
    # benchmark, but with a different arrangement. This profile is held out
    # from parameter selection and is used only as a layout-transfer check.
    'hu_layout_b': {
        'arena': (-10.0, -5.0, 10.0, 5.0),
        'obstacles': [
            (-7.0, -2.0, -3.0, 0.0),
            (-1.0, 0.0, 1.0, 3.0),
            (4.0, -2.5, 7.0, -0.5),
        ],
    },
    # A second held-out arrangement. It keeps the Hu-aligned arena size and
    # obstacle count but changes obstacle orientation and spatial balance.
    'hu_layout_c': {
        'arena': (-10.0, -5.0, 10.0, 5.0),
        'obstacles': [
            (-7.5, -3.5, -4.5, -0.5),
            (-1.5, -1.0, 1.5, 1.0),
            (4.5, 1.0, 7.5, 3.0),
        ],
    },
}


def configure_world(profile):
    """Select arena constants before constructing a coordinator instance."""
    if profile not in WORLD_PROFILES:
        raise ValueError('world_profile must be one of %s' %
                         (tuple(sorted(WORLD_PROFILES)),))
    global ARENA, OBSTACLES, NX, NY
    config = WORLD_PROFILES[profile]
    ARENA = tuple(config['arena'])
    OBSTACLES = list(config['obstacles'])
    NX = int(round((ARENA[2] - ARENA[0]) / GRID_RES))
    NY = int(round((ARENA[3] - ARENA[1]) / GRID_RES))


configure_world('research')

SENSING_RANGE = 1.3   # rs, from the paper
ALPHA = 0.8            # utility-function tuning parameter used in the paper
ARRIVE_THRESH = 0.5
UPDATE_HZ = 4.0

# How far ahead along the planned A* path (or the straight-line fallback) to
# place the published waypoint, re-issued every tick as the robot advances
# -- a simple pure-pursuit-style lookahead distance.
MAX_WAYPOINT_DIST = 2.5

# Keep frontier goals away from walls/obstacles by this much. Without this,
# a frontier cell can end up right against a wall or box -- a goal position
# the DDPG policy never saw during training (targets were always in open
# corridor space) -- and the robot just freezes/creeps at the boundary
# instead of confidently approaching it.
WALL_MARGIN = 0.8
OBSTACLE_MARGIN = 0.7

# When picking which frontier to chase, also reject candidates whose direct
# line-of-sight from the robot clips an obstacle -- otherwise the goal can
# be in open space but *behind* a box from the robot's current position,
# aiming the bounded waypoint straight at/through the obstacle corner.
LOS_CHECK_MARGIN = 0.35

# Every tick re-derives frontiers from the live-updating grid, so two nearby
# candidates can swap utility rank from one tick to the next and cause rapid
# goal flip-flopping. Require a goal to be held at least this long before
# ownership/line-of-sight invalidation can force a reselect (arrival is
# exempt -- reaching the goal should always immediately trigger a new pick).
MIN_GOAL_HOLD_TIME = 2.0

# radius fake-marked "explored" around a spot where a robot got physically
# wedged and couldn't be freed by its own back-up maneuver
STUCK_BLACKLIST_RADIUS = 1.8
RECOVERY_SUCCESS_DISTANCE = 0.15

# The last handful of frontier cells can end up in a corner the DDPG policy
# physically cannot enter (tight concave geometry it never saw in training).
# Without a fallback, robots would retry that same spot forever -- observed
# during testing as ~3500 stuck/blacklist cycles over ~2 hours with zero
# further coverage gained. So: if coverage is already high and every robot
# has had nothing reachable to chase for a while, call it done rather than
# spin indefinitely. The strict "zero frontiers left" completion check above
# still fires immediately/normally when the arena is genuinely fully covered.
COVERAGE_DONE_THRESHOLD = 97.0   # percent
IDLE_DONE_HOLD = 30.0             # seconds all robots must be idle
STARTUP_GRACE_PERIOD = 5.0        # seconds before zero-frontier completion is valid
EMPTY_FRONTIER_HOLD = 5.0         # seconds frontiers must remain continuously absent

# --- vision-guided frontier selection -------------------------------------
# Each robot camera is split into left/centre/right thirds. The default
# score estimates the fraction of each third occupied by projected obstacle
# surfaces, using the frame's own sky and floor colours as background
# references. This avoids the old inverse-edge-density failure where a large
# flat wall had very few Canny edges and was incorrectly scored as open.
# The score only biases high-level frontier selection; low-level collision
# avoidance remains laser-only, matching the original trained DDPG policy.
CAMERA_FOV = 1.085595   # rad, matches horizontal_fov in turtlebot3_waffle_pi.gazebo.xacro
VISION_WEIGHT = 3.0      # utility-score bonus for choosing a visually-open direction

STATUS_WRITE_PERIOD = 1.0  # seconds

METRIC_FIELDS = [
    'run_id', 'experiment_seed', 'wall_time', 'elapsed_s',
    'elapsed_wall_s', 'elapsed_sim_s', 'observed_real_time_factor',
    'selection_mode', 'mapping_mode', 'pose_source',
    'world_profile', 'communication_delay_s',
    'communication_drop_probability', 'communication_bandwidth_bytes_per_s',
    'communication_messages_sent', 'communication_messages_dropped',
    'communication_bandwidth_blocked',
    'coverage_pct', 'coverage_explored_cells',
    'coverage_explorable_cells', 'truth_obstacle_cells',
    'coverage_mask_sha256', 'grid_resolution_m',
    'raw_frontiers', 'candidate_frontiers', 'blacklisted_spots',
    'stuck_blacklist_ttl_s',
    'goal_assignments', 'goals_reached', 'no_path_fallbacks',
    'no_path_start_occupied', 'no_path_goal_occupied',
    'no_path_disconnected',
    'stuck_events', 'unique_stuck_regions', 'recovery_direction_mode',
    'recovery_results', 'recovery_successes', 'recovery_success_rate',
    'recovery_mean_displacement_m', 'recovery_success_distance_m',
    'near_collision_events', 'near_collision_active_s',
    'near_collision_sample_ratio', 'minimum_laser_clearance_m',
    'near_collision_range_m', 'near_collision_clear_range_m',
    'path_length_total_m', 'execution_path_cost_mode',
    'robot1_x', 'robot1_y', 'robot2_x', 'robot2_y',
    'robot3_x', 'robot3_y',
    'vision_enabled', 'vision_scoring_mode', 'vision_projection_mode',
    'vision_feature_scale',
    'vision_ready_robots',
    'vision_image_mean_openness', 'vision_token_mean_openness',
    'vision_token_nonzero_ratio', 'vision_observation_count',
    'vision_observation_mean_openness', 'vision_observation_std_openness',
    'vision_direction_contrast_mean',
    'scan_ready_robots', 'mapped_obstacle_cells',
    'idle_robots', 'policy_actions_received', 'policy_select_requests',
    'policy_select_accepted', 'policy_select_stale',
    'policy_select_owner_rejected', 'policy_action_timeouts',
    'goals_dropped_owner', 'goals_dropped_blocked', 'goals_cleared_wait',
    'goals_replaced_policy', 'hybrid_fallback_active',
    'hybrid_fallback_activations', 'hybrid_fallback_goal_assignments',
    'adaptive_execution_weight', 'tail_completion_active',
    'tail_completion_activations', 'tail_completion_goal_assignments',
    'mission_done', 'evaluation_horizon_reached',
]


def segment_intersects_rect(p0, p1, rect, margin=0.0):
    """Liang-Barsky segment-vs-axis-aligned-rectangle intersection test."""
    x0, y0, x1, y1 = rect
    x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - x0), (dx, x1 - p0[0]), (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return False
                elif r > t0:
                    t0 = r
            else:
                if r < t0:
                    return False
                elif r < t1:
                    t1 = r
    return True


def line_of_sight_clear(p0, p1, obstacles, margin=LOS_CHECK_MARGIN):
    return not any(segment_intersects_rect(p0, p1, rect, margin) for rect in obstacles)


_NEIGHBORS_8 = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]


def astar_grid(grid, start_cell, goal_cell, max_expansions=20000,
               allow_unknown=False, unknown_cost=2.0):
    """8-connected A* over `grid`. A cell is passable if grid[iy,ix]==1
    (explored free space), plus the goal cell itself is always allowed even
    if it's still unknown (it's a frontier point, right at the edge of
    sensed territory). Returns a list of (ix, iy) cells from start to goal,
    or None if unreachable."""
    ny, nx = grid.shape

    def in_bounds(c):
        return 0 <= c[0] < nx and 0 <= c[1] < ny

    def passable(c):
        value = grid[c[1], c[0]]
        return c == goal_cell or value == 1 or (allow_unknown and value == 0)

    if not in_bounds(start_cell) or not in_bounds(goal_cell):
        return None
    if not passable(start_cell):
        return None

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_heap = [(0.0, start_cell)]
    came_from = {}
    g_score = {start_cell: 0.0}
    closed = set()
    expansions = 0

    while open_heap and expansions < max_expansions:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_cell:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)
        expansions += 1

        for dx, dy in _NEIGHBORS_8:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in closed or not in_bounds(neighbor) or not passable(neighbor):
                continue
            step_cost = 1.41421356 if dx and dy else 1.0
            if allow_unknown and grid[neighbor[1], neighbor[0]] == 0:
                step_cost *= unknown_cost
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal_cell)
                heapq.heappush(open_heap, (f, neighbor))

    return None


def waypoint_along_path(path_xy, current_pos, lookahead):
    """Walk forward along path_xy (list of (x,y), already ordered start to
    goal) starting from the point nearest current_pos, and return the point
    `lookahead` meters further along -- a simple pure-pursuit-style
    path-following waypoint."""
    if not path_xy:
        return None
    path_arr = np.array(path_xy)
    d = np.linalg.norm(path_arr - np.array(current_pos), axis=1)
    idx = int(np.argmin(d))
    acc = 0.0
    for i in range(idx, len(path_arr) - 1):
        seg = np.linalg.norm(path_arr[i + 1] - path_arr[i])
        if acc + seg >= lookahead:
            remain = lookahead - acc
            direction = (path_arr[i + 1] - path_arr[i]) / (seg + 1e-9)
            return path_arr[i] + direction * remain
        acc += seg
    return path_arr[-1]


class Coordinator(object):
    def _control_now(self):
        """Return simulation time for control logic, with a startup fallback."""
        if self.use_sim_time:
            return rospy.get_time()
        return time.time()

    def __init__(self):
        self.use_sim_time = bool(rospy.get_param('/use_sim_time', False))
        self.world_profile = rospy.get_param('~world_profile', 'research')
        configure_world(self.world_profile)
        self.selection_mode = rospy.get_param('~selection_mode', 'vision')
        if self.selection_mode not in COORDINATOR_SELECTION_MODES:
            raise ValueError('selection_mode must be one of %s' %
                             (COORDINATOR_SELECTION_MODES,))
        self.vision_enabled = bool(rospy.get_param('~vision_enabled', True))
        self.vision_scoring_mode = rospy.get_param(
            '~vision_scoring_mode', 'projected_obstacle')
        if self.vision_scoring_mode not in VISION_SCORING_MODES:
            raise ValueError('vision_scoring_mode must be one of %s' %
                             (VISION_SCORING_MODES,))
        self.vision_projection_mode = rospy.get_param(
            '~vision_projection_mode', 'thirds')
        if self.vision_projection_mode not in VISION_PROJECTION_MODES:
            raise ValueError('vision_projection_mode must be one of %s' %
                             (VISION_PROJECTION_MODES,))
        self.vision_feature_scale = float(rospy.get_param(
            '~vision_feature_scale', 1.0))
        if not 0.0 <= self.vision_feature_scale <= 1.0:
            raise ValueError('vision_feature_scale must be in [0, 1]')
        self.mapping_mode = rospy.get_param('~mapping_mode', 'ground_truth_radius')
        if self.mapping_mode not in MAPPING_MODES:
            raise ValueError('mapping_mode must be one of %s' % (MAPPING_MODES,))
        self.pose_source = rospy.get_param('~pose_source', 'gazebo_truth')
        if self.pose_source not in POSE_SOURCES:
            raise ValueError('pose_source must be one of %s' % (POSE_SOURCES,))
        self.scan_mapping_range = float(rospy.get_param(
            '~scan_mapping_range', SENSING_RANGE))
        self.scan_ray_stride = max(1, int(rospy.get_param('~scan_ray_stride', 4)))
        self.scan_inflation_cells = max(
            0, int(rospy.get_param('~scan_inflation_cells', 1)))
        self.execution_path_cost_mode = rospy.get_param(
            '~execution_path_cost_mode', 'euclidean')
        if self.execution_path_cost_mode not in ('euclidean', 'line_risk'):
            raise ValueError(
                'execution_path_cost_mode must be euclidean or line_risk')
        self.cluster_frontiers = bool(rospy.get_param('~cluster_frontiers', True))
        self.frontier_cluster_min_cells = int(rospy.get_param('~frontier_cluster_min_cells', 2))
        self.frontier_cells_per_candidate = int(
            rospy.get_param('~frontier_cells_per_candidate', 12))
        self.max_frontier_candidates = int(rospy.get_param('~max_frontier_candidates', 48))
        self.information_radius = float(rospy.get_param('~information_radius', 1.2))
        self.communication_range = float(rospy.get_param('~communication_range', 5.0))
        self.communication_delay_s = float(rospy.get_param(
            '~communication_delay_s', 0.0))
        self.communication_drop_probability = float(rospy.get_param(
            '~communication_drop_probability', 0.0))
        self.communication_bandwidth_bytes_per_s = float(rospy.get_param(
            '~communication_bandwidth_bytes_per_s', 0.0))
        self.communication_message_bytes = int(rospy.get_param(
            '~communication_message_bytes', 48))
        self.communication_stale_after_s = float(rospy.get_param(
            '~communication_stale_after_s', 2.0))
        self.metrics_period = float(rospy.get_param('~metrics_period', 1.0))
        self.local_bev_size = int(rospy.get_param('~local_bev_size', 64))
        self.policy_action_timeout = float(rospy.get_param('~policy_action_timeout', 2.0))
        self.frontier_track_radius = float(rospy.get_param('~frontier_track_radius', 1.0))
        self.frontier_track_ttl = float(rospy.get_param('~frontier_track_ttl', 3.0))
        self.coverage_done_threshold = float(rospy.get_param(
            '~coverage_done_threshold', COVERAGE_DONE_THRESHOLD))
        self.idle_done_hold = float(rospy.get_param(
            '~idle_done_hold', IDLE_DONE_HOLD))
        self.shutdown_on_done = bool(rospy.get_param('~shutdown_on_done', False))
        self.complete_on_coverage = bool(rospy.get_param(
            '~complete_on_coverage', False))
        self.max_simulation_seconds = float(rospy.get_param(
            '~max_simulation_seconds', 0.0))
        self.hybrid_fallback_stall_seconds = float(rospy.get_param(
            '~hybrid_fallback_stall_seconds', 60.0))
        self.hybrid_fallback_active_seconds = float(rospy.get_param(
            '~hybrid_fallback_active_seconds', 30.0))
        self.hybrid_fallback_progress_epsilon = float(rospy.get_param(
            '~hybrid_fallback_progress_epsilon', 0.2))
        self.hybrid_fallback_minimum_coverage = float(rospy.get_param(
            '~hybrid_fallback_minimum_coverage', 20.0))
        self.adaptive_execution_start_coverage = float(rospy.get_param(
            '~adaptive_execution_start_coverage', 40.0))
        self.adaptive_execution_full_coverage = float(rospy.get_param(
            '~adaptive_execution_full_coverage', 90.0))
        if (self.adaptive_execution_full_coverage <=
                self.adaptive_execution_start_coverage):
            raise ValueError(
                'adaptive_execution_full_coverage must exceed start coverage')
        self.tail_completion_enabled = bool(rospy.get_param(
            '~tail_completion_enabled', False))
        self.tail_completion_coverage = float(rospy.get_param(
            '~tail_completion_coverage', 90.0))
        self.stuck_blacklist_ttl = float(rospy.get_param(
            '~stuck_blacklist_ttl', 90.0))
        self.recovery_direction_mode = rospy.get_param(
            '~recovery_direction_mode', 'random')
        if self.recovery_direction_mode not in ('random', 'clearance'):
            raise ValueError(
                'recovery_direction_mode must be random or clearance')
        self.recovery_success_distance = float(rospy.get_param(
            '~recovery_success_distance', RECOVERY_SUCCESS_DISTANCE))
        recovery_succeeded(0.0, self.recovery_success_distance)
        self.near_collision_range = float(rospy.get_param(
            '~near_collision_range', 0.18))
        self.near_collision_clear_range = float(rospy.get_param(
            '~near_collision_clear_range', 0.25))
        if self.near_collision_range <= 0.0:
            raise ValueError('near_collision_range must be positive')
        if self.near_collision_clear_range <= self.near_collision_range:
            raise ValueError(
                'near_collision_clear_range must exceed near_collision_range')
        default_metrics = str(PROJECT_ROOT / 'outputs' / 'gazebo_metrics.csv')
        self.metrics_path = os.path.expanduser(rospy.get_param('~metrics_path', default_metrics))
        self.status_html_path = os.path.expanduser(rospy.get_param(
            '~status_html_path', str(PROJECT_ROOT / 'outputs' /
                                     'exploration_status.html')))
        self.run_id = rospy.get_param('/run_id', time.strftime('%Y%m%d-%H%M%S'))
        self.experiment_seed = int(rospy.get_param('~experiment_seed', 0))
        self.communication_model = CommunicationModel(
            delay_s=self.communication_delay_s,
            drop_probability=self.communication_drop_probability,
            bandwidth_bytes_per_s=self.communication_bandwidth_bytes_per_s,
            message_bytes=self.communication_message_bytes,
            stale_after_s=self.communication_stale_after_s,
            seed=self.experiment_seed)

        self.grid = np.zeros((NY, NX), dtype=np.uint8)  # 0 unknown, 1 explored free, 2 obstacle
        xs = ARENA[0] + (np.arange(NX) + 0.5) * GRID_RES
        ys = ARENA[1] + (np.arange(NY) + 0.5) * GRID_RES
        self.xs, self.ys = xs, ys
        self.XX, self.YY = np.meshgrid(xs, ys)
        self.truth_obstacle_mask = np.zeros_like(self.grid, dtype=bool)
        for (x0, y0, x1, y1) in OBSTACLES:
            self.truth_obstacle_mask |= (
                (self.XX >= x0 - OBSTACLE_MARGIN) &
                (self.XX <= x1 + OBSTACLE_MARGIN) &
                (self.YY >= y0 - OBSTACLE_MARGIN) &
                (self.YY <= y1 + OBSTACLE_MARGIN))
        self.truth_obstacle_mask |= self.XX <= ARENA[0] + WALL_MARGIN
        self.truth_obstacle_mask |= self.XX >= ARENA[2] - WALL_MARGIN
        self.truth_obstacle_mask |= self.YY <= ARENA[1] + WALL_MARGIN
        self.truth_obstacle_mask |= self.YY >= ARENA[3] - WALL_MARGIN
        if self.mapping_mode == 'ground_truth_radius':
            self.grid[self.truth_obstacle_mask] = 2
        else:
            # The finite map domain is known, but all interior obstacles start
            # unknown and must be discovered from laser scans.
            self.grid[[0, -1], :] = 2
            self.grid[:, [0, -1]] = 2
        # total cells that could ever legitimately become "explored" (i.e.
        # everything that isn't permanently wall/obstacle) -- the denominator
        # for the coverage percentage reported below.
        self.total_explorable = int(np.sum(~self.truth_obstacle_mask))
        self.coverage_mask_sha256 = hashlib.sha256(
            self.truth_obstacle_mask.astype(np.uint8).tobytes()).hexdigest()
        self.ever_explored_truth_free = np.zeros_like(self.grid, dtype=bool)
        self.last_coverage_explored_cells = 0

        # (x, y) points reported by _stuck_event_cb. Unlike the old approach,
        # these do NOT touch self.grid -- they only steer goal *selection*
        # away from a spot a robot got wedged at. The grid itself stays a
        # 100%-honest record of what's actually been sensed, so coverage%
        # and the "exploration complete" check below can never be fooled by
        # a stuck-recovery blacklist into thinking unsensed area is done.
        self.blacklist = TimedSpatialBlacklist(
            ttl_seconds=self.stuck_blacklist_ttl,
            merge_radius=0.5 * STUCK_BLACKLIST_RADIUS)

        initial_poses = dict(
            (name, (
                float(rospy.get_param('~%s_initial_x' % name, 0.0)),
                float(rospy.get_param('~%s_initial_y' % name, 0.0)),
                float(rospy.get_param('~%s_initial_yaw' % name, 0.0))))
            for name in ROBOT_NAMES)
        if self.pose_source == 'wheel_odometry':
            self.positions = dict(
                (name, pose[:2]) for name, pose in initial_poses.items())
            self.yaws = dict(
                (name, pose[2]) for name, pose in initial_poses.items())
        else:
            self.positions = {}       # name -> (x, y)
            self.yaws = {}            # name -> heading (rad)
        self.last_wheel_positions = {}
        self.vision_scores = {}   # name -> (left, centre, right) openness in [0,1]
        self.vision_token_count = 0
        self.vision_token_nonzero_count = 0
        self.vision_token_openness_sum = 0.0
        self.vision_observation_count = 0
        self.vision_observation_value_count = 0
        self.vision_observation_sum = 0.0
        self.vision_observation_square_sum = 0.0
        self.vision_direction_contrast_sum = 0.0
        self.scan_ready = set()
        self.first_node_pos = {}  # name -> (x, y), recorded on first sighting
        self.current_goal = {}    # name -> (x, y) or None -- the selected frontier point
        self.current_goal_id = {} # name -> persistent frontier-track ID or None
        self.goal_set_time = {}   # name -> control-clock time the goal was chosen
        self.current_path = {}    # name -> list of (x, y) waypoints from A*, or None
        self.previous_positions = {}
        self.path_length = dict((name, 0.0) for name in ROBOT_NAMES)
        self.stuck_events = dict((name, 0) for name in ROBOT_NAMES)
        # `stuck_events` counts recovery activations and may include repeated
        # attempts at one corner. This second counter records only newly
        # blacklisted spatial regions, which is easier to interpret in the
        # paper as distinct failure locations.
        self.unique_stuck_regions = 0
        self.recovery_results = 0
        self.recovery_successes = 0
        self.recovery_displacement_sum = 0.0
        self.near_collision_events = dict((name, 0) for name in ROBOT_NAMES)
        self.clearance_telemetry = ClearanceTelemetry(
            event_threshold=self.near_collision_range)
        self.goal_assignments = 0
        self.goals_reached = 0
        self.no_path_fallbacks = 0
        self.no_path_start_occupied = 0
        self.no_path_goal_occupied = 0
        self.no_path_disconnected = 0
        self.latest_policy_actions = {}
        self.last_consumed_policy_action = dict((name, 0.0) for name in ROBOT_NAMES)
        self.policy_actions_received = 0
        self.policy_select_requests = 0
        self.policy_select_accepted = 0
        self.policy_select_stale = 0
        self.policy_select_owner_rejected = 0
        self.policy_action_timeouts = 0
        self.goals_dropped_owner = 0
        self.goals_dropped_blocked = 0
        self.goals_cleared_wait = 0
        self.goals_replaced_policy = 0
        self.hybrid_fallback_goal_assignments = 0
        self.adaptive_execution_weight = 0.0
        self.tail_completion_active = False
        self.tail_completion_activations = 0
        self.tail_completion_goal_assignments = 0
        self.hybrid_fallback = CoverageStallFallback(
            stall_seconds=self.hybrid_fallback_stall_seconds,
            active_seconds=self.hybrid_fallback_active_seconds,
            progress_epsilon=self.hybrid_fallback_progress_epsilon,
            minimum_coverage=self.hybrid_fallback_minimum_coverage)
        self.frontier_tracks = {}  # id -> (last position, last seen control time)
        self.next_frontier_track_id = 0
        self._last_metrics_write = 0.0
        self._metrics_file = None
        self._metrics_writer = None
        self._metrics_lock = threading.Lock()
        self.targets_pub = {
            name: rospy.Publisher('/%s/exploration_target' % name, Point, queue_size=1, latch=True)
            for name in ROBOT_NAMES
        }
        self.done_pub = {
            name: rospy.Publisher('/%s/exploration_done' % name, Bool, queue_size=1, latch=True)
            for name in ROBOT_NAMES
        }
        self.observation_pub = {
            name: rospy.Publisher('/%s/frontier_observation' % name,
                                  FrontierObservation, queue_size=1)
            for name in ROBOT_NAMES
        }
        self.local_bev_pub = {
            name: rospy.Publisher('/%s/local_bev' % name,
                                  OccupancyGrid, queue_size=1)
            for name in ROBOT_NAMES
        }

        if self.pose_source == 'gazebo_truth':
            rospy.Subscriber(
                '/gazebo/model_states', ModelStates, self._model_states_cb)
        for name in ROBOT_NAMES:
            if self.pose_source == 'wheel_odometry':
                rospy.Subscriber('/%s/joint_states' % name, JointState,
                                 self._joint_states_cb, callback_args=name)
            rospy.Subscriber('/%s/stuck_event' % name, Point, self._stuck_event_cb,
                             callback_args=name)
            rospy.Subscriber(
                '/%s/recovery_displacement_m' % name, Float32,
                self._recovery_displacement_cb, callback_args=name)
            rospy.Subscriber('/%s/near_collision_event' % name, Bool,
                             self._near_collision_cb, callback_args=name)
            rospy.Subscriber('/%s/min_clearance_m' % name, Float32,
                             self._clearance_cb, callback_args=name)
            rospy.Subscriber('/%s/camera/rgb/image_raw' % name, Image, self._camera_cb, callback_args=name)
            rospy.Subscriber('/%s/scan' % name, LaserScan,
                             self._scan_cb, callback_args=name)
            rospy.Subscriber('/%s/frontier_action' % name, FrontierAction,
                             self._policy_action_cb, callback_args=name)
        self.exploration_done_logged = False
        self.evaluation_horizon_reached = False
        self.start_time = time.time()
        # Completion time must be measured on Gazebo's /clock rather than
        # only on the host wall clock. The simulator can run slower when the
        # CPU is busy, which otherwise makes identical policies look faster or
        # slower across repeated trials. Keep both clocks for diagnostics and
        # backwards compatibility with earlier result files.
        initial_sim_time = rospy.get_time()
        self.start_sim_time = initial_sim_time if self.use_sim_time else None
        self.start_control_time = (
            initial_sim_time if self.use_sim_time else self.start_time)
        self._last_status_write = 0.0
        self.all_idle_since = None
        self.no_frontier_since = None
        self._open_metrics_file()
        rospy.on_shutdown(self._close_metrics_file)
        rospy.loginfo('[coordinator] selection_mode=%s, mapping_mode=%s, pose_source=%s, world_profile=%s, clustered_frontiers=%s, metrics=%s',
                      self.selection_mode, self.mapping_mode, self.pose_source,
                      self.world_profile, self.cluster_frontiers,
                      self.metrics_path)

    def _camera_cb(self, msg, name):
        if not self.vision_enabled:
            return
        if msg.encoding not in ('rgb8', 'bgr8') or msg.height == 0 or msg.width == 0:
            return
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        except ValueError:
            return
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        scores = openness_from_image(img, mode=self.vision_scoring_mode)
        self.vision_scores[name] = scores
        self.vision_observation_count += 1
        self.vision_observation_value_count += len(scores)
        self.vision_observation_sum += float(sum(scores))
        self.vision_observation_square_sum += float(
            sum(value * value for value in scores))
        self.vision_direction_contrast_sum += float(max(scores) - min(scores))

    def _scan_cb(self, msg, name):
        if self.mapping_mode != 'laser_scan':
            return
        position = self.positions.get(name)
        yaw = self.yaws.get(name)
        if position is None or yaw is None or msg.range_max <= 0.0:
            return
        effective_max = min(float(msg.range_max), self.scan_mapping_range)
        update_grid_from_scan(
            self.grid, position, yaw, msg.ranges, msg.angle_min,
            msg.angle_increment, msg.range_min, effective_max, ARENA,
            GRID_RES, ray_stride=self.scan_ray_stride,
            inflation_cells=self.scan_inflation_cells)
        clear_robot_footprints(
            self.grid, list(self.positions.values()), ARENA, GRID_RES,
            radius_cells=max(1, self.scan_inflation_cells))
        self.ever_explored_truth_free |= (
            (self.grid == 1) & ~self.truth_obstacle_mask)
        self.scan_ready.add(name)

    def _policy_action_cb(self, msg, name):
        if msg.robot_name and msg.robot_name != name:
            rospy.logwarn_throttle(
                5.0, '[coordinator] ignoring action for %s received on %s topic',
                msg.robot_name, name)
            return
        self.policy_actions_received += 1
        self.latest_policy_actions[name] = (msg, self._control_now())

    def _consume_policy_action(self, name, frontiers, frontier_ids, owner, robot_index):
        """Return a fresh validated high-level action once.

        Frontier IDs are tracked across nearby cluster motion, so a policy
        does not accidentally select a different target when clustering order
        or the frontier boundary changes between observation and action.
        """
        pending = self.latest_policy_actions.get(name)
        if pending is None:
            return None, None
        msg, received_at = pending
        if received_at <= self.last_consumed_policy_action[name]:
            return None, None
        if self._control_now() - received_at > self.policy_action_timeout:
            self.last_consumed_policy_action[name] = received_at
            self.policy_action_timeouts += 1
            return None, None
        self.last_consumed_policy_action[name] = received_at

        if msg.action_type == FrontierAction.WAIT:
            return 'wait', None
        if msg.action_type == FrontierAction.CONTINUE:
            return 'continue', None
        if msg.action_type != FrontierAction.SELECT:
            rospy.logwarn('[coordinator] invalid action_type=%d from %s',
                          msg.action_type, name)
            return None, None
        self.policy_select_requests += 1

        matching = [index for index, frontier_id in enumerate(frontier_ids)
                    if frontier_id == msg.frontier_id]
        if not matching:
            # A frontier cluster can split/merge while an action is in flight.
            # Rebind only to a nearby candidate in the same responsibility
            # region; never redirect an old action to an arbitrary frontier.
            tracked = self.frontier_tracks.get(msg.frontier_id)
            if tracked is not None and len(frontiers):
                distances = np.linalg.norm(
                    frontiers - np.asarray(tracked[0]), axis=1)
                nearest = int(np.argmin(distances))
                if (distances[nearest] <= 1.5 * self.frontier_track_radius and
                        int(owner[nearest]) == robot_index):
                    matching = [nearest]
        if not matching:
            rospy.logwarn_throttle(
                2.0, '[coordinator] stale frontier_id=%d from %s',
                msg.frontier_id, name)
            self.policy_select_stale += 1
            return None, None
        index = matching[0]
        if int(owner[index]) != robot_index:
            rospy.logwarn_throttle(
                2.0, '[coordinator] masked frontier_id=%d is not assigned to %s',
                msg.frontier_id, name)
            self.policy_select_owner_rejected += 1
            return None, None
        self.policy_select_accepted += 1
        return 'select', index

    def _stuck_event_cb(self, msg, name):
        # a robot physically could not make progress here (e.g. wedged in a
        # concave corner the DDPG policy never trained on) -- remember the
        # spot so goal-selection avoids sending any robot back there. This
        # does NOT mark the area explored -- it stays genuinely unknown/
        # frontier-eligible so coverage% and completion stay honest, and a
        # different robot approaching from another angle can still claim it.
        rospy.logwarn("[coordinator] blacklisting stuck area near (%.2f, %.2f)", msg.x, msg.y)
        self.stuck_events[name] += 1
        if self.blacklist.add(msg.x, msg.y, self._control_now()):
            self.unique_stuck_regions += 1
        for name in ROBOT_NAMES:
            goal = self.current_goal.get(name)
            if goal is not None and math.hypot(goal[0] - msg.x, goal[1] - msg.y) < STUCK_BLACKLIST_RADIUS:
                self.current_goal[name] = None
                self.current_goal_id[name] = None
                self.current_path[name] = None

    def _near_collision_cb(self, msg, name):
        if msg.data:
            self.near_collision_events[name] += 1

    def _recovery_displacement_cb(self, msg, name):
        displacement = max(0.0, float(msg.data))
        self.recovery_results += 1
        self.recovery_displacement_sum += displacement
        if recovery_succeeded(
                displacement, self.recovery_success_distance):
            self.recovery_successes += 1

    def _clearance_cb(self, msg, name):
        self.clearance_telemetry.observe(
            name, msg.data, self._control_now())

    def _model_states_cb(self, msg):
        for name in ROBOT_NAMES:
            if name in msg.name:
                idx = msg.name.index(name)
                p = msg.pose[idx].position
                o = msg.pose[idx].orientation
                siny_cosp = 2 * (o.w * o.z + o.x * o.y)
                cosy_cosp = 1 - 2 * (o.y * o.y + o.z * o.z)
                self._update_pose(
                    name, p.x, p.y, math.atan2(siny_cosp, cosy_cosp))

    def _joint_states_cb(self, msg, name):
        try:
            left_index = msg.name.index('wheel_left_joint')
            right_index = msg.name.index('wheel_right_joint')
            left = float(msg.position[left_index])
            right = float(msg.position[right_index])
        except (ValueError, IndexError):
            return
        previous = self.last_wheel_positions.get(name)
        self.last_wheel_positions[name] = (left, right)
        if previous is None:
            return
        x, y = self.positions[name]
        yaw = self.yaws[name]
        next_pose = integrate_wheel_positions(
            (x, y, yaw), previous, (left, right),
            WHEEL_RADIUS, WHEEL_SEPARATION)
        self._update_pose(name, *next_pose)

    def _update_pose(self, name, x, y, yaw):
        self.positions[name] = (x, y)
        self.yaws[name] = yaw
        if name not in self.first_node_pos:
            self.first_node_pos[name] = (x, y)
        previous = self.previous_positions.get(name)
        if previous is not None:
            step = math.hypot(x - previous[0], y - previous[1])
            # Ignore spawn/reset jumps in path-length metrics.
            if step < 1.0:
                self.path_length[name] += step
        self.previous_positions[name] = (x, y)

    def vision_bias(self, robot_pos, robot_yaw, candidate_pos, name):
        """Openness bonus for a candidate frontier that falls within the
        robot's current camera view, based on which third of the frame its
        bearing lands in. 0 if outside the FOV or no image received yet."""
        scores = self.vision_scores.get(name)
        if scores is None:
            return 0.0
        bearing = math.atan2(candidate_pos[1] - robot_pos[1], candidate_pos[0] - robot_pos[0])
        rel = (bearing - robot_yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(rel) > CAMERA_FOV / 2.0:
            return 0.0
        return directional_openness(
            scores, rel, CAMERA_FOV, mode=self.vision_projection_mode)

    def _cell_index(self, x, y):
        ix = int(np.clip((x - ARENA[0]) / GRID_RES, 0, NX - 1))
        iy = int(np.clip((y - ARENA[1]) / GRID_RES, 0, NY - 1))
        return ix, iy

    def track_frontiers(self, frontiers):
        """Assign persistent IDs to moving frontier-cluster representatives."""
        now = self._control_now()
        active = dict((track_id, state)
                      for track_id, state in self.frontier_tracks.items()
                      if now - state[1] <= self.frontier_track_ttl)
        pairs = []
        for index, frontier in enumerate(frontiers):
            for track_id, (position, _) in active.items():
                distance = float(np.linalg.norm(frontier - np.asarray(position)))
                if distance <= self.frontier_track_radius:
                    pairs.append((distance, index, track_id))
        pairs.sort()

        ids = [None] * len(frontiers)
        used_tracks = set()
        for _, index, track_id in pairs:
            if ids[index] is None and track_id not in used_tracks:
                ids[index] = track_id
                used_tracks.add(track_id)
        for index in range(len(frontiers)):
            if ids[index] is None:
                ids[index] = self.next_frontier_track_id
                self.next_frontier_track_id += 1
            self.frontier_tracks[ids[index]] = (tuple(frontiers[index]), now)
        self.frontier_tracks = dict(
            (track_id, state) for track_id, state in self.frontier_tracks.items()
            if now - state[1] <= self.frontier_track_ttl)
        return ids

    def _local_bev_message(self, name):
        """Publish a fixed-size, world-aligned local occupancy crop.

        ROS occupancy values are -1 unknown, 0 free and 100 occupied. Keeping
        this as a standard OccupancyGrid makes the observation inspectable in
        RViz and easy to convert to a MAPPO tensor later.
        """
        size = max(8, self.local_bev_size)
        pos = self.positions[name]
        center_x, center_y = self._cell_index(*pos)
        half = size // 2
        x0, y0 = center_x - half, center_y - half
        x1, y1 = x0 + size, y0 + size

        crop = np.full((size, size), -1, dtype=np.int8)
        src_x0, src_y0 = max(0, x0), max(0, y0)
        src_x1, src_y1 = min(NX, x1), min(NY, y1)
        if src_x1 > src_x0 and src_y1 > src_y0:
            values = self.grid[src_y0:src_y1, src_x0:src_x1]
            encoded = np.full(values.shape, -1, dtype=np.int8)
            encoded[values == 1] = 0
            encoded[values == 2] = 100
            dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
            crop[dst_y0:dst_y0 + encoded.shape[0],
                 dst_x0:dst_x0 + encoded.shape[1]] = encoded

        msg = OccupancyGrid()
        msg.header = Header(stamp=rospy.Time.now(), frame_id='map')
        msg.info.resolution = GRID_RES
        msg.info.width = size
        msg.info.height = size
        msg.info.origin.position.x = ARENA[0] + x0 * GRID_RES
        msg.info.origin.position.y = ARENA[1] + y0 * GRID_RES
        msg.info.origin.orientation.w = 1.0
        msg.data = crop.reshape(-1).astype(np.int8).tolist()
        return msg

    def mark_explored(self, x, y, radius):
        ix, iy = self._cell_index(x, y)
        r_cells = int(radius / GRID_RES) + 1
        x0, x1 = max(0, ix - r_cells), min(NX, ix + r_cells + 1)
        y0, y1 = max(0, iy - r_cells), min(NY, iy + r_cells + 1)
        sub_xx, sub_yy = self.XX[y0:y1, x0:x1], self.YY[y0:y1, x0:x1]
        dist = np.hypot(sub_xx - x, sub_yy - y)
        mask = (dist <= radius) & (self.grid[y0:y1, x0:x1] == 0)
        self.grid[y0:y1, x0:x1][mask] = 1

    def detect_frontiers(self):
        unknown = self.grid == 0
        explored = self.grid == 1
        adj = np.zeros_like(explored)
        adj[1:, :] |= explored[:-1, :]
        adj[:-1, :] |= explored[1:, :]
        adj[:, 1:] |= explored[:, :-1]
        adj[:, :-1] |= explored[:, 1:]
        frontier_mask = unknown & adj
        iy, ix = np.nonzero(frontier_mask)
        if len(ix) == 0:
            return np.empty((0, 2))
        return np.stack([self.xs[ix], self.ys[iy]], axis=1)

    def voronoi_owner(self, query_pts, robot_names, robot_positions):
        d = np.linalg.norm(query_pts[:, None, :] - robot_positions[None, :, :], axis=2)
        return np.argmin(d, axis=1)

    def coverage_pct(self):
        self.ever_explored_truth_free |= (
            (self.grid == 1) & ~self.truth_obstacle_mask)
        self.last_coverage_explored_cells = int(
            np.sum(self.ever_explored_truth_free))
        return (100.0 * self.last_coverage_explored_cells /
                self.total_explorable)

    def cluster_frontier_points(self, points):
        """Convert pixel frontiers into stable cluster representatives.

        Frontier-level MAPPO needs a bounded action set. The same candidates
        are also used by every rule baseline so later comparisons stay fair.
        """
        if not self.cluster_frontiers or len(points) == 0:
            return points[:self.max_frontier_candidates]
        mask = np.zeros_like(self.grid, dtype=np.uint8)
        for x, y in points:
            ix, iy = self._cell_index(x, y)
            mask[iy, ix] = 1
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.frontier_cluster_min_cells:
                continue
            cx, cy = centroids[label]
            cells_y, cells_x = np.nonzero(labels == label)
            # A long connected boundary still needs several high-level
            # actions. Split it by angle around its centroid instead of
            # collapsing an entire room boundary into one frontier token.
            angles = np.arctan2(cells_y - cy, cells_x - cx)
            order = np.argsort(angles)
            groups = max(1, int(math.ceil(
                float(area) / max(1, self.frontier_cells_per_candidate))))
            for group in np.array_split(order, groups):
                if group.size == 0:
                    continue
                gx, gy = np.mean(cells_x[group]), np.mean(cells_y[group])
                nearest = group[np.argmin(
                    (cells_x[group] - gx) ** 2 + (cells_y[group] - gy) ** 2)]
                ix, iy = int(cells_x[nearest]), int(cells_y[nearest])
                candidates.append((len(group), self.xs[ix], self.ys[iy]))
        candidates.sort(reverse=True)
        return np.asarray([(x, y) for _, x, y in candidates[:self.max_frontier_candidates]],
                          dtype=np.float64)

    def information_gain(self, point):
        ix, iy = self._cell_index(*point)
        r = max(1, int(self.information_radius / GRID_RES))
        x0, x1 = max(0, ix - r), min(NX, ix + r + 1)
        y0, y1 = max(0, iy - r), min(NY, iy + r + 1)
        unknown = self.grid[y0:y1, x0:x1] == 0
        return float(np.sum(unknown)) / float(max(1, unknown.size))

    def obstacle_risk(self, point):
        """Return 0 in open space and 1 at the configured safety boundary."""
        x, y = point
        clearance = min(x - ARENA[0], ARENA[2] - x, y - ARENA[1], ARENA[3] - y)
        if self.mapping_mode == 'ground_truth_radius':
            for x0, y0, x1, y1 in OBSTACLES:
                dx = max(x0 - x, 0.0, x - x1)
                dy = max(y0 - y, 0.0, y - y1)
                clearance = min(clearance, math.hypot(dx, dy))
        else:
            ix, iy = self._cell_index(x, y)
            radius = max(1, int(math.ceil(
                max(WALL_MARGIN, OBSTACLE_MARGIN) / GRID_RES)))
            x0, x1 = max(0, ix - radius), min(NX, ix + radius + 1)
            y0, y1 = max(0, iy - radius), min(NY, iy + radius + 1)
            occupied_y, occupied_x = np.nonzero(self.grid[y0:y1, x0:x1] == 2)
            if occupied_x.size:
                obstacle_x = self.xs[occupied_x + x0]
                obstacle_y = self.ys[occupied_y + y0]
                clearance = min(
                    clearance,
                    float(np.min(np.hypot(obstacle_x - x, obstacle_y - y))))
        safety = max(WALL_MARGIN, OBSTACLE_MARGIN, 1e-6)
        return float(np.clip(1.0 - clearance / safety, 0.0, 1.0))

    def _line_of_sight_clear(self, p0, p1):
        if self.mapping_mode == 'ground_truth_radius':
            return line_of_sight_clear(p0, p1, OBSTACLES)
        start = self._cell_index(*p0)
        end = self._cell_index(*p1)
        return not any(self.grid[iy, ix] == 2
                       for ix, iy in bresenham_cells(start, end))

    def _plan_path(self, start_cell, goal_cell):
        """Plan conservatively in known maps and risk-aware in unknown maps."""
        # Laser callbacks run on ROS subscriber threads. Plan against a stable
        # snapshot and always treat the robot's physical pose as traversable,
        # even if a teammate scan arrives at the same instant.
        planning_grid = self.grid.copy()
        planning_grid[start_cell[1], start_cell[0]] = 1
        return astar_grid(
            planning_grid, start_cell, goal_cell,
            allow_unknown=self.mapping_mode == 'laser_scan',
            unknown_cost=2.0)

    def _execution_costs(self, frontiers, robot_positions):
        """Build cheap finite costs for dynamic ownership at control rate.

        Exact A* path costs are evaluated only when a robot needs a new goal;
        running A* for every robot/frontier pair at 4 Hz would dominate Gazebo.
        """
        n_robots, n_frontiers = len(ROBOT_NAMES), len(frontiers)
        distance = np.linalg.norm(
            robot_positions[:, None, :] - frontiers[None, :, :], axis=2)
        path_cost = np.zeros((n_robots, n_frontiers), dtype=np.float64)
        collision = np.zeros_like(path_cost)
        overlap = np.zeros_like(path_cost)
        communication = np.zeros_like(path_cost)
        vision = np.zeros_like(path_cost)

        obstacle = np.broadcast_to(
            np.asarray([self.obstacle_risk(f) for f in frontiers])[None, :],
            (n_robots, n_frontiers)).copy()
        information = np.broadcast_to(
            np.asarray([self.information_gain(f) for f in frontiers])[None, :],
            (n_robots, n_frontiers)).copy()
        active_goals = dict((name, self.current_goal.get(name)) for name in ROBOT_NAMES)

        for i, name in enumerate(ROBOT_NAMES):
            other_positions = np.delete(robot_positions, i, axis=0)
            yaw = self.yaws.get(name, 0.0)
            for j, frontier in enumerate(frontiers):
                if self.execution_path_cost_mode == 'line_risk':
                    path_cost[i, j] = line_risk_path_cost(
                        self.grid, self._cell_index(*robot_positions[i]),
                        self._cell_index(*frontier), GRID_RES)
                else:
                    path_cost[i, j] = distance[i, j]

                nearest_robot = np.min(np.linalg.norm(other_positions - frontier, axis=1))
                collision[i, j] = math.exp(-nearest_robot / 1.5)

                other_goals = [g for robot, g in active_goals.items()
                               if robot != name and g is not None]
                if other_goals:
                    nearest_goal = np.min(np.linalg.norm(np.asarray(other_goals) - frontier, axis=1))
                    overlap[i, j] = math.exp(-nearest_goal / 1.5)

                future_link = np.min(np.linalg.norm(other_positions - frontier, axis=1))
                communication[i, j] = np.clip(
                    (future_link - self.communication_range) / max(self.communication_range, 1e-6),
                    0.0, 1.0)
                if self.vision_enabled:
                    vision[i, j] = self.vision_feature_scale * self.vision_bias(
                        robot_positions[i], yaw, frontier, name)

        self.vision_token_count += int(vision.size)
        self.vision_token_nonzero_count += int(np.count_nonzero(vision > 1e-6))
        self.vision_token_openness_sum += float(np.sum(vision))

        costs = combine_cost_terms(
            'execution_aware', distance=distance, path=path_cost,
            obstacle=obstacle, collision=collision, overlap=overlap,
            communication=communication, information=information, vision=vision)
        ownership_costs = combine_assignment_cost_terms(
            distance=distance, path=path_cost, obstacle=obstacle,
            collision=collision, overlap=overlap,
            communication=communication, information=information,
            vision=vision)
        features = {
            'distance': distance,
            'path': path_cost,
            'obstacle': obstacle,
            'collision': collision,
            'overlap': overlap,
            'communication': communication,
            'information': information,
            'vision': vision,
        }
        return costs, ownership_costs, features

    def _publish_policy_observations(self, coverage, frontiers, frontier_ids, owner,
                                     execution_costs, features):
        stamp = rospy.Time.now()
        for i, name in enumerate(ROBOT_NAMES):
            x, y = self.positions[name]
            observation = FrontierObservation()
            observation.header = Header(stamp=stamp, frame_id='map')
            observation.robot_name = name
            observation.coverage_pct = coverage
            observation.has_active_goal = self.current_goal.get(name) is not None
            observation.robot_pose = Pose2D(x=x, y=y, theta=self.yaws.get(name, 0.0))
            local_bev = self._local_bev_message(name)
            local_bev.header.stamp = stamp
            observation.local_bev = local_bev

            tokens = []
            active_goal_id = self.current_goal_id.get(name)
            for j, frontier in enumerate(frontiers):
                token = FrontierToken()
                token.id = frontier_ids[j]
                token.position = Point(x=float(frontier[0]), y=float(frontier[1]), z=0.0)
                token.analytic_cost = float(execution_costs[i, j])
                token.distance = float(features['distance'][i, j])
                token.path_cost = float(features['path'][i, j])
                token.obstacle_risk = float(features['obstacle'][i, j])
                token.collision_risk = float(features['collision'][i, j])
                token.overlap_risk = float(features['overlap'][i, j])
                token.communication_risk = float(features['communication'][i, j])
                token.information_gain = float(features['information'][i, j])
                token.vision_openness = float(features['vision'][i, j])
                token.assigned = int(owner[j]) == i
                # Keep the initial action mask finite and permissive. Exact A*
                # reachability is checked after SELECT, avoiding the previous
                # prototype failure where every frontier became masked by inf.
                token.reachable = bool(np.isfinite(execution_costs[i, j]))
                token.selectable = bool(
                    token.assigned and token.reachable and
                    (active_goal_id is None or token.id == active_goal_id))
                tokens.append(token)
            observation.frontiers = tokens
            teammate_intents = []
            communication_now = self._control_now()
            for other_name in ROBOT_NAMES:
                if other_name == name:
                    continue
                intent = TeammateIntent()
                intent.robot_name = other_name
                other_pos = self.positions[other_name]
                separation = math.hypot(other_pos[0] - x, other_pos[1] - y)
                other_goal_id = self.current_goal_id.get(other_name)
                other_goal = self.current_goal.get(other_name)
                payload = {
                    'pose': (other_pos[0], other_pos[1],
                             self.yaws.get(other_name, 0.0)),
                    'target_frontier_id': (
                        int(other_goal_id) if other_goal_id is not None else -1),
                    'target_position': (
                        tuple(other_goal) if other_goal is not None else None),
                }
                delivered = self.communication_model.exchange(
                    other_name, name, communication_now,
                    separation <= self.communication_range, payload)
                intent.received = delivered is not None
                intent.link_quality = (
                    float(np.clip(
                        1.0 - separation / max(self.communication_range, 1e-6),
                        0.0, 1.0)) *
                    (1.0 - self.communication_drop_probability)
                    if intent.received else 0.0)
                intent.target_frontier_id = -1
                if intent.received:
                    delivered_pose = delivered['pose']
                    intent.pose = Pose2D(
                        x=delivered_pose[0], y=delivered_pose[1],
                        theta=delivered_pose[2])
                    intent.target_frontier_id = delivered['target_frontier_id']
                    delivered_goal = delivered['target_position']
                    if delivered_goal is not None:
                        intent.target_position = Point(
                            x=delivered_goal[0], y=delivered_goal[1], z=0.0)
                teammate_intents.append(intent)
            observation.teammate_intents = teammate_intents
            self.observation_pub[name].publish(observation)
            self.local_bev_pub[name].publish(local_bev)

    def _open_metrics_file(self):
        directory = os.path.dirname(self.metrics_path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        new_file = not os.path.exists(self.metrics_path) or os.path.getsize(self.metrics_path) == 0
        self._metrics_file = open(self.metrics_path, 'a', newline='')
        self._metrics_writer = csv.DictWriter(self._metrics_file, fieldnames=METRIC_FIELDS)
        if new_file:
            self._metrics_writer.writeheader()
            self._metrics_file.flush()

    def _close_metrics_file(self):
        with self._metrics_lock:
            if self._metrics_file is not None:
                self._metrics_file.flush()
                self._metrics_file.close()
                self._metrics_file = None
                self._metrics_writer = None

    def _write_metrics(self, coverage, raw_frontiers, candidate_frontiers,
                       force=False):
        now = time.time()
        if not force and now - self._last_metrics_write < self.metrics_period:
            return
        self._last_metrics_write = now
        elapsed_wall = now - self.start_time
        sim_now = rospy.get_time()
        if sim_now > 0.0:
            if self.start_sim_time is None or sim_now < self.start_sim_time:
                self.start_sim_time = sim_now
            elapsed_sim = max(0.0, sim_now - self.start_sim_time)
        else:
            elapsed_sim = None
        observed_real_time_factor = (
            elapsed_sim / elapsed_wall
            if elapsed_sim is not None and elapsed_wall > 0.0 else None)
        image_scores = [value for scores in self.vision_scores.values()
                        for value in scores]
        # Keep numerator and percentage from the same coverage snapshot.
        # Laser callbacks may update the live grid while this row is written.
        explored_cells = self.last_coverage_explored_cells
        clearance = self.clearance_telemetry.snapshot()
        vision_observation_mean = (
            self.vision_observation_sum / self.vision_observation_value_count
            if self.vision_observation_value_count else None)
        vision_observation_variance = (
            max(0.0,
                self.vision_observation_square_sum /
                self.vision_observation_value_count -
                vision_observation_mean * vision_observation_mean)
            if vision_observation_mean is not None else None)
        communication = self.communication_model.diagnostics()
        row = {
            'run_id': self.run_id,
            'experiment_seed': self.experiment_seed,
            'wall_time': '%.3f' % now,
            # `elapsed_s` is retained as the historical wall-clock field.
            # The summarizer prefers `elapsed_sim_s` when it is available.
            'elapsed_s': '%.3f' % elapsed_wall,
            'elapsed_wall_s': '%.3f' % elapsed_wall,
            'elapsed_sim_s': (
                '%.3f' % elapsed_sim if elapsed_sim is not None else ''),
            'observed_real_time_factor': (
                '%.6f' % observed_real_time_factor
                if observed_real_time_factor is not None else ''),
            'selection_mode': self.selection_mode,
            'mapping_mode': self.mapping_mode,
            'pose_source': self.pose_source,
            'world_profile': self.world_profile,
            'communication_delay_s': '%.3f' % self.communication_delay_s,
            'communication_drop_probability': (
                '%.4f' % self.communication_drop_probability),
            'communication_bandwidth_bytes_per_s': (
                '%.3f' % self.communication_bandwidth_bytes_per_s),
            'communication_messages_sent': (
                communication['communication_messages_sent']),
            'communication_messages_dropped': (
                communication['communication_messages_dropped']),
            'communication_bandwidth_blocked': (
                communication['communication_bandwidth_blocked']),
            'coverage_pct': '%.4f' % coverage,
            'coverage_explored_cells': explored_cells,
            'coverage_explorable_cells': self.total_explorable,
            'truth_obstacle_cells': int(np.sum(self.truth_obstacle_mask)),
            'coverage_mask_sha256': self.coverage_mask_sha256,
            'grid_resolution_m': '%.4f' % GRID_RES,
            'raw_frontiers': len(raw_frontiers),
            'candidate_frontiers': len(candidate_frontiers),
            'blacklisted_spots': self.blacklist.count(self._control_now()),
            'stuck_blacklist_ttl_s': '%.3f' % self.stuck_blacklist_ttl,
            'goal_assignments': self.goal_assignments,
            'goals_reached': self.goals_reached,
            'no_path_fallbacks': self.no_path_fallbacks,
            'no_path_start_occupied': self.no_path_start_occupied,
            'no_path_goal_occupied': self.no_path_goal_occupied,
            'no_path_disconnected': self.no_path_disconnected,
            'stuck_events': sum(self.stuck_events.values()),
            'unique_stuck_regions': self.unique_stuck_regions,
            'recovery_direction_mode': self.recovery_direction_mode,
            'recovery_results': self.recovery_results,
            'recovery_successes': self.recovery_successes,
            'recovery_success_rate': (
                '%.6f' % (float(self.recovery_successes) /
                          self.recovery_results)
                if self.recovery_results else ''),
            'recovery_mean_displacement_m': (
                '%.6f' % (self.recovery_displacement_sum /
                          self.recovery_results)
                if self.recovery_results else ''),
            'recovery_success_distance_m': (
                '%.4f' % self.recovery_success_distance),
            'near_collision_events': sum(self.near_collision_events.values()),
            'near_collision_active_s': '%.4f' % clearance['active_seconds'],
            'near_collision_sample_ratio': '%.6f' % clearance['sample_ratio'],
            'minimum_laser_clearance_m': (
                '%.4f' % clearance['minimum_clearance']
                if clearance['minimum_clearance'] is not None else ''),
            'near_collision_range_m': '%.4f' % self.near_collision_range,
            'near_collision_clear_range_m': (
                '%.4f' % self.near_collision_clear_range),
            'path_length_total_m': '%.4f' % sum(self.path_length.values()),
            'execution_path_cost_mode': self.execution_path_cost_mode,
            'robot1_x': self.positions.get('robot1', ('', ''))[0],
            'robot1_y': self.positions.get('robot1', ('', ''))[1],
            'robot2_x': self.positions.get('robot2', ('', ''))[0],
            'robot2_y': self.positions.get('robot2', ('', ''))[1],
            'robot3_x': self.positions.get('robot3', ('', ''))[0],
            'robot3_y': self.positions.get('robot3', ('', ''))[1],
            'vision_enabled': int(self.vision_enabled),
            'vision_scoring_mode': self.vision_scoring_mode,
            'vision_projection_mode': self.vision_projection_mode,
            'vision_feature_scale': '%.4f' % self.vision_feature_scale,
            'vision_ready_robots': len(self.vision_scores),
            'vision_image_mean_openness': (
                '%.6f' % float(np.mean(image_scores)) if image_scores else ''),
            'vision_token_mean_openness': (
                '%.6f' % (self.vision_token_openness_sum /
                          self.vision_token_count)
                if self.vision_token_count else ''),
            'vision_token_nonzero_ratio': (
                '%.6f' % (float(self.vision_token_nonzero_count) /
                          self.vision_token_count)
                if self.vision_token_count else ''),
            'vision_observation_count': self.vision_observation_count,
            'vision_observation_mean_openness': (
                '%.6f' % vision_observation_mean
                if vision_observation_mean is not None else ''),
            'vision_observation_std_openness': (
                '%.6f' % math.sqrt(vision_observation_variance)
                if vision_observation_variance is not None else ''),
            'vision_direction_contrast_mean': (
                '%.6f' % (self.vision_direction_contrast_sum /
                          self.vision_observation_count)
                if self.vision_observation_count else ''),
            'scan_ready_robots': len(self.scan_ready),
            'mapped_obstacle_cells': int(np.sum(self.grid == 2)),
            'idle_robots': sum(self.current_goal.get(name) is None for name in ROBOT_NAMES),
            'policy_actions_received': self.policy_actions_received,
            'policy_select_requests': self.policy_select_requests,
            'policy_select_accepted': self.policy_select_accepted,
            'policy_select_stale': self.policy_select_stale,
            'policy_select_owner_rejected': self.policy_select_owner_rejected,
            'policy_action_timeouts': self.policy_action_timeouts,
            'goals_dropped_owner': self.goals_dropped_owner,
            'goals_dropped_blocked': self.goals_dropped_blocked,
            'goals_cleared_wait': self.goals_cleared_wait,
            'goals_replaced_policy': self.goals_replaced_policy,
            'hybrid_fallback_active': int(self.hybrid_fallback.active),
            'hybrid_fallback_activations': self.hybrid_fallback.activations,
            'hybrid_fallback_goal_assignments': self.hybrid_fallback_goal_assignments,
            'adaptive_execution_weight': '%.4f' % self.adaptive_execution_weight,
            'tail_completion_active': int(self.tail_completion_active),
            'tail_completion_activations': self.tail_completion_activations,
            'tail_completion_goal_assignments': self.tail_completion_goal_assignments,
            'mission_done': int(self.exploration_done_logged),
            'evaluation_horizon_reached': int(
                self.evaluation_horizon_reached),
        }
        # ROS shutdown callbacks can run while the final control iteration is
        # building this row. Serialize the actual write with file closure so a
        # timeout cannot produce an otherwise spurious node traceback.
        with self._metrics_lock:
            if self._metrics_writer is None or self._metrics_file is None:
                return
            self._metrics_writer.writerow(row)
            self._metrics_file.flush()

    def filter_blacklisted(self, pts):
        """Drop any point within STUCK_BLACKLIST_RADIUS of a reported stuck
        spot. Used only for goal-selection candidates -- never for the raw
        frontier count that drives the completion check."""
        active_points = self.blacklist.active_points(self._control_now())
        if not active_points or len(pts) == 0:
            return pts
        bl = np.array(active_points)
        d = np.linalg.norm(pts[:, None, :] - bl[None, :, :], axis=2)
        keep = d.min(axis=1) >= STUCK_BLACKLIST_RADIUS
        return pts[keep]

    def _safe_sleep(self, rate):
        # rate.sleep() raises ROSTimeMovedBackwardsException if the simulated
        # /clock ever ticks non-monotonically (observed occasionally under
        # WSL2). Recreating the Rate object resyncs it to the current time.
        try:
            rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            return rate
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            rospy.logwarn("ROS time moved backwards, resyncing rate")
            return rospy.Rate(UPDATE_HZ)
        return rate

    def _write_status_file(self, coverage, all_frontiers):
        now = time.time()
        if now - self._last_status_write < STATUS_WRITE_PERIOD:
            return
        self._last_status_write = now
        try:
            rows = []
            for name in ROBOT_NAMES:
                pos = self.positions.get(name)
                goal = self.current_goal.get(name)
                pos_s = "(%.2f, %.2f)" % pos if pos else "--"
                goal_s = "(%.2f, %.2f)" % goal if goal else "空闲 (无目标)"
                rows.append((name, pos_s, goal_s))

            done = self.exploration_done_logged
            status_word = "探索完成" if done else "探索中"
            status_color = "#2e7d32" if done else "#1565c0"
            elapsed = now - self.start_time
            rows_html = "\n".join(
                "<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % r for r in rows
            )
            html = """<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="2">
<title>探索进度</title>
<style>
body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; padding: 24px; background:#fafafa; }}
h1 {{ font-size: 20px; }}
.status {{ font-size: 42px; font-weight: bold; color: {status_color}; }}
.cov {{ font-size: 64px; font-weight: bold; margin: 12px 0; }}
.bar {{ background:#e0e0e0; border-radius: 8px; height: 24px; width: 100%; max-width: 500px; overflow:hidden; }}
.bar-fill {{ background: {status_color}; height: 100%; width: {coverage:.1f}%; transition: width 1s; }}
table {{ border-collapse: collapse; margin-top: 20px; }}
td, th {{ border: 1px solid #ccc; padding: 6px 14px; text-align: left; }}
.meta {{ color: #666; margin-top: 16px; font-size: 13px; }}
</style></head>
<body>
<h1>Voronoi 多机器人探索 -- 实时进度</h1>
<div class="status">{status_word}</div>
<div class="cov">{coverage:.1f}%</div>
<div class="bar"><div class="bar-fill"></div></div>
<table>
<tr><th>机器人</th><th>当前位置</th><th>当前目标</th></tr>
{rows_html}
</table>
<div class="meta">
剩余前沿点(未探索边界): {n_frontiers} 个 &nbsp;|&nbsp;
卡住拉黑次数: {n_blacklist} &nbsp;|&nbsp;
已运行: {elapsed:.0f} 秒 &nbsp;|&nbsp;
本页每2秒自动刷新,直接开着浏览器标签页看就行,不用敲命令
</div>
</body></html>""".format(
                status_color=status_color, coverage=coverage, status_word=status_word,
                rows_html=rows_html, n_frontiers=len(all_frontiers),
                n_blacklist=self.blacklist.count(self._control_now()), elapsed=elapsed)
            status_directory = os.path.dirname(self.status_html_path)
            if status_directory and not os.path.isdir(status_directory):
                os.makedirs(status_directory)
            tmp_path = self.status_html_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(html)
            os.replace(tmp_path, self.status_html_path)
        except (IOError, OSError) as e:
            rospy.logwarn_throttle(10.0, "[coordinator] could not write status file: %s", e)

    def spin(self):
        rate = rospy.Rate(UPDATE_HZ)
        while not rospy.is_shutdown():
            if len(self.positions) < len(ROBOT_NAMES):
                rospy.loginfo_throttle(2.0, "waiting for all robot poses... have %s",
                                        list(self.positions.keys()))
                rate = self._safe_sleep(rate)
                continue

            if (self.vision_enabled and
                    self.selection_mode in (('vision',) +
                                            EXECUTION_SELECTION_MODES) and
                    time.time() - self.start_time > 10.0 and
                    len(self.vision_scores) < len(ROBOT_NAMES)):
                missing = [name for name in ROBOT_NAMES if name not in self.vision_scores]
                rospy.logerr_throttle(
                    10.0, '[coordinator] camera input missing for %s; vision term is zero',
                    ', '.join(missing))

            if (self.mapping_mode == 'laser_scan' and
                    time.time() - self.start_time > 10.0 and
                    len(self.scan_ready) < len(ROBOT_NAMES)):
                missing = [name for name in ROBOT_NAMES if name not in self.scan_ready]
                rospy.logerr_throttle(
                    10.0, '[coordinator] laser mapping input missing for %s',
                    ', '.join(missing))

            if self.mapping_mode == 'ground_truth_radius':
                for name in ROBOT_NAMES:
                    x, y = self.positions[name]
                    self.mark_explored(x, y, SENSING_RANGE)

            coverage = self.coverage_pct()
            fallback_active = False
            if self.selection_mode in HYBRID_SELECTION_MODES:
                fallback_activated, fallback_active = self.hybrid_fallback.update(
                    coverage, self._control_now())
                if fallback_activated:
                    rospy.logwarn(
                        '[coordinator] coverage stalled at %.2f%%; using execution-aware '
                        'frontier selection for %.0f seconds',
                        coverage, self.hybrid_fallback_active_seconds)
                    # A bounded recovery window must be able to replace stale
                    # goals; otherwise fallback has no effect until the low-level
                    # controller independently reports a failure.
                    for name in ROBOT_NAMES:
                        self.current_goal[name] = None
                        self.current_goal_id[name] = None
                        self.current_path[name] = None
            tail_active = bool(
                self.tail_completion_enabled and
                self.selection_mode in POLICY_SELECTION_MODES and
                coverage >= self.tail_completion_coverage)
            if tail_active and not self.tail_completion_active:
                self.tail_completion_activations += 1
                rospy.logwarn(
                    '[coordinator] coverage reached %.2f%%; switching to '
                    'deterministic execution-aware tail completion', coverage)
                for name in ROBOT_NAMES:
                    self.current_goal[name] = None
                    self.current_goal_id[name] = None
                    self.current_path[name] = None
            self.tail_completion_active = tail_active
            analytic_override_active = fallback_active or tail_active
            # raw frontier count drives the completion check -- this is a
            # ground-truth read of the grid, untouched by blacklisting, so a
            # handful of stuck-recovery blacklist points can never make the
            # coordinator believe a mostly-unexplored arena is finished.
            all_frontiers = self.detect_frontiers()
            self._write_status_file(coverage, all_frontiers)

            if len(all_frontiers) == 0:
                if self.no_frontier_since is None:
                    self.no_frontier_since = self._control_now()
            else:
                self.no_frontier_since = None

            all_idle = all(self.current_goal.get(name) is None for name in ROBOT_NAMES)
            if all_idle:
                if self.all_idle_since is None:
                    self.all_idle_since = self._control_now()
            else:
                self.all_idle_since = None

            if not self.exploration_done_logged:
                control_now = self._control_now()
                sensing_ready = (
                    self.mapping_mode != 'laser_scan' or
                    len(self.scan_ready) == len(ROBOT_NAMES))
                strictly_done = (
                    sensing_ready and
                    (control_now - self.start_control_time) >= STARTUP_GRACE_PERIOD and
                    self.no_frontier_since is not None and
                    (control_now - self.no_frontier_since) >= EMPTY_FRONTIER_HOLD)
                coverage_done = (
                    self.complete_on_coverage and sensing_ready and
                    (control_now - self.start_control_time) >= STARTUP_GRACE_PERIOD and
                    coverage >= self.coverage_done_threshold)
                practically_done = (
                    not self.complete_on_coverage and
                    all_idle and self.all_idle_since is not None and
                    coverage >= self.coverage_done_threshold and
                    (control_now - self.all_idle_since) >= self.idle_done_hold)
                if strictly_done or coverage_done or practically_done:
                    if strictly_done:
                        reason = "no frontier points remain"
                    elif coverage_done:
                        reason = "coverage %.1f%% reached the %.0f%% evaluation threshold" % (
                            coverage, self.coverage_done_threshold)
                    else:
                        reason = (
                            "coverage %.1f%% >= %.0f%% and all robots idle for >=%.0fs "
                            "(%d remaining frontier(s) likely unreachable)" %
                            (coverage, self.coverage_done_threshold,
                             self.idle_done_hold, len(all_frontiers)))
                    rospy.loginfo("Exploration complete: %s. Signalling all robots to stop.", reason)
                    self.exploration_done_logged = True
                    for name in ROBOT_NAMES:
                        self.done_pub[name].publish(Bool(data=True))

            control_elapsed = self._control_now() - self.start_control_time
            if (not self.exploration_done_logged and
                    not self.evaluation_horizon_reached and
                    self.max_simulation_seconds > 0.0 and
                    control_elapsed >= self.max_simulation_seconds):
                self.evaluation_horizon_reached = True
                rospy.loginfo(
                    'Evaluation horizon reached at %.2f simulation seconds; '
                    'stopping without marking the coverage objective complete.',
                    control_elapsed)
                for name in ROBOT_NAMES:
                    self.done_pub[name].publish(Bool(data=True))

            if self.exploration_done_logged or self.evaluation_horizon_reached:
                self._write_metrics(
                    coverage, all_frontiers, np.empty((0, 2)), force=True)
                if self.shutdown_on_done:
                    shutdown_reason = (
                        'exploration mission complete'
                        if self.exploration_done_logged else
                        'evaluation simulation horizon reached')
                    rospy.signal_shutdown(shutdown_reason)
                    break
                rate = self._safe_sleep(rate)
                continue

            rospy.loginfo_throttle(
                5.0, "[coordinator] coverage %.1f%%, frontiers remaining: %d, blacklisted spots: %d",
                coverage, len(all_frontiers), self.blacklist.count(self._control_now()))

            # candidates actually offered to robots -- same frontier set,
            # minus points near a reported stuck spot. If this empties out
            # while all_frontiers is still non-empty, every robot just goes
            # idle this tick rather than falsely declaring the mission done.
            frontiers = self.cluster_frontier_points(self.filter_blacklisted(all_frontiers))
            self._write_metrics(coverage, all_frontiers, frontiers)
            if len(frontiers) == 0:
                rate = self._safe_sleep(rate)
                continue
            frontier_ids = self.track_frontiers(frontiers)

            robot_positions = np.array([self.positions[n] for n in ROBOT_NAMES])
            execution_costs, ownership_costs, execution_features = self._execution_costs(
                frontiers, robot_positions)
            if self.selection_mode in EXECUTION_SELECTION_MODES:
                effective_ownership_costs = ownership_costs
                if self.selection_mode in ADAPTIVE_SELECTION_MODES:
                    adaptive_coverage = (
                        self.adaptive_execution_full_coverage
                        if tail_active else coverage)
                    effective_ownership_costs, self.adaptive_execution_weight = \
                        blend_phase_costs(
                            execution_features['distance'], ownership_costs,
                            adaptive_coverage,
                            start_coverage=self.adaptive_execution_start_coverage,
                            full_coverage=self.adaptive_execution_full_coverage,
                            cross_robot=True)
                else:
                    self.adaptive_execution_weight = 1.0
                owner = assign_by_cost(effective_ownership_costs)
            else:
                self.adaptive_execution_weight = 0.0
                owner = self.voronoi_owner(frontiers, ROBOT_NAMES, robot_positions)
            self._publish_policy_observations(
                coverage, frontiers, frontier_ids, owner,
                execution_costs, execution_features)

            for i, name in enumerate(ROBOT_NAMES):
                pos = np.array(self.positions[name])
                goal = self.current_goal.get(name)

                reached = goal is not None and np.linalg.norm(pos - np.array(goal)) < ARRIVE_THRESH
                if reached:
                    self.goals_reached += 1

                held_long_enough = (
                    self._control_now() - self.goal_set_time.get(name, 0.0)
                    >= MIN_GOAL_HOLD_TIME)

                # Voronoi partitions are dynamic: as robots move, a goal
                # picked earlier can end up closer to a *different* robot.
                # Re-check ownership so a robot drops a goal once it's no
                # longer "its" cell, instead of wandering into territory
                # another robot already owns -- but only once the goal has
                # been held a little while, to avoid flip-flopping between
                # near-tied candidates every single tick.
                still_owned = True
                if goal is not None and held_long_enough:
                    if self.selection_mode in EXECUTION_SELECTION_MODES:
                        goal_id = self.current_goal_id.get(name)
                        matching = [index for index, frontier_id in enumerate(frontier_ids)
                                    if frontier_id == goal_id]
                        if matching:
                            still_owned = int(owner[matching[0]]) == i
                    else:
                        dists_to_goal = np.linalg.norm(robot_positions - np.array(goal), axis=1)
                        still_owned = int(np.argmin(dists_to_goal)) == i

                # In the legacy known-obstacle mode, a newly invalid direct
                # line is enough to drop a goal. In an incrementally mapped
                # world, discovering an obstacle should instead trigger A*
                # replanning; only an occupied goal cell invalidates the goal.
                blocked = False
                if goal is not None and held_long_enough:
                    if self.mapping_mode == 'ground_truth_radius':
                        blocked = not self._line_of_sight_clear(
                            pos, np.array(goal))
                    else:
                        start_cell = self._cell_index(*pos)
                        goal_cell = self._cell_index(*goal)
                        blocked = self.grid[goal_cell[1], goal_cell[0]] == 2
                        if not blocked:
                            replanned = self._plan_path(start_cell, goal_cell)
                            if replanned:
                                self.current_path[name] = [
                                    (self.xs[ix], self.ys[iy])
                                    for ix, iy in replanned]

                invalid_goal = reached or not still_owned or blocked
                if invalid_goal:
                    if not still_owned:
                        self.goals_dropped_owner += 1
                    if blocked:
                        self.goals_dropped_blocked += 1
                    goal = None
                    self.current_goal[name] = None
                    self.current_goal_id[name] = None
                    self.current_path[name] = None

                policy_controls_selection = (
                    self.selection_mode in POLICY_SELECTION_MODES and
                    not analytic_override_active)
                policy_index = None
                if policy_controls_selection:
                    action_kind, action_index = self._consume_policy_action(
                        name, frontiers, frontier_ids, owner, i)
                    if action_kind == 'wait':
                        if goal is not None:
                            self.goals_cleared_wait += 1
                        goal = None
                        self.current_goal[name] = None
                        self.current_goal_id[name] = None
                        self.current_path[name] = None
                    elif action_kind == 'select':
                        selected_id = frontier_ids[action_index]
                        same_goal = (goal is not None and
                                     self.current_goal_id.get(name) == selected_id)
                        # The action mask normally prevents a new SELECT while
                        # navigation is active. Keep this authoritative guard
                        # as well because an observation and its ROS action can
                        # straddle a goal assignment in another control tick.
                        if not same_goal and goal is None:
                            policy_index = action_index
                    # CONTINUE intentionally keeps the existing target.

                should_select = (
                    policy_index is not None if policy_controls_selection
                    else goal is None)
                if should_select:
                    replacing_policy_goal = (
                        policy_controls_selection and goal is not None)
                    own_indices = (np.asarray([policy_index], dtype=np.int64)
                                   if policy_controls_selection
                                   else np.flatnonzero(owner == i))
                    if len(own_indices) > 0:
                        # prefer line-of-sight-clear candidates, but fall back
                        # to the unfiltered set rather than leaving the robot
                        # with nothing to chase (better to attempt a path and
                        # lean on the DDPG's own reactive avoidance than to
                        # freeze in place).
                        if self.mapping_mode == 'ground_truth_radius':
                            clear_mask = np.array([
                                self._line_of_sight_clear(pos, frontiers[index])
                                for index in own_indices])
                            if clear_mask.any():
                                own_indices = own_indices[clear_mask]
                    if len(own_indices) == 0:
                        if reached or not still_owned or blocked:
                            goal = None
                            self.current_goal[name] = None
                            self.current_goal_id[name] = None
                    else:
                        own_frontiers = frontiers[own_indices]
                        selected_path = None
                        if self.selection_mode in EXECUTION_SELECTION_MODES:
                            start_cell = self._cell_index(*pos)
                            candidate_paths = []
                            candidate_path_costs = []
                            for candidate in own_frontiers:
                                path = self._plan_path(
                                    start_cell, self._cell_index(*candidate))
                                candidate_paths.append(path)
                                candidate_path_costs.append(
                                    (len(path) - 1) * GRID_RES if path else
                                    np.linalg.norm(candidate - pos) * 1.5 + 1.0)
                            refined_execution_cost = (
                                execution_costs[i, own_indices]
                                + 0.25 * normalize_rows(np.asarray(candidate_path_costs)))
                            if self.selection_mode in ADAPTIVE_SELECTION_MODES:
                                adaptive_coverage = (
                                    self.adaptive_execution_full_coverage
                                    if tail_active else coverage)
                                refined_cost, _ = blend_phase_costs(
                                    execution_features['distance'][i, own_indices],
                                    refined_execution_cost, adaptive_coverage,
                                    start_coverage=self.adaptive_execution_start_coverage,
                                    full_coverage=self.adaptive_execution_full_coverage)
                            else:
                                refined_cost = refined_execution_cost
                            local_best = int(np.argmin(refined_cost))
                            best_index = own_indices[local_best]
                            selected_path = candidate_paths[local_best]
                        else:
                            d = np.linalg.norm(own_frontiers - pos, axis=1)
                            first_pos = np.array(self.first_node_pos[name])
                            delta = np.linalg.norm(own_frontiers - first_pos, axis=1)
                            utility = ALPHA * d + (1 - ALPHA) * delta
                            if self.selection_mode == 'vision':
                                yaw = self.yaws.get(name, 0.0)
                                vision = np.array([
                                    self.vision_bias(pos, yaw, f, name) for f in own_frontiers])
                                utility -= VISION_WEIGHT * vision
                            best_index = own_indices[np.argmin(utility)]
                        best = frontiers[best_index]
                        if replacing_policy_goal:
                            self.goals_replaced_policy += 1
                        goal = tuple(best)
                        self.current_goal[name] = goal
                        self.current_goal_id[name] = frontier_ids[best_index]
                        self.goal_set_time[name] = self._control_now()
                        self.goal_assignments += 1
                        if (self.selection_mode in HYBRID_SELECTION_MODES and
                                fallback_active):
                            self.hybrid_fallback_goal_assignments += 1
                        if tail_active:
                            self.tail_completion_goal_assignments += 1

                        # Algorithm 1 line 9: move via shortest path through
                        # already-explored (sensed, collision-free) space
                        # rather than a beeline that can cut through walls
                        # or obstacle corners.
                        start_cell = self._cell_index(*pos)
                        goal_cell = self._cell_index(*best)
                        path_cells = selected_path
                        if (path_cells is None and self.selection_mode not in
                                EXECUTION_SELECTION_MODES):
                            path_cells = self._plan_path(start_cell, goal_cell)
                        if path_cells:
                            self.current_path[name] = [(self.xs[ix], self.ys[iy]) for (ix, iy) in path_cells]
                        else:
                            self.current_path[name] = None  # fall back to straight line below
                            self.no_path_fallbacks += 1
                            start_value = self.grid[start_cell[1], start_cell[0]]
                            goal_value = self.grid[goal_cell[1], goal_cell[0]]
                            if start_value == 2:
                                self.no_path_start_occupied += 1
                            elif goal_value == 2:
                                self.no_path_goal_occupied += 1
                            else:
                                self.no_path_disconnected += 1
                        rospy.loginfo("[coordinator] mode=%s goal for %s: (%.2f, %.2f) [path: %s]",
                                      self.selection_mode, name, best[0], best[1],
                                      "%d waypoints" % len(path_cells) if path_cells else "none, using beeline")

                if goal is None:
                    continue

                path_xy = self.current_path.get(name)
                if path_xy:
                    waypoint = waypoint_along_path(path_xy, pos, MAX_WAYPOINT_DIST)
                else:
                    # no plannable path (e.g. goal not yet reachable through
                    # sensed space) -- fall back to a bounded-lookahead
                    # straight line, same as before this feature existed.
                    vec = np.array(goal) - pos
                    dist_to_goal = np.linalg.norm(vec)
                    if dist_to_goal > MAX_WAYPOINT_DIST:
                        waypoint = pos + vec / dist_to_goal * MAX_WAYPOINT_DIST
                    else:
                        waypoint = np.array(goal)

                msg = Point(x=waypoint[0], y=waypoint[1], z=0.0)
                self.targets_pub[name].publish(msg)

            rate = self._safe_sleep(rate)


def main():
    rospy.init_node('exploration_coordinator', anonymous=False)
    coord = Coordinator()
    coord.spin()


if __name__ == "__main__":
    main()
