#!/usr/bin/env bash
set -eo pipefail

# Held-out layout transfer check. It deliberately reuses every learned model
# and controller parameter from the primary Hu-aligned experiment.
repeats="${1:-3}"
episode_seconds="${2:-900}"
coverage_threshold="${3:-97}"
seed_base="${4:-9500}"

package_root="/home/ros/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/turtlebot_ddpg"
export EVALUATION_WORLD_PROFILE=hu_layout_b
export EVALUATION_WORLD_FILE="${package_root}/worlds/hu_layout_b_arena.world"

exec "${package_root}/scripts/run_hu_aligned_evaluation.sh" \
  "${repeats}" "${episode_seconds}" "${coverage_threshold}" \
  hu_final_pair "${seed_base}"
