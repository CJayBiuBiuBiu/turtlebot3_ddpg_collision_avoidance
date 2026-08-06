#!/usr/bin/env bash
set -eo pipefail

repeats="${1:-3}"
episode_seconds="${2:-90}"
evaluation_seed_base="${3:-9400}"
read -r -a training_seeds <<< "${TRAINING_SEEDS:-1101 2202 3303}"

project_root="/home/ros/catkin_ws/src/turtlebot3_ddpg_collision_avoidance"
package_root="${project_root}/turtlebot_ddpg"
checkpoint_dir="${INDEPENDENT_CHECKPOINT_DIR:-${project_root}/outputs/checkpoints/independent_seed_study}"
evaluation_root="${INDEPENDENT_EVALUATION_ROOT:-${project_root}/outputs/evaluation/independent_training_seed_study}"
mkdir -p "${evaluation_root}"

for training_seed in "${training_seeds[@]}"; do
  checkpoint="${checkpoint_dir}/mappo_from_scratch_seed_${training_seed}.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing independent checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  echo "[independent_seed_eval] training seed ${training_seed}"
  MAPPO_CHECKPOINT="${checkpoint}" \
  OUTPUT_DIR="${evaluation_root}/training_seed_${training_seed}" \
    "${package_root}/scripts/run_hu_aligned_evaluation.sh" \
      "${repeats}" "${episode_seconds}" 97 mappo_vision \
      "${evaluation_seed_base}"
done

echo "[independent_seed_eval] complete: ${evaluation_root}"
