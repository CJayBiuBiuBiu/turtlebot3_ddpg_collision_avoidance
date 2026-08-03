#!/usr/bin/env bash
set -eo pipefail

episodes="${1:-6}"
episode_seconds="${2:-600}"
rollout_length="${3:-32}"
decision_interval="${4:-2.5}"
seeds=(101 202 303)

project_root="/home/ros/catkin_ws/src/turtlebot3_ddpg_collision_avoidance"
package_root="${project_root}/turtlebot_ddpg"
base_checkpoint="${project_root}/outputs/checkpoints/mappo_update22_pre_hu_training.pt"
study_dir="${project_root}/outputs/checkpoints/continuation_seed_study"

if [[ ! -f "${base_checkpoint}" ]]; then
  echo "Base checkpoint not found: ${base_checkpoint}" >&2
  exit 2
fi
mkdir -p "${study_dir}"

for seed in "${seeds[@]}"; do
  checkpoint="${study_dir}/mappo_seed_${seed}.pt"
  if [[ -e "${checkpoint}" ]]; then
    echo "Refusing to overwrite an existing seed checkpoint: ${checkpoint}" >&2
    echo "Move it aside or evaluate it before restarting the study." >&2
    exit 2
  fi
  cp "${base_checkpoint}" "${checkpoint}"
  echo "[seed_study] seed ${seed}: ${episodes} Hu-aligned continuation episodes"
  "${package_root}/scripts/run_mappo_episodes.sh" \
    "${episodes}" "${episode_seconds}" "${rollout_length}" \
    "${decision_interval}" hu_aligned "${seed}" "${checkpoint}"
done

echo "[seed_study] checkpoints written to ${study_dir}"
