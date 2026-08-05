#!/usr/bin/env bash
set -eo pipefail

target_updates="${1:-18}"
episode_seconds="${2:-120}"
rollout_length="${3:-32}"
decision_interval="${4:-2.5}"
seeds=(1101 2202 3303)

if ! [[ "${target_updates}" =~ ^[1-9][0-9]*$ ]]; then
  echo "target_updates must be a positive integer" >&2
  exit 2
fi

project_root="/home/ros/catkin_ws/src/turtlebot3_ddpg_collision_avoidance"
package_root="${project_root}/turtlebot_ddpg"
source_dir="${project_root}/outputs/checkpoints/independent_seed_study"
target_dir="${project_root}/outputs/checkpoints/independent_seed_study_long"
source /home/ros/ddpg_env/bin/activate
mkdir -p "${target_dir}"

checkpoint_updates() {
  python3 - "$1" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(int(payload.get('metadata', {}).get('updates', 0)))
PY
}

for seed in "${seeds[@]}"; do
  source_checkpoint="${source_dir}/mappo_from_scratch_seed_${seed}.pt"
  target_checkpoint="${target_dir}/mappo_from_scratch_seed_${seed}.pt"
  if [[ ! -f "${source_checkpoint}" ]]; then
    echo "Missing source checkpoint: ${source_checkpoint}" >&2
    exit 2
  fi
  if [[ ! -f "${target_checkpoint}" ]]; then
    cp --preserve=timestamps "${source_checkpoint}" "${target_checkpoint}"
  fi
  current_updates="$(checkpoint_updates "${target_checkpoint}")"
  if (( current_updates > target_updates )); then
    echo "Checkpoint ${target_checkpoint} is already at ${current_updates} updates" >&2
    exit 2
  fi
  while (( current_updates < target_updates )); do
    additional_updates=$((target_updates - current_updates))
    echo "[independent_seed_continuation] seed ${seed}: ${current_updates} -> ${target_updates} updates"
    "${package_root}/scripts/run_mappo_episodes.sh" \
      "${additional_updates}" "${episode_seconds}" "${rollout_length}" \
      "${decision_interval}" hu_aligned "${seed}" "${target_checkpoint}" true
    next_updates="$(checkpoint_updates "${target_checkpoint}")"
    if (( next_updates <= current_updates )); then
      echo "Training made no update progress for seed ${seed}" >&2
      exit 1
    fi
    current_updates="${next_updates}"
  done
  echo "[independent_seed_continuation] seed ${seed}: reached ${target_updates} updates"
done

echo "[independent_seed_continuation] checkpoints written to ${target_dir}"
