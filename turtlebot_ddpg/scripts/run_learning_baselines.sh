#!/usr/bin/env bash
set -eo pipefail

# Train MAPPO and the controlled IPPO baseline with matched seeds and inputs.
episodes="${1:-20}"
episode_seconds="${2:-120}"
rollout_length="${3:-32}"
decision_interval="${4:-2.5}"
robot_count="${5:-3}"
seeds=(1101 2202 3303)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
study_id="$(date +%Y%m%d_%H%M%S)"
study_dir="${project_root}/outputs/training/learning_baselines_${study_id}"
mkdir -p "${study_dir}"

for algorithm in ippo mappo; do
  for seed in "${seeds[@]}"; do
    run_dir="${study_dir}/${algorithm}/seed_${seed}"
    checkpoint="${run_dir}/${algorithm}_seed_${seed}.pt"
    if [[ -e "${checkpoint}" ]]; then
      echo "Refusing to overwrite ${checkpoint}" >&2
      exit 2
    fi
    echo "[learning_baselines] ${algorithm}, seed ${seed}, ${robot_count} robots"
    TRAINING_OUTPUT_DIR="${run_dir}" \
      "${package_root}/scripts/run_mappo_episodes.sh" \
      "${episodes}" "${episode_seconds}" "${rollout_length}" \
      "${decision_interval}" research "${seed}" "${checkpoint}" \
      false "${algorithm}" "${robot_count}"
  done
done

python3 "${package_root}/scripts/plot_learning_comparison.py" \
  "${study_dir}" --output "${study_dir}/learning_comparison.png"
echo "[learning_baselines] results: ${study_dir}"
