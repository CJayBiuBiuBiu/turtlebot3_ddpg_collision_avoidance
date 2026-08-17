#!/usr/bin/env bash
set -eo pipefail

# Evaluate every matched IPPO/MAPPO training lineage on common Gazebo seeds.
training_dir="${1:?Usage: $0 TRAINING_DIR [eval_repeats] [episode_seconds] [seed_base]}"
eval_repeats="${2:-3}"
episode_seconds="${3:-120}"
seed_base="${4:-17000}"
training_seeds=(1101 2202 3303)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
training_dir="$(readlink -f "${training_dir}")"
output_dir="${OUTPUT_DIR:-${training_dir}/evaluation}"
mkdir -p "${output_dir}"
output_dir="$(readlink -f "${output_dir}")"

for training_seed in "${training_seeds[@]}"; do
  ippo_checkpoint="${training_dir}/ippo/seed_${training_seed}/ippo_seed_${training_seed}.pt"
  mappo_checkpoint="${training_dir}/mappo/seed_${training_seed}/mappo_seed_${training_seed}.pt"
  for checkpoint in "${ippo_checkpoint}" "${mappo_checkpoint}"; do
    if [[ ! -f "${checkpoint}" ]]; then
      echo "Checkpoint not found: ${checkpoint}" >&2
      exit 2
    fi
  done
  lineage_dir="${output_dir}/training_seed_${training_seed}"
  echo "[policy_study] training seed ${training_seed}"
  IPPO_CHECKPOINT="${ippo_checkpoint}" \
  MAPPO_CHECKPOINT="${mappo_checkpoint}" \
  OUTPUT_DIR="${lineage_dir}" \
    "${package_root}/scripts/run_policy_baseline_evaluation.sh" \
    "${eval_repeats}" "${episode_seconds}" "${seed_base}"
done

python3 "${package_root}/scripts/summarize_policy_baseline_study.py" \
  "${output_dir}" --output "${output_dir}/policy_baseline_summary.csv" \
  --runs-output "${output_dir}/policy_baseline_runs.csv" \
  --figure "${output_dir}/policy_baseline_comparison.png"
echo "[policy_study] results: ${output_dir}"
