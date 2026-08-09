#!/usr/bin/env bash
set -eo pipefail

episodes="${1:-6}"
episode_seconds="${2:-120}"
rollout_length="${3:-32}"
decision_interval="${4:-2.5}"
seeds=(1101 2202 3303)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
study_dir="${project_root}/outputs/checkpoints/independent_seed_study"
mkdir -p "${study_dir}"

for seed in "${seeds[@]}"; do
  checkpoint="${study_dir}/mappo_from_scratch_seed_${seed}.pt"
  if [[ -e "${checkpoint}" ]]; then
    echo "Refusing to overwrite independent checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  echo "[independent_seed_study] seed ${seed}: starting from scratch"
  "${package_root}/scripts/run_mappo_episodes.sh" \
    "${episodes}" "${episode_seconds}" "${rollout_length}" \
    "${decision_interval}" hu_aligned "${seed}" "${checkpoint}" true
done

echo "[independent_seed_study] checkpoints written to ${study_dir}"
