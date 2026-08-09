#!/usr/bin/env bash
set -eo pipefail

target_updates="${1:-18}"
episode_seconds="${2:-120}"
rollout_length="${3:-32}"
decision_interval="${4:-2.5}"
seeds=(4404 5505)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
target_dir="${project_root}/outputs/checkpoints/independent_seed_study_long"
mkdir -p "${target_dir}"

for seed in "${seeds[@]}"; do
  checkpoint="${target_dir}/mappo_from_scratch_seed_${seed}.pt"
  if [[ -e "${checkpoint}" ]]; then
    echo "Refusing to overwrite independent checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  echo "[independent_seed_expansion] seed ${seed}: training ${target_updates} updates from scratch"
  "${package_root}/scripts/run_mappo_episodes.sh" \
    "${target_updates}" "${episode_seconds}" "${rollout_length}" \
    "${decision_interval}" hu_aligned "${seed}" "${checkpoint}" true
done

echo "[independent_seed_expansion] checkpoints written to ${target_dir}"
