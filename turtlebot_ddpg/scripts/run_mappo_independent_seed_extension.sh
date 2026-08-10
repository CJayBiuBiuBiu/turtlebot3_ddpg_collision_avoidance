#!/usr/bin/env bash
set -eo pipefail

# Extend the short-budget independent-training audit from three to five seeds.
# Existing checkpoints are never overwritten.
episodes="${1:-6}"
episode_seconds="${2:-120}"
rollout_length="${3:-32}"
decision_interval="${4:-2.5}"
read -r -a seeds <<< "${TRAINING_SEEDS:-4404 5505}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
checkpoint_dir="${project_root}/outputs/checkpoints/independent_seed_study"
mkdir -p "${checkpoint_dir}"

for seed in "${seeds[@]}"; do
  checkpoint="${checkpoint_dir}/mappo_from_scratch_seed_${seed}.pt"
  if [[ -e "${checkpoint}" ]]; then
    echo "Refusing to overwrite independent checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  echo "[independent_seed_extension] seed ${seed}: ${episodes} fixed-budget episodes from scratch"
  "${package_root}/scripts/run_mappo_episodes.sh" \
    "${episodes}" "${episode_seconds}" "${rollout_length}" \
    "${decision_interval}" hu_aligned "${seed}" "${checkpoint}" true
done

echo "[independent_seed_extension] checkpoints written to ${checkpoint_dir}"
