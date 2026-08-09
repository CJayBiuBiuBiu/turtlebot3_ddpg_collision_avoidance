#!/usr/bin/env bash
set -euo pipefail

repeats="${VALIDATION_REPEATS:-3}"
episode_seconds="${VALIDATION_EPISODE_SECONDS:-300}"
seed_base="${VALIDATION_SEED_BASE:-7000}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
run_id="$(date +%Y%m%d_%H%M%S)"
validation_root="${VALIDATION_OUTPUT_DIR:-${project_root}/outputs/checkpoint_validation/${run_id}}"
mkdir -p "${validation_root}"

if [[ "$#" -gt 0 ]]; then
  checkpoints=("$@")
else
  shopt -s nullglob
  checkpoints=("${project_root}"/outputs/checkpoints/snapshots/mappo_update_*.pt)
  shopt -u nullglob
fi
if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "No checkpoint snapshots supplied or found." >&2
  exit 2
fi

for checkpoint in "${checkpoints[@]}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found: ${checkpoint}" >&2
    exit 2
  fi
  checkpoint="$(readlink -f "${checkpoint}")"
  digest="$(sha256sum "${checkpoint}" | awk '{print substr($1,1,12)}')"
  candidate_dir="${validation_root}/candidate_${digest}"
  MAPPO_CHECKPOINT="${checkpoint}" \
  EVALUATION_WORLD_PROFILE=research \
  OUTPUT_DIR="${candidate_dir}" \
    "${package_root}/scripts/run_hu_aligned_evaluation.sh" \
      "${repeats}" "${episode_seconds}" 97 mappo_vision "${seed_base}"
done

python3 "${package_root}/scripts/rank_checkpoint_validation.py" \
  "${validation_root}" --world-profile research
echo "[checkpoint_validation] complete: ${validation_root}"
