#!/usr/bin/env bash
set -eo pipefail

# Train matched MAPPO policies from scratch. Each seed is used once with the
# visual feature and once with that feature disabled; no checkpoint is shared
# between conditions.
episodes="${1:-6}"
episode_seconds="${2:-600}"
rollout_length="${3:-32}"
decision_interval="${4:-2.5}"
read -r -a seeds <<< "${VISION_STUDY_SEEDS:-101 202 303}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
study_dir="${project_root}/outputs/checkpoints/vision_seed_study"
mkdir -p "${study_dir}"

for seed in "${seeds[@]}"; do
  for condition in with_vision without_vision; do
    if [[ "${condition}" == "with_vision" ]]; then
      vision_enabled=true
    else
      vision_enabled=false
    fi
    checkpoint="${study_dir}/${condition}_seed_${seed}.pt"
    if [[ -e "${checkpoint}" ]]; then
      echo "Refusing to overwrite study checkpoint: ${checkpoint}" >&2
      exit 2
    fi
    echo "[vision_seed_study] ${condition}, seed ${seed}: training from scratch"
    "${package_root}/scripts/run_mappo_episodes.sh" \
      "${episodes}" "${episode_seconds}" "${rollout_length}" \
      "${decision_interval}" hu_aligned "${seed}" "${checkpoint}" \
      "${vision_enabled}"
  done
done

echo "[vision_seed_study] independent checkpoints written to ${study_dir}"
