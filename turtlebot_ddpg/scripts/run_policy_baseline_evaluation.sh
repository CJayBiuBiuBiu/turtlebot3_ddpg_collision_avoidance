#!/usr/bin/env bash
set -eo pipefail

# Matched-seed evaluation of independently trained IPPO and MAPPO actors.
repeats="${1:-5}"
episode_seconds="${2:-180}"
seed_base="${3:-17000}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
set -u

mappo_checkpoint="${MAPPO_CHECKPOINT:-${project_root}/outputs/checkpoints/mappo_latest.pt}"
ippo_checkpoint="${IPPO_CHECKPOINT:-${project_root}/outputs/checkpoints/ippo_latest.pt}"
for checkpoint in "${mappo_checkpoint}" "${ippo_checkpoint}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found: ${checkpoint}" >&2
    exit 2
  fi
done
run_id="$(date +%Y%m%d_%H%M%S)"
output_dir="${OUTPUT_DIR:-${project_root}/outputs/evaluation/policy_baselines_${run_id}}"
mkdir -p "${output_dir}"
output_dir="$(readlink -f "${output_dir}")"
printf '%s\n' \
  'field,value' \
  'comparison,IPPO_vs_MAPPO' \
  "repeats,${repeats}" \
  "episode_seconds,${episode_seconds}" \
  "seed_base,${seed_base}" \
  "mappo_checkpoint,${mappo_checkpoint}" \
  "mappo_checkpoint_sha256,$(sha256sum "${mappo_checkpoint}" | awk '{print $1}')" \
  "ippo_checkpoint,${ippo_checkpoint}" \
  "ippo_checkpoint_sha256,$(sha256sum "${ippo_checkpoint}" | awk '{print $1}')" \
  'vision_enabled,false' \
  'fallback_enabled,false' \
  >"${output_dir}/experiment_manifest.csv"

render_prefix=()
if ! timeout 2s xdpyinfo >/dev/null 2>&1; then
  if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "No working display is available and xvfb-run is not installed" >&2
    exit 2
  fi
  render_prefix=(xvfb-run -a -s "-screen 0 1280x720x24")
fi
cleanup_ros() {
  set +e
  timeout 5s rosnode kill -a >/dev/null 2>&1
  pkill -INT -x roslaunch >/dev/null 2>&1
  pkill -INT -x gzserver >/dev/null 2>&1
  pkill -INT -x gzclient >/dev/null 2>&1
  pkill -INT -x rosmaster >/dev/null 2>&1
  sleep 3
  pkill -KILL -x gzserver >/dev/null 2>&1
  pkill -KILL -x gzclient >/dev/null 2>&1
  pkill -KILL -x rosmaster >/dev/null 2>&1
  set -e
}
trap cleanup_ros EXIT
cleanup_ros

for ((repeat = 1; repeat <= repeats; repeat++)); do
  seed=$((seed_base + repeat))
  for algorithm in ippo mappo; do
    if [[ "${algorithm}" == "ippo" ]]; then
      checkpoint="${ippo_checkpoint}"
    else
      checkpoint="${mappo_checkpoint}"
    fi
    stem="repeat_$(printf '%02d' "${repeat}")_${algorithm}"
    metrics_path="${output_dir}/${stem}.csv"
    launch_log="${output_dir}/${stem}.log"
    echo "[policy_baseline] ${algorithm}, repeat ${repeat}/${repeats}, seed ${seed}"
    set +e
    timeout --signal=INT --kill-after=20s "$((episode_seconds * 3 + 60))s" \
      "${render_prefix[@]}" roslaunch turtlebot_ddpg gazebo_research_experiment.launch \
        gui:=false verbose:=false world_profile:=research \
        world_name:="${package_root}/worlds/voronoi_exploration_arena.world" \
        selection_mode:="${algorithm}" vision_enabled:=false \
        mapping_mode:=laser_scan start_network_policy:=true \
        mappo_checkpoint:="${checkpoint}" network_deterministic:=true \
        max_simulation_seconds:="${episode_seconds}" \
        complete_on_coverage:=false shutdown_on_done:=true \
        experiment_seed:="${seed}" gazebo_seed:="${seed}" \
        metrics_path:="${metrics_path}" >"${launch_log}" 2>&1
    status=$?
    set -e
    cleanup_ros
    if [[ ${status} -ne 0 && ${status} -ne 124 && ${status} -ne 130 ]]; then
      tail -40 "${launch_log}" >&2
      exit "${status}"
    fi
    if [[ ! -s "${metrics_path}" ]]; then
      echo "No metrics produced for ${stem}" >&2
      exit 1
    fi
  done
done

python3 "${package_root}/scripts/summarize_evaluation.py" "${output_dir}" \
  --positive-method mappo --negative-method ippo \
  --paired-output mappo_vs_ippo.csv
echo "[policy_baseline] results: ${output_dir}"
