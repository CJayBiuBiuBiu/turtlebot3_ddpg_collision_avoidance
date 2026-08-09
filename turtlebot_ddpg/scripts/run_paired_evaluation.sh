#!/usr/bin/env bash
set -eo pipefail

repeats="${1:-5}"
episode_seconds="${2:-90}"
seed_base="${3:-1000}"
if ! [[ "${repeats}" =~ ^[1-9][0-9]*$ &&
        "${episode_seconds}" =~ ^[1-9][0-9]*$ &&
        "${seed_base}" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [repeats] [episode_seconds] [seed_base]" >&2
  exit 2
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
set -u

checkpoint="${project_root}/outputs/checkpoints/mappo_latest.pt"
run_id="$(date +%Y%m%d_%H%M%S)"
output_dir="${project_root}/outputs/evaluation/paired_${run_id}"
mkdir -p "${output_dir}"

if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  exit 2
fi
checkpoint_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
printf '%s\n' \
  'field,value' \
  'protocol,seed_matched_core_evaluation' \
  'methods,mappo;execution_aware' \
  "repeats,${repeats}" \
  "episode_seconds,${episode_seconds}" \
  "seed_base,${seed_base}" \
  "checkpoint,${checkpoint}" \
  "checkpoint_sha256,${checkpoint_sha256}" \
  'mapping_mode,laser_scan' \
  'vision_enabled,true' \
  >"${output_dir}/experiment_manifest.csv"

cleanup_ros() {
  set +e
  if rostopic list >/dev/null 2>&1; then
    timeout 5s rosnode kill -a >/dev/null 2>&1
  fi
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
trap 'exit 130' INT TERM
cleanup_ros

run_method() {
  local repeat_id="$1"
  local method="$2"
  local experiment_seed="$3"
  local metrics_path="${output_dir}/repeat_${repeat_id}_${method}.csv"
  local launch_log="${output_dir}/repeat_${repeat_id}_${method}.log"
  local network_policy=false
  if [[ "${method}" == "mappo" ]]; then
    network_policy=true
  fi

  echo "[paired_eval] repeat ${repeat_id}, seed ${experiment_seed}, method ${method}"
  set +e
  timeout --signal=INT --kill-after=20s "${episode_seconds}s" \
    roslaunch turtlebot_ddpg gazebo_research_experiment.launch \
      gui:=false verbose:=false selection_mode:="${method}" \
      start_smoke_policy:=false \
      start_network_policy:="${network_policy}" \
      start_mappo_training:=false \
      mappo_checkpoint:="${checkpoint}" \
      network_deterministic:=true \
      experiment_seed:="${experiment_seed}" \
      metrics_path:="${metrics_path}" >"${launch_log}" 2>&1
  local status=$?
  set -e
  cleanup_ros
  if [[ ${status} -ne 0 && ${status} -ne 124 && ${status} -ne 130 ]]; then
    echo "[paired_eval] ${method} failed with status ${status}" >&2
    tail -40 "${launch_log}" >&2
    exit "${status}"
  fi
  if [[ ! -s "${metrics_path}" ]]; then
    echo "[paired_eval] ${method} produced no metrics" >&2
    exit 1
  fi
  python3 - "${metrics_path}" "${method}" <<'PY'
import csv
import sys

path, method = sys.argv[1:]
with open(path, newline='') as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit('%s has only a CSV header' % method)
last = rows[-1]
if int(last.get('scan_ready_robots', 0) or 0) != 3:
    raise SystemExit('%s did not receive all three laser streams' % method)
if method == 'mappo' and int(last.get('vision_ready_robots', 0) or 0) != 3:
    raise SystemExit('%s did not receive all three camera streams' % method)
PY
}

for ((repeat = 1; repeat <= repeats; repeat++)); do
  repeat_id="$(printf '%02d' "${repeat}")"
  experiment_seed=$((seed_base + repeat - 1))
  if (( repeat % 2 == 1 )); then
    run_method "${repeat_id}" mappo "${experiment_seed}"
    run_method "${repeat_id}" execution_aware "${experiment_seed}"
  else
    run_method "${repeat_id}" execution_aware "${experiment_seed}"
    run_method "${repeat_id}" mappo "${experiment_seed}"
  fi
done

python3 "${project_root}/turtlebot_ddpg/scripts/summarize_evaluation.py" "${output_dir}"
