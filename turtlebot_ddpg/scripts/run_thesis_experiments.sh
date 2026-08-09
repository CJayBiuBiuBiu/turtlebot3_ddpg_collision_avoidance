#!/usr/bin/env bash
set -eo pipefail

repeats="${1:-5}"
episode_seconds="${2:-90}"
method_filter="${3:-all}"
seed_base="${4:-1000}"
if ! [[ "${repeats}" =~ ^[1-9][0-9]*$ &&
        "${episode_seconds}" =~ ^[1-9][0-9]*$ &&
        "${seed_base}" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [repeats] [episode_seconds] [method|all] [seed_base]" >&2
  exit 2
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
set -u

checkpoint="${MAPPO_CHECKPOINT:-${project_root}/outputs/checkpoints/mappo_latest.pt}"
run_id="$(date +%Y%m%d_%H%M%S)"
output_dir="${project_root}/outputs/evaluation/thesis_laser_${run_id}"
methods=(geometric execution_aware mappo_no_vision mappo_vision)
if [[ "${method_filter}" != "all" ]]; then
  case "${method_filter}" in
    geometric|execution_aware|mappo_no_vision|mappo_vision|mappo_hybrid)
      methods=("${method_filter}") ;;
    *) echo "Unknown method filter: ${method_filter}" >&2; exit 2 ;;
  esac
fi
mkdir -p "${output_dir}"

if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  exit 2
fi

run_method() {
  local repeat_id="$1"
  local method="$2"
  local experiment_seed="$3"
  local selection_mode="${method}"
  local vision_enabled=false
  local network_policy=false

  case "${method}" in
    geometric|execution_aware) ;;
    mappo_no_vision)
      selection_mode=mappo
      network_policy=true
      ;;
    mappo_vision)
      selection_mode=mappo
      vision_enabled=true
      network_policy=true
      ;;
    mappo_hybrid)
      selection_mode=mappo_hybrid
      vision_enabled=true
      network_policy=true
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 2
      ;;
  esac

  local metrics_path="${output_dir}/repeat_${repeat_id}_${method}.csv"
  local launch_log="${output_dir}/repeat_${repeat_id}_${method}.log"
  echo "[thesis_eval] repeat ${repeat_id}, seed ${experiment_seed}, method ${method}"

  set +e
  timeout --signal=INT --kill-after=20s "${episode_seconds}s" \
    roslaunch turtlebot_ddpg gazebo_research_experiment.launch \
      gui:=false verbose:=false \
      selection_mode:="${selection_mode}" \
      mapping_mode:=laser_scan \
      vision_enabled:="${vision_enabled}" \
      vision_scoring_mode:=projected_obstacle \
      start_smoke_policy:=false \
      start_network_policy:="${network_policy}" \
      start_mappo_training:=false \
      mappo_checkpoint:="${checkpoint}" \
      network_deterministic:=true \
      experiment_seed:="${experiment_seed}" \
      metrics_path:="${metrics_path}" >"${launch_log}" 2>&1
  local status=$?
  set -e
  if [[ ${status} -ne 0 && ${status} -ne 124 && ${status} -ne 130 ]]; then
    echo "[thesis_eval] ${method} failed with status ${status}" >&2
    tail -50 "${launch_log}" >&2
    exit "${status}"
  fi
  if [[ ! -s "${metrics_path}" ]]; then
    echo "[thesis_eval] ${method} produced no metrics" >&2
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
if method in ('mappo_vision', 'mappo_hybrid'):
    if int(last.get('vision_ready_robots', 0) or 0) != 3:
        raise SystemExit('%s did not receive all three camera streams' % method)
PY
  sleep 3
}

for ((repeat = 1; repeat <= repeats; repeat++)); do
  repeat_id="$(printf '%02d' "${repeat}")"
  experiment_seed=$((seed_base + repeat - 1))
  offset=$(((repeat - 1) % ${#methods[@]}))
  for ((index = 0; index < ${#methods[@]}; index++)); do
    method_index=$(((offset + index) % ${#methods[@]}))
    run_method "${repeat_id}" "${methods[method_index]}" "${experiment_seed}"
  done
done

summarizer="${project_root}/turtlebot_ddpg/scripts/summarize_evaluation.py"
python3 "${summarizer}" "${output_dir}" \
  --positive-method mappo_vision \
  --negative-method mappo_no_vision \
  --paired-output visual_ablation.csv
python3 "${summarizer}" "${output_dir}" \
  --positive-method mappo_vision \
  --negative-method execution_aware \
  --paired-output mappo_vs_execution_aware.csv >/dev/null
python3 "${summarizer}" "${output_dir}" \
  --positive-method execution_aware \
  --negative-method geometric \
  --paired-output execution_aware_vs_geometric.csv >/dev/null

echo "[thesis_eval] complete: ${output_dir}"
