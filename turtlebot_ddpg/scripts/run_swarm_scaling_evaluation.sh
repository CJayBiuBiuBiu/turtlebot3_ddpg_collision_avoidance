#!/usr/bin/env bash
set -eo pipefail

# Fixed-horizon Gazebo scaling study for 3, 5, and 7 homogeneous robots.
repeats="${1:-5}"
episode_seconds="${2:-180}"
seed_base="${3:-15000}"
team_sizes=(3 5 7)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
set -u
checkpoint="${MAPPO_CHECKPOINT:-${project_root}/outputs/checkpoints/mappo_latest.pt}"
if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  exit 2
fi
run_id="$(date +%Y%m%d_%H%M%S)"
output_dir="${OUTPUT_DIR:-${project_root}/outputs/evaluation/scaling_${run_id}}"
mkdir -p "${output_dir}"

render_prefix=()
if ! timeout 2s xdpyinfo >/dev/null 2>&1; then
  if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "No display is available and xvfb-run is not installed" >&2
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

for robot_count in "${team_sizes[@]}"; do
  case "${robot_count}" in
    3) robot_names='[robot1, robot2, robot3]' ;;
    5) robot_names='[robot1, robot2, robot3, robot4, robot5]' ;;
    7) robot_names='[robot1, robot2, robot3, robot4, robot5, robot6, robot7]' ;;
  esac
  for ((repeat = 1; repeat <= repeats; repeat++)); do
    seed=$((seed_base + robot_count * 100 + repeat))
    stem="robots_${robot_count}_repeat_$(printf '%02d' "${repeat}")"
    metrics_path="${output_dir}/${stem}.csv"
    launch_log="${output_dir}/${stem}.log"
    echo "[scaling] ${robot_count} robots, repeat ${repeat}/${repeats}, seed ${seed}"
    set +e
    timeout --signal=INT --kill-after=20s "$((episode_seconds * 3 + 60))s" \
      "${render_prefix[@]}" roslaunch turtlebot_ddpg gazebo_research_experiment.launch \
        gui:=false verbose:=false world_profile:=research \
        world_name:="${package_root}/worlds/voronoi_exploration_arena.world" \
        robot_count:="${robot_count}" robot_names:="${robot_names}" \
        selection_mode:=mappo_hybrid vision_enabled:=true \
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
    python3 - "${metrics_path}" "${robot_count}" <<'PY'
import csv
import sys

path, expected = sys.argv[1], int(sys.argv[2])
with open(path, newline='') as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit('metrics file contains no rows')
last = rows[-1]
if int(last.get('robot_count', 0) or 0) != expected:
    raise SystemExit('coordinator robot count does not match launch request')
if int(last.get('scan_ready_robots', 0) or 0) != expected:
    raise SystemExit('not all robot laser streams became ready')
if int(last.get('vision_ready_robots', 0) or 0) != expected:
    raise SystemExit('not all robot camera streams became ready')
PY
  done
done

python3 "${package_root}/scripts/summarize_scaling_study.py" \
  "${output_dir}" --output "${output_dir}/scaling_summary.csv" \
  --figure "${output_dir}/scaling_summary.png"
echo "[scaling] results: ${output_dir}"
