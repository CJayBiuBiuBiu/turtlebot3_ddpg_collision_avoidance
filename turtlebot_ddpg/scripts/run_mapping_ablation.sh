#!/usr/bin/env bash
set -eo pipefail

repeats="${1:-2}"
episode_seconds="${2:-90}"
if ! [[ "${repeats}" =~ ^[1-9][0-9]*$ &&
        "${episode_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 [repeats] [episode_seconds]" >&2
  exit 2
fi

source /opt/ros/noetic/setup.bash
source /usr/share/gazebo/setup.sh
source /home/ros/catkin_ws/devel/setup.bash
source /home/ros/ddpg_env/bin/activate
set -u

project_root="/home/ros/catkin_ws/src/turtlebot3_ddpg_collision_avoidance"
checkpoint="${project_root}/outputs/checkpoints/mappo_latest.pt"
run_id="$(date +%Y%m%d_%H%M%S)"
output_dir="${project_root}/outputs/evaluation/mapping_ablation_${run_id}"
mkdir -p "${output_dir}"

run_method() {
  local repeat_id="$1"
  local method="$2"
  local mapping_mode=ground_truth_radius
  if [[ "${method}" == "mappo_laser_scan" ]]; then
    mapping_mode=laser_scan
  fi
  local metrics_path="${output_dir}/repeat_${repeat_id}_${method}.csv"
  local launch_log="${output_dir}/repeat_${repeat_id}_${method}.log"

  echo "[mapping_ablation] repeat ${repeat_id}, method ${method}"
  set +e
  timeout --signal=INT --kill-after=20s "${episode_seconds}s" \
    roslaunch turtlebot_ddpg gazebo_research_experiment.launch \
      gui:=false verbose:=false selection_mode:=mappo \
      mapping_mode:="${mapping_mode}" \
      vision_enabled:=true vision_scoring_mode:=projected_obstacle \
      start_smoke_policy:=false start_network_policy:=true \
      start_mappo_training:=false \
      mappo_checkpoint:="${checkpoint}" \
      network_deterministic:=true \
      metrics_path:="${metrics_path}" >"${launch_log}" 2>&1
  local status=$?
  set -e
  if [[ ${status} -ne 0 && ${status} -ne 124 && ${status} -ne 130 ]]; then
    echo "[mapping_ablation] ${method} failed with status ${status}" >&2
    tail -40 "${launch_log}" >&2
    exit "${status}"
  fi
  sleep 3
}

for ((repeat = 1; repeat <= repeats; repeat++)); do
  repeat_id="$(printf '%02d' "${repeat}")"
  if (( repeat % 2 == 1 )); then
    run_method "${repeat_id}" mappo_laser_scan
    run_method "${repeat_id}" mappo_ground_truth
  else
    run_method "${repeat_id}" mappo_ground_truth
    run_method "${repeat_id}" mappo_laser_scan
  fi
done

python3 "${project_root}/turtlebot_ddpg/scripts/summarize_evaluation.py" \
  "${output_dir}" \
  --positive-method mappo_laser_scan \
  --negative-method mappo_ground_truth
