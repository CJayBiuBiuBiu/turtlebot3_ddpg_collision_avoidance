#!/usr/bin/env bash
set -eo pipefail

episodes="${1:-20}"
episode_seconds="${2:-120}"
rollout_length="${3:-32}"
decision_interval="${4:-2.5}"
world_profile="${5:-research}"
training_seed="${6:-0}"
checkpoint_override="${7:-}"
vision_enabled="${8:-true}"

if ! [[ "${episodes}" =~ ^[1-9][0-9]*$ &&
        "${episode_seconds}" =~ ^[1-9][0-9]*$ &&
        "${rollout_length}" =~ ^[1-9][0-9]*$ &&
        "${decision_interval}" =~ ^[0-9]+([.][0-9]+)?$ &&
        "${training_seed}" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [episodes] [episode_seconds] [rollout_length] [decision_interval] [research|hu_aligned] [training_seed] [checkpoint] [true|false vision]" >&2
  exit 2
fi
if [[ "${vision_enabled}" != "true" && "${vision_enabled}" != "false" ]]; then
  echo "vision_enabled must be true or false" >&2
  exit 2
fi
if [[ "${world_profile}" != "research" && "${world_profile}" != "hu_aligned" ]]; then
  echo "Unknown world profile: ${world_profile}" >&2
  exit 2
fi

source /opt/ros/noetic/setup.bash
source /usr/share/gazebo/setup.sh
source /home/ros/catkin_ws/devel/setup.bash
source /home/ros/ddpg_env/bin/activate
set -u

project_root="/home/ros/catkin_ws/src/turtlebot3_ddpg_collision_avoidance"
run_id="$(date +%Y%m%d_%H%M%S)"
output_dir="${project_root}/outputs/training/${world_profile}_${run_id}"
checkpoint="${project_root}/outputs/checkpoints/mappo_latest.pt"
if [[ -n "${checkpoint_override}" ]]; then
  checkpoint="${checkpoint_override}"
fi
if [[ "${world_profile}" == "research" ]]; then
  training_log="${project_root}/outputs/mappo_training.csv"
  diagnostics_log="${project_root}/outputs/mappo_diagnostics.csv"
  world_args=()
else
  training_log="${project_root}/outputs/mappo_training_hu_aligned.csv"
  diagnostics_log="${project_root}/outputs/mappo_diagnostics_hu_aligned.csv"
  world_args=(
    world_profile:=hu_aligned
    world_name:="${project_root}/turtlebot_ddpg/worlds/hu_aligned_arena.world"
    robot1_x:=-0.8 robot1_y:=4.0 robot1_yaw:=-1.5708
    robot2_x:=0.0 robot2_y:=4.0 robot2_yaw:=-1.5708
    robot3_x:=0.8 robot3_y:=4.0 robot3_yaw:=-1.5708
    mapping_mode:=laser_scan
    complete_on_coverage:=true coverage_done_threshold:=97
    shutdown_on_done:=true
  )
fi
if [[ -n "${checkpoint_override}" ]]; then
  checkpoint_label="$(basename "${checkpoint}" .pt)"
  training_log="${output_dir}/${checkpoint_label}_training.csv"
  diagnostics_log="${output_dir}/${checkpoint_label}_diagnostics.csv"
fi
mkdir -p "${output_dir}"

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

for ((episode = 1; episode <= episodes; episode++)); do
  episode_id="$(printf '%03d' "${episode}")"
  experiment_seed=$((training_seed + episode - 1))
  metrics_path="${output_dir}/episode_${episode_id}.csv"
  launch_log="${output_dir}/episode_${episode_id}.log"
  echo "[mappo_episodes] episode ${episode}/${episodes} (${episode_seconds}s, training seed ${training_seed}, experiment seed ${experiment_seed}, vision ${vision_enabled})"

  set +e
  timeout --signal=INT --kill-after=20s "${episode_seconds}s" \
    roslaunch turtlebot_ddpg gazebo_research_experiment.launch \
      gui:=false \
      verbose:=false \
      selection_mode:=mappo \
      vision_enabled:="${vision_enabled}" \
      start_smoke_policy:=false \
      start_network_policy:=false \
      start_mappo_training:=true \
      training_rollout_length:="${rollout_length}" \
      training_decision_interval:="${decision_interval}" \
      training_seed:="${training_seed}" \
      experiment_seed:="${experiment_seed}" \
      mappo_training_checkpoint:="${checkpoint}" \
      mappo_training_log:="${training_log}" \
      mappo_diagnostics_log:="${diagnostics_log}" \
      "${world_args[@]}" \
      metrics_path:="${metrics_path}" >"${launch_log}" 2>&1
  status=$?
  set -e
  cleanup_ros

  if [[ ${status} -ne 0 && ${status} -ne 124 && ${status} -ne 130 ]]; then
    echo "[mappo_episodes] episode ${episode} failed with status ${status}" >&2
    tail -40 "${launch_log}" >&2
    exit "${status}"
  fi
  if [[ ! -s "${metrics_path}" ]]; then
    echo "[mappo_episodes] episode ${episode} produced no metrics" >&2
    exit 1
  fi
  if grep -Eq 'Traceback|REQUIRED process \[robot[123]_explorer|\[gazebo-[0-9]+\] process has died' "${launch_log}"; then
    echo "[mappo_episodes] episode ${episode} contains a node failure" >&2
    exit 1
  fi
  grep -E '\[mappo_train\] (resumed|update=)' "${launch_log}" | tail -5 || true
done

echo "[mappo_episodes] complete"
echo "  checkpoint: ${checkpoint}"
echo "  training log: ${training_log}"
echo "  diagnostics log: ${diagnostics_log}"
echo "  episode metrics: ${output_dir}/episode_*.csv"
