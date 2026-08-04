#!/usr/bin/env bash
set -eo pipefail

repeats="${1:-5}"
episode_seconds="${2:-900}"
coverage_threshold="${3:-97}"
method_filter="${4:-all}"
seed_base="${5:-2000}"
hybrid_stall_seconds="${HYBRID_FALLBACK_STALL_SECONDS:-60.0}"
hybrid_active_seconds="${HYBRID_FALLBACK_ACTIVE_SECONDS:-30.0}"
hybrid_progress_epsilon="${HYBRID_FALLBACK_PROGRESS_EPSILON:-0.2}"
hybrid_minimum_coverage="${HYBRID_FALLBACK_MINIMUM_COVERAGE:-20.0}"
adaptive_start_coverage="${ADAPTIVE_EXECUTION_START_COVERAGE:-40.0}"
adaptive_full_coverage="${ADAPTIVE_EXECUTION_FULL_COVERAGE:-90.0}"
tail_completion_coverage="${TAIL_COMPLETION_COVERAGE:-90.0}"
vision_feature_scale="${VISION_FEATURE_SCALE:-1.0}"
execution_path_cost_mode="${EXECUTION_PATH_COST_MODE:-euclidean}"
if ! [[ "${repeats}" =~ ^[1-9][0-9]*$ &&
        "${episode_seconds}" =~ ^[1-9][0-9]*$ &&
        "${coverage_threshold}" =~ ^[0-9]+([.][0-9]+)?$ &&
        "${seed_base}" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [repeats] [episode_seconds] [coverage_threshold] [method|all] [seed_base]" >&2
  exit 2
fi
wall_timeout_seconds=$((episode_seconds * 2 + 60))

source /opt/ros/noetic/setup.bash
source /usr/share/gazebo/setup.sh
source /home/ros/catkin_ws/devel/setup.bash
source /home/ros/ddpg_env/bin/activate
set -u

project_root="/home/ros/catkin_ws/src/turtlebot3_ddpg_collision_avoidance"
package_root="${project_root}/turtlebot_ddpg"
checkpoint="${MAPPO_CHECKPOINT:-${project_root}/outputs/checkpoints/mappo_latest.pt}"
world_profile="${EVALUATION_WORLD_PROFILE:-hu_aligned}"
case "${world_profile}" in
  research) default_world_file="${package_root}/worlds/voronoi_exploration_arena.world" ;;
  hu_aligned) default_world_file="${package_root}/worlds/hu_aligned_arena.world" ;;
  hu_layout_b) default_world_file="${package_root}/worlds/hu_layout_b_arena.world" ;;
  hu_layout_c) default_world_file="${package_root}/worlds/hu_layout_c_arena.world" ;;
  *) echo "Unsupported evaluation world profile: ${world_profile}" >&2; exit 2 ;;
esac
world_file="${EVALUATION_WORLD_FILE:-${default_world_file}}"
run_id="$(date +%Y%m%d_%H%M%S)"
output_dir="${OUTPUT_DIR:-${project_root}/outputs/evaluation/${world_profile}_${run_id}}"
methods=(geometric execution_aware mappo_vision)
if [[ "${method_filter}" != "all" ]]; then
  case "${method_filter}" in
    hu_pair) methods=(geometric mappo_hybrid) ;;
    hu_final_pair) methods=(geometric mappo_tail) ;;
    core_comparison) methods=(geometric execution_aware mappo_vision) ;;
    execution_pair) methods=(geometric execution_aware) ;;
    core_vision_ablation) methods=(geometric execution_aware mappo_no_vision mappo_vision) ;;
    vision_pair) methods=(mappo_no_vision mappo_vision) ;;
    improvement_pair) methods=(mappo_hybrid mappo_tail) ;;
    stack_ablation) methods=(mappo_vision mappo_hybrid mappo_tail) ;;
    deployment_pair) methods=(mappo_tail mappo_tail_decentralized) ;;
    improvement_ablation) methods=(geometric execution_aware adaptive_execution_aware mappo_hybrid mappo_tail mappo_adaptive_tail) ;;
    hu_ablation) methods=(geometric execution_aware mappo_no_vision mappo_vision mappo_hybrid) ;;
    geometric|execution_aware|adaptive_execution_aware|mappo_no_vision|mappo_vision|mappo_hybrid|mappo_tail|mappo_tail_decentralized|mappo_adaptive_tail|mappo_adaptive_tail_decentralized) methods=("${method_filter}") ;;
    *) echo "Unknown method filter: ${method_filter}" >&2; exit 2 ;;
  esac
fi
mkdir -p "${output_dir}"

render_backend="native_display"
render_prefix=()
if ! timeout 2s xdpyinfo >/dev/null 2>&1; then
  if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "No working display is available and xvfb-run is not installed" >&2
    exit 2
  fi
  render_backend="xvfb"
  render_prefix=(xvfb-run -a -s "-screen 0 1280x720x24")
fi

if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  exit 2
fi
checkpoint_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
checkpoint_update="$(python3 - "${checkpoint}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(payload.get('metadata', {}).get('updates', 'unknown'))
PY
)"
world_file_sha256="$(sha256sum "${world_file}" | awk '{print $1}')"
launch_file_sha256="$(sha256sum "${package_root}/launch/gazebo_research_experiment.launch" | awk '{print $1}')"
config_file_sha256="$(sha256sum "${package_root}/config/gazebo_research_experiment.yaml" | awk '{print $1}')"
source_snapshot_sha256="$(
  find "${package_root}/scripts" "${package_root}/launch" \
       "${package_root}/config" "${package_root}/worlds" -type f \
       ! -path '*/__pycache__/*' -print0 |
    sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
)"
git_revision="$(git -C "${project_root}" rev-parse HEAD 2>/dev/null || echo unavailable)"
git_dirty_files="$(git -C "${project_root}" status --porcelain 2>/dev/null | wc -l)"
ros_distribution="$(rosversion -d 2>/dev/null || echo unknown)"
gazebo_version="$( (gazebo --version 2>/dev/null || true) | tr '\n,' ' ;')"
python_version="$(python3 --version 2>&1 | tr ',' ';')"
torch_version="$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null | tr ',' ';')"
actor_deployment="single_process"
if [[ "${method_filter}" == "deployment_pair" ]]; then
  actor_deployment="single_process_vs_per_robot_processes"
fi
printf '%s\n' \
  'field,value' \
  "protocol,${world_profile}_frozen_ablation" \
  "methods,$(IFS=';'; echo "${methods[*]}")" \
  "repeats,${repeats}" \
  "episode_seconds,${episode_seconds}" \
  "wall_timeout_safety_budget,${wall_timeout_seconds}" \
  "coverage_threshold,${coverage_threshold}" \
  "seed_base,${seed_base}" \
  'gazebo_seed,matched to experiment_seed for each repeat' \
  "checkpoint,${checkpoint}" \
  "checkpoint_sha256,${checkpoint_sha256}" \
  "checkpoint_update,${checkpoint_update}" \
  'checkpoint_selection_rule,frozen endpoint supplied before evaluation; no test-seed ranking' \
  "git_revision,${git_revision}" \
  "git_dirty_files,${git_dirty_files}" \
  "source_snapshot_sha256,${source_snapshot_sha256}" \
  "world_profile,${world_profile}" \
  "world_file,${world_file}" \
  "world_file_sha256,${world_file_sha256}" \
  "launch_file_sha256,${launch_file_sha256}" \
  "config_file_sha256,${config_file_sha256}" \
  "ros_distribution,${ros_distribution}" \
  "gazebo_version,${gazebo_version}" \
  "render_backend,${render_backend}" \
  "python_version,${python_version}" \
  "torch_version,${torch_version}" \
  'network_deterministic,true' \
  'coverage_definition,cumulative sensed truth-free cells divided by fixed explorable-cell mask' \
  'near_collision_definition,laser clearance event at <=0.18 m rearmed at >=0.25 m' \
  'primary_early_metric,mean_coverage_0_90_pct for fixed 90-second ablation; mean_coverage_0_300_pct for completion benchmark' \
  'primary_completion_metric,time_to_97_s' \
  'secondary_metrics,time_to_80_s;time_to_90_s;time_to_95_s;path_length_total_m;stuck_events;unique_stuck_regions;near_collision_events' \
  'parameter_tuning,false' \
  "paired_actor_deployment,${actor_deployment}" \
  "adaptive_execution_start_coverage,${adaptive_start_coverage}" \
  "adaptive_execution_full_coverage,${adaptive_full_coverage}" \
  "tail_completion_coverage,${tail_completion_coverage}" \
  "vision_feature_scale,${vision_feature_scale}" \
  "execution_path_cost_mode,${execution_path_cost_mode}" \
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
  pkill -INT -f '/turtlebot_ddpg/scripts/exploration_coordinator_node.py' >/dev/null 2>&1
  pkill -INT -f '/turtlebot_ddpg/scripts/multi_robot_explorer_node.py' >/dev/null 2>&1
  pkill -INT -f '/turtlebot_ddpg/scripts/mappo_policy_node.py' >/dev/null 2>&1
  sleep 3
  pkill -KILL -x gzserver >/dev/null 2>&1
  pkill -KILL -x gzclient >/dev/null 2>&1
  pkill -KILL -x rosmaster >/dev/null 2>&1
  set -e
}

validate_run() {
  local metrics_path="$1"
  local launch_log="$2"
  local method="$3"
  if [[ ! -s "${metrics_path}" ]]; then
    echo "[hu_eval] ${method} produced no metrics" >&2
    return 1
  fi
  if grep -Eq 'Traceback|Segmentation fault|REQUIRED process \[robot[123]_explorer|\[gazebo-[0-9]+\] process has died' "${launch_log}"; then
    echo "[hu_eval] ${method} log contains a node failure" >&2
    grep -En 'Traceback|Segmentation fault|REQUIRED process \[robot[123]_explorer|\[gazebo-[0-9]+\] process has died' \
      "${launch_log}" | tail -20 >&2
    return 1
  fi
  if ! python3 - "${metrics_path}" "${method}" "${episode_seconds}" "${world_profile}" <<'PY'
import csv
import sys

path, method, episode_seconds, expected_world_profile = sys.argv[1:]
episode_seconds = float(episode_seconds)
with open(path, newline='') as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit('%s has only a CSV header' % method)
last = rows[-1]
elapsed = float(last.get('elapsed_s', 0.0) or 0.0)
mission_done = bool(int(last.get('mission_done', 0) or 0))
has_horizon_field = 'evaluation_horizon_reached' in last
horizon_reached = bool(int(last.get('evaluation_horizon_reached', 0) or 0))
# The external timeout includes Gazebo/model startup while elapsed_s starts at
# coordinator construction. Allow 30 seconds for startup and shutdown, but do
# not accept a short CSV fragment left by an interrupted experiment.
minimum_budget_elapsed = max(5.0, episode_seconds - 30.0)
if has_horizon_field and not mission_done and not horizon_reached:
    raise SystemExit(
        '%s is incomplete: neither mission completion nor simulation horizon '
        'was recorded' % method)
if not has_horizon_field and not mission_done and elapsed < minimum_budget_elapsed:
    raise SystemExit(
        '%s is incomplete: elapsed %.1fs, expected mission completion or >=%.1fs' %
        (method, elapsed, minimum_budget_elapsed))
if last.get('world_profile') != expected_world_profile:
    raise SystemExit('%s used the wrong world profile' % method)
if int(last.get('scan_ready_robots', 0) or 0) != 3:
    raise SystemExit('%s did not receive all three laser streams' % method)
explored = int(last.get('coverage_explored_cells', 0) or 0)
explorable = int(last.get('coverage_explorable_cells', 0) or 0)
reported_coverage = float(last.get('coverage_pct', 0.0) or 0.0)
if explorable <= 0:
    raise SystemExit('%s has no auditable coverage denominator' % method)
recomputed_coverage = 100.0 * explored / explorable
if abs(recomputed_coverage - reported_coverage) > 0.02:
    raise SystemExit(
        '%s coverage mismatch: reported %.4f, recomputed %.4f' %
        (method, reported_coverage, recomputed_coverage))
if len(last.get('coverage_mask_sha256', '')) != 64:
    raise SystemExit('%s has no coverage-mask fingerprint' % method)
if method in ('mappo_vision', 'mappo_hybrid', 'mappo_tail', 'mappo_tail_decentralized', 'mappo_adaptive_tail', 'mappo_adaptive_tail_decentralized') and int(last.get('vision_ready_robots', 0) or 0) != 3:
    raise SystemExit('%s did not receive all three camera streams' % method)
if method.startswith('mappo_') and int(last.get('policy_actions_received', 0) or 0) <= 0:
    raise SystemExit('%s did not receive any actor actions' % method)
PY
  then
    return 1
  fi
  if [[ "${method}" == *"_decentralized" ]]; then
    for robot in robot1 robot2 robot3; do
      if ! grep -Fq "actors=${robot} " "${launch_log}"; then
        echo "[hu_eval] ${method} did not start the ${robot} actor process" >&2
        return 1
      fi
    done
  fi
}

trap cleanup_ros EXIT
trap 'exit 130' INT TERM
cleanup_ros

run_method() {
  local repeat_id="$1"
  local method="$2"
  local experiment_seed="$3"
  local selection_mode="${method}"
  local vision_enabled=false
  local network_policy=false
  local decentralized_actor_nodes=false
  local tail_completion_enabled=false

  if [[ "${method}" == "mappo_no_vision" ||
        "${method}" == "mappo_vision" || "${method}" == "mappo_hybrid" ||
        "${method}" == "mappo_tail" ||
        "${method}" == "mappo_tail_decentralized" ||
        "${method}" == "mappo_adaptive_tail" ||
        "${method}" == "mappo_adaptive_tail_decentralized" ]]; then
    selection_mode=mappo
    network_policy=true
  fi
  if [[ "${method}" == "mappo_vision" || "${method}" == "mappo_hybrid" ||
        "${method}" == "mappo_tail" ||
        "${method}" == "mappo_tail_decentralized" ||
        "${method}" == "mappo_adaptive_tail" ||
        "${method}" == "mappo_adaptive_tail_decentralized" ]]; then
    vision_enabled=true
  fi
  if [[ "${method}" == "mappo_hybrid" ]]; then
    selection_mode=mappo_hybrid
  fi
  if [[ "${method}" == "mappo_tail" ||
        "${method}" == "mappo_tail_decentralized" ]]; then
    selection_mode=mappo_hybrid
    tail_completion_enabled=true
  fi
  if [[ "${method}" == "mappo_tail_decentralized" ]]; then
    decentralized_actor_nodes=true
  fi
  if [[ "${method}" == "mappo_adaptive_tail" ||
        "${method}" == "mappo_adaptive_tail_decentralized" ]]; then
    selection_mode=mappo_adaptive_hybrid
    tail_completion_enabled=true
  fi
  if [[ "${method}" == "mappo_adaptive_tail_decentralized" ]]; then
    decentralized_actor_nodes=true
  fi

  local metrics_path="${output_dir}/repeat_${repeat_id}_${method}.csv"
  local launch_log="${output_dir}/repeat_${repeat_id}_${method}.log"
  if [[ -s "${metrics_path}" && -s "${launch_log}" ]] &&
      validate_run "${metrics_path}" "${launch_log}" "${method}"; then
    echo "[hu_eval] repeat ${repeat_id}, seed ${experiment_seed}, method ${method}: already valid, skipping"
    return
  fi
  if [[ -e "${metrics_path}" || -e "${launch_log}" ]]; then
    local invalid_stamp
    invalid_stamp="$(date +%Y%m%d_%H%M%S)"
    [[ ! -e "${metrics_path}" ]] || mv "${metrics_path}" \
      "${metrics_path}.invalid_${invalid_stamp}"
    [[ ! -e "${launch_log}" ]] || mv "${launch_log}" \
      "${launch_log}.invalid_${invalid_stamp}"
  fi
  echo "[hu_eval] repeat ${repeat_id}, seed ${experiment_seed}, method ${method}"

  set +e
  local fatal_startup=0
  timeout --signal=INT --kill-after=20s "${wall_timeout_seconds}s" \
    "${render_prefix[@]}" roslaunch turtlebot_ddpg gazebo_research_experiment.launch \
      gui:=false verbose:=false \
      world_profile:="${world_profile}" world_name:="${world_file}" \
      robot1_x:=-0.8 robot1_y:=4.0 robot1_yaw:=-1.5708 \
      robot2_x:=0.0 robot2_y:=4.0 robot2_yaw:=-1.5708 \
      robot3_x:=0.8 robot3_y:=4.0 robot3_yaw:=-1.5708 \
      selection_mode:="${selection_mode}" \
      mapping_mode:=laser_scan \
      execution_path_cost_mode:="${execution_path_cost_mode}" \
      vision_enabled:="${vision_enabled}" \
      vision_scoring_mode:=projected_obstacle \
      vision_feature_scale:="${vision_feature_scale}" \
      start_smoke_policy:=false \
      start_network_policy:="${network_policy}" \
      decentralized_actor_nodes:="${decentralized_actor_nodes}" \
      start_mappo_training:=false \
      mappo_checkpoint:="${checkpoint}" \
      network_deterministic:=true \
      experiment_seed:="${experiment_seed}" \
      gazebo_seed:="${experiment_seed}" \
      coverage_done_threshold:="${coverage_threshold}" \
      max_simulation_seconds:="${episode_seconds}" \
      idle_done_hold:=30 \
      complete_on_coverage:=true \
      hybrid_fallback_stall_seconds:="${hybrid_stall_seconds}" \
      hybrid_fallback_active_seconds:="${hybrid_active_seconds}" \
      hybrid_fallback_progress_epsilon:="${hybrid_progress_epsilon}" \
      hybrid_fallback_minimum_coverage:="${hybrid_minimum_coverage}" \
      adaptive_execution_start_coverage:="${adaptive_start_coverage}" \
      adaptive_execution_full_coverage:="${adaptive_full_coverage}" \
      tail_completion_enabled:="${tail_completion_enabled}" \
      tail_completion_coverage:="${tail_completion_coverage}" \
      shutdown_on_done:=true \
      metrics_path:="${metrics_path}" >"${launch_log}" 2>&1 &
  local launch_pid=$!
  while kill -0 "${launch_pid}" >/dev/null 2>&1; do
    sleep 5
    if grep -Eq 'Traceback|Segmentation fault|REQUIRED process \[robot[123]_explorer|\[gazebo-[0-9]+\] process has died' "${launch_log}"; then
      fatal_startup=1
      kill -INT "${launch_pid}" >/dev/null 2>&1 || true
      break
    fi
  done
  wait "${launch_pid}"
  local status=$?
  set -e
  cleanup_ros
  if [[ ${fatal_startup} -eq 1 ]]; then
    echo "[hu_eval] ${method} aborted after a fatal simulator/node event" >&2
    grep -En 'Traceback|Segmentation fault|REQUIRED process \[robot[123]_explorer|\[gazebo-[0-9]+\] process has died' \
      "${launch_log}" | tail -20 >&2
    return 1
  fi
  if [[ ${status} -ne 0 && ${status} -ne 124 && ${status} -ne 130 ]]; then
    echo "[hu_eval] ${method} failed with status ${status}" >&2
    tail -50 "${launch_log}" >&2
    exit "${status}"
  fi
  validate_run "${metrics_path}" "${launch_log}" "${method}"
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

summarizer="${package_root}/scripts/summarize_evaluation.py"
if [[ "${method_filter}" == "hu_pair" ]]; then
  python3 "${summarizer}" "${output_dir}" \
    --positive-method mappo_hybrid \
    --negative-method geometric \
    --paired-output full_method_vs_hu_reproduction.csv
elif [[ "${method_filter}" == "hu_final_pair" ]]; then
  python3 "${summarizer}" "${output_dir}" \
    --positive-method mappo_tail \
    --negative-method geometric \
    --paired-output final_method_vs_hu_reproduction.csv
elif [[ "${method_filter}" == "improvement_pair" ||
        "${method_filter}" == "improvement_ablation" ||
        "${method_filter}" == "stack_ablation" ]]; then
  python3 "${summarizer}" "${output_dir}" \
    --positive-method mappo_tail \
    --negative-method mappo_hybrid \
    --paired-output tail_completion_vs_previous_full_method.csv
  if [[ "${method_filter}" == "stack_ablation" ]]; then
    python3 "${summarizer}" "${output_dir}" \
      --positive-method mappo_hybrid \
      --negative-method mappo_vision \
      --paired-output fallback_vs_plain_mappo.csv >/dev/null
    python3 "${summarizer}" "${output_dir}" \
      --positive-method mappo_tail \
      --negative-method mappo_vision \
      --paired-output complete_stack_vs_plain_mappo.csv >/dev/null
  fi
  if [[ "${method_filter}" == "improvement_ablation" ]]; then
    python3 "${summarizer}" "${output_dir}" \
      --positive-method adaptive_execution_aware \
      --negative-method execution_aware \
      --paired-output adaptive_vs_fixed_execution.csv >/dev/null
  fi
elif [[ "${method_filter}" == "deployment_pair" ]]; then
  python3 "${summarizer}" "${output_dir}" \
    --positive-method mappo_tail_decentralized \
    --negative-method mappo_tail \
    --paired-output decentralized_vs_single_actor.csv
elif [[ "${method_filter}" == "vision_pair" ]]; then
  python3 "${summarizer}" "${output_dir}" \
    --positive-method mappo_vision \
    --negative-method mappo_no_vision \
    --paired-output vision_feature_ablation.csv
elif [[ "${method_filter}" == "execution_pair" ]]; then
  python3 "${summarizer}" "${output_dir}" \
    --positive-method execution_aware \
    --negative-method geometric \
    --paired-output execution_aware_vs_geometric.csv
elif [[ "${method_filter}" == "core_vision_ablation" ]]; then
  python3 "${summarizer}" "${output_dir}" \
    --positive-method mappo_vision \
    --negative-method geometric \
    --paired-output mappo_vision_vs_geometric.csv
  python3 "${summarizer}" "${output_dir}" \
    --positive-method execution_aware \
    --negative-method geometric \
    --paired-output execution_aware_vs_geometric.csv >/dev/null
  python3 "${summarizer}" "${output_dir}" \
    --positive-method mappo_no_vision \
    --negative-method execution_aware \
    --paired-output mappo_no_vision_vs_execution_aware.csv >/dev/null
  python3 "${summarizer}" "${output_dir}" \
    --positive-method mappo_vision \
    --negative-method mappo_no_vision \
    --paired-output vision_feature_ablation.csv >/dev/null
elif [[ "${method_filter}" == "hu_ablation" ]]; then
  python3 "${summarizer}" "${output_dir}" \
    --positive-method mappo_hybrid \
    --negative-method geometric \
    --paired-output full_method_vs_hu_reproduction.csv
  python3 "${summarizer}" "${output_dir}" \
    --positive-method execution_aware \
    --negative-method geometric \
    --paired-output execution_aware_vs_geometric.csv >/dev/null
  python3 "${summarizer}" "${output_dir}" \
    --positive-method mappo_no_vision \
    --negative-method execution_aware \
    --paired-output mappo_no_vision_vs_execution_aware.csv >/dev/null
  python3 "${summarizer}" "${output_dir}" \
    --positive-method mappo_vision \
    --negative-method mappo_no_vision \
    --paired-output vision_feature_ablation.csv >/dev/null
  python3 "${summarizer}" "${output_dir}" \
    --positive-method mappo_hybrid \
    --negative-method mappo_vision \
    --paired-output fallback_ablation.csv >/dev/null
else
  python3 "${summarizer}" "${output_dir}" \
    --positive-method mappo_vision \
    --negative-method execution_aware \
    --paired-output mappo_vs_execution_aware.csv
  python3 "${summarizer}" "${output_dir}" \
    --positive-method execution_aware \
    --negative-method geometric \
    --paired-output execution_aware_vs_geometric.csv >/dev/null
fi

echo "[hu_eval] complete: ${output_dir}"
