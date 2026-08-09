#!/usr/bin/env bash
set -eo pipefail

selection_mode="${1:-execution_aware}"
case "${selection_mode}" in
  geometric|vision|execution_aware|mappo) ;;
  *)
    echo "Usage: $0 [geometric|vision|execution_aware|mappo]" >&2
    exit 2
    ;;
esac

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
set -u

start_smoke_policy=false
if [[ "${selection_mode}" == "mappo" ]]; then
  start_smoke_policy=true
fi

exec roslaunch turtlebot_ddpg gazebo_research_experiment.launch \
  selection_mode:="${selection_mode}" \
  start_smoke_policy:="${start_smoke_policy}"
