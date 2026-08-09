#!/usr/bin/env bash

# Resolve the research workspace from this file instead of a user-specific path.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="${TURTLEBOT_DDPG_PACKAGE_ROOT:-$(cd "${script_dir}/.." && pwd)}"
project_root="${TURTLEBOT_DDPG_PROJECT_ROOT:-$(cd "${package_root}/.." && pwd)}"
catkin_ws="${TURTLEBOT_DDPG_CATKIN_WS:-$(cd "${project_root}/../.." && pwd)}"
python_env="${TURTLEBOT_DDPG_VENV:-${HOME}/ddpg_env}"

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  source /opt/ros/noetic/setup.bash
fi
if [[ -f /usr/share/gazebo/setup.sh ]]; then
  source /usr/share/gazebo/setup.sh
fi
if [[ -f "${catkin_ws}/devel/setup.bash" ]]; then
  source "${catkin_ws}/devel/setup.bash"
else
  echo "Catkin workspace is not built: ${catkin_ws}" >&2
  return 2
fi
if [[ -f "${python_env}/bin/activate" ]]; then
  source "${python_env}/bin/activate"
else
  echo "Python environment is missing: ${python_env}" >&2
  echo "Set TURTLEBOT_DDPG_VENV to override it." >&2
  return 2
fi

export TURTLEBOT_DDPG_PACKAGE_ROOT="${package_root}"
export TURTLEBOT_DDPG_PROJECT_ROOT="${project_root}"
export TURTLEBOT_DDPG_CATKIN_WS="${catkin_ws}"

