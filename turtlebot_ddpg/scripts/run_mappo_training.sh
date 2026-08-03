#!/usr/bin/env bash
set -eo pipefail

rollout_length="${1:-32}"
decision_interval="${2:-2.5}"

source /opt/ros/noetic/setup.bash
source /usr/share/gazebo/setup.sh
source /home/ros/catkin_ws/devel/setup.bash
source /home/ros/ddpg_env/bin/activate
set -u

exec roslaunch turtlebot_ddpg gazebo_research_experiment.launch \
  selection_mode:=mappo \
  start_smoke_policy:=false \
  start_network_policy:=false \
  start_mappo_training:=true \
  training_rollout_length:="${rollout_length}" \
  training_decision_interval:="${decision_interval}"
