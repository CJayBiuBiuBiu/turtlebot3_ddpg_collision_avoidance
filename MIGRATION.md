# Moving the project to another computer

The GitHub repository is the source-of-truth backup for code, launch files,
configuration, the selected dissertation checkpoint, and the compact raw
evidence bundle. The dissertation/Overleaf source is intentionally maintained
outside this repository.

## Files preserved in Git

- `turtlebot_ddpg/scripts/`: coordinator, MAPPO/IPPO, experiment and plotting code
- `turtlebot_ddpg/launch/`, `config/`, and `worlds/`: Gazebo setup
- `turtlebot_ddpg/msg/`: ROS interfaces
- `research_artifacts/dissertation_v1/`: selected checkpoint and compact evidence
- `requirements-noetic.txt`: Python package versions
- `README.md` and `REPRODUCIBILITY.md`: build and experiment commands

Large temporary `outputs/`, ROS bags, catkin build products, and the dissertation
source are excluded from Git.

## Restore procedure

```bash
# Ubuntu 20.04 / ROS Noetic
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/CJayBiuBiuBiu/turtlebot3_ddpg_collision_avoidance.git

# Install or clone the standard TurtleBot3 message/runtime dependencies first.
python3 -m venv ~/ddpg_env --system-site-packages
~/ddpg_env/bin/pip install -r \
  ~/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/requirements-noetic.txt

cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
catkin build turtlebot_ddpg
source devel/setup.bash

cd ~/catkin_ws/src/turtlebot3_ddpg_collision_avoidance
python3 research_artifacts/dissertation_v1/verify_artifact.py
~/ddpg_env/bin/python -m unittest discover \
  -s turtlebot_ddpg/tests -p 'test_*.py' -v
```

Use the selected checkpoint at:

```text
research_artifacts/dissertation_v1/checkpoint/mappo_selected_validation_v1.pt
```

Before changing devices, confirm that `git status` is clean and that the
latest commit is visible on the GitHub `main` branch.
