# Vision-MAPPO Multi-Robot Exploration in Gazebo

This repository extends the original TurtleBot3 DDPG collision-avoidance and
Hu-style Voronoi exploration project into a dissertation prototype for:

> Vision-Guided Multi-Agent Reinforcement Learning for Swarm Robotic
> Exploration in Unknown Environments

The current system runs with ROS Noetic and Gazebo Classic. Three TurtleBot3
Waffle Pi robots build a shared sensor-derived occupancy map. A shared MAPPO
actor selects frontier targets from robot-local observations, while the
reproduced DDPG controller provides low-level motion. The final hierarchy also
contains timed spatial recovery, bounded coverage-stall fallback, and
deterministic tail completion. The default stuck recovery keeps the reproduced
random turn. An optional clearance-guided mode is retained for controlled
ablation, and every completed recovery publishes its measured displacement.

## Current architecture

```text
laser scans + RGB camera
          |
          v
sensor-built occupancy map and local BEV observations
          |
          v
frontier tokens + visual openness + teammate intent
          |
          v
shared MAPPO frontier policy (centralized critic during training)
          |
          v
timed recovery / bounded fallback / tail completion
          |
          v
frozen DDPG low-level controller in Gazebo
```

The final evaluation uses local-observation action selection in one ROS policy
process. It is a Gazebo research prototype, not a physically distributed
multi-computer deployment.

## Main entry points

```bash
# Python environment used by the dissertation experiments
python3 -m venv ~/ddpg_env --system-site-packages
~/ddpg_env/bin/pip install -r requirements-noetic.txt

# Build
cd ~/catkin_ws
catkin build turtlebot_ddpg

# Launch the Gazebo research system
cd ~/catkin_ws/src/turtlebot3_ddpg_collision_avoidance
turtlebot_ddpg/scripts/run_gazebo_research.sh

# Run the Hu-aligned paired evaluation with the archived checkpoint
MAPPO_CHECKPOINT="$PWD/research_artifacts/dissertation_v1/checkpoint/mappo_selected_validation_v1.pt" \
  turtlebot_ddpg/scripts/run_hu_aligned_evaluation.sh \
  5 900 97 hu_final_pair 9200

# Isolate execution-aware allocation and timed failure memory
EVAL_WORLD=hu_aligned EXECUTION_PATH_COST_MODE=line_risk \
  turtlebot_ddpg/scripts/run_hu_aligned_evaluation.sh \
  5 90 100 execution_pair 9700
EVAL_WORLD=hu_aligned \
  turtlebot_ddpg/scripts/run_hu_aligned_evaluation.sh \
  5 120 100 memory_ablation 9800

# Compare laser-guided recovery with the reproduced random-turn baseline
EVALUATION_WORLD_PROFILE=hu_aligned \
  turtlebot_ddpg/scripts/run_hu_aligned_evaluation.sh \
  5 90 100 recovery_direction_pair 11450

# Repeat that one-factor comparison in the complete 97% evaluation stack
EVALUATION_WORLD_PROFILE=hu_aligned \
  turtlebot_ddpg/scripts/run_hu_aligned_evaluation.sh \
  5 900 97 final_recovery_pair 12100

# Isolate the camera feature in the same complete 97% control stack
EVALUATION_WORLD_PROFILE=hu_aligned \
  turtlebot_ddpg/scripts/run_hu_aligned_evaluation.sh \
  5 900 97 final_vision_pair 12300

# Test confidence-gated camera input and wheel-odometry registration
EVAL_WORLD=layout_c \
  turtlebot_ddpg/scripts/run_hu_aligned_evaluation.sh \
  5 90 100 gated_vision_pair 9900
EVAL_WORLD=hu_aligned \
  turtlebot_ddpg/scripts/run_hu_aligned_evaluation.sh \
  5 90 100 pose_source_pair 9970

# Plot recorded Gazebo trajectories from one evaluation directory
python turtlebot_ddpg/scripts/plot_gazebo_trajectory.py \
  outputs/evaluation/pose_source_pair_n5_20260806/repeat_01_mappo_tail_wheel_odom.csv \
  outputs/evaluation/pose_source_pair_n5_20260806/trajectory_wheel_odom.png

# Run unit tests
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
~/ddpg_env/bin/python -m unittest discover \
  -s turtlebot_ddpg/tests -p 'test_*.py' -v
```

The scripts discover the repository, catkin workspace, and virtual environment
from their own locations. Custom layouts can set `TURTLEBOT_DDPG_PROJECT_ROOT`,
`TURTLEBOT_DDPG_CATKIN_WS`, and `TURTLEBOT_DDPG_VENV`. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the frozen software versions,
artifact checks, and the exact evidence directories used by the dissertation.

The communication channel keeps the original range-only behaviour by default.
Delay, packet loss, and bandwidth limits can be enabled for controlled studies:

```bash
COMMUNICATION_DELAY_SECONDS=0.25 \
COMMUNICATION_DROP_PROBABILITY=0.10 \
COMMUNICATION_BANDWIDTH_BYTES_PER_SECOND=512 \
  turtlebot_ddpg/scripts/run_hu_aligned_evaluation.sh \
  5 90 100 communication_audit 10100
```

The pose-source audit integrates TurtleBot3 wheel-joint positions from the
known initial placement. It removes continuous Gazebo model pose from scan
registration, but it is not a substitute for loop closure or multi-robot map
merging. The `projected_obstacle_gated` vision mode suppresses directional
camera features when frame contrast or sky--floor reference separation is too
weak.

The compact dissertation evidence bundle is versioned under
`research_artifacts/dissertation_v1/`. It contains the validation-selected
checkpoint, raw valid-run CSV files, manifests, summaries, and SHA-256
checksums used by the reported tables. Large exploratory logs and videos remain
excluded from Git.

## Archived upstream project notes

The material below describes the original ROS Kinetic project. It is retained
for provenance and is not the installation guide for the Noetic dissertation
prototype above.

# Mapless Collision Avoidance of TurtleBot3 Using DDPG and Prioritized Experience Replay
This work is implemented in paper [Accelerated Sim-to-Real Deep Reinforcement Learning: Learning Collision Avoidance from Human Player](https://arxiv.org/abs/2102.10711) published in 2021 IEEE/SICE International Symposium on System Integration (SII) and [Voronoi-Based Multi-Robot Autonomous Exploration in Unknown Environments via Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9244647) published in IEEE Transactions on Vehicular Technology.

Demo Video 1: [Link](https://youtu.be/BmwxevgsdGc) 

Demo Video 2: [Link](https://youtu.be/XYvwVYhxP-o) 

Single Turtlebot3 Collision Avoidance:

![](single-turtlebot-collision-avoidance.gif)

Multiple Turtlebot3 Collision Avoidance:

![](multiple-turtlebot-collision-avoidance.gif)

This code is for training and testing ddpg algorithm on turtlebot3 waffle pi.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install the software

```
Ubuntu 16.04
ROS Kinetic
tensorflow-gpu == 1.13.1 or 1.14.0 (in case you have nvidia 10 series or 20 series gpu) or tensorflow-cpu==1.15.0 (in case you don't have nvidia gpu)
Keras == 2.3.1
numpy == 1.16.6
```


### Virtual Environment

You need to make a virtual environment called 'ddpg_env' and install the following library for it. If you are using ros-noetic not ros-kinetic, you need to install tensorflow-cpu==1.15.0.

```
tensorflow-gpu == 1.13.1 or 1.14.0 (in case you have nvidia 10 series or 20 series gpu) or tensorflow-cpu==1.15.0 (in case you don't have nvidia gpu)
Keras == 2.3.1
numpy == 1.16.6
```

### Installing

The next step is to install dependent packages for TurtleBot3 control on Remote PC. For more details, please refer to [turtlebot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/#pc-setup).

```
$ sudo apt-get update
$ sudo apt-get upgrade
$ wget https://raw.githubusercontent.com/ROBOTIS-GIT/robotis_tools/master/install_ros_kinetic.sh && chmod 755 ./install_ros_kinetic.sh && bash ./install_ros_kinetic.sh

$ sudo apt-get install ros-kinetic-joy ros-kinetic-teleop-twist-joy ros-kinetic-teleop-twist-keyboard ros-kinetic-laser-proc ros-kinetic-rgbd-launch ros-kinetic-depthimage-to-laserscan ros-kinetic-rosserial-arduino ros-kinetic-rosserial-python ros-kinetic-rosserial-server ros-kinetic-rosserial-client ros-kinetic-rosserial-msgs ros-kinetic-amcl ros-kinetic-map-server ros-kinetic-move-base ros-kinetic-urdf ros-kinetic-xacro ros-kinetic-compressed-image-transport ros-kinetic-rqt-image-view ros-kinetic-gmapping ros-kinetic-navigation ros-kinetic-interactive-markers

$ cd ~/catkin_ws/src/
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
$ cd ~/catkin_ws && catkin_make
```


## Setting up the network between PC and turtlebot

Please refer to [Turtlebot3 Setup](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/#pc-setup)

## Git clone ddpg scripts

```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance.git
$ cd ~/catkin_ws && catkin_make
$ echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc
```

### Please use the new turtlebot3_waffle_pi.urdf.xacro and turtlebot3_waffle_pi.gazebo.xacro in the turtlebot3_description folder instead of using the original ones from this repository to change the laser from 360 rays to be 24 rays.
 

### Start gazebo world (Change your world file location based on your setting)
To launch the corridor world
```
$ roslaunch turtlebot_ddpg turtlebot3_empty_world.launch world_file:='/home/hanlin/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/turtlebot_ddpg/worlds/turtlebot3_modified_corridor2.world'
```
To launch the maze world
```
$ roslaunch turtlebot_ddpg turtlebot3_empty_world.launch world_file:='/home/hanlin/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/turtlebot_ddpg/worlds/turtlebot3_modified_maze.world'
```

### Start a new terminal and Open python virtual environment

```
$ source ~/ddpg_env/bin/activate
```

For train and play with original ddpg

```
$ cd ~/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/turtlebot_ddpg/scripts/original_ddpg
$ rosrun turtlebot_ddpg ddpg_network_turtlebot3_original_ddpg.py
```


For train and play with ddpg with human data
```
$ cd ~/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/turtlebot_ddpg/scripts/fd_replay/play_human_data
$ rosrun turtlebot_ddpg ddpg_network_turtlebot3_amcl_fd_replay_human.py
```

For training
```
please change train_indicator=1 under ddpg_network_turtlebot3_original_ddpg.py or ddpg_network_turtlebot3_amcl_fd_replay_human.py
```

For playing trained weights
```
 please change train_indicator=0 under ddpg_network_turtlebot3_original_ddpg.py or ddpg_network_turtlebot3_amcl_fd_replay_human.py
```



## Authors

* **Hanlin Niu** - [Personal Page](https://scholar.google.com/citations?user=8rF8ZdgAAAAJ&hl=en)


### Paper
If you use this code in your research, please cite our SII paper or IEEE Transactions on Vehicular Technology paper:
```
@inproceedings{niu2021accelerated,
  title={Accelerated sim-to-real deep reinforcement learning: Learning collision avoidance from human player},
  author={Niu, Hanlin and Ji, Ze and Arvin, Farshad and Lennox, Barry and Yin, Hujun and Carrasco, Joaquin},
  booktitle={2021 IEEE/SICE International Symposium on System Integration (SII)},
  pages={144--149},
  year={2021},
  organization={IEEE}
}
```
```
@article{hu2020voronoi,
  title={Voronoi-based multi-robot autonomous exploration in unknown environments via deep reinforcement learning},
  author={Hu, Junyan and Niu, Hanlin and Carrasco, Joaquin and Lennox, Barry and Arvin, Farshad},
  journal={IEEE Transactions on Vehicular Technology},
  volume={69},
  number={12},
  pages={14413--14423},
  year={2020},
  publisher={IEEE}
}
```
