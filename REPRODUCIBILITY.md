# Reproducibility

The dissertation results use ROS Noetic, Gazebo Classic 11.15.1, Python 3.8.10,
PyTorch 2.4.1 CPU, and the package versions in `requirements-noetic.txt`.
ROS Noetic and Gazebo Classic are retained to reproduce the Hu-style legacy
stack; both platforms are end-of-life and are not recommended for a new robot
deployment.

## Paths

Experiment scripts resolve the project from their own location. The following
variables override the detected locations when needed:

```bash
export TURTLEBOT_DDPG_PROJECT_ROOT=/path/to/turtlebot3_ddpg_collision_avoidance
export TURTLEBOT_DDPG_CATKIN_WS=/path/to/catkin_ws
export TURTLEBOT_DDPG_VENV=/path/to/python/environment
```

## Build and test

```bash
cd /path/to/catkin_ws
source /opt/ros/noetic/setup.bash
catkin build turtlebot_ddpg
source devel/setup.bash
/path/to/python/environment/bin/python -m unittest discover \
  -s src/turtlebot3_ddpg_collision_avoidance/turtlebot_ddpg/tests -v
```

## Dissertation artifact

`research_artifacts/dissertation_v1/` contains the frozen MAPPO checkpoint,
raw valid-run CSV files, run manifests, generated summaries, and SHA-256
checksums used by the dissertation. Run the verification script before using
the data:

```bash
python3 research_artifacts/dissertation_v1/verify_artifact.py
```

The top-level experiment manifest maps each dissertation table to its source
directories. Invalid simulator starts are retained in the source experiment
directories when an invalid log was produced.

## Controlled learning and scaling comparisons

The learning comparison holds the actor architecture, observation adapter,
action mask, reward, and low-level DDPG controller fixed. IPPO changes only the
critic from centralized team-value estimation to a shared local value function.

```bash
turtlebot_ddpg/scripts/run_learning_baselines.sh 20 120 32 2.5 3
```

The scaling protocol uses the larger `research` world and evaluates 3, 5, and
7 robots at a fixed simulation-time horizon. It records Gazebo's real-time
factor so additional robots are not mistaken for an algorithmic slowdown.

```bash
MAPPO_CHECKPOINT="$PWD/research_artifacts/dissertation_v1/checkpoint/mappo_selected_validation_v1.pt" \
  turtlebot_ddpg/scripts/run_swarm_scaling_evaluation.sh 5 180 15000
```
