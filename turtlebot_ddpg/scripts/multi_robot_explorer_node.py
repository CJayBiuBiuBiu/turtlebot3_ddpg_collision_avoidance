#!/usr/bin/env python3
"""ROS executable wrapper for one namespaced low-level DDPG controller."""

import os
import sys


MODULE_DIR = os.path.join(os.path.dirname(__file__), 'voronoi_exploration')
sys.path.insert(0, MODULE_DIR)

from multi_robot_explorer import main


if __name__ == '__main__':
    main()
