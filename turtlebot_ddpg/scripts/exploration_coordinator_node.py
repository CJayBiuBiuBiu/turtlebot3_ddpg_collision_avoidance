#!/usr/bin/env python3
"""ROS executable wrapper for the research exploration coordinator."""

import os
import sys


MODULE_DIR = os.path.join(os.path.dirname(__file__), 'voronoi_exploration')
sys.path.insert(0, MODULE_DIR)

from exploration_coordinator import main


if __name__ == '__main__':
    main()
