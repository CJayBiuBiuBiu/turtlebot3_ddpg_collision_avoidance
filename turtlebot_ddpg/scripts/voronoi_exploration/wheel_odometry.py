"""Differential-drive wheel odometry used by the Gazebo pose audit."""

import math


def integrate_wheel_positions(pose, previous, current, wheel_radius,
                              wheel_separation, max_wheel_delta=2.0):
    """Integrate one pair of wheel-position samples.

    Returns the updated ``(x, y, yaw)`` pose. A large discontinuity is
    treated as a simulator reset and leaves the pose unchanged.
    """
    if wheel_radius <= 0.0 or wheel_separation <= 0.0:
        raise ValueError('wheel dimensions must be positive')
    delta_left = float(current[0]) - float(previous[0])
    delta_right = float(current[1]) - float(previous[1])
    if (abs(delta_left) > max_wheel_delta or
            abs(delta_right) > max_wheel_delta):
        return tuple(pose)
    distance = wheel_radius * (delta_right + delta_left) / 2.0
    delta_yaw = wheel_radius * (
        delta_right - delta_left) / wheel_separation
    x, y, yaw = pose
    midpoint_yaw = yaw + 0.5 * delta_yaw
    return (
        x + distance * math.cos(midpoint_yaw),
        y + distance * math.sin(midpoint_yaw),
        (yaw + delta_yaw + math.pi) % (2.0 * math.pi) - math.pi)
