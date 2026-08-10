"""Small, testable helpers for the optional low-level laser safety shield."""

import math

import numpy as np


def frontal_clearance(ranges, angle_min, angle_increment,
                      half_angle=math.pi / 3.0):
    """Return minimum finite clearance inside a forward-facing scan sector."""
    values = np.asarray(ranges, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('ranges must be a non-empty one-dimensional array')
    if angle_increment <= 0.0:
        raise ValueError('angle_increment must be positive')
    if not 0.0 < half_angle <= math.pi:
        raise ValueError('half_angle must be in (0, pi]')

    angles = angle_min + np.arange(values.size) * angle_increment
    wrapped = (angles + math.pi) % (2.0 * math.pi) - math.pi
    mask = np.abs(wrapped) <= half_angle
    finite = values[mask & np.isfinite(values)]
    if finite.size == 0:
        finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else math.inf


def clearance_speed_scale(clearance, stop_range=0.20, slow_range=0.35):
    """Linearly reduce forward speed near the predeclared safety boundary."""
    if stop_range <= 0.0:
        raise ValueError('stop_range must be positive')
    if slow_range <= stop_range:
        raise ValueError('slow_range must exceed stop_range')
    if not math.isfinite(clearance):
        return 1.0
    return float(np.clip(
        (clearance - stop_range) / (slow_range - stop_range), 0.0, 1.0))


def clearance_guided_turn_direction(ranges, angle_min, angle_increment,
                                    fallback_direction=1.0,
                                    deadband=0.05):
    """Choose the recovery turn toward the side with more laser clearance.

    Positive angular velocity turns left. A lower-quartile sector score is
    used so that one distant ray cannot hide a nearby obstacle. When both
    sides are effectively equal, the caller-provided fallback is preserved.
    """
    values = np.asarray(ranges, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('ranges must be a non-empty one-dimensional array')
    if angle_increment <= 0.0:
        raise ValueError('angle_increment must be positive')
    if fallback_direction not in (-1.0, 1.0):
        raise ValueError('fallback_direction must be -1 or 1')
    if deadband < 0.0:
        raise ValueError('deadband must be non-negative')

    angles = angle_min + np.arange(values.size) * angle_increment
    wrapped = (angles + math.pi) % (2.0 * math.pi) - math.pi
    finite = np.isfinite(values) & (values >= 0.0)
    left = values[finite & (wrapped >= math.pi / 12.0) &
                  (wrapped <= 5.0 * math.pi / 6.0)]
    right = values[finite & (wrapped <= -math.pi / 12.0) &
                   (wrapped >= -5.0 * math.pi / 6.0)]
    if left.size == 0 or right.size == 0:
        return float(fallback_direction)
    left_score = float(np.percentile(left, 25))
    right_score = float(np.percentile(right, 25))
    if left_score > right_score + deadband:
        return 1.0
    if right_score > left_score + deadband:
        return -1.0
    return float(fallback_direction)
