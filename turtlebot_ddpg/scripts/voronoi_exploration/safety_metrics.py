"""Run-level safety telemetry independent of ROS message handling."""

import math


class ClearanceTelemetry(object):
    """Accumulate laser-clearance exposure without frame-count inflation.

    The event count is still emitted by the robot-side hysteresis detector.
    This class complements it with time below the event threshold, the share
    of received samples below that threshold, and the minimum observed range.
    """

    def __init__(self, event_threshold=0.18, max_sample_gap=0.5):
        if event_threshold <= 0.0:
            raise ValueError('event_threshold must be positive')
        if max_sample_gap <= 0.0:
            raise ValueError('max_sample_gap must be positive')
        self.event_threshold = float(event_threshold)
        self.max_sample_gap = float(max_sample_gap)
        self._last = {}
        self.active_seconds = 0.0
        self.sample_count = 0
        self.below_threshold_samples = 0
        self.minimum_clearance = math.inf

    def observe(self, robot_name, clearance_m, timestamp_s):
        clearance_m = float(clearance_m)
        timestamp_s = float(timestamp_s)
        if not math.isfinite(clearance_m) or clearance_m < 0.0:
            return

        previous = self._last.get(robot_name)
        if previous is not None:
            previous_clearance, previous_time = previous
            elapsed = timestamp_s - previous_time
            if elapsed > 0.0 and previous_clearance <= self.event_threshold:
                self.active_seconds += min(elapsed, self.max_sample_gap)

        self._last[robot_name] = (clearance_m, timestamp_s)
        self.sample_count += 1
        if clearance_m <= self.event_threshold:
            self.below_threshold_samples += 1
        self.minimum_clearance = min(self.minimum_clearance, clearance_m)

    def snapshot(self):
        return {
            'active_seconds': self.active_seconds,
            'sample_count': self.sample_count,
            'below_threshold_samples': self.below_threshold_samples,
            'sample_ratio': (
                float(self.below_threshold_samples) / self.sample_count
                if self.sample_count else 0.0),
            'minimum_clearance': (
                self.minimum_clearance
                if math.isfinite(self.minimum_clearance) else None),
        }
