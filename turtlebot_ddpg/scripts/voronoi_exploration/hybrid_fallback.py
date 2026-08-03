#!/usr/bin/env python3
"""Coverage-stall detector for the MAPPO execution-aware fallback."""


class CoverageStallFallback(object):
    """Open a bounded rule-control window after measurable coverage stalls."""

    def __init__(self, stall_seconds=60.0, active_seconds=30.0,
                 progress_epsilon=0.2, minimum_coverage=20.0):
        self.stall_seconds = float(stall_seconds)
        self.active_seconds = float(active_seconds)
        self.progress_epsilon = float(progress_epsilon)
        self.minimum_coverage = float(minimum_coverage)
        if self.stall_seconds <= 0.0 or self.active_seconds <= 0.0:
            raise ValueError('fallback durations must be positive')
        if self.progress_epsilon <= 0.0:
            raise ValueError('progress_epsilon must be positive')
        self.anchor_coverage = None
        self.anchor_time = None
        self.active_until = None
        self.activations = 0

    @property
    def active(self):
        return self.active_until is not None

    def update(self, coverage, now):
        """Return ``(activated_now, active)`` for the current sample."""
        coverage = float(coverage)
        now = float(now)
        if self.anchor_coverage is None:
            self.anchor_coverage = coverage
            self.anchor_time = now
            return False, False

        if self.active:
            if now < self.active_until:
                return False, True
            self.active_until = None
            self.anchor_coverage = coverage
            self.anchor_time = now
            return False, False

        if coverage >= self.anchor_coverage + self.progress_epsilon:
            self.anchor_coverage = coverage
            self.anchor_time = now
            return False, False

        stalled = now - self.anchor_time >= self.stall_seconds
        if coverage >= self.minimum_coverage and stalled:
            self.active_until = now + self.active_seconds
            self.anchor_coverage = coverage
            self.anchor_time = now
            self.activations += 1
            return True, True
        return False, False
