#!/usr/bin/env python3
"""Coverage-stall detector for the MAPPO execution-aware fallback."""

import math


REFERENCE_COVERAGE = 'reference_coverage'
OBSERVED_FREE_AREA = 'observed_free_area'


def validate_progress_mode(mode, mapping_mode, max_seconds, uses_coverage_phase,
                           starts_training):
    """Reject combinations that would use truth coverage in a fixed-horizon audit."""
    if mode not in (REFERENCE_COVERAGE, OBSERVED_FREE_AREA):
        raise ValueError('unknown control progress mode: %s' % mode)
    if mode == REFERENCE_COVERAGE:
        return
    if mapping_mode != 'laser_scan':
        raise ValueError('observed_free_area requires laser_scan mapping')
    if not math.isfinite(max_seconds) or max_seconds <= 0.0:
        raise ValueError('observed_free_area requires a positive simulation horizon')
    if uses_coverage_phase:
        raise ValueError(
            'observed_free_area is incompatible with coverage-based completion, '
            'tail completion or adaptive execution')
    if starts_training:
        raise ValueError('observed_free_area is an evaluation mode for frozen policies')


class CoverageStallFallback(object):
    """Open a bounded rule-control window after scalar progress stalls."""

    def __init__(self, stall_seconds=60.0, active_seconds=30.0,
                 progress_epsilon=0.2, minimum_coverage=20.0):
        self.stall_seconds = float(stall_seconds)
        self.active_seconds = float(active_seconds)
        self.progress_epsilon = float(progress_epsilon)
        self.minimum_coverage = float(minimum_coverage)
        if self.stall_seconds <= 0.0 or self.active_seconds <= 0.0:
            raise ValueError('fallback durations must be positive')
        if not math.isfinite(self.progress_epsilon) or self.progress_epsilon <= 0.0:
            raise ValueError('progress_epsilon must be finite and positive')
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
