#!/usr/bin/env python3
"""Bounded spatial memory for temporarily unsafe exploration targets."""

import math


class TimedSpatialBlacklist(object):
    """Merge nearby reports and expire them so frontiers remain recoverable."""

    def __init__(self, ttl_seconds=90.0, merge_radius=0.9):
        self.ttl_seconds = float(ttl_seconds)
        self.merge_radius = float(merge_radius)
        if self.ttl_seconds <= 0.0 or self.merge_radius < 0.0:
            raise ValueError('invalid blacklist lifetime or merge radius')
        self._entries = []

    def _prune(self, now):
        self._entries = [entry for entry in self._entries
                         if entry[2] > now]

    def add(self, x, y, now):
        now = float(now)
        x, y = float(x), float(y)
        self._prune(now)
        for index, (old_x, old_y, _) in enumerate(self._entries):
            if math.hypot(x - old_x, y - old_y) <= self.merge_radius:
                self._entries[index] = (x, y, now + self.ttl_seconds)
                return False
        self._entries.append((x, y, now + self.ttl_seconds))
        return True

    def active_points(self, now):
        self._prune(float(now))
        return [(x, y) for x, y, _ in self._entries]

    def count(self, now):
        return len(self.active_points(now))
