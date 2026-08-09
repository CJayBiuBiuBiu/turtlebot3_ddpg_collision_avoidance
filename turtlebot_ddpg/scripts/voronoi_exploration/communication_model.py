#!/usr/bin/env python3
"""Deterministic, optional communication impairments for teammate intent."""

from collections import defaultdict, deque
import random


class CommunicationModel(object):
    """Model range, delay, packet loss, staleness, and sender bandwidth.

    Zero delay, zero loss, and non-positive bandwidth reproduce the original
    range-only communication behavior.
    """

    def __init__(self, delay_s=0.0, drop_probability=0.0,
                 bandwidth_bytes_per_s=0.0, message_bytes=48,
                 stale_after_s=2.0, seed=0):
        self.delay_s = float(delay_s)
        self.drop_probability = float(drop_probability)
        self.bandwidth_bytes_per_s = float(bandwidth_bytes_per_s)
        self.message_bytes = int(message_bytes)
        self.stale_after_s = float(stale_after_s)
        if self.delay_s < 0.0:
            raise ValueError('delay_s must be non-negative')
        if not 0.0 <= self.drop_probability <= 1.0:
            raise ValueError('drop_probability must be in [0, 1]')
        if self.message_bytes <= 0:
            raise ValueError('message_bytes must be positive')
        if self.stale_after_s <= 0.0:
            raise ValueError('stale_after_s must be positive')
        self.random = random.Random(int(seed))
        self.queues = defaultdict(deque)
        self.latest = {}
        self.sender_tokens = {}
        self.sender_token_time = {}
        self.sent = 0
        self.dropped = 0
        self.bandwidth_blocked = 0

    def _has_bandwidth(self, sender, now):
        if self.bandwidth_bytes_per_s <= 0.0:
            return True
        capacity = max(float(self.message_bytes), self.bandwidth_bytes_per_s)
        last_time = self.sender_token_time.get(sender, float(now))
        tokens = self.sender_tokens.get(sender, capacity)
        elapsed = max(0.0, float(now) - last_time)
        tokens = min(capacity, tokens + elapsed * self.bandwidth_bytes_per_s)
        self.sender_token_time[sender] = float(now)
        if tokens + 1e-9 < self.message_bytes:
            self.sender_tokens[sender] = tokens
            self.bandwidth_blocked += 1
            return False
        self.sender_tokens[sender] = tokens - self.message_bytes
        return True

    def exchange(self, sender, receiver, now, in_range, payload):
        """Queue one message and return the freshest non-stale delivery."""
        now = float(now)
        link = (str(sender), str(receiver))
        if in_range and self._has_bandwidth(link[0], now):
            if self.random.random() < self.drop_probability:
                self.dropped += 1
            else:
                queue = self.queues[link]
                queue.append((now + self.delay_s, payload))
                if len(queue) > 1024:
                    queue.popleft()
                self.sent += 1

        queue = self.queues[link]
        while queue and queue[0][0] <= now + 1e-9:
            _, delivered_payload = queue.popleft()
            self.latest[link] = (now, delivered_payload)

        latest = self.latest.get(link)
        if latest is None or now - latest[0] > self.stale_after_s:
            return None
        return latest[1]

    def diagnostics(self):
        return {
            'communication_messages_sent': self.sent,
            'communication_messages_dropped': self.dropped,
            'communication_bandwidth_blocked': self.bandwidth_blocked,
        }

