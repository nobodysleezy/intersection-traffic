"""Count tracked vehicles that cross a horizontal line."""

from __future__ import annotations

import time

from traffic_dashboard.detector import Track


class LineCounter:
    def __init__(self) -> None:
        self.total = 0
        self._prev_y: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}
        self._counted_at: dict[int, float] = {}

    def reset(self) -> None:
        self.total = 0
        self._prev_y.clear()
        self._last_seen.clear()
        self._counted_at.clear()

    def update(self, tracks: list[Track], line_y: int, direction: str) -> int:
        now = time.monotonic()
        seen: set[int] = set()
        new_counts = 0
        for track in tracks:
            seen.add(track.track_id)
            curr_y = track.centroid[1]
            prev_y = self._prev_y.get(track.track_id)
            self._prev_y[track.track_id] = curr_y
            self._last_seen[track.track_id] = now
            if prev_y is None or track.track_id in self._counted_at:
                continue
            if _crossed(prev_y, curr_y, line_y, direction):
                self._counted_at[track.track_id] = now
                self.total += 1
                new_counts += 1
        self._forget_stale(now, seen)
        return new_counts

    def _forget_stale(self, now: float, seen: set[int]) -> None:
        stale = [
            track_id
            for track_id, seen_at in self._last_seen.items()
            if track_id not in seen and now - seen_at > 30
        ]
        for track_id in stale:
            self._prev_y.pop(track_id, None)
            self._last_seen.pop(track_id, None)
        expired = [
            track_id
            for track_id, counted_at in self._counted_at.items()
            if now - counted_at > 90
        ]
        for track_id in expired:
            self._counted_at.pop(track_id, None)


def _crossed(prev_y: float, curr_y: float, line_y: int, direction: str) -> bool:
    went_down = prev_y < line_y <= curr_y
    went_up = prev_y > line_y >= curr_y
    if direction == "down":
        return went_down
    if direction == "up":
        return went_up
    return went_down or went_up
