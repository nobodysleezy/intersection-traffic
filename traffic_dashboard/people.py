"""Count people who cross a zebra, skipping anyone in the parking lot."""

from __future__ import annotations

import time

from traffic_dashboard.detector import Track
from traffic_dashboard.zones import contains, crosswalks, parking, segments_cross


class PeopleCounter:
    def __init__(self) -> None:
        self.total = 0
        self._prev: dict[int, tuple[float, float]] = {}
        self._counted: set[int] = set()
        self._last_seen: dict[int, float] = {}

    def reset(self) -> None:
        self.__init__()

    def update(self, tracks: list[Track], width: int, height: int, shift_y: float) -> list[Track]:
        now = time.monotonic()
        lot = parking(shift_y)
        lines = crosswalks(shift_y)
        kept: list[Track] = []
        seen: set[int] = set()
        for track in tracks:
            if track.label != "person":
                continue
            _x1, y1, _x2, y2 = track.xyxy
            if y2 - y1 < 14:
                continue
            foot = (((track.xyxy[0] + track.xyxy[2]) / 2) / width, y2 / height)
            if contains(foot, lot):
                continue
            seen.add(track.track_id)
            self._last_seen[track.track_id] = now
            kept.append(track)
            previous = self._prev.get(track.track_id)
            self._prev[track.track_id] = foot
            if previous is None or track.track_id in self._counted:
                continue
            if any(segments_cross(previous, foot, *line) for line in lines):
                self._counted.add(track.track_id)
                self.total += 1
        stale = [
            track_id
            for track_id, seen_at in self._last_seen.items()
            if track_id not in seen and now - seen_at > 8
        ]
        for track_id in stale:
            self._prev.pop(track_id, None)
            self._last_seen.pop(track_id, None)
        return kept
