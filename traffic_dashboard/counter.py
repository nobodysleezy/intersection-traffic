"""Count vehicles by the lane they leave, and ignore the parking lot."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from traffic_dashboard.detector import Track
from traffic_dashboard.zones import Lane, contains, lanes, parking, segments_cross

# How far a shared-lane vehicle must travel before the turn is decided.
_FORWARD_PX = 28.0
_LEFT_RATIO = 0.55
_PENDING_S = 2.2


@dataclass
class LaneCounts:
    left: int = 0
    straight: int = 0
    right: int = 0
    tags: dict[int, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.left + self.straight + self.right


class ApproachCounter:
    """Count only vehicles that drive out of the four approach lanes."""

    def __init__(self) -> None:
        self.counts = LaneCounts()
        self._prev: dict[int, tuple[float, float]] = {}
        self._counted: dict[int, float] = {}
        self._pending: dict[int, tuple[float, tuple[float, float], tuple[float, float]]] = {}
        self._last_seen: dict[int, float] = {}

    def reset(self) -> None:
        self.__init__()

    def update(self, tracks: list[Track], width: int, height: int, shift_y: float) -> list[Track]:
        now = time.monotonic()
        road = lanes(shift_y)
        lot = parking(shift_y)
        kept: list[Track] = []
        seen: set[int] = set()

        for track in tracks:
            if track.label == "person":
                continue
            seen.add(track.track_id)
            self._last_seen[track.track_id] = now
            foot = _foot(track, width, height)
            if contains(foot, lot):
                self._prev.pop(track.track_id, None)
                continue
            lane = _lane_at(foot, road)
            previous = self._prev.get(track.track_id)
            self._prev[track.track_id] = foot
            previous_lane = _lane_at(previous, road) if previous is not None else None
            if lane is not None and track.track_id not in self.counts.tags:
                self.counts.tags[track.track_id] = lane.kind
            if (
                lane is not None
                or previous_lane is not None
                or track.track_id in self._pending
                or track.track_id in self._counted
            ):
                kept.append(track)
            moved_forward = previous is not None and foot[1] < previous[1]
            if previous is None or track.track_id in self._counted:
                self._advance_pending(track.track_id, foot, width, height, now)
                continue
            if (
                previous_lane is not None
                and moved_forward
                and segments_cross(previous, foot, *previous_lane.stop_line)
            ):
                self._leave_lane(track.track_id, previous_lane, foot, now)
            else:
                self._advance_pending(track.track_id, foot, width, height, now)

        self._drop_stale(now, seen)
        return kept

    def _leave_lane(self, track_id: int, lane: Lane, foot: tuple[float, float], now: float) -> None:
        if lane.kind == "left":
            self._add(track_id, "left", now)
        elif lane.kind == "right":
            self._add(track_id, "right", now)
        elif lane.kind == "straight":
            self._add(track_id, "straight", now)
        else:
            self._pending[track_id] = (now, foot, lane.left)

    def _advance_pending(
        self,
        track_id: int,
        foot: tuple[float, float],
        width: int,
        height: int,
        now: float,
    ) -> None:
        pending = self._pending.get(track_id)
        if pending is None:
            return
        started, origin, left_axis = pending
        dx = (foot[0] - origin[0]) * width
        dy = (foot[1] - origin[1]) * height
        forward = -dy
        leftward = dx * left_axis[0] + dy * left_axis[1]
        if forward >= _FORWARD_PX and leftward >= forward * _LEFT_RATIO:
            self._add(track_id, "left", now)
            self._pending.pop(track_id, None)
        elif forward >= _FORWARD_PX:
            self._add(track_id, "straight", now)
            self._pending.pop(track_id, None)
        elif now - started > _PENDING_S:
            self._add(track_id, "straight", now)
            self._pending.pop(track_id, None)

    def _add(self, track_id: int, direction: str, now: float) -> None:
        if track_id in self._counted:
            return
        self._counted[track_id] = now
        self.counts.tags[track_id] = direction
        if direction == "left":
            self.counts.left += 1
        elif direction == "right":
            self.counts.right += 1
        else:
            self.counts.straight += 1

    def _drop_stale(self, now: float, seen: set[int]) -> None:
        stale = [
            track_id
            for track_id, seen_at in self._last_seen.items()
            if track_id not in seen and now - seen_at > 8
        ]
        for track_id in stale:
            self._prev.pop(track_id, None)
            self._last_seen.pop(track_id, None)
            self._pending.pop(track_id, None)
            if track_id not in self._counted:
                self.counts.tags.pop(track_id, None)
        expired = [track_id for track_id, at in self._counted.items() if now - at > 120]
        for track_id in expired:
            self._counted.pop(track_id, None)
            self.counts.tags.pop(track_id, None)


def _foot(track: Track, width: int, height: int) -> tuple[float, float]:
    x1, _y1, x2, y2 = track.xyxy
    return ((x1 + x2) / 2 / width, y2 / height)


def _lane_at(point: tuple[float, float], road: tuple[Lane, ...]) -> Lane | None:
    for lane in road:
        if contains(point, lane.polygon):
            return lane
    return None
