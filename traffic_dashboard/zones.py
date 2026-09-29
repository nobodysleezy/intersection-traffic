"""Lane, parking, and crosswalk geometry for the near approach.

The camera looks down Osvoboditelů. Four lanes run from the bottom of the
frame up to the stop line:

1. left only
2. left or straight
3. straight
4. right only

Parked cars sit in the lot on the left and are outside every lane.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Bottom edge is wider because of perspective. Coordinates are fractions of the frame.
_LANE_BOTTOM_LEFT = (0.26, 0.99)
_LANE_TOP_LEFT = (0.36, 0.70)
_LANE_BOTTOM_RIGHT = (0.62, 0.99)
_LANE_TOP_RIGHT = (0.56, 0.66)

_PARKING = (
    (0.00, 0.58),
    (0.33, 0.58),
    (0.27, 0.80),
    (0.18, 0.99),
    (0.00, 0.99),
)

# Near zebra, then the zebra on the right side of the intersection.
_CROSSWALKS = (
    ((0.40, 0.655), (0.58, 0.605)),
    ((0.70, 0.50), (0.86, 0.47)),
)

_LANE_KINDS = ("left", "left_or_straight", "straight", "right")


@dataclass(frozen=True)
class Lane:
    kind: str
    polygon: tuple[tuple[float, float], ...]

    @property
    def stop_line(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Top edge of the lane, where a vehicle enters the intersection."""
        return self.polygon[3], self.polygon[2]

    @property
    def forward(self) -> tuple[float, float]:
        """Unit vector from the lane entrance toward the intersection."""
        bottom = _mid(self.polygon[0], self.polygon[1])
        top = _mid(self.polygon[3], self.polygon[2])
        return _unit((top[0] - bottom[0], top[1] - bottom[1]))

    @property
    def left(self) -> tuple[float, float]:
        fx, fy = self.forward
        return _unit((fy, -fx))


def lanes(shift_y: float = 0.0) -> tuple[Lane, ...]:
    built = []
    for index, kind in enumerate(_LANE_KINDS):
        start, end = index / 4, (index + 1) / 4
        polygon = (
            _mix(_LANE_BOTTOM_LEFT, _LANE_BOTTOM_RIGHT, start, shift_y),
            _mix(_LANE_BOTTOM_LEFT, _LANE_BOTTOM_RIGHT, end, shift_y),
            _mix(_LANE_TOP_LEFT, _LANE_TOP_RIGHT, end, shift_y),
            _mix(_LANE_TOP_LEFT, _LANE_TOP_RIGHT, start, shift_y),
        )
        built.append(Lane(kind, polygon))
    return tuple(built)


def parking(shift_y: float = 0.0) -> tuple[tuple[float, float], ...]:
    return tuple((x, _clip_y(y + shift_y)) for x, y in _PARKING)


def crosswalks(shift_y: float = 0.0) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    lines = []
    for start, end in _CROSSWALKS:
        lines.append(
            (
                (start[0], _clip_y(start[1] + shift_y)),
                (end[0], _clip_y(end[1] + shift_y)),
            )
        )
    return tuple(lines)


def contains(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def segments_cross(
    start: tuple[float, float],
    end: tuple[float, float],
    line_a: tuple[float, float],
    line_b: tuple[float, float],
) -> bool:
    """True when the path from start to end cuts the line segment."""
    return _ccw(start, line_a, line_b) != _ccw(end, line_a, line_b) and _ccw(
        start, end, line_a
    ) != _ccw(start, end, line_b)


def _mix(
    start: tuple[float, float],
    end: tuple[float, float],
    weight: float,
    shift_y: float,
) -> tuple[float, float]:
    x = start[0] + (end[0] - start[0]) * weight
    y = start[1] + (end[1] - start[1]) * weight
    return (x, _clip_y(y + shift_y))


def _clip_y(value: float) -> float:
    return float(min(0.995, max(0.02, value)))


def _mid(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _unit(vector: tuple[float, float]) -> tuple[float, float]:
    norm = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
    if norm < 1e-6:
        return (0.0, -1.0)
    return (vector[0] / norm, vector[1] / norm)


def _ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
