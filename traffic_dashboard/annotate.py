"""Draw lanes, tracks, and the traffic-light box."""

from __future__ import annotations

import cv2
import numpy as np

from traffic_dashboard.detector import Track
from traffic_dashboard.zones import crosswalks, lanes, parking

_COLORS = {
    "left": (255, 150, 40),
    "left_or_straight": (40, 190, 255),
    "straight": (40, 200, 40),
    "right": (200, 40, 220),
    "person": (255, 255, 255),
}
_TEXT = (255, 255, 255)
_LABELS = {
    "left": "left",
    "left_or_straight": "left/straight",
    "straight": "straight",
    "right": "right",
    "person": "person",
}


def annotate(
    frame: np.ndarray,
    tracks: list[Track],
    tags: dict[int, str],
    light_box: tuple[int, int, int, int],
    red_active: bool,
    counts: tuple[int, int, int, int],
    shift_y: float,
    red_mask_roi: np.ndarray | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    _draw_zones(canvas, width, height, shift_y)

    x1, y1, x2, y2 = light_box
    if red_mask_roi is not None and y2 > y1 and x2 > x1:
        colored = cv2.cvtColor(red_mask_roi, cv2.COLOR_GRAY2BGR)
        colored[:, :, 2] = np.maximum(colored[:, :, 2], red_mask_roi)
        roi = canvas[y1:y2, x1:x2]
        if roi.shape[:2] == colored.shape[:2]:
            canvas[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.45, colored, 0.55, 0)

    for track in tracks:
        tag = "person" if track.label == "person" else tags.get(track.track_id, "straight")
        color = _COLORS.get(tag, _COLORS["straight"])
        bx1, by1, bx2, by2 = track.xyxy
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2)
        name = _LABELS.get(tag, tag)
        cv2.putText(
            canvas,
            f"{name} {track.track_id}",
            (bx1, max(16, by1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    signal_color = (0, 0, 255) if red_active else (80, 180, 80)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), signal_color, 2)

    left, straight, right, people = counts
    cv2.rectangle(canvas, (0, 0), (430, 36), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        f"L {left}   straight {straight}   R {right}   people {people}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _TEXT,
        2,
        cv2.LINE_AA,
    )
    return canvas


def _draw_zones(canvas: np.ndarray, width: int, height: int, shift_y: float) -> None:
    lot = _pixels(parking(shift_y), width, height)
    cv2.polylines(canvas, [lot], True, (40, 40, 200), 1)
    for lane in lanes(shift_y):
        color = _COLORS[lane.kind]
        cv2.polylines(canvas, [_pixels(lane.polygon, width, height)], True, color, 1)
        a, b = lane.stop_line
        cv2.line(canvas, _px(a, width, height), _px(b, width, height), color, 2)
    for line in crosswalks(shift_y):
        cv2.line(
            canvas,
            _px(line[0], width, height),
            _px(line[1], width, height),
            (0, 255, 255),
            2,
        )


def _pixels(points: tuple[tuple[float, float], ...], width: int, height: int) -> np.ndarray:
    return np.array([_px(point, width, height) for point in points], np.int32)


def _px(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    return (int(point[0] * width), int(point[1] * height))
