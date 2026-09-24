"""Draw tracks, the counting line, and the traffic-light box."""

from __future__ import annotations

import cv2
import numpy as np

from traffic_dashboard.detector import Track

_BOX_COLOR = (40, 190, 255)
_LINE_COLOR = (0, 220, 255)
_TEXT = (255, 255, 255)


def annotate(
    frame: np.ndarray,
    tracks: list[Track],
    line_y: int,
    light_box: tuple[int, int, int, int],
    red_active: bool,
    cars_passed: int,
    red_mask_roi: np.ndarray | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    x1, y1, x2, y2 = light_box

    if red_mask_roi is not None and y2 > y1 and x2 > x1:
        colored = cv2.cvtColor(red_mask_roi, cv2.COLOR_GRAY2BGR)
        colored[:, :, 2] = np.maximum(colored[:, :, 2], red_mask_roi)
        roi = canvas[y1:y2, x1:x2]
        if roi.shape[:2] == colored.shape[:2]:
            canvas[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.45, colored, 0.55, 0)

    cv2.line(canvas, (0, line_y), (width - 1, line_y), _LINE_COLOR, 2)
    cv2.putText(
        canvas,
        "COUNT LINE",
        (12, max(22, line_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _LINE_COLOR,
        2,
        cv2.LINE_AA,
    )

    for track in tracks:
        bx1, by1, bx2, by2 = track.xyxy
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), _BOX_COLOR, 2)
        label = f"{track.label} {track.track_id}"
        cv2.putText(
            canvas,
            label,
            (bx1, max(16, by1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _BOX_COLOR,
            1,
            cv2.LINE_AA,
        )

    signal_color = (0, 0, 255) if red_active else (80, 180, 80)
    signal_text = "SIGNAL: RED" if red_active else "SIGNAL"
    cv2.rectangle(canvas, (x1, y1), (x2, y2), signal_color, 2)
    cv2.putText(
        canvas,
        signal_text,
        (x1, max(16, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        signal_color,
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(canvas, (0, 0), (280, 36), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        f"Cars passed: {cars_passed}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        _TEXT,
        2,
        cv2.LINE_AA,
    )
    return canvas
