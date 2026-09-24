"""Detect a red traffic signal with HSV thresholding and time each phase."""

from __future__ import annotations

import time

import cv2
import numpy as np

from traffic_dashboard.config import RuntimeConfig

_KERNEL = np.ones((3, 3), np.uint8)
# Require the color to hold this long before a phase starts or ends.
_DEBOUNCE_S = 0.5


def red_mask(bgr_roi: np.ndarray) -> np.ndarray:
    """Return a binary mask of red pixels. Hue wraps around 0 in OpenCV."""
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, (0, 90, 70), (12, 255, 255))
    high = cv2.inRange(hsv, (165, 90, 70), (180, 255, 255))
    mask = cv2.bitwise_or(low, high)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)


def red_ratio(bgr_roi: np.ndarray) -> tuple[float, np.ndarray]:
    mask = red_mask(bgr_roi)
    ratio = float(cv2.countNonZero(mask)) / float(mask.size)
    return ratio, mask


def crop_light(frame: np.ndarray, config: RuntimeConfig) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    x1 = int(np.clip(config.light_x, 0, 1) * width)
    y1 = int(np.clip(config.light_y, 0, 1) * height)
    x2 = int(np.clip(config.light_x + config.light_w, 0, 1) * width)
    y2 = int(np.clip(config.light_y + config.light_h, 0, 1) * height)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, (x1, y1, x2, y2)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


class RedPhaseTracker:
    """Count red phases and record how long each one stays red."""

    def __init__(self) -> None:
        self.appearances = 0
        self.durations: list[float] = []
        self.active = False
        self.started_at: float | None = None
        self._pending: bool | None = None
        self._pending_since = 0.0

    def reset(self) -> None:
        self.__init__()

    def update(self, is_red: bool, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if is_red == self.active:
            self._pending = None
            return
        if self._pending is not is_red:
            self._pending = is_red
            self._pending_since = now
            return
        if now - self._pending_since < _DEBOUNCE_S:
            return
        self._pending = None
        if is_red:
            self.active = True
            self.started_at = now
            self.appearances += 1
            return
        if self.started_at is not None:
            self.durations.append(now - self.started_at)
        self.active = False
        self.started_at = None

    def current_duration(self, now: float | None = None) -> float:
        if not self.active or self.started_at is None:
            return 0.0
        now = time.monotonic() if now is None else now
        return max(0.0, now - self.started_at)

    @property
    def average_duration(self) -> float:
        if not self.durations:
            return 0.0
        return float(sum(self.durations) / len(self.durations))

    @property
    def last_duration(self) -> float:
        if not self.durations:
            return 0.0
        return float(self.durations[-1])
