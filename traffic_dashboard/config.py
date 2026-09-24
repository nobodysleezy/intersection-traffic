"""Shared settings for the traffic dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_STREAM_URL = "https://www.youtube.com/watch?v=0Ua8_c0Nphg"

# COCO class ids used by YOLOv8.
VEHICLE_CLASS_IDS = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}

# Near approach (red bus lane), just below the crosswalk.
DEFAULT_LINE_Y = 0.78

# Signal facing that approach, near the middle of the intersection.
DEFAULT_LIGHT_ROI = (0.46, 0.40, 0.07, 0.14)


@dataclass
class RuntimeConfig:
    """Controls the worker reads each frame. Safe to copy across threads."""

    line_y: float = DEFAULT_LINE_Y
    direction: str = "both"
    light_x: float = DEFAULT_LIGHT_ROI[0]
    light_y: float = DEFAULT_LIGHT_ROI[1]
    light_w: float = DEFAULT_LIGHT_ROI[2]
    light_h: float = DEFAULT_LIGHT_ROI[3]
    red_ratio: float = 0.06
    confidence: float = 0.35
    classes: tuple[str, ...] = field(
        default_factory=lambda: ("car", "bus", "truck", "motorcycle")
    )
    show_red_mask: bool = False


@dataclass
class Stats:
    cars_passed: int = 0
    red_appearances: int = 0
    red_durations: list[float] = field(default_factory=list)
    red_active: bool = False
    current_red_duration: float = 0.0
    average_red_duration: float = 0.0
    last_red_duration: float = 0.0
    red_ratio: float = 0.0
    fps: float = 0.0
    device: str = "mps"
    status: str = "Idle"
    error: str = ""
    tracks: int = 0
