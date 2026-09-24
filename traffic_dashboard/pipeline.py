"""Background worker that ties capture, tracking, and metrics together."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace

import cv2

from traffic_dashboard.annotate import annotate
from traffic_dashboard.capture import YouTubeCapture
from traffic_dashboard.config import RuntimeConfig, Stats
from traffic_dashboard.counter import LineCounter
from traffic_dashboard.detector import VehicleTracker
from traffic_dashboard.traffic_light import RedPhaseTracker, crop_light, red_ratio

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._config = RuntimeConfig()
        self._stats = Stats()
        self._frame: bytes | None = None
        self._stop = threading.Event()
        self._reset = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracker: VehicleTracker | None = None
        self._url = ""
        self._quality = "720p"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def update_config(self, config: RuntimeConfig) -> None:
        with self._lock:
            self._config = config

    def snapshot(self) -> tuple[bytes | None, Stats]:
        with self._lock:
            return self._frame, replace(self._stats)

    def start(self, url: str, quality: str) -> None:
        if self.running:
            return
        self._url = url.strip()
        self._quality = quality
        self._stop.clear()
        with self._lock:
            self._stats = Stats(status="Starting", device=self._stats.device)
            self._frame = None
        self._thread = threading.Thread(target=self._loop, name="traffic-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
        self._thread = None
        with self._lock:
            self._stats.status = "Stopped"
            self._stats.error = ""

    def reset_counts(self) -> None:
        self._reset.set()

    def _publish(self, frame_bgr, stats: Stats) -> None:
        ok, encoded = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        payload = encoded.tobytes() if ok else None
        with self._lock:
            if payload is not None:
                self._frame = payload
            self._stats = stats

    def _set_status(self, status: str, error: str = "") -> None:
        with self._lock:
            self._stats.status = status
            self._stats.error = error

    def _loop(self) -> None:
        capture: YouTubeCapture | None = None
        misses = 0
        last_tick = time.monotonic()
        fps = 0.0
        try:
            self._set_status("Connecting to YouTube")
            capture = YouTubeCapture(self._url, self._quality)
            self._set_status("Loading YOLOv8")
            if self._tracker is None:
                self._tracker = VehicleTracker()
            else:
                self._tracker.reset()
            device = self._tracker.device
            counter = LineCounter()
            phases = RedPhaseTracker()
            self._set_status("Live")

            while not self._stop.is_set():
                if self._reset.is_set():
                    counter.reset()
                    phases.reset()
                    self._reset.clear()

                frame = capture.read()
                if frame is None:
                    misses += 1
                    if misses >= 20:
                        self._set_status("Reconnecting")
                        capture.reconnect()
                        misses = 0
                    continue
                misses = 0

                with self._lock:
                    config = self._config

                tracks = self._tracker.track(frame, config.classes, config.confidence)
                line_y = int(config.line_y * frame.shape[0])
                counter.update(tracks, line_y, config.direction)

                roi, light_box = crop_light(frame, config)
                ratio = 0.0
                mask = None
                if roi is not None:
                    ratio, mask = red_ratio(roi)
                phases.update(ratio >= config.red_ratio)

                now = time.monotonic()
                dt = now - last_tick
                last_tick = now
                if dt > 0:
                    fps = (0.85 * fps) + (0.15 * (1.0 / dt))

                annotated = annotate(
                    frame,
                    tracks,
                    line_y,
                    light_box,
                    phases.active,
                    counter.total,
                    mask if config.show_red_mask else None,
                )
                durations = list(phases.durations[-12:])
                self._publish(
                    annotated,
                    Stats(
                        cars_passed=counter.total,
                        red_appearances=phases.appearances,
                        red_durations=durations,
                        red_active=phases.active,
                        current_red_duration=phases.current_duration(now),
                        average_red_duration=phases.average_duration,
                        last_red_duration=phases.last_duration,
                        red_ratio=ratio,
                        fps=fps,
                        device=device,
                        status="Live",
                        tracks=len(tracks),
                    ),
                )
        except Exception as exc:
            logger.exception("Pipeline failed")
            self._set_status("Error", str(exc))
        finally:
            if capture is not None:
                capture.close()
