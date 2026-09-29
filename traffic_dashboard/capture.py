"""Capture frames from the YouTube live camera."""

from __future__ import annotations

import logging
import subprocess
import threading
import time

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from streamlink import Streamlink
from streamlink.exceptions import PluginError
from yt_dlp import YoutubeDL

logger = logging.getLogger(__name__)

FRAME_W = 1280
FRAME_H = 720
# Decode a bit faster than inference. The reader keeps only the newest frame,
# so a slow detector never falls behind the live camera.
_FPS = 12

_HEIGHT = {"720p": 720, "480p": 480, "360p": 360, "best": 720}


def _pick_stream(streams: dict, quality: str):
    if quality in streams:
        return streams[quality]
    for name, stream in streams.items():
        if str(name).startswith(quality):
            return stream
    if "best" in streams:
        return streams["best"]
    if not streams:
        return None
    return next(iter(streams.values()))


def _streamlink_url(url: str, quality: str) -> str | None:
    """Use streamlink when the video exposes a playlist it can open."""
    session = Streamlink()
    session.set_option("hls-live-edge", 2)
    try:
        streams = session.streams(url)
    except PluginError as exc:
        logger.info("Streamlink could not open this video: %s", exc)
        return None
    stream = _pick_stream(streams, quality)
    media = getattr(stream, "url", None)
    return media or None


def _ytdlp_url(url: str, quality: str) -> str:
    """YouTube live cameras often omit the HLS link streamlink expects.

    The Android player client still publishes a live HLS playlist.
    """
    height = _HEIGHT.get(quality, 720)
    options = {
        "quiet": True,
        "no_warnings": True,
        "format": f"b[height<={height}][vcodec^=avc1]/b[height<={height}]/b",
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    media = (info or {}).get("url")
    if not media:
        raise RuntimeError(
            "No playable stream. The YouTube live feed may be offline."
        )
    return media


def resolve_media_url(url: str, quality: str) -> str:
    media = _streamlink_url(url, quality)
    if media:
        logger.info("Resolved stream with streamlink")
        return media
    logger.info("Resolving the live HLS playlist")
    return _ytdlp_url(url, quality)


class YouTubeCapture:
    """Read BGR frames from a YouTube live stream.

    Streamlink is tried first. This Zlín camera does not publish a playlist
    that plugin can open, so the app then asks the YouTube Android client for
    the live HLS URL and decodes it to raw frames. OpenCV analyzes those frames.
    """

    def __init__(self, url: str, quality: str = "720p") -> None:
        self.url = url
        self.quality = quality
        self._ff: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stop_reader = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._seq = 0
        self._consumed = -1
        self._stderr: list[str] = []
        self._frame_bytes = FRAME_W * FRAME_H * 3
        self.open()

    def open(self) -> None:
        self.close()
        media = resolve_media_url(self.url, self.quality)
        ffmpeg = get_ffmpeg_exe()
        self._stderr = []
        self._ff = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-user_agent",
                "com.google.android.youtube/19.09.37 (Linux; U; Android 14)",
                "-fflags",
                "nobuffer+discardcorrupt",
                "-flags",
                "low_delay",
                "-i",
                media,
                "-an",
                "-vf",
                f"scale={FRAME_W}:{FRAME_H}:flags=fast_bilinear,fps={_FPS}",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._reader = threading.Thread(target=self._pump, name="frame-pump", daemon=True)
        self._reader.start()

    def _drain_stderr(self) -> None:
        proc = self._ff
        if proc is None or proc.stderr is None:
            return
        for line in iter(proc.stderr.readline, b""):
            text = line.decode("utf-8", "replace").strip()
            if text:
                self._stderr.append(text)
                del self._stderr[:-8]

    def _pump(self) -> None:
        """Keep the newest frame only, so inference cannot build a backlog."""
        while not self._stop_reader.is_set():
            frame = self._read_raw()
            if frame is None:
                return
            with self._lock:
                self._latest = frame
                self._seq += 1

    def _read_raw(self) -> np.ndarray | None:
        proc = self._ff
        if proc is None or proc.stdout is None:
            return None
        raw = proc.stdout.read(self._frame_bytes)
        if len(raw) != self._frame_bytes:
            if self._stderr:
                logger.warning("ffmpeg: %s", " | ".join(self._stderr))
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((FRAME_H, FRAME_W, 3))
        return frame.copy()

    @property
    def alive(self) -> bool:
        proc = self._ff
        return proc is not None and proc.poll() is None

    def read(self, timeout: float = 2.0) -> np.ndarray | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._stop_reader.is_set():
            with self._lock:
                if self._latest is not None and self._seq != self._consumed:
                    self._consumed = self._seq
                    return self._latest.copy()
            if not self.alive:
                return None
            time.sleep(0.01)
        return None

    def reconnect(self) -> None:
        logger.info("Reconnecting to the live stream")
        self.open()

    def close(self) -> None:
        self._stop_reader.set()
        proc = self._ff
        self._ff = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        reader = self._reader
        if reader is not None and reader.is_alive() and reader is not threading.current_thread():
            reader.join(timeout=1.5)
        self._reader = None
        self._stop_reader.clear()
        with self._lock:
            self._latest = None
            self._seq = 0
            self._consumed = -1
