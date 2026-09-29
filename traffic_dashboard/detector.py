"""YOLOv8 vehicle detection and tracking on the Apple MPS backend."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from pathlib import Path

from traffic_dashboard.config import PERSON_CLASS_ID, VEHICLE_CLASS_IDS

_TRACKER = str(Path(__file__).with_name("bytetrack.yaml"))


@dataclass
class Track:
    track_id: int
    xyxy: tuple[int, int, int, int]
    centroid: tuple[float, float]
    label: str
    confidence: float


def resolve_device() -> str:
    """Use Metal when this Mac supports it."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class VehicleTracker:
    """Track cars, buses, trucks, and motorcycles with YOLOv8."""

    def __init__(self, weights: str = "yolov8n.pt") -> None:
        self.device = resolve_device()
        self.model = YOLO(weights)
        self._names = self.model.names
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(source=blank, device=self.device, verbose=False, half=False)

    def reset(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) or []
        for tracker in trackers:
            if hasattr(tracker, "reset"):
                tracker.reset()

    def track(
        self,
        frame,
        class_names: tuple[str, ...],
        confidence: float,
    ) -> list[Track]:
        class_ids = [
            VEHICLE_CLASS_IDS[name]
            for name in class_names
            if name in VEHICLE_CLASS_IDS
        ]
        class_ids.append(PERSON_CLASS_ID)
        if not class_ids:
            return []

        results = self.model.track(
            frame,
            persist=True,
            device=self.device,
            classes=class_ids,
            conf=min(confidence, 0.2),
            iou=0.5,
            imgsz=640,
            tracker=_TRACKER,
            verbose=False,
            half=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        tracks: list[Track] = []
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()
        for track_id, box, conf, cls_id in zip(ids, xyxy, confs, clss):
            x1, y1, x2, y2 = (int(v) for v in box)
            label = str(self._names.get(cls_id, cls_id))
            if label == "person":
                if conf < 0.22:
                    continue
            elif conf < confidence:
                continue
            tracks.append(
                Track(
                    track_id=int(track_id),
                    xyxy=(x1, y1, x2, y2),
                    centroid=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    label=label,
                    confidence=float(conf),
                )
            )
        return tracks
