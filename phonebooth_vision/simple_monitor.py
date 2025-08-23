"""Simple real-time object monitor using YOLOv8.

This script starts two threads:
1. FrameGrabber: reads frames from the default webcam and always keeps the
   most recent frame available.
2. Detector loop (main thread): pulls latest frame, performs inference, counts
   objects in view, and writes `objects.json` to disk.

The JSON file looks like:
    {"person": 2, "bottle": 1}

Missing classes are omitted; counts are updated every inference run.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import cv2  # type: ignore
from ultralytics import YOLO  # type: ignore
import torch  # new import

OUTPUT_PATH = Path("objects.json")
MODEL_NAME = "yolov8n.pt"  # lightweight default model
INFERENCE_INTERVAL = 0.0  # run as fast as possible; adjust to limit FPS

OUTPUT_IMAGE = Path("latest.jpg")
IMG_SIZE = 320  # smaller inference resolution for CPU
JPEG_QUALITY = 60  # lower quality for faster encoding
ANNOTATE_INTERVAL = 3  # draw rectangles every N frames

latest_jpeg: bytes | None = None
jpeg_lock = threading.Lock()


def _bgr_color(name: str):
    h = hash(name) & 0xFFFFFF
    # convert hex to BGR tuple
    r = (h >> 16) & 0xFF
    g = (h >> 8) & 0xFF
    b = h & 0xFF
    return (b, g, r)


def _hex_color(name: str) -> str:
    return f"#{hash(name) & 0xFFFFFF:06x}"


def _collect(result) -> tuple[Dict[str, Dict[str, object]], list[tuple[tuple[int,int,int,int], str]]]:
    names = result.names
    counts: Dict[str, Dict[str, object]] = {}
    boxes = []
    for xyxy, cls_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist()):
        name = names[int(cls_id)]
        color_hex = _hex_color(name)
        entry = counts.setdefault(name, {"count": 0, "color": color_hex})
        entry["count"] += 1
        boxes.append((tuple(map(int, xyxy)), name))
    return counts, boxes


class FrameGrabber(threading.Thread):
    """Continuously grabs frames from webcam, keeping only the latest."""

    def __init__(self, camera_index: int = 0):
        super().__init__(daemon=True)
        self._latest: Optional[SimpleNamespace] = None  # stores frame + timestamp
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Try indexes camera_index…camera_index+9
        self._cap = None
        for idx in range(camera_index, camera_index + 50):
            found = False
            for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    print(f"Opened camera {idx} via backend {backend}")
                    self._cap = cap
                    found = True
                    break
            if found:
                break
        if self._cap is None:
            raise RuntimeError("Unable to open any webcam (indexes 0–9)")

    def run(self) -> None:  # type: ignore[override]
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest = SimpleNamespace(frame=frame, ts=time.time())

    def get_frame(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)
        self._cap.release()


def main():  # noqa: CCR001
    grabber = FrameGrabber()
    grabber.start()

    model = YOLO(MODEL_NAME)
    device = 0 if torch.cuda.is_available() else "cpu"
    half = torch.cuda.is_available()

    try:
        frame_idx = 0
        last_fps_t = time.time()
        frames = 0
        while True:
            latest = grabber.get_frame()
            if latest is None:
                time.sleep(0.01)
                continue

            frame_idx += 1

            result = model.predict(
                latest.frame,
                imgsz=IMG_SIZE,
                conf=0.25,
                device=device,
                half=half,
                verbose=False,
            )[0]

            counts, boxes = _collect(result)

            # annotate only every ANNOTATE_INTERVAL frames
            if frame_idx % ANNOTATE_INTERVAL == 0:
                annotated = latest.frame.copy()
                for (x1, y1, x2, y2), name in boxes:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), _bgr_color(name), 2)
                    cv2.putText(annotated, name, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _bgr_color(name), 2)

                _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                with jpeg_lock:
                    global latest_jpeg
                    latest_jpeg = buf.tobytes()

                tmp = OUTPUT_IMAGE.with_suffix(".tmp")
                tmp.write_bytes(latest_jpeg)
                for _ in range(5):
                    try:
                        tmp.replace(OUTPUT_IMAGE)
                        break
                    except PermissionError:
                        time.sleep(0.05)

            OUTPUT_PATH.write_text(json.dumps(counts, indent=2))

            frames += 1
            now = time.time()
            if now - last_fps_t >= 1.0:
                print(f"{frames/(now-last_fps_t):.1f} FPS")
                frames = 0
                last_fps_t = now

            if INFERENCE_INTERVAL:
                time.sleep(INFERENCE_INTERVAL)
    except KeyboardInterrupt:
        print("Stopping…")
    finally:
        grabber.stop()


if __name__ == "__main__":
    main()
