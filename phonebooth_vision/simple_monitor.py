"""Enhanced real-time object monitor using YOLOv8 with clothing detection.

This script starts multiple threads:
1. FrameGrabber: reads frames from the default webcam and always keeps the
   most recent frame available.
2. Detector loop (main thread): pulls latest frame, performs inference, counts
   objects in view, and writes `objects.json` to disk.
3. Clothing detection: generates descriptive clothing labels for detected persons.
4. JSON server: serves clothing detection data to network clients.

The JSON file looks like:
    {"person": 2, "bottle": 1}

Clothing detections are served as:
    {
      "timestamp": "2025-08-23T15:42:10Z",
      "detections": [
        {
          "id": 1,
          "bbox": [100, 150, 200, 400],
          "description": "green shirt with bee design, blue jeans"
        }
      ]
    }
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, List, Tuple, Any

import cv2  # type: ignore
from ultralytics import YOLO  # type: ignore
import torch  # new import

# Import clothing detection and server modules
from .clothing_detector import ClothingDetector, SimpleClothingDetector
from .json_server import create_server
from .config.manager import get_settings

OUTPUT_PATH = Path("objects.json")
MODEL_NAME = "yolov8n.pt"  # lightweight default model
INFERENCE_INTERVAL = 0.0  # run as fast as possible; adjust to limit FPS

OUTPUT_IMAGE = Path("latest.jpg")
IMG_SIZE = 320  # smaller inference resolution for CPU
JPEG_QUALITY = 60  # lower quality for faster encoding
ANNOTATE_INTERVAL = 3  # draw rectangles every N frames

latest_jpeg: bytes | None = None
jpeg_lock = threading.Lock()

# Clothing detection and server variables
clothing_detector = None
clothing_server = None
last_clothing_update = 0.0
current_clothing_detections = []  # Store current clothing detections for visualization


def _bgr_color(name: str):
    h = hash(name) & 0xFFFFFF
    # convert hex to BGR tuple
    r = (h >> 16) & 0xFF
    g = (h >> 8) & 0xFF
    b = h & 0xFF
    return (b, g, r)


def _hex_color(name: str) -> str:
    return f"#{hash(name) & 0xFFFFFF:06x}"


def draw_clothing_descriptions(frame: np.ndarray, clothing_detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw clothing descriptions on the frame.
    
    Args:
        frame: Input frame
        clothing_detections: List of clothing detection results
        
    Returns:
        Frame with clothing descriptions drawn
    """
    annotated_frame = frame.copy()
    
    for detection in clothing_detections:
        bbox = detection['bbox']
        description = detection['description']
        detection_id = detection['id']
        
        x1, y1, x2, y2 = bbox
        
        # Draw clothing description outline (dashed rectangle)
        color = (0, 255, 255)  # Cyan color for clothing descriptions
        thickness = 2
        
        # Draw dashed rectangle
        dash_length = 10
        for i in range(0, x2 - x1, dash_length * 2):
            # Top edge
            cv2.line(annotated_frame, (x1 + i, y1), (min(x1 + i + dash_length, x2), y1), color, thickness)
            # Bottom edge
            cv2.line(annotated_frame, (x1 + i, y2), (min(x1 + i + dash_length, x2), y2), color, thickness)
        
        for i in range(0, y2 - y1, dash_length * 2):
            # Left edge
            cv2.line(annotated_frame, (x1, y1 + i), (x1, min(y1 + i + dash_length, y2)), color, thickness)
            # Right edge
            cv2.line(annotated_frame, (x2, y1 + i), (x2, min(y1 + i + dash_length, y2)), color, thickness)
        
        # Draw clothing description text
        text = f"Clothing {detection_id}: {description}"
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Position text above the bounding box
        text_x = x1
        text_y = max(y1 - 10, text_height + 5)
        
        # Draw text background
        cv2.rectangle(annotated_frame, 
                     (text_x - 2, text_y - text_height - 2), 
                     (text_x + text_width + 2, text_y + baseline + 2), 
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(annotated_frame, text, (text_x, text_y), 
                   font, font_scale, color, font_thickness)
        
        # Draw detailed clothing items if available
        if 'clothing_items' in detection and detection['clothing_items']:
            detail_y = text_y + text_height + 5
            detail_font_scale = 0.4
            detail_font_thickness = 1
            
            for region, item_desc in detection['clothing_items'].items():
                if region != 'full_body':  # Skip full body to avoid redundancy
                    region_name = region.replace('_', ' ').title()
                    detail_text = f"{region_name}: {item_desc}"
                    
                    # Get detail text size
                    (detail_width, detail_height), detail_baseline = cv2.getTextSize(detail_text, font, detail_font_scale, detail_font_thickness)
                    
                    # Draw detail text background
                    cv2.rectangle(annotated_frame, 
                                 (text_x - 2, detail_y - detail_height - 2), 
                                 (text_x + detail_width + 2, detail_y + detail_baseline + 2), 
                                 (0, 0, 0), -1)
                    
                    # Draw detail text
                    cv2.putText(annotated_frame, detail_text, (text_x, detail_y), 
                               font, detail_font_scale, color, detail_font_thickness)
                    
                    detail_y += detail_height + 2
        
        # Draw small ID indicator
        cv2.circle(annotated_frame, (x1 + 10, y1 + 10), 8, color, -1)
        cv2.putText(annotated_frame, str(detection_id), (x1 + 5, y1 + 15), 
                   font, 0.4, (255, 255, 255), 1)
    
    return annotated_frame


def _collect(result) -> tuple[Dict[str, Dict[str, object]], list[tuple[tuple[int,int,int,int], str]]]:
    names = result.names
    counts: Dict[str, Dict[str, object]] = {}
    boxes = []
    for xyxy, cls_id, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
        name = names[int(cls_id)]
        color_hex = _hex_color(name)
        entry = counts.setdefault(name, {"count": 0, "color": color_hex})
        entry["count"] += 1
        boxes.append((tuple(map(int, xyxy)), name, conf))
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


def initialize_clothing_detection():
    """Initialize clothing detection and server based on configuration."""
    global clothing_detector, clothing_server
    
    try:
        logging.info("Initializing clothing detection system...")
        settings = get_settings()
        clothing_config = settings.app.clothing
        server_config = settings.app.server
        
        logging.info(f"Clothing config: enabled={clothing_config.enabled}, model_type={clothing_config.model_type}")
        logging.info(f"Server config: enabled={server_config.enabled}, server_type={server_config.server_type}")
        
        # Initialize clothing detector
        if clothing_config.enabled:
            if clothing_config.model_type == "advanced":
                try:
                    logging.info(f"Attempting to initialize advanced clothing detector: {clothing_config.model_name}")
                    clothing_detector = ClothingDetector(
                        model_name=clothing_config.model_name,
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )
                    logging.info(f"Successfully initialized advanced clothing detector: {clothing_config.model_name}")
                except ImportError as e:
                    logging.warning(f"Advanced clothing detection not available: {e}")
                    logging.info("Falling back to simple clothing detector")
                    clothing_detector = SimpleClothingDetector()
                    logging.info("Successfully initialized simple clothing detector")
            else:
                logging.info("Initializing simple clothing detector")
                clothing_detector = SimpleClothingDetector()
                logging.info("Successfully initialized simple clothing detector")
        else:
            logging.info("Clothing detection is disabled in configuration")
        
        # Initialize server
        if server_config.enabled:
            logging.info(f"Initializing {server_config.server_type} server...")
            if server_config.server_type == "http":
                clothing_server = create_server(
                    server_type=server_config.server_type,
                    host=server_config.host,
                    port=server_config.port
                )
                logging.info(f"Created HTTP server instance for {server_config.host}:{server_config.port}")
            else:
                clothing_server = create_server(
                    server_type=server_config.server_type,
                    output_path=server_config.output_file
                )
                logging.info(f"Created file-based server instance for {server_config.output_file}")
            
            if server_config.server_type == "http":
                clothing_server.start(background=True)
                logging.info(f"Started clothing data server on {server_config.host}:{server_config.port}")
            else:
                logging.info(f"Initialized file-based server: {server_config.output_file}")
        else:
            logging.info("Server is disabled in configuration")
                
    except Exception as e:
        logging.error(f"Failed to initialize clothing detection: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        clothing_detector = None
        clothing_server = None


def process_clothing_detections(frame, detections: List[Tuple[Tuple[int, int, int, int], str, float]]):
    """Process clothing detections and update server."""
    global clothing_detector, clothing_server, last_clothing_update, current_clothing_detections
    
    if not clothing_detector or not clothing_server:
        logging.debug("Clothing detector or server not initialized")
        return
    
    try:
        settings = get_settings()
        clothing_config = settings.app.clothing
        
        # Check if it's time to update clothing detections
        current_time = time.time()
        if current_time - last_clothing_update < clothing_config.update_frequency:
            logging.debug(f"Not time to update clothing yet. Last update: {last_clothing_update}, current: {current_time}")
            return
        
        # Filter person detections based on confidence
        person_detections = [
            (bbox, class_name) for bbox, class_name, conf in detections 
            if class_name.lower() == "person" and conf >= clothing_config.min_person_confidence
        ]
        
        logging.info(f"Found {len(person_detections)} person detections with confidence >= {clothing_config.min_person_confidence}")
        
        if person_detections:
            # Generate clothing descriptions
            logging.debug(f"Processing clothing for {len(person_detections)} persons")
            clothing_results = clothing_detector.process_detections(frame, person_detections)
            
            # Update server
            clothing_server.update_detections(clothing_results)
            last_clothing_update = current_time
            
            # Store current clothing detections for visualization
            current_clothing_detections = clothing_results
            
            logging.info(f"Updated clothing detections: {len(clothing_results)} persons")
            for result in clothing_results:
                logging.debug(f"Clothing result: ID {result['id']}, Description: {result['description']}")
        else:
            logging.debug("No person detections found for clothing analysis")
            current_clothing_detections = []
            
    except Exception as e:
        logging.error(f"Error processing clothing detections: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")


def main():  # noqa: CCR001
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize clothing detection
    initialize_clothing_detection()
    
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

            # Debug: Log detection results
            if boxes:
                logging.debug(f"YOLO detections: {len(boxes)} objects")
                for bbox, name, conf in boxes:
                    logging.debug(f"  - {name}: confidence={conf:.2f}, bbox={bbox}")

            # Process clothing detections
            process_clothing_detections(latest.frame, boxes)

            # annotate only every ANNOTATE_INTERVAL frames
            if frame_idx % ANNOTATE_INTERVAL == 0:
                annotated = latest.frame.copy()
                
                # Draw YOLO object detections
                for (x1, y1, x2, y2), name, conf in boxes:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), _bgr_color(name), 2)
                    label = f"{name} {conf:.2f}"
                    cv2.putText(annotated, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _bgr_color(name), 2)
                
                # Draw clothing descriptions on top
                if current_clothing_detections:
                    annotated = draw_clothing_descriptions(annotated, current_clothing_detections)

                _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                with jpeg_lock:
                    global latest_jpeg
                    latest_jpeg = buf.tobytes()

                # Use atomic file writing to prevent read conflicts
                tmp = OUTPUT_IMAGE.with_suffix(".tmp")
                try:
                    # Write to temporary file first
                    tmp.write_bytes(latest_jpeg)
                    
                    # Atomic replace with retry logic
                    for attempt in range(10):
                        try:
                            tmp.replace(OUTPUT_IMAGE)
                            break
                        except PermissionError:
                            # File is being read, wait a bit longer
                            time.sleep(0.1)
                        except Exception as e:
                            print(f"Error replacing image file: {e}")
                            break
                except Exception as e:
                    print(f"Error writing image file: {e}")

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
