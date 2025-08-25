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
import numpy as np  # needed for Fashionpedia detector

# Initialize configuration FIRST, before any other imports
from .config.manager import set_settings_path
config_path = Path(__file__).parent.parent / "config.toml"
if config_path.exists():
    set_settings_path(config_path)
    print(f"Loaded configuration from: {config_path}")

# Import clothing detection and server modules
from .clothing_detector import ClothingDetector, SimpleClothingDetector
from .fashionpedia_detector import FashionpediaDetector
from .json_server import create_server
from .config.manager import get_settings

MODEL_NAME = "yolov8n.pt"  # lightweight default model
INFERENCE_INTERVAL = 0.0  # run as fast as possible; adjust to limit FPS

OUTPUT_IMAGE = Path("latest.jpg")
IMG_SIZE = 320  # smaller inference resolution for better performance
JPEG_QUALITY = 60  # lower quality for faster encoding
ANNOTATE_INTERVAL = 5  # draw rectangles every N frames (reduced for better performance)

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

        # Try indexes camera_index…camera_index+3 with better backend priority
        self._cap = None
        # Prioritize DSHOW over MSMF to avoid the common MSMF errors
        backends = [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF), 
            ("ANY", cv2.CAP_ANY)
        ]
        
        for idx in range(camera_index, camera_index + 3):
            found = False
            for backend_name, backend in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        # Test if we can actually read a frame
                        ret, test_frame = cap.read()
                        if ret:
                            print(f"Successfully opened camera {idx} via {backend_name} backend")
                            self._cap = cap
                            found = True
                            break
                        else:
                            cap.release()
                    else:
                        cap.release()
                except Exception as e:
                    print(f"Failed to open camera {idx} with {backend_name}: {e}")
                    continue
            if found:
                break
        
        if self._cap is None:
            print("Warning: Unable to open webcam. Using synthetic frames for testing.")
            # Create a synthetic frame generator for testing
            self._synthetic_mode = True
            self._frame_count = 0
        else:
            self._synthetic_mode = False

    def run(self) -> None:  # type: ignore[override]
        if self._synthetic_mode:
            # Generate synthetic frames for testing
            while not self._stop_event.is_set():
                # Create a synthetic frame with some variation
                frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                
                # Add some "person-like" shapes for testing
                if self._frame_count % 30 == 0:  # Every 30 frames, add a "person"
                    # Draw a simple rectangle to simulate a person
                    cv2.rectangle(frame, (400, 200), (600, 600), (100, 100, 100), -1)
                    cv2.rectangle(frame, (450, 150), (550, 200), (80, 80, 80), -1)  # Head
                
                with self._lock:
                    self._latest = SimpleNamespace(frame=frame, ts=time.time())
                
                self._frame_count += 1
                time.sleep(0.033)  # ~30 FPS
        else:
            # Real webcam capture
            # Add buffer settings for more stable capture
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
            self._cap.set(cv2.CAP_PROP_FPS, 30)  # Set consistent frame rate
            
            consecutive_failures = 0
            max_failures = 10
            
            while not self._stop_event.is_set():
                ret, frame = self._cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"Warning: {consecutive_failures} consecutive frame read failures")
                        consecutive_failures = 0  # Reset counter
                    time.sleep(0.05)
                    continue
                
                consecutive_failures = 0  # Reset on successful read
                with self._lock:
                    self._latest = SimpleNamespace(frame=frame, ts=time.time())

    def get_frame(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)
        if not self._synthetic_mode and self._cap:
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
            # Check model type from configuration
            if clothing_config.model_type == "fashionpedia":
                try:
                    logging.info(f"Initializing Fashionpedia detector for fashion-specific clothing analysis")
                    clothing_detector = FashionpediaDetector(
                        model_name=clothing_config.model_name,
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )
                    logging.info(f"Successfully initialized Fashionpedia detector")
                except ImportError as e:
                    logging.warning(f"Fashionpedia detection not available: {e}")
                    logging.info("Falling back to simple clothing detector")
                    clothing_detector = SimpleClothingDetector()
                    logging.info("Successfully initialized simple clothing detector")
                except Exception as e:
                    logging.error(f"Failed to initialize Fashionpedia detector: {e}")
                    logging.info("Falling back to simple clothing detector")
                    clothing_detector = SimpleClothingDetector()
                    logging.info("Successfully initialized simple clothing detector")
            elif clothing_config.model_type == "advanced":
                try:
                    logging.info(f"Initializing advanced clothing detector: {clothing_config.model_name}")
                    # Extract BLIP configuration parameters
                    blip_config = {
                        'max_length': clothing_config.max_length,
                        'num_beams': clothing_config.num_beams,
                        'temperature': clothing_config.temperature,
                        'do_sample': clothing_config.do_sample,
                        'top_p': clothing_config.top_p
                    }
                    clothing_detector = ClothingDetector(
                        model_name=clothing_config.model_name,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        config=blip_config
                    )
                    logging.info(f"Successfully initialized advanced clothing detector")
                except ImportError as e:
                    logging.warning(f"Advanced clothing detection not available: {e}")
                    logging.info("Falling back to simple clothing detector")
                    clothing_detector = SimpleClothingDetector()
                    logging.info("Successfully initialized simple clothing detector")
                except Exception as e:
                    logging.error(f"Failed to initialize advanced clothing detector: {e}")
                    logging.info("Falling back to simple clothing detector")
                    clothing_detector = SimpleClothingDetector()
                    logging.info("Successfully initialized simple clothing detector")
            else:  # simple
                logging.info("Initializing simple clothing detector")
                clothing_detector = SimpleClothingDetector()
                logging.info("Successfully initialized simple clothing detector")
        else:
            logging.info("Clothing detection is disabled in configuration")
        
        # Initialize file-based server for clothing data
        if server_config.enabled:
            logging.info(f"Initializing file-based server for {server_config.output_file}")
            clothing_server = create_server(
                server_type="file",
                output_path=server_config.output_file
            )
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
    
    # Load configuration
    settings = get_settings()
    model_config = settings.app.model
    
    grabber = FrameGrabber()
    grabber.start()

    model = YOLO(MODEL_NAME)
    
    # Optimize GPU configuration based on settings
    if torch.cuda.is_available() and model_config.use_gpu:
        device = 0  # Use first GPU
        half = model_config.enable_half_precision  # Use config setting
        # Set GPU memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(model_config.gpu_memory_fraction)
        # Enable cudnn benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True
        logging.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        logging.info(f"Half-precision: {half}")
        logging.info(f"GPU memory fraction: {model_config.gpu_memory_fraction}")
    else:
        device = "cpu"
        half = False
        if not torch.cuda.is_available():
            logging.warning("CUDA not available - using CPU (performance will be limited)")
        elif not model_config.use_gpu:
            logging.info("GPU disabled in configuration - using CPU")
    
    # Move model to device
    if torch.cuda.is_available() and model_config.use_gpu:
        model.to(device)
    
    # Configure detection parameters based on settings
    confidence_threshold = model_config.confidence
    iou_threshold = model_config.iou
    
    # Set up class filtering for person-only mode
    classes_to_detect = None
    if model_config.person_only:
        # Only detect person class (class ID 0 in COCO dataset)
        classes_to_detect = [0]
        logging.info("Person-only detection mode enabled for better performance")
    else:
        # Use specified classes if available
        if hasattr(model_config, 'classes') and model_config.classes:
            # Convert class names to IDs
            class_names = [name.strip() for name in model_config.classes]
            class_ids = []
            for name in class_names:
                if name in model.names:
                    class_ids.append(model.names.index(name))
                else:
                    logging.warning(f"Unknown class name: {name}")
            if class_ids:
                classes_to_detect = class_ids
                logging.info(f"Detecting classes: {class_names}")
    
    if classes_to_detect:
        logging.info(f"Filtering detections to class IDs: {classes_to_detect}")

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
                conf=confidence_threshold,
                iou=iou_threshold,
                device=device,
                half=half,
                verbose=False,
                classes=classes_to_detect,  # Filter to specific classes
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

            # Removed objects.json output as requested
            # OUTPUT_PATH.write_text(json.dumps(counts, indent=2))

            frames += 1
            now = time.time()
            if now - last_fps_t >= 1.0:
                fps = frames/(now-last_fps_t)
                print(f"{fps:.1f} FPS", end="")
                
                # Show GPU memory usage if available
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    cached = torch.cuda.memory_reserved() / 1024**3
                    print(f" | GPU: {allocated:.2f}GB/{cached:.2f}GB", end="")
                
                print()  # New line
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
