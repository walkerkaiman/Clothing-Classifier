# YOLO Classifier

Local, privacy-preserving computer-vision service that detects objects with YOLOv8, publishes live counts via a JSON API, and serves a real-time annotated video stream over HTTP with advanced clothing detection capabilities.

---

## Features

* 📷 Automatic webcam discovery (Windows)
* ⚡ GPU acceleration (CUDA) or CPU fallback
* 🖼️ Live MJPEG stream with color-matched bounding-boxes
* 📊 `/objects.json` endpoint – dictionary `{class: {count,color}}`
* 👕 **Advanced clothing detection with multi-region analysis**
  * Segments persons into body regions (head, upper body, lower body)
  * Generates detailed clothing descriptions for each region
  * Supports both simple (color/pattern) and advanced (AI model) detection
  * Visual overlays with dashed bounding boxes and detailed text
* 🌐 CORS enabled – any LAN client can fetch the JSON
* 🔧 Web UI with two-panel layout (data on left, camera on right)
* 🗄️ Model files dropped in `models/` appear in UI dropdown
* 📝 Config overrides persisted to `settings_override.json`
* 🚀 **Real-time JSON server for clothing data**
* 🔄 **Atomic file operations** for robust image streaming
* 📱 **Responsive web UI** with scrollable data sections

---

## Quick-start

### 🔧 Manual Setup

```powershell
# clone repo
git clone https://github.com/walkerkaiman/YOLO-Classifier
cd YOLO-Classifier

# create Python 3.11+ venv
python -m venv .venv
. .venv\Scripts\Activate.ps1

# install CUDA PyTorch (replace cu121 with cu118 if needed)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# install project dependencies
pip install -r requirements.txt

# run monitor (starts inference loop + clothing detection)
python -m phonebooth_vision.simple_monitor

# in another terminal
python -m phonebooth_vision.http_server
```

Open `http://<PC-IP>:8000/` to see the UI.

---

## Endpoints

| Method | Path           | Description                           |
|--------|----------------|---------------------------------------|
| GET    | `/`            | Web UI (video + counts + clothing + settings) |
| GET    | `/stream`      | MJPEG stream (multipart)              |
| GET    | `/objects.json`| Live counts JSON                      |
| GET    | `/clothing.json`| Live clothing detections JSON         |
| GET    | `/detections`  | Combined objects + clothing data      |
| GET    | `/cameras`     | Available camera list                 |
| GET    | `/models`      | Model files in `models/`              |
| POST   | `/settings`    | `{camera:int, model:str}` override    |

---

## Configuration

* **Camera & model** – use the Settings form in the UI.
* **IMG_SIZE / JPEG_QUALITY** – tweak constants in `simple_monitor.py`.
* **Stream FPS** – change the sleep in `/stream` generator (default `await asyncio.sleep(0.05)`).

### Clothing Detection Configuration

Create a `config.toml` file to customize clothing detection:

```toml
[app.clothing]
enabled = true
model_type = "simple"  # "simple" or "advanced"
model_name = "Salesforce/blip-image-captioning-base"
update_frequency = 1.0  # seconds between updates
min_person_confidence = 0.5

[app.server]
enabled = true
server_type = "http"  # "http" or "file"
host = "0.0.0.0"
port = 8001
output_file = "clothing_detections.json"
```

**Clothing Detection Modes:**
- **Simple**: Uses color and pattern analysis (fast, no external models)
  * Analyzes dominant colors and basic patterns
  * Segments body into regions for detailed analysis
  * Lightweight and privacy-preserving
- **Advanced**: Uses vision-language models for detailed descriptions (requires internet for model download)
  * BLIP or other vision-language models for rich descriptions
  * More accurate clothing identification
  * Requires transformers library and model download

**Multi-Region Analysis:**
The system automatically segments each detected person into:
- **Head region** (top 25%): Hats, hair, accessories
- **Upper body** (25-60%): Shirts, jackets, tops
- **Lower body** (60%+): Pants, skirts, shoes
- **Full body**: Fallback analysis when regions are unclear

### JSON Server

The clothing detection data is served in real-time via:

**HTTP Server** (default):
- Endpoint: `http://<PC-IP>:8001/detections`
- Returns: `{"timestamp": "...", "detections": [...]}`
- CORS enabled for cross-origin requests

**File-based Server**:
- Output: `clothing_detections.json`
- Updated in real-time as detections change

**Example JSON Response:**
```json
{
  "timestamp": "2025-08-23T15:42:10Z",
  "detections": [
    {
      "id": 1,
      "bbox": [100, 150, 200, 400],
      "class": "person",
      "description": "blue shirt, black pants",
      "clothing_items": {
        "upper_body": "blue shirt",
        "lower_body": "black pants",
        "head": "dark hair"
      },
      "timestamp": 1732294930.123
    }
  ]
}
```

---

## Hardware notes

* GPU strongly recommended. Install the matching CUDA wheel for PyTorch.
* On CPU, reduce `IMG_SIZE` to 320 and keep `JPEG_QUALITY` ≤ 60.
* Clothing detection works on both CPU and GPU (advanced mode benefits from GPU acceleration).

## Web UI Features

* **Two-panel layout**: Data panels on the left, camera feed on the right
* **Scrollable sections**: Object counts and clothing detections in scrollable containers
* **Real-time updates**: Data refreshes automatically every second
* **Status indicators**: Visual indicators for data connection status
* **Detailed clothing breakdown**: Shows individual clothing items by body region
* **Responsive design**: Adapts to different screen sizes
* **Visual overlays**: Clothing descriptions displayed directly on video feed with dashed bounding boxes

---

## Development

```bash
# lint
ruff check .

# run unit tests (placeholder)
pytest

# test clothing detection
python test_clothing.py

# run clothing detection examples
python example_clothing_detection.py

# test the clothing detection server
curl http://localhost:8001/detections

# test the main web UI
curl http://localhost:8000/clothing.json
```

## Troubleshooting

* **Clothing detections not updating**: Check that the clothing detection server is running on port 8001
* **Webcam images disappearing**: The system now uses atomic file operations to prevent read/write conflicts
* **500 errors on /objects.json**: The endpoint now returns empty data instead of errors when files are corrupted
* **Advanced clothing detection not working**: Ensure transformers library is installed (`pip install transformers`)
* **Performance issues**: Reduce `update_frequency` in config.toml or switch to simple detection mode

The project structure:

```
phonebooth_vision/
  __init__.py
  simple_monitor.py     # capture → infer → annotate → publish
  http_server.py        # FastAPI app + Web UI
  clothing_detector.py  # Multi-region clothing analysis
  json_server.py        # Real-time clothing data server
  config/               # Pydantic models & manager
    manager.py          # Configuration management
    models.py           # Pydantic configuration models
models/                 # drop your .pt /.pth weights here
config.toml            # Clothing detection configuration
test_clothing.py       # Clothing detection test script
example_clothing_detection.py  # Example usage
```

---

## License

MIT © 2025 Phonebooth Vision Team
