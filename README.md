# Phonebooth-Vision

Local, privacy-preserving computer-vision service that detects objects with YOLOv8, publishes live counts via a JSON API, and serves a real-time annotated video stream over HTTP.

---

## Features

* 📷 Automatic webcam discovery (Windows)
* ⚡ GPU acceleration (CUDA) or CPU fallback
* 🖼️ Live MJPEG stream with color-matched bounding-boxes
* 📊 `/objects.json` endpoint – dictionary `{class: {count,color}}`
* 👕 **NEW: Clothing detection with descriptive labels**
* 🌐 CORS enabled – any LAN client can fetch the JSON
* 🔧 Web UI to view video, counts, clothing, and change camera/model
* 🗄️ Model files dropped in `models/` appear in UI dropdown
* 📝 Config overrides persisted to `settings_override.json`
* 🚀 **NEW: Real-time JSON server for clothing data**

---

## Quick-start

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

# install project
 pip install -e .

# run monitor (starts inference loop + clothing detection)
 python -m phonebooth_vision.simple_monitor

# in another terminal
 uvicorn phonebooth_vision.http_server:app --host 0.0.0.0 --port 8000
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
- **Advanced**: Uses vision-language models for detailed descriptions (requires internet for model download)

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
      "description": "green shirt with bee design, blue jeans",
      "timestamp": 1732294930.123
    }
  ]
}
```

---

## Hardware notes

* GPU strongly recommended. Install the matching CUDA wheel for PyTorch.
* On CPU, reduce `IMG_SIZE` to 320 and keep `JPEG_QUALITY` ≤ 60.

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
```

The project structure:

```
phonebooth_vision/
  __init__.py
  simple_monitor.py     # capture → infer → annotate → publish
  http_server.py        # FastAPI app + Web UI
  config/               # Pydantic models & manager
models/                 # drop your .pt /.pth weights here
```

---

## License

MIT © 2025 Phonebooth Vision Team
