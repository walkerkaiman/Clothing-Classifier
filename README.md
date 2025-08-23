# Phonebooth-Vision

Local, privacy-preserving computer-vision service that detects objects with YOLOv8, publishes live counts via a JSON API, and serves a real-time annotated video stream over HTTP.

---

## Features

* 📷 Automatic webcam discovery (Windows)
* ⚡ GPU acceleration (CUDA) or CPU fallback
* 🖼️ Live MJPEG stream with color-matched bounding-boxes
* 📊 `/objects.json` endpoint – dictionary `{class: {count,color}}`
* 🌐 CORS enabled – any LAN client can fetch the JSON
* 🔧 Web UI to view video, counts, and change camera/model
* 🗄️ Model files dropped in `models/` appear in UI dropdown
* 📝 Config overrides persisted to `settings_override.json`

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

# run monitor (starts inference loop)
 python -m phonebooth_vision.simple_monitor

# in another terminal
 uvicorn phonebooth_vision.http_server:app --host 0.0.0.0 --port 8000
```

Open `http://<PC-IP>:8000/` to see the UI.

---

## Endpoints

| Method | Path           | Description                           |
|--------|----------------|---------------------------------------|
| GET    | `/`            | Web UI (video + counts + settings)    |
| GET    | `/stream`      | MJPEG stream (multipart)              |
| GET    | `/objects.json`| Live counts JSON                      |
| GET    | `/cameras`     | Available camera list                 |
| GET    | `/models`      | Model files in `models/`              |
| POST   | `/settings`    | `{camera:int, model:str}` override    |

---

## Configuration

* **Camera & model** – use the Settings form in the UI.
* **IMG_SIZE / JPEG_QUALITY** – tweak constants in `simple_monitor.py`.
* **Stream FPS** – change the sleep in `/stream` generator (default `await asyncio.sleep(0.05)`).

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
