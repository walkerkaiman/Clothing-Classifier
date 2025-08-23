"""Minimal HTTP server to expose objects.json and a live UI."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2  # add import
import asyncio
from fastapi.middleware.cors import CORSMiddleware

OUTPUT_PATH = Path("objects.json")
STATIC_DIR = Path(__file__).with_suffix("").parent / "static"
IMAGE_PATH = Path("latest.jpg")
MODELS_DIR = Path("models")

app = FastAPI(title="Object Monitor UI")

# Mount static directory under /static
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/objects.json", response_class=JSONResponse)
def get_objects():
    if not OUTPUT_PATH.exists():
        return {}
    try:
        data = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return data


@app.get("/latest.jpg")
def latest_image():
    if not IMAGE_PATH.exists():
        raise HTTPException(status_code=404)
    return FileResponse(IMAGE_PATH, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# --- MJPEG stream endpoint ---
@app.get("/stream")
async def mjpeg_stream():
    async def gen():
        boundary = b"frame"
        while True:
            if IMAGE_PATH.exists():
                frame = IMAGE_PATH.read_bytes()
                yield (b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.05)  # ~20 FPS
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    # fallback basic HTML
    return HTMLResponse("""
    <html>
      <head>
        <title>Object Monitor</title>
        <style>
          :root { --bg: #1e1e1e; --fg: #e0e0e0; --accent: #00bcd4; }
          html, body { margin: 0; padding: 0; height: 100%; background: var(--bg); color: var(--fg); font-family: "Segoe UI", Arial, sans-serif; }
          .container { max-width: 1100px; margin: 0 auto; padding: 2rem; }
          h1, h2 { color: var(--accent); margin-top: 0; }
          #cam { width: 100%; border: 2px solid var(--accent); border-radius: 6px; }
          table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
          th { text-align: left; background: #2b2b2b; color: var(--accent); }
          th, td { padding: 8px 12px; border-bottom: 1px solid #333; }
          tr:nth-child(even) { background: #252525; }
          tr:hover { background: #303030; }
          select, button { background: #2b2b2b; color: var(--fg); border: 1px solid var(--accent); border-radius: 4px; padding: 6px 10px; margin-top: 4px; }
          button { cursor: pointer; }
          button:hover { background: var(--accent); color: var(--bg); }
          a { color: var(--accent); }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Live Objects</h1>
          <img id="cam" src="/stream" />
          <table id="obj-table">
            <thead><tr><th>Object</th><th>Count</th></tr></thead>
            <tbody></tbody>
          </table>

          <h2>Settings</h2>
          <form id="settings-form">
            <label>Camera:
              <select id="camera" name="camera"></select>
            </label><br/>
            <label>Model:
              <select id="model" name="model"></select>
            </label><br/>
            <button type="submit">Change Settings</button>
          </form>
        </div>

        <script>
          const camImg = document.getElementById('cam');
          document.addEventListener('DOMContentLoaded', async () => {
            // populate dropdowns
            const camSel = document.getElementById('camera');
            const modelSel = document.getElementById('model');
            const cams = await fetch('/cameras').then(r=>r.json());
            cams.forEach(c => {
              const opt = document.createElement('option');
              opt.value = c.id;
              opt.textContent = c.name;
              camSel.appendChild(opt);
            });
            const models = await fetch('/models').then(r=>r.json());
            models.forEach(m => {
              const opt = document.createElement('option');
              opt.value = m;
              opt.textContent = m;
              modelSel.appendChild(opt);
            });
          });

          async function refreshCounts() {
            const res = await fetch('/objects.json');
            const data = await res.json();
            const tbody = document.querySelector('#obj-table tbody');
            tbody.innerHTML = '';
            for (const [name, info] of Object.entries(data)) {
              const row = document.createElement('tr');
              row.innerHTML = `<td style=\"color:${info.color}\">${name}</td><td>${info.count}</td>`;
              tbody.appendChild(row);
            }
          }
          setInterval(refreshCounts, 200); // 5 times per second
          refreshCounts();

          document.getElementById('settings-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const body = {
              camera: parseInt(formData.get('camera')),
              model: formData.get('model')
            };
            await fetch('/settings', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
            alert('Settings updated. Please restart monitor to apply.');
          });
        </script>
      </body>
    </html>""")


@app.post("/settings")
def update_settings(payload: dict):
    # simplistic: write to settings.json which user can reload manually
    path = Path("settings_override.json")
    path.write_text(json.dumps(payload, indent=2))
    return {"status": "saved", "path": str(path)}


@app.get("/cameras")
def list_cameras():
    cams = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cams.append({"id": idx, "name": f"Camera {idx}"})
            cap.release()
    return cams

@app.get("/models")
def list_models():
    if not MODELS_DIR.exists():
        return []
    return [p.name for p in MODELS_DIR.glob("*.pt")]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("phonebooth_vision.http_server:app", host="0.0.0.0", port=8000, reload=False)
