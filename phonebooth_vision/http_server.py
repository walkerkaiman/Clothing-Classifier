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
from pathlib import Path

OUTPUT_PATH = Path("objects.json")
STATIC_DIR = Path(__file__).with_suffix("").parent / "static"
IMAGE_PATH = Path("latest.jpg")
MODELS_DIR = Path("models")
CLOTHING_DETECTIONS_PATH = Path("clothing_detections.json")

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
        content = OUTPUT_PATH.read_text()
        if not content.strip():
            return {}
        data = json.loads(content)
        return data
    except json.JSONDecodeError as e:
        # Return empty data instead of 500 error
        return {}
    except Exception as e:
        # Log the error but return empty data
        print(f"Error reading objects.json: {e}")
        return {}


@app.get("/clothing.json", response_class=JSONResponse)
def get_clothing_detections():
    """Get current clothing detections."""
    try:
        # Try to fetch from the clothing detection server first
        import requests
        response = requests.get("http://localhost:8001/detections", timeout=1.0)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        # Fallback to file if server is not available
        pass
    
    # Fallback to file-based approach
    if not CLOTHING_DETECTIONS_PATH.exists():
        return {"timestamp": "", "detections": []}
    try:
        data = json.loads(CLOTHING_DETECTIONS_PATH.read_text())
    except json.JSONDecodeError as e:
        return {"timestamp": "", "detections": []}
    return data


@app.get("/detections", response_class=JSONResponse)
def get_all_detections():
    """Get both object counts and clothing detections."""
    objects_data = get_objects()
    clothing_data = get_clothing_detections()
    
    return {
        "objects": objects_data,
        "clothing": clothing_data,
        "timestamp": clothing_data.get("timestamp", "")
    }


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
        last_frame = None
        
        while True:
            try:
                if IMAGE_PATH.exists():
                    # Try to read the file with retry logic
                    frame = None
                    for attempt in range(3):  # Try up to 3 times
                        try:
                            frame = IMAGE_PATH.read_bytes()
                            break
                        except PermissionError:
                            # File is being written, wait a bit and retry
                            await asyncio.sleep(0.01)
                            continue
                        except Exception as e:
                            print(f"Error reading image file: {e}")
                            break
                    
                    if frame and frame != last_frame:
                        yield (b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                        last_frame = frame
                    elif last_frame:
                        # Send the last known good frame if we can't read a new one
                        yield (b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + last_frame + b"\r\n")
                        
            except Exception as e:
                print(f"Stream error: {e}")
                # Send a placeholder frame or continue
                pass
                
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
          .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
          .main-layout { display: flex; gap: 2rem; align-items: flex-start; }
          .left-panel { flex: 1; min-width: 400px; }
          .right-panel { flex: 1; }
          h1, h2 { color: var(--accent); margin-top: 0; }
          #cam { width: 100%; max-width: 600px; border: 2px solid var(--accent); border-radius: 6px; }
          table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
          th { text-align: left; background: #2b2b2b; color: var(--accent); }
          th, td { padding: 8px 12px; border-bottom: 1px solid #333; }
          tr:nth-child(even) { background: #252525; }
          tr:hover { background: #303030; }
          select, button { background: #2b2b2b; color: var(--fg); border: 1px solid var(--accent); border-radius: 4px; padding: 6px 10px; margin-top: 4px; }
          button { cursor: pointer; }
          button:hover { background: var(--accent); color: var(--bg); }
          a { color: var(--accent); }
          .section { margin-bottom: 2rem; }
          .settings-section { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #333; }
          
          /* Scrollable content area */
          .scrollable-content {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 1rem;
            background: #252525;
          }
          
          /* Responsive design */
          @media (max-width: 1200px) {
            .main-layout { flex-direction: column; }
            .left-panel, .right-panel { min-width: auto; }
            #cam { max-width: 100%; }
          }
          
          /* Status indicators */
          .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
          }
          .status-active { background-color: #4caf50; }
          .status-inactive { background-color: #f44336; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Live Objects & Clothing Detection</h1>
          
          <div class="main-layout">
            <!-- Left Panel: Objects and Clothing Data -->
            <div class="left-panel">
              <div class="section">
                <h2>Clothing Detections <span class="status-indicator status-active" id="clothing-status"></span></h2>
                <div class="scrollable-content">
                                     <table id="clothing-table">
                     <thead><tr><th>ID</th><th>Description</th><th>Details</th><th>BBox</th></tr></thead>
                     <tbody></tbody>
                   </table>
                </div>
              </div>

              <div class="section">
                <h2>Object Counts <span class="status-indicator status-active" id="objects-status"></span></h2>
                <div class="scrollable-content">
                  <table id="obj-table">
                    <thead><tr><th>Object</th><th>Count</th></tr></thead>
                    <tbody></tbody>
                  </table>
                </div>
              </div>
            </div>

            <!-- Right Panel: Camera Feed and Settings -->
            <div class="right-panel">
              <div class="section">
                <h2>Live Camera Feed</h2>
                <img id="cam" src="/stream" />
              </div>

              <div class="settings-section">
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
            </div>
          </div>
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
            try {
              const res = await fetch('/objects.json');
              const data = await res.json();
              const tbody = document.querySelector('#obj-table tbody');
              tbody.innerHTML = '';
              for (const [name, info] of Object.entries(data)) {
                const row = document.createElement('tr');
                row.innerHTML = `<td style=\"color:${info.color}\">${name}</td><td>${info.count}</td>`;
                tbody.appendChild(row);
              }
              document.getElementById('objects-status').className = 'status-indicator status-active';
            } catch (error) {
              document.getElementById('objects-status').className = 'status-indicator status-inactive';
            }
          }

                     async function refreshClothing() {
             try {
               const res = await fetch('/clothing.json');
               const data = await res.json();
               const tbody = document.querySelector('#clothing-table tbody');
               tbody.innerHTML = '';
               for (const detection of data.detections || []) {
                 const row = document.createElement('tr');
                 
                 // Create detailed clothing breakdown
                 let detailsHtml = '';
                 if (detection.clothing_items) {
                   const details = [];
                   for (const [region, item] of Object.entries(detection.clothing_items)) {
                     const regionName = region.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
                     details.push(`<strong>${regionName}:</strong> ${item}`);
                   }
                   detailsHtml = details.join('<br>');
                 }
                 
                 row.innerHTML = `<td>${detection.id}</td><td>${detection.description}</td><td>${detailsHtml}</td><td>${detection.bbox.join(', ')}</td>`;
                 tbody.appendChild(row);
               }
               document.getElementById('clothing-status').className = 'status-indicator status-active';
             } catch (error) {
               document.getElementById('clothing-status').className = 'status-indicator status-inactive';
             }
           }
          
          setInterval(refreshCounts, 200); // 5 times per second
          setInterval(refreshClothing, 1000); // 1 time per second
          refreshCounts();
          refreshClothing();

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
