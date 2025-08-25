"""Minimal HTTP server to expose objects.json and a live UI."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2  # add import
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

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
          
                     .settings-form {
             background: #252525;
             padding: 1rem;
             border: 1px solid #333;
             border-radius: 4px;
             margin-top: 10px;
           }
           
           .setting-group {
             display: flex;
             align-items: center;
             margin-bottom: 10px;
             gap: 10px;
             position: relative;
           }
           
           .setting-group label {
             min-width: 120px;
             font-weight: 500;
             color: var(--fg);
             cursor: help;
           }
           
           .setting-group input[type="range"] {
             flex: 1;
             background: #2b2b2b;
             border: 1px solid var(--accent);
             border-radius: 4px;
             height: 6px;
             outline: none;
           }
           
           .setting-group input[type="range"]::-webkit-slider-thumb {
             background: var(--accent);
             border-radius: 50%;
             width: 16px;
             height: 16px;
             cursor: pointer;
           }
           
           .setting-group input[type="range"]::-moz-range-thumb {
             background: var(--accent);
             border-radius: 50%;
             width: 16px;
             height: 16px;
             cursor: pointer;
             border: none;
           }
           
           .setting-group span {
             min-width: 40px;
             text-align: right;
             font-weight: 500;
             color: var(--fg);
           }
           
           .setting-group input[type="checkbox"] {
             margin-left: 10px;
             width: 16px;
             height: 16px;
             accent-color: var(--accent);
           }
           
           .settings-form button {
             background: #2b2b2b;
             color: var(--fg);
             border: 1px solid var(--accent);
             border-radius: 4px;
             padding: 8px 16px;
             cursor: pointer;
             margin-top: 10px;
             font-weight: 500;
           }
           
           .settings-form button:hover {
             background: var(--accent);
             color: var(--bg);
           }
           
           /* Tooltip styles */
           .tooltip {
             position: relative;
             display: inline-block;
           }
           
           .tooltip .tooltiptext {
             visibility: hidden;
             width: 250px;
             background-color: #1e1e1e;
             color: var(--fg);
             text-align: left;
             border-radius: 6px;
             padding: 8px 12px;
             position: absolute;
             z-index: 1;
             bottom: 125%;
             left: 50%;
             margin-left: -125px;
             opacity: 0;
             transition: opacity 0.3s;
             border: 1px solid var(--accent);
             font-size: 12px;
             line-height: 1.4;
           }
           
           .tooltip .tooltiptext::after {
             content: "";
             position: absolute;
             top: 100%;
             left: 50%;
             margin-left: -5px;
             border-width: 5px;
             border-style: solid;
             border-color: var(--accent) transparent transparent transparent;
           }
           
                       .tooltip:hover .tooltiptext {
              visibility: visible;
              opacity: 1;
            }
            
            /* Description styling */
            .description-item {
              background: #2b2b2b;
              border: 1px solid #333;
              border-radius: 4px;
              padding: 12px;
              margin-bottom: 10px;
              font-size: 14px;
              line-height: 1.5;
            }
            
            .description-header {
              font-weight: 600;
              color: var(--accent);
              margin-bottom: 8px;
              font-size: 13px;
            }
            
            .description-text {
              color: var(--fg);
              font-style: italic;
            }
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
                     <thead><tr><th>ID</th><th>Head</th><th>Upper Body</th><th>Lower Body</th></tr></thead>
                     <tbody></tbody>
                   </table>
                 </div>
               </div>
               
               <div class="section">
                 <h2>Person Descriptions <span class="status-indicator status-active" id="description-status"></span></h2>
                 <div class="scrollable-content">
                   <div id="descriptions-container">
                     <!-- Person descriptions will be populated here -->
                   </div>
                 </div>
               </div>
            </div>

            <!-- Right Panel: Camera Feed and Settings -->
            <div class="right-panel">
              <div class="section">
                <h2>Live Camera Feed</h2>
                <img id="cam" src="/stream" />
              </div>
              
                             <div class="section">
                 <h2>BLIP Settings</h2>
                 <div class="settings-form">
                   <div class="setting-group">
                     <label for="max_length" class="tooltip">Max Length:
                       <span class="tooltiptext">Maximum length of the generated description. Higher values allow for more detailed descriptions but may be slower to generate.</span>
                     </label>
                     <input type="range" id="max_length" min="50" max="300" value="150" />
                     <span id="max_length_value">150</span>
                   </div>
                   <div class="setting-group">
                     <label for="temperature" class="tooltip">Temperature:
                       <span class="tooltiptext">Controls creativity vs. conservatism. Higher values (0.8-1.0) make BLIP more creative and less conservative, while lower values (0.1-0.5) make it more focused and predictable.</span>
                     </label>
                     <input type="range" id="temperature" min="0.1" max="1.0" step="0.1" value="0.9" />
                     <span id="temperature_value">0.9</span>
                   </div>
                   <div class="setting-group">
                     <label for="num_beams" class="tooltip">Beam Search:
                       <span class="tooltiptext">Number of search paths for text generation. Lower values (1-3) create more variety, while higher values (5-10) produce more focused and consistent results.</span>
                     </label>
                     <input type="range" id="num_beams" min="1" max="10" value="3" />
                     <span id="num_beams_value">3</span>
                   </div>
                   <div class="setting-group">
                     <label for="top_p" class="tooltip">Top P:
                       <span class="tooltiptext">Nucleus sampling parameter that controls diversity. Higher values (0.8-1.0) make BLIP more permissive and creative, while lower values (0.1-0.5) make it more conservative and focused.</span>
                     </label>
                     <input type="range" id="top_p" min="0.1" max="1.0" step="0.1" value="0.9" />
                     <span id="top_p_value">0.9</span>
                   </div>
                   <div class="setting-group">
                     <label for="do_sample" class="tooltip">Enable Sampling:
                       <span class="tooltiptext">When enabled, BLIP uses sampling for text generation, creating more varied and creative descriptions. When disabled, it uses deterministic generation for more consistent results.</span>
                     </label>
                     <input type="checkbox" id="do_sample" checked />
                   </div>
                   <button onclick="updateBLIPSettings()">Update Settings</button>
                 </div>
               </div>
            </div>
          </div>
        </div>

        <script>
          const camImg = document.getElementById('cam');
          
          // Update value displays when sliders change
          document.getElementById('max_length').addEventListener('input', function() {
            document.getElementById('max_length_value').textContent = this.value;
          });
          
          document.getElementById('temperature').addEventListener('input', function() {
            document.getElementById('temperature_value').textContent = this.value;
          });
          
          document.getElementById('num_beams').addEventListener('input', function() {
            document.getElementById('num_beams_value').textContent = this.value;
          });
          
          document.getElementById('top_p').addEventListener('input', function() {
            document.getElementById('top_p_value').textContent = this.value;
          });
          
          async function updateBLIPSettings() {
            const settings = {
              max_length: parseInt(document.getElementById('max_length').value),
              temperature: parseFloat(document.getElementById('temperature').value),
              num_beams: parseInt(document.getElementById('num_beams').value),
              top_p: parseFloat(document.getElementById('top_p').value),
              do_sample: document.getElementById('do_sample').checked
            };
            
            try {
              const response = await fetch('/update_blip_settings', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify(settings)
              });
              
              if (response.ok) {
                alert('BLIP settings updated successfully! The changes will take effect on the next clothing detection update.');
              } else {
                alert('Failed to update settings');
              }
            } catch (error) {
              console.error('Error updating settings:', error);
              alert('Error updating settings');
            }
          }
          
                     document.addEventListener('DOMContentLoaded', async () => {
             // Load current BLIP settings from server
             try {
               const response = await fetch('/get_blip_settings');
               if (response.ok) {
                 const settings = await response.json();
                 
                 // Update sliders and values
                 document.getElementById('max_length').value = settings.max_length;
                 document.getElementById('max_length_value').textContent = settings.max_length;
                 
                 document.getElementById('temperature').value = settings.temperature;
                 document.getElementById('temperature_value').textContent = settings.temperature;
                 
                 document.getElementById('num_beams').value = settings.num_beams;
                 document.getElementById('num_beams_value').textContent = settings.num_beams;
                 
                 document.getElementById('top_p').value = settings.top_p;
                 document.getElementById('top_p_value').textContent = settings.top_p;
                 
                 document.getElementById('do_sample').checked = settings.do_sample;
               }
             } catch (error) {
               console.error('Error loading BLIP settings:', error);
             }
           });

                                           async function refreshClothing() {
              try {
                const res = await fetch('/detections');
                const data = await res.json();
                
                // Update clothing table
                const tbody = document.querySelector('#clothing-table tbody');
                tbody.innerHTML = '';
                
                // Update descriptions container
                const descriptionsContainer = document.getElementById('descriptions-container');
                descriptionsContainer.innerHTML = '';
                
                for (const detection of data.detections || []) {
                  // Create clothing table row
                  const row = document.createElement('tr');
                  
                  // Extract clothing items by region (now handling arrays)
                  let headItems = [];
                  let upperBodyItems = [];
                  let lowerBodyItems = [];
                  
                  if (detection.clothing_items) {
                    // Handle both old format (strings) and new format (arrays)
                    if (Array.isArray(detection.clothing_items.head)) {
                      headItems = detection.clothing_items.head;
                    } else if (detection.clothing_items.head) {
                      headItems = [detection.clothing_items.head];
                    }
                    
                    if (Array.isArray(detection.clothing_items.upper_body)) {
                      upperBodyItems = detection.clothing_items.upper_body;
                    } else if (detection.clothing_items.upper_body) {
                      upperBodyItems = [detection.clothing_items.upper_body];
                    }
                    
                    if (Array.isArray(detection.clothing_items.lower_body)) {
                      lowerBodyItems = detection.clothing_items.lower_body;
                    } else if (detection.clothing_items.lower_body) {
                      lowerBodyItems = [detection.clothing_items.lower_body];
                    }
                  }
                  
                  // Join multiple items with commas
                  const headText = headItems.join(', ');
                  const upperBodyText = upperBodyItems.join(', ');
                  const lowerBodyText = lowerBodyItems.join(', ');
                  
                  row.innerHTML = `<td>${detection.id}</td><td>${headText}</td><td>${upperBodyText}</td><td>${lowerBodyText}</td>`;
                  tbody.appendChild(row);
                  
                  // Create description item
                  if (detection.description) {
                    const descriptionItem = document.createElement('div');
                    descriptionItem.className = 'description-item';
                    descriptionItem.innerHTML = `
                      <div class="description-header">Person ${detection.id}</div>
                      <div class="description-text">"${detection.description}"</div>
                    `;
                    descriptionsContainer.appendChild(descriptionItem);
                  }
                }
                
                document.getElementById('clothing-status').className = 'status-indicator status-active';
                document.getElementById('description-status').className = 'status-indicator status-active';
              } catch (error) {
                document.getElementById('clothing-status').className = 'status-indicator status-inactive';
                document.getElementById('description-status').className = 'status-indicator status-inactive';
              }
            }
          
          setInterval(refreshClothing, 1000); // 1 time per second
          refreshClothing();
        </script>
      </body>
    </html>""")


# Removed settings endpoint since the Settings UI section was removed


# Removed cameras and models endpoints since Settings UI was removed


@app.get("/detections", response_class=JSONResponse)
def get_detections():
    """Get current clothing detections."""
    # Read directly from the file that simple_monitor writes to
    clothing_path = Path("clothing_detections.json")
    if not clothing_path.exists():
        return {"timestamp": "", "detections": []}
    try:
        data = json.loads(clothing_path.read_text())
    except json.JSONDecodeError as e:
        return {"timestamp": "", "detections": []}
    return data


@app.get("/get_blip_settings")
def get_blip_settings():
    """Get current BLIP generation settings."""
    try:
        # Read current config
        config_path = Path("config.toml")
        if not config_path.exists():
            return JSONResponse({"error": "Config file not found"}, status_code=404)
        
        # Parse TOML
        import tomli
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        
        # Get BLIP settings with defaults
        clothing_config = config.get("app", {}).get("clothing", {})
        
        settings = {
            "max_length": clothing_config.get("max_length", 150),
            "temperature": clothing_config.get("temperature", 0.9),
            "num_beams": clothing_config.get("num_beams", 3),
            "top_p": clothing_config.get("top_p", 0.9),
            "do_sample": clothing_config.get("do_sample", True)
        }
        
        return JSONResponse(settings)
        
    except Exception as e:
        print(f"Error in get_blip_settings: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/update_blip_settings")
async def update_blip_settings(request: Request):
    """Update BLIP generation settings."""
    try:
        settings_data = await request.json()
        print(f"Received settings: {settings_data}")  # Debug log
        
        # Read current config
        config_path = Path("config.toml")
        if not config_path.exists():
            return JSONResponse({"error": "Config file not found"}, status_code=404)
        
        # Parse TOML
        import tomli
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        
        # Ensure the clothing section exists
        if "app" not in config:
            config["app"] = {}
        if "clothing" not in config["app"]:
            config["app"]["clothing"] = {}
        
        # Update BLIP settings
        config["app"]["clothing"]["max_length"] = settings_data["max_length"]
        config["app"]["clothing"]["temperature"] = settings_data["temperature"]
        config["app"]["clothing"]["num_beams"] = settings_data["num_beams"]
        config["app"]["clothing"]["top_p"] = settings_data["top_p"]
        config["app"]["clothing"]["do_sample"] = settings_data["do_sample"]
        
        print(f"Updated config: {config['app']['clothing']}")  # Debug log
        
        # Write back to config file
        import tomli_w
        with open(config_path, "wb") as f:
            tomli_w.dump(config, f)
        
        print("Config file written successfully")  # Debug log
        return JSONResponse({"message": "Settings updated successfully"})
        
    except Exception as e:
        print(f"Error in update_blip_settings: {e}")  # Debug log
        import traceback
        traceback.print_exc()  # Print full stack trace
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("phonebooth_vision.http_server:app", host="0.0.0.0", port=8000, reload=False)
