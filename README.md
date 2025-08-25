# YOLO Classifier - Advanced Computer Vision with AI Clothing Detection

A real-time computer vision application that combines YOLOv8 object detection with advanced AI-powered clothing analysis. Features a modern web interface, GPU acceleration, and intelligent clothing categorization by body regions.

## 🚀 Features

### Core Functionality
- **📷 Real-time Object Detection**: YOLOv8-based detection with GPU acceleration
- **👕 AI Clothing Analysis**: BLIP-powered clothing description and categorization
- **🌐 Live Web Interface**: Modern, responsive UI with real-time video streaming
- **⚡ High Performance**: GPU-accelerated inference with CUDA support
- **🔧 Configurable**: Extensive configuration options via `config.toml`

### Clothing Detection Features
- **Multi-Region Analysis**: Automatically segments clothing by body areas (Head, Upper Body, Lower Body)
- **Intelligent Categorization**: Uses lookup tables to properly categorize clothing items
- **Comprehensive Descriptions**: BLIP generates detailed, natural language descriptions
- **Real-time Updates**: Clothing data updates every second with configurable frequency
- **Multiple Items Support**: Detects and displays multiple clothing items per body region

### Web Interface Features
- **Live Video Stream**: MJPEG streaming with real-time annotations
- **Clothing Detections Table**: Organized display of clothing by body regions
- **Person Descriptions**: Natural language descriptions from BLIP analysis
- **BLIP Settings Panel**: Real-time adjustment of AI generation parameters
- **Status Indicators**: Visual feedback for system status
- **Responsive Design**: Works on desktop and mobile devices

### Technical Features
- **GPU Acceleration**: CUDA support for optimal performance
- **Atomic File Operations**: Robust image streaming without conflicts
- **CORS Support**: Cross-origin requests enabled for external integrations
- **Error Handling**: Graceful degradation and error recovery
- **Logging**: Comprehensive logging for debugging and monitoring

## 📋 Requirements

### System Requirements
- **OS**: Windows 10/11 (tested), Linux/macOS (should work)
- **Python**: 3.11 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for models and dependencies

### Hardware Recommendations
- **GPU**: NVIDIA RTX 3060 or better for optimal performance
- **CPU**: Modern multi-core processor (for CPU fallback)
- **Webcam**: USB webcam or built-in camera
- **Network**: Stable internet connection for model downloads

## 🛠️ Installation

### 1. Clone the Repository
```powershell
git clone https://github.com/walkerkaiman/YOLO-Classifier
cd YOLO-Classifier
```

### 2. Create Virtual Environment
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install CUDA PyTorch (Recommended)
```powershell
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. Download YOLO Model
The application will automatically download `yolov8n.pt` on first run, or you can manually download it to the project root.

## 🚀 Quick Start

### Option 1: Using the Convenience Script
```powershell
.\run.ps1
```
This script automatically:
- Activates the virtual environment
- Starts the monitor in a new window
- Starts the web server
- Opens your browser to the interface

### Option 2: Manual Start
```powershell
# Terminal 1: Start the monitor (object detection + clothing analysis)
python -m phonebooth_vision.simple_monitor

# Terminal 2: Start the web server
python -m phonebooth_vision.http_server
```

### Option 3: Using PowerShell Scripts
```powershell
# Start monitor only
Start-Process powershell -ArgumentList @('-NoExit', '-Command', 'python -m phonebooth_vision.simple_monitor')

# Start server only
Start-Process powershell -ArgumentList @('-NoExit', '-Command', 'python -m phonebooth_vision.http_server')
```

## 🌐 Web Interface

Once running, access the web interface at:
- **Local**: `http://localhost:8000`
- **Network**: `http://<your-ip>:8000`

### Interface Sections

#### 1. Clothing Detections Table
- **ID**: Person identification number
- **Head**: Hats, hair accessories, glasses, facial features (beards, mustaches), hair styles
- **Upper Body**: Shirts, jackets, tops, sweaters, blouses, outerwear
- **Lower Body**: Pants, skirts, dresses, shorts, leggings
- **Feet**: Shoes, boots, sandals, sneakers, footwear

#### 2. Person Descriptions
- Natural language descriptions from BLIP analysis
- Full clothing descriptions for each detected person
- Updates in real-time

#### 3. Live Camera Feed
- Real-time MJPEG video stream
- Annotated with bounding boxes and labels
- Responsive design for different screen sizes

#### 4. BLIP Settings Panel
- **Max Length**: Maximum description length (50-300)
- **Temperature**: Creativity vs. conservatism (0.1-1.0)
- **Beam Search**: Number of search paths (1-10)
- **Top P**: Nucleus sampling parameter (0.1-1.0)
- **Enable Sampling**: Toggle for deterministic vs. random generation

## ⚙️ Configuration

### Main Configuration File: `config.toml`

```toml
[app]
# General application settings
name = "YOLO Classifier"
version = "1.0.0"

[app.model]
# YOLO model configuration
model_path = "yolov8n.pt"
confidence = 0.5
iou_threshold = 0.45
person_only = true
classes = ["person"]

# GPU optimization
use_gpu = true
enable_half_precision = true
gpu_memory_fraction = 0.8

[app.clothing]
# Clothing detection settings
enabled = true
model_type = "advanced"  # "simple" or "advanced"
model_name = "Salesforce/blip-image-captioning-base"
update_frequency = 1.0  # seconds between updates
min_person_confidence = 0.5

# BLIP generation parameters
max_length = 150
num_beams = 3
temperature = 0.9
do_sample = true
top_p = 0.9

[app.server]
# Web server configuration
enabled = true
host = "0.0.0.0"
port = 8000
```

### Configuration Options

#### Model Settings
- **`confidence`**: Minimum confidence for object detection (0.1-1.0)
- **`iou_threshold`**: Intersection over Union threshold for NMS
- **`person_only`**: Limit detection to person class only
- **`classes`**: List of classes to detect

#### GPU Settings
- **`use_gpu`**: Enable GPU acceleration
- **`enable_half_precision`**: Use FP16 for faster inference
- **`gpu_memory_fraction`**: GPU memory usage limit (0.1-1.0)

#### Clothing Detection Settings
- **`model_type`**: "simple" (color analysis) or "advanced" (BLIP AI)
- **`update_frequency`**: Seconds between clothing updates
- **`min_person_confidence`**: Minimum confidence for person detection

#### BLIP Generation Parameters
- **`max_length`**: Maximum description length
- **`temperature`**: Higher = more creative, Lower = more focused
- **`num_beams`**: Beam search paths (lower = more variety)
- **`do_sample`**: Enable random sampling vs. deterministic
- **`top_p`**: Nucleus sampling parameter (higher = more permissive)

## 📡 API Endpoints

### Web Interface
- **`GET /`**: Main web interface
- **`GET /stream`**: MJPEG video stream
- **`GET /latest.jpg`**: Latest captured image

### Data Endpoints
- **`GET /detections`**: Current clothing detections JSON
- **`GET /get_blip_settings`**: Current BLIP configuration
- **`POST /update_blip_settings`**: Update BLIP parameters

### Example API Usage

#### Get Clothing Detections
```bash
curl http://localhost:8000/detections
```

Response:
```json
{
  "timestamp": "2025-01-27T10:30:00Z",
  "detections": [
    {
      "id": 1,
      "bbox": [100, 150, 200, 400],
      "class": "person",
      "confidence": 0.95,
      "description": "a person wearing a blue shirt and black pants",
      "clothing_items": {
        "head": ["dark hair"],
        "upper_body": ["blue shirt"],
        "lower_body": ["black pants"]
      },
      "timestamp": 1737985800.123
    }
  ]
}
```

#### Update BLIP Settings
```bash
curl -X POST http://localhost:8000/update_blip_settings \
  -H "Content-Type: application/json" \
  -d '{
    "max_length": 200,
    "temperature": 0.8,
    "num_beams": 2,
    "top_p": 0.9,
    "do_sample": true
  }'
```

## 🏗️ Project Structure

```
YOLO-Classifier/
├── phonebooth_vision/           # Main application package
│   ├── __init__.py
│   ├── simple_monitor.py        # Main detection loop
│   ├── http_server.py           # Web server and UI
│   ├── clothing_detector.py     # Clothing analysis engine
│   ├── json_server.py           # JSON data server
│   ├── fashionpedia_detector.py # Alternative clothing detector
│   ├── auto_launch.py           # Auto-launch utilities
│   └── config/                  # Configuration management
│       ├── __init__.py
│       ├── manager.py           # Config loading/saving
│       └── models.py            # Pydantic models
├── config.toml                  # Main configuration file
├── requirements.txt             # Python dependencies
├── run.ps1                      # Convenience startup script
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## 🔧 Advanced Configuration

### Custom Model Support
Place custom YOLO models in the project root:
- `yolov8n.pt` (default)
- `yolov8s.pt`
- `yolov8m.pt`
- `yolov8l.pt`
- `yolov8x.pt`

### Clothing Detection Modes

#### Simple Mode
- Uses color and pattern analysis
- Fast, no external models required
- Privacy-preserving
- Basic clothing identification

#### Advanced Mode (BLIP)
- Uses vision-language AI models
- Detailed, natural language descriptions
- Requires internet for model download
- More accurate and comprehensive
- **BLIP Capabilities**: Can detect clothing items, accessories, personal features, colors, materials, and styles
- **Examples**: "blue t-shirt", "black jeans", "beard", "glasses", "long hair", "formal wear", "cotton shirt", "plaid pattern"

### Performance Tuning

#### For High FPS
```toml
[app.model]
confidence = 0.3
iou_threshold = 0.5
enable_half_precision = true
gpu_memory_fraction = 0.9

[app.clothing]
update_frequency = 2.0
```

#### For High Accuracy
```toml
[app.model]
confidence = 0.7
iou_threshold = 0.3

[app.clothing]
max_length = 200
temperature = 0.7
num_beams = 5
```

## 🐛 Troubleshooting

### Common Issues

#### Low FPS
- **Check GPU**: Ensure CUDA is properly installed
- **Reduce confidence**: Lower `confidence` in config.toml
- **Enable half precision**: Set `enable_half_precision = true`
- **Increase update frequency**: Set `update_frequency = 2.0` or higher

#### CUDA Errors
```powershell
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Web Interface Not Loading
- **Check ports**: Ensure port 8000 is not in use
- **Firewall**: Allow Python through Windows Firewall
- **Network**: Check if `0.0.0.0` binding works on your system

#### Clothing Detection Not Working
- **Check internet**: BLIP model requires internet for download
- **Verify config**: Ensure `model_type = "advanced"` in config.toml
- **Check transformers**: `pip install transformers`

#### Memory Issues
- **Reduce batch size**: Lower `gpu_memory_fraction`
- **Use smaller model**: Switch to `yolov8n.pt`
- **Close other applications**: Free up system memory

### Debug Mode
Enable verbose logging by modifying the startup commands:
```powershell
python -m phonebooth_vision.simple_monitor --verbose
python -m phonebooth_vision.http_server --log-level debug
```

### Log Files
- Check console output for error messages
- Monitor GPU memory usage
- Verify file permissions for image writing

## 🔄 Updates and Maintenance

### Updating Dependencies
```powershell
pip install --upgrade -r requirements.txt
```

### Backup Configuration
```powershell
cp config.toml config.toml.backup
```

### Clean Installation
```powershell
# Remove virtual environment
Remove-Item -Recurse -Force .venv

# Recreate and reinstall
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Development Setup
```powershell
# Install development dependencies
pip install -r requirements.txt
pip install pytest ruff black

# Run tests
pytest

# Format code
black phonebooth_vision/
ruff check phonebooth_vision/
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for the object detection model
- **BLIP**: Salesforce for the vision-language model
- **FastAPI**: Modern web framework
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review the configuration options
- Open an issue on GitHub with detailed information

---

**Version**: 1.0.0  
**Last Updated**: January 2025  
**Maintainer**: Phonebooth Vision Team
