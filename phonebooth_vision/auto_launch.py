#!/usr/bin/env python3
"""Auto-launch script for Phonebooth Vision - runs everything silently."""

import sys
import time
import threading
import webbrowser
import subprocess
from pathlib import Path
from typing import Optional

# Import our modules
from .simple_monitor import main as monitor_main
from .http_server import app
import uvicorn


def open_browser_delayed(url: str, delay: float = 3.0):
    """Open browser after a delay to allow servers to start."""
    time.sleep(delay)
    try:
        webbrowser.open(url)
        print(f"🌐 Opened browser to: {url}")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print(f"📋 Please manually open: {url}")


def start_servers():
    """Start both the main server and clothing detection server."""
    try:
        # Start the main web server
        print("🚀 Starting Phonebooth Vision...")
        print("📷 YOLO object detection + AI clothing analysis")
        print("🌐 Web UI will open automatically in your browser")
        print("=" * 60)
        
        # Start browser opening in background thread
        browser_thread = threading.Thread(
            target=open_browser_delayed, 
            args=("http://localhost:8000", 3.0)
        )
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start the monitor (which includes both servers)
        monitor_main()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Phonebooth Vision...")
    except Exception as e:
        print(f"❌ Error starting Phonebooth Vision: {e}")
        print("Press any key to exit...")
        input()


def main():
    """Main entry point for the auto-launch executable."""
    # Set up logging to file instead of console
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('phonebooth_vision.log'),
            logging.StreamHandler(sys.stdout)  # Still show some output for debugging
        ]
    )
    
    # Start everything
    start_servers()


if __name__ == "__main__":
    main()
