"""JSON server for real-time clothing detection data.

This module provides a lightweight HTTP server that serves clothing detection
data as JSON to clients on the local network.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logger = logging.getLogger(__name__)


class ClothingDataServer:
    """Server for serving clothing detection data as JSON."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        """Initialize the clothing data server.
        
        Args:
            host: Host to bind to (0.0.0.0 for all interfaces)
            port: Port to serve on
        """
        self.host = host
        self.port = port
        self.app = FastAPI(title="Clothing Detection API")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Current detection data
        self._current_data: Dict[str, Any] = {
            "timestamp": "",
            "detections": []
        }
        self._data_lock = Lock()
        
        # Setup routes
        self._setup_routes()
        
        # Statistics
        self._request_count = 0
        self._last_request_time = time.time()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "service": "Clothing Detection API",
                "version": "1.0.0",
                "endpoints": {
                    "/detections": "Get current clothing detections",
                    "/stats": "Get server statistics",
                    "/health": "Health check"
                }
            }
        
        @self.app.get("/detections")
        async def get_detections():
            """Get current clothing detections."""
            with self._data_lock:
                self._request_count += 1
                self._last_request_time = time.time()
                return self._current_data
        
        @self.app.get("/detections/latest")
        async def get_latest_detections():
            """Get latest clothing detections (alias for /detections)."""
            return await get_detections()
        
        @self.app.get("/stats")
        async def get_stats():
            """Get server statistics."""
            with self._data_lock:
                return {
                    "total_requests": self._request_count,
                    "last_request": datetime.fromtimestamp(self._last_request_time).isoformat(),
                    "uptime": time.time() - self._last_request_time,
                    "current_detections": len(self._current_data.get("detections", [])),
                    "server_time": datetime.now().isoformat()
                }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/config")
        async def get_config():
            """Get current server configuration."""
            return {
                "host": self.host,
                "port": self.port,
                "cors_enabled": True
            }
    
    def update_detections(self, detections: List[Dict[str, Any]]):
        """Update the current detection data.
        
        Args:
            detections: List of detection dictionaries
        """
        with self._data_lock:
            self._current_data = {
                "timestamp": datetime.now().isoformat(),
                "detections": detections
            }
            logger.debug(f"Updated detections: {len(detections)} items")
    
    def start(self, background: bool = False):
        """Start the server.
        
        Args:
            background: If True, start in background thread
        """
        if background:
            import threading
            server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            server_thread.start()
            logger.info(f"Started clothing data server in background on {self.host}:{self.port}")
        else:
            self._run_server()
    
    def _run_server(self):
        """Run the uvicorn server."""
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise


class FileBasedServer:
    """Simple file-based server that writes JSON to disk."""
    
    def __init__(self, output_path: str = "clothing_detections.json"):
        """Initialize the file-based server.
        
        Args:
            output_path: Path to write JSON data
        """
        self.output_path = Path(output_path)
        self._lock = Lock()
        logger.info(f"File-based server initialized, output: {self.output_path}")
    
    def update_detections(self, detections: List[Dict[str, Any]]):
        """Update detection data by writing to file.
        
        Args:
            detections: List of detection dictionaries
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "detections": detections
        }
        
        with self._lock:
            try:
                # Write to temporary file first, then rename
                temp_path = self.output_path.with_suffix(".tmp")
                temp_path.write_text(json.dumps(data, indent=2))
                temp_path.replace(self.output_path)
                logger.debug(f"Updated clothing detections file: {len(detections)} items")
            except Exception as e:
                logger.error(f"Failed to write detections to file: {e}")
    
    def get_latest_data(self) -> Dict[str, Any]:
        """Get the latest data from file.
        
        Returns:
            Latest detection data
        """
        try:
            if self.output_path.exists():
                return json.loads(self.output_path.read_text())
            else:
                return {"timestamp": "", "detections": []}
        except Exception as e:
            logger.error(f"Failed to read detections file: {e}")
            return {"timestamp": "", "detections": []}


# Convenience function to create server based on configuration
def create_server(server_type: str = "http", **kwargs) -> ClothingDataServer | FileBasedServer:
    """Create a server instance based on type.
    
    Args:
        server_type: Type of server ("http" or "file")
        **kwargs: Additional arguments for server initialization
        
    Returns:
        Server instance
    """
    if server_type.lower() == "http":
        return ClothingDataServer(**kwargs)
    elif server_type.lower() == "file":
        return FileBasedServer(**kwargs)
    else:
        raise ValueError(f"Unknown server type: {server_type}")
