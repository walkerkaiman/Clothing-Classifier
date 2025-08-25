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

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


class ClothingDataServer:  # Deprecated; retained for backward compatibility
    pass


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
    
    # Removed unused get_latest_data method per pruning report


# Convenience function to create server based on configuration
def create_server(server_type: str = "file", **kwargs) -> FileBasedServer:
    """Create a server instance based on type.
    
    Args:
        server_type: Type of server ("http" or "file")
        **kwargs: Additional arguments for server initialization
        
    Returns:
        Server instance
    """
    # Only file-based server is supported in current flow
    return FileBasedServer(**kwargs)
