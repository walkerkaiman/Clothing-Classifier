#!/usr/bin/env python3
"""
Test script to verify clothing detection is working with run.ps1 setup.
This will test the advanced clothing detection system.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
import sys
import time
import requests

# Add the project to the path
sys.path.insert(0, str(Path(__file__).parent))

from phonebooth_vision.clothing_detector import ClothingDetector

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_clothing_detection_with_run_ps1():
    """Test the clothing detection system that should be running via run.ps1."""
    
    logger.info("=== TESTING CLOTHING DETECTION WITH RUN.PS1 SETUP ===")
    
    # Test 1: Check if the servers are running
    logger.info("1. Testing server connectivity...")
    
    try:
        # Test main HTTP server
        response = requests.get("http://localhost:8000/", timeout=5)
        logger.info(f"Main server (port 8000): Status {response.status_code}")
        
        # Test clothing detection server
        response = requests.get("http://localhost:8001/detections", timeout=5)
        logger.info(f"Clothing server (port 8001): Status {response.status_code}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Server connectivity test failed: {e}")
        return
    
    # Test 2: Test the advanced clothing detector directly
    logger.info("2. Testing advanced clothing detector directly...")
    
    try:
        # Initialize the advanced detector (same as in config.toml)
        detector = ClothingDetector(model_name="microsoft/git-base-coco")
        logger.info("Advanced clothing detector initialized successfully!")
        
        # Try to capture a frame from webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return
        
        logger.info("Press 'c' to capture and analyze a frame, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read frame")
                break
            
            # Display the frame
            cv2.imshow('Clothing Detection Test (run.ps1)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                logger.info("=== ANALYZING CURRENT FRAME ===")
                
                # Create a mock person detection (full frame)
                h, w = frame.shape[:2]
                bbox = (0, 0, w, h)  # Full frame as person bbox
                
                # Segment into regions
                regions = detector.segment_person_regions(frame, bbox)
                logger.info(f"Segmented regions: {list(regions.keys())}")
                
                # Analyze each region with advanced detection
                for region_name, region_image in regions.items():
                    logger.info(f"\n--- Analyzing {region_name} ---")
                    
                    # Save the region image for inspection
                    region_filename = f"run_ps1_test_{region_name}.jpg"
                    cv2.imwrite(region_filename, region_image)
                    logger.info(f"Saved {region_name} image to {region_filename}")
                    
                    # Test advanced clothing description
                    start_time = time.time()
                    description = detector.generate_clothing_description(region_image)
                    end_time = time.time()
                    
                    logger.info(f"Advanced description: {description}")
                    logger.info(f"Processing time: {end_time - start_time:.2f} seconds")
                
                logger.info("=== ANALYSIS COMPLETE ===")
        
        cap.release()
        cv2.destroyAllWindows()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure transformers library is installed: pip install transformers")
    except Exception as e:
        logger.error(f"Error in clothing detection test: {e}")
    
    # Test 3: Check configuration
    logger.info("3. Checking configuration...")
    
    try:
        from phonebooth_vision.config.manager import get_settings
        settings = get_settings()
        logger.info(f"Clothing enabled: {settings.clothing.enabled}")
        logger.info(f"Model type: {settings.clothing.model_type}")
        logger.info(f"Model name: {settings.clothing.model_name}")
        logger.info(f"Server type: {settings.server.server_type}")
        logger.info(f"Server port: {settings.server.port}")
    except Exception as e:
        logger.error(f"Configuration check failed: {e}")

if __name__ == "__main__":
    test_clothing_detection_with_run_ps1()
