#!/usr/bin/env python3
"""
Test script for advanced clothing detection.
This will test the new Fashionpedia-style clothing detection system.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
import sys
import time

# Add the project to the path
sys.path.insert(0, str(Path(__file__).parent))

from phonebooth_vision.clothing_detector import ClothingDetector

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_advanced_clothing_detection():
    """Test the advanced clothing detection system."""
    
    try:
        # Initialize the advanced detector
        logger.info("Initializing advanced clothing detector...")
        detector = ClothingDetector(model_name="Salesforce/blip-image-captioning-base")
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
            cv2.imshow('Advanced Clothing Detection Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                logger.info("=== ANALYZING CURRENT FRAME WITH ADVANCED DETECTOR ===")
                
                # Create a mock person detection (full frame)
                h, w = frame.shape[:2]
                bbox = (0, 0, w, h)  # Full frame as person bbox
                
                # Segment into regions
                regions = detector.segment_person_regions(frame, bbox)
                logger.info(f"Segmented regions: {list(regions.keys())}")
                
                # Analyze each region with advanced detection
                for region_name, region_image in regions.items():
                    logger.info(f"\n--- Analyzing {region_name} with advanced detector ---")
                    
                    # Save the region image for inspection
                    region_filename = f"advanced_debug_{region_name}.jpg"
                    cv2.imwrite(region_filename, region_image)
                    logger.info(f"Saved {region_name} image to {region_filename}")
                    
                    # Test advanced clothing description
                    start_time = time.time()
                    description = detector.generate_clothing_description(region_image)
                    end_time = time.time()
                    
                    logger.info(f"Advanced description: {description}")
                    logger.info(f"Processing time: {end_time - start_time:.2f} seconds")
                
                logger.info("=== ADVANCED ANALYSIS COMPLETE ===")
        
        cap.release()
        cv2.destroyAllWindows()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure transformers library is installed: pip install transformers")
    except Exception as e:
        logger.error(f"Error in advanced clothing detection: {e}")

if __name__ == "__main__":
    test_advanced_clothing_detection()
