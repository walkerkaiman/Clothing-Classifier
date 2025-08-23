#!/usr/bin/env python3
"""Test script for clothing detection functionality."""

import cv2
import numpy as np
from phonebooth_vision.clothing_detector import SimpleClothingDetector, ClothingDetector


def test_simple_detector():
    """Test the simple clothing detector."""
    print("Testing Simple Clothing Detector...")
    
    # Create a test image (simulate a person)
    test_image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Add some colored regions to simulate clothing
    test_image[50:150, 50:150] = [0, 100, 0]  # Green shirt area
    test_image[150:200, 50:150] = [100, 0, 0]  # Red pants area
    
    detector = SimpleClothingDetector()
    
    # Test color detection
    color = detector.get_dominant_color(test_image)
    print(f"  Dominant color: {color}")
    
    # Test pattern detection
    patterns = detector.detect_patterns(test_image)
    print(f"  Patterns: {patterns}")
    
    # Test full description
    description = detector.generate_simple_description(test_image)
    print(f"  Description: {description}")
    
    # Test with detections
    detections = [((50, 50, 150, 200), "person")]
    results = detector.process_detections(test_image, detections)
    print(f"  Detection results: {results}")
    
    return True


def test_advanced_detector():
    """Test the advanced clothing detector (if transformers available)."""
    print("\nTesting Advanced Clothing Detector...")
    
    try:
        # Create a test image
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        test_image[50:150, 50:150] = [0, 100, 0]  # Green shirt area
        
        # Note: This will download the model on first run
        detector = ClothingDetector(model_name="Salesforce/blip-image-captioning-base")
        
        # Test description generation
        description = detector.generate_clothing_description(test_image)
        print(f"  Description: {description}")
        
        # Test with detections
        detections = [((50, 50, 150, 200), "person")]
        results = detector.process_detections(test_image, detections)
        print(f"  Detection results: {results}")
        
        return True
        
    except Exception as e:
        print(f"  Advanced detector test failed: {e}")
        print("  This is expected if transformers is not installed or no internet connection")
        return False


def test_json_server():
    """Test the JSON server functionality."""
    print("\nTesting JSON Server...")
    
    try:
        from phonebooth_vision.json_server import FileBasedServer
        
        # Test file-based server
        server = FileBasedServer(output_path="test_clothing.json")
        
        test_detections = [
            {
                "id": 1,
                "bbox": [100, 150, 200, 400],
                "class": "person",
                "description": "green shirt with bee design",
                "timestamp": 1234567890.123
            }
        ]
        
        server.update_detections(test_detections)
        print("  File-based server test passed")
        
        # Clean up
        import os
        if os.path.exists("test_clothing.json"):
            os.remove("test_clothing.json")
        
        return True
        
    except Exception as e:
        print(f"  JSON server test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running clothing detection tests...\n")
    
    tests = [
        ("Simple Detector", test_simple_detector),
        ("Advanced Detector", test_advanced_detector),
        ("JSON Server", test_json_server),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("All tests passed! ✅")
    else:
        print("Some tests failed. Check the output above. ❌")
    
    return all_passed


if __name__ == "__main__":
    main()
