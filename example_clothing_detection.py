#!/usr/bin/env python3
"""Example script demonstrating clothing detection functionality."""

import cv2
import time
from phonebooth_vision.clothing_detector import SimpleClothingDetector, ClothingDetector
from phonebooth_vision.json_server import create_server


def example_simple_detection():
    """Example using simple clothing detection."""
    print("=== Simple Clothing Detection Example ===")
    
    # Initialize detector
    detector = SimpleClothingDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Simulate person detection (in real usage, this comes from YOLO)
            # For demo, we'll use the entire frame as a "person"
            h, w = frame.shape[:2]
            bbox = (0, 0, w, h)
            
            # Generate clothing description
            description = detector.generate_simple_description(frame)
            
            # Display results
            cv2.putText(frame, f"Clothing: {description}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            cv2.imshow('Simple Clothing Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"Saved frame with description: {description}")
                cv2.imwrite('clothing_example.jpg', frame)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def example_with_server():
    """Example using clothing detection with JSON server."""
    print("=== Clothing Detection with JSON Server Example ===")
    
    # Initialize detector and server
    detector = SimpleClothingDetector()
    server = create_server(server_type="file", output_path="example_clothing.json")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    print("Check 'example_clothing.json' for real-time updates")
    
    last_update = 0
    update_interval = 2.0  # Update every 2 seconds
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            
            # Update clothing detections periodically
            if current_time - last_update >= update_interval:
                # Simulate person detection
                h, w = frame.shape[:2]
                detections = [((0, 0, w, h), "person")]
                
                # Process detections
                results = detector.process_detections(frame, detections)
                
                # Update server
                server.update_detections(results)
                
                print(f"Updated detections: {len(results)} persons")
                for result in results:
                    print(f"  - {result['description']}")
                
                last_update = current_time
            
            # Display frame
            cv2.imshow('Clothing Detection with Server', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def example_advanced_detection():
    """Example using advanced clothing detection (requires transformers)."""
    print("=== Advanced Clothing Detection Example ===")
    print("Note: This requires internet connection for model download")
    
    try:
        # Initialize advanced detector
        detector = ClothingDetector(model_name="Salesforce/blip-image-captioning-base")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Simulate person detection
                h, w = frame.shape[:2]
                bbox = (0, 0, w, h)
                
                # Generate advanced clothing description
                description = detector.generate_clothing_description(frame)
                
                # Display results
                cv2.putText(frame, f"Advanced: {description}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                
                cv2.imshow('Advanced Clothing Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Advanced detection failed: {e}")
        print("Make sure transformers is installed and you have internet connection")


def main():
    """Run examples based on user choice."""
    print("Clothing Detection Examples")
    print("=" * 30)
    print("1. Simple detection (webcam)")
    print("2. Simple detection with JSON server")
    print("3. Advanced detection (requires transformers)")
    print("4. Run all examples")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        example_simple_detection()
    elif choice == "2":
        example_with_server()
    elif choice == "3":
        example_advanced_detection()
    elif choice == "4":
        example_simple_detection()
        print("\n" + "="*50 + "\n")
        example_with_server()
        print("\n" + "="*50 + "\n")
        example_advanced_detection()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
