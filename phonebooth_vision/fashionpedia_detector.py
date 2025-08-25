#!/usr/bin/env python3
"""Fashionpedia-style clothing detector with attribute vectors."""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, FashionpediaDetector will not work")

logger = logging.getLogger(__name__)


class FashionpediaDetector:
    """Advanced clothing detector using Fashionpedia-style attribute analysis."""
    
    def __init__(self, model_name: str = "microsoft/resnet-50", device: str = "cpu"):
        """Initialize the Fashionpedia detector."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for FashionpediaDetector")
        
        self.device = device
        self.model_name = model_name
        
        # Initialize the feature extractor and model
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            logger.info(f"Initialized FashionpediaDetector with {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Define fashion attributes
        self.fashion_attributes = {
            'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'brown', 'purple', 'pink', 'orange'],
            'patterns': ['solid', 'striped', 'polka_dot', 'floral', 'geometric', 'abstract', 'plaid', 'checkered'],
            'categories': ['shirt', 'pants', 'dress', 'skirt', 'jacket', 'coat', 'sweater', 't-shirt', 'jeans', 'shorts'],
            'sleeve_types': ['short_sleeve', 'long_sleeve', 'sleeveless', 'cap_sleeve'],
            'necklines': ['round_neck', 'v_neck', 'crew_neck', 'scoop_neck', 'square_neck']
        }
    
    def crop_person(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Crop the person from the frame using the bounding box."""
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2]
    
    def segment_person_regions(self, frame: np.ndarray, bbox: List[int]) -> Dict[str, np.ndarray]:
        """Segment a person into different body regions."""
        x1, y1, x2, y2 = bbox
        person_height = y2 - y1
        person_width = x2 - x1
        
        # Define region boundaries
        head_height = int(person_height * 0.15)
        upper_body_height = int(person_height * 0.35)
        lower_body_height = person_height - head_height - upper_body_height
        
        regions = {}
        
        # Head region (top 15%)
        head_y1 = y1
        head_y2 = y1 + head_height
        regions['head'] = frame[head_y1:head_y2, x1:x2]
        
        # Upper body region (15% - 50%)
        upper_y1 = y1 + head_height
        upper_y2 = y1 + head_height + upper_body_height
        regions['upper_body'] = frame[upper_y1:upper_y2, x1:x2]
        
        # Lower body region (50% - bottom)
        lower_y1 = y1 + head_height + upper_body_height
        lower_y2 = y2
        regions['lower_body'] = frame[lower_y1:lower_y2, x1:x2]
        
        # Full body
        regions['full_body'] = frame[y1:y2, x1:x2]
        
        return regions
    
    def extract_fashion_attributes(self, person_image: np.ndarray, region_name: str) -> Dict[str, str]:
        """Extract fashion attributes from a person image region."""
        if person_image.size == 0:
            return {'category': 'unknown', 'color': 'unknown', 'pattern': 'unknown'}
        
        # Resize image for model input
        resized_image = cv2.resize(person_image, (224, 224))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Extract features using the model
        try:
            inputs = self.feature_extractor(images=rgb_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                features = self.model(**inputs)
                feature_vector = features.logits.squeeze().cpu().numpy()
        except Exception as e:
            logger.warning(f"Failed to extract features: {e}")
            return {'category': 'unknown', 'color': 'unknown', 'pattern': 'unknown'}
        
        # Analyze fashion features
        attributes = self._analyze_fashion_features(feature_vector, person_image, region_name)
        
        return attributes
    
    def _analyze_fashion_features(self, features: np.ndarray, image: np.ndarray, region_name: str) -> Dict[str, str]:
        """Analyze fashion features from extracted features and image."""
        attributes = {}
        
        # Analyze color
        attributes['color'] = self._analyze_color(image)
        
        # Analyze pattern
        attributes['pattern'] = self._analyze_pattern(image)
        
        # Analyze category based on region
        attributes['category'] = self._detect_category(features, image)
        
        # Add region-specific attributes
        if region_name == 'head':
            attributes['category'] = self._detect_head_category(image)
        elif region_name == 'lower_body':
            attributes['category'] = self._detect_lower_body_category(image)
        elif region_name == 'upper_body':
            attributes['sleeve_type'] = self._detect_sleeve_type(features, image)
            attributes['neckline'] = self._detect_neckline(features, image)
        
        return attributes
    
    def _analyze_color(self, image: np.ndarray) -> str:
        """Analyze the dominant color in the image."""
        if image.size == 0:
            return 'unknown'
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges (more lenient thresholds)
        color_ranges = {
            'red': [(0, 10, 50, 255), (170, 180, 50, 255)],
            'blue': [(100, 130, 50, 255)],
            'green': [(35, 85, 50, 255)],
            'yellow': [(20, 35, 50, 255)],
            'black': [(0, 180, 0, 50)],
            'white': [(0, 180, 0, 30)],
            'gray': [(0, 180, 0, 80)],
            'brown': [(10, 20, 50, 255)],
            'purple': [(130, 170, 50, 255)],
            'pink': [(140, 170, 50, 255)],
            'orange': [(10, 20, 50, 255)]
        }
        
        # Find dominant color
        max_pixels = 0
        dominant_color = 'unknown'
        
        for color, ranges in color_ranges.items():
            total_pixels = 0
            for lower, upper in ranges:
                if len(lower) == 3:  # HSV tuple
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                else:  # HSV range tuple
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                total_pixels += cv2.countNonZero(mask)
            
            if total_pixels > max_pixels and total_pixels > image.size * 0.03:  # 3% threshold
                max_pixels = total_pixels
                dominant_color = color
        
        # Fallback: use mean HSV values
        if dominant_color == 'unknown':
            mean_hsv = np.mean(hsv, axis=(0, 1))
            hue = mean_hsv[0]
            saturation = mean_hsv[1]
            value = mean_hsv[2]
            
            if value < 50:
                dominant_color = 'black'
            elif value > 200 and saturation < 50:
                dominant_color = 'white'
            elif saturation < 50:
                dominant_color = 'gray'
            elif hue < 10 or hue > 170:
                dominant_color = 'red'
            elif hue < 35:
                dominant_color = 'orange'
            elif hue < 85:
                dominant_color = 'green'
            elif hue < 130:
                dominant_color = 'blue'
            elif hue < 170:
                dominant_color = 'purple'
        
        return dominant_color
    
    def _analyze_pattern(self, image: np.ndarray) -> str:
        """Analyze the pattern in the image."""
        if image.size == 0:
            return 'unknown'
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Pattern classification
        if edge_density < 0.02:
            return 'solid'
        elif edge_density > 0.08:
            # Check for stripes
            horizontal_edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            vertical_edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            
            horizontal_strength = np.mean(np.abs(horizontal_edges))
            vertical_strength = np.mean(np.abs(vertical_edges))
            
            if horizontal_strength > vertical_strength * 1.5:
                return 'striped'
            elif vertical_strength > horizontal_strength * 1.5:
                return 'striped'
            else:
                return 'patterned'
        else:
            return 'patterned'
    
    def _detect_category(self, features: np.ndarray, image: np.ndarray) -> str:
        """Detect clothing category based on features and image analysis."""
        # Simple heuristic based on image aspect ratio
        height, width = image.shape[:2]
        aspect_ratio = width / height if height > 0 else 1
        
        if aspect_ratio > 1.5:  # Wide rectangle
            return 'pants'
        elif aspect_ratio < 0.8:  # Tall rectangle
            return 'dress'
        else:
            return 'shirt'
    
    def _detect_head_category(self, image: np.ndarray) -> str:
        """Detect headwear category."""
        return 'hat'
    
    def _detect_lower_body_category(self, image: np.ndarray) -> str:
        """Detect lower body clothing category."""
        height, width = image.shape[:2]
        aspect_ratio = width / height if height > 0 else 1
        
        if aspect_ratio > 1.2:
            return 'pants'
        else:
            return 'shorts'
    
    def _detect_sleeve_type(self, features: np.ndarray, image: np.ndarray) -> str:
        """Detect sleeve type."""
        return 'short_sleeve'  # Default
    
    def _detect_neckline(self, features: np.ndarray, image: np.ndarray) -> str:
        """Detect neckline type."""
        return 'round_neck'  # Default
    
    def generate_fashionpedia_description(self, attributes: Dict[str, str]) -> str:
        """Generate a concise description from fashion attributes."""
        parts = []
        
        # Add color if not unknown
        if attributes.get('color', 'unknown') != 'unknown':
            parts.append(attributes['color'])
        
        # Add pattern if not solid or unknown
        pattern = attributes.get('pattern', 'unknown')
        if pattern not in ['unknown', 'solid']:
            parts.append(pattern)
        
        # Add category
        category = attributes.get('category', 'clothing')
        if category != 'unknown':
            parts.append(category)
        
        # Add sleeve type for upper body
        if 'sleeve_type' in attributes and attributes['sleeve_type'] != 'unknown':
            parts.append(attributes['sleeve_type'])
        
        # Add neckline for upper body
        if 'neckline' in attributes and attributes['neckline'] != 'round_neck':
            parts.append(attributes['neckline'])
        
        if not parts:
            return 'clothing'
        
        return ' '.join(parts)
    
    def process_detections(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """Process detections and generate clothing descriptions."""
        clothing_items = []
        
        for detection in detections:
            if detection['class'] == 'person':
                bbox = detection['bbox']
                
                # Segment person into regions
                regions = self.segment_person_regions(frame, bbox)
                
                region_descriptions = {}
                
                # Analyze each region
                for region_name, region_image in regions.items():
                    if region_image.size > 0:
                        attributes = self.extract_fashion_attributes(region_image, region_name)
                        description = self.generate_fashionpedia_description(attributes)
                        region_descriptions[region_name] = description
                    else:
                        region_descriptions[region_name] = 'clothing not visible'
                
                # Combine descriptions
                combined_description = self._combine_clothing_descriptions(region_descriptions)
                
                clothing_items.append({
                    'id': detection.get('id', len(clothing_items) + 1),
                    'bbox': bbox,
                    'description': combined_description,
                    'clothing_items': region_descriptions
                })
        
        return {
            'timestamp': detections[0]['timestamp'] if detections else None,
            'detections': clothing_items
        }
    
    def _combine_clothing_descriptions(self, region_descriptions: Dict[str, str]) -> str:
        """Combine regional clothing descriptions into a single description."""
        descriptions = []
        
        for region, desc in region_descriptions.items():
            if desc and desc not in ['clothing not visible', 'clothing']:
                descriptions.append(desc)
        
        if not descriptions:
            return 'clothing detected'
        
        return ', '.join(descriptions)

