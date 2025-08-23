"""Fashionpedia clothing detection module.

This module implements proper Fashionpedia integration for fashion-specific
clothing detection using attribute vectors and the Fashionpedia dataset.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)

# Fashionpedia-specific imports
try:
    from transformers import (
        AutoProcessor, 
        AutoModelForImageClassification,
        AutoFeatureExtractor
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Fashionpedia detection will not work.")


class FashionpediaDetector:
    """Fashionpedia-based clothing detector using attribute vectors."""
    
    def __init__(self, model_name: str = "microsoft/resnet-50", device: Optional[str] = None):
        """Initialize the Fashionpedia detector.
        
        Args:
            model_name: Name of the base model for feature extraction
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for Fashionpedia detection. Install with: pip install transformers")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Fashionpedia attribute categories
        self.fashion_attributes = {
            'category': [
                'shirt', 'pants', 'dress', 'skirt', 'jacket', 'coat', 'sweater', 'hoodie',
                'jeans', 't-shirt', 'blouse', 'vest', 'scarf', 'gloves', 'socks', 'belt',
                'tie', 'bow tie', 'suit', 'blazer', 'polo', 'tank top', 'shorts', 'leggings',
                'jumpsuit', 'overalls', 'hat', 'cap', 'shoes', 'sneakers', 'boots'
            ],
            'color': [
                'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown',
                'white', 'gray', 'black', 'beige', 'navy', 'olive', 'maroon', 'teal'
            ],
            'pattern': [
                'solid', 'striped', 'polka dot', 'floral', 'plaid', 'checkered', 'geometric',
                'abstract', 'animal print', 'camouflage', 'tie dye', 'paisley', 'chevron'
            ],
            'sleeve_type': [
                'sleeveless', 'short sleeve', 'long sleeve', 'three quarter sleeve',
                'bell sleeve', 'puff sleeve', 'raglan sleeve', 'kimono sleeve'
            ],
            'neckline': [
                'round neck', 'v neck', 'scoop neck', 'square neck', 'cowl neck',
                'boat neck', 'halter neck', 'off shoulder', 'one shoulder'
            ],
            'fit': [
                'loose', 'regular', 'tight', 'oversized', 'slim', 'relaxed', 'fitted'
            ],
            'material': [
                'cotton', 'polyester', 'wool', 'silk', 'denim', 'leather', 'suede',
                'linen', 'rayon', 'nylon', 'spandex', 'cashmere', 'velvet', 'satin'
            ]
        }
        
        # Initialize model and processor
        try:
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
            
            logger.info(f"Loaded Fashionpedia detector model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Fashionpedia model {model_name}: {e}")
            raise
    
    def crop_person(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop a person from the frame using the bounding box.
        
        Args:
            frame: Input frame as numpy array
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Cropped person image
        """
        x1, y1, x2, y2 = bbox
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Ensure valid crop dimensions
        if x2 <= x1 or y2 <= y1:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        cropped = frame[y1:y2, x1:x2]
        
        # Resize to standard size for model input
        cropped = cv2.resize(cropped, (224, 224))
        return cropped
    
    def segment_person_regions(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, np.ndarray]:
        """Segment a person into different body regions for detailed fashion analysis.
        
        Args:
            frame: Input frame as numpy array
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Dictionary of region crops: {'upper_body', 'lower_body', 'head', 'full_body'}
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return {}
        
        person_height = y2 - y1
        person_width = x2 - x1
        
        regions = {}
        
        # Full body (original crop)
        regions['full_body'] = cv2.resize(frame[y1:y2, x1:x2], (224, 224))
        
        # Head region (top 25% of person)
        head_height = int(person_height * 0.25)
        head_y1 = y1
        head_y2 = min(y1 + head_height, y2)
        if head_y2 > head_y1:
            regions['head'] = cv2.resize(frame[head_y1:head_y2, x1:x2], (224, 224))
        
        # Upper body (25% to 60% of person height)
        upper_y1 = y1 + int(person_height * 0.25)
        upper_y2 = y1 + int(person_height * 0.60)
        if upper_y2 > upper_y1:
            regions['upper_body'] = cv2.resize(frame[upper_y1:upper_y2, x1:x2], (224, 224))
        
        # Lower body (60% to bottom)
        lower_y1 = y1 + int(person_height * 0.60)
        lower_y2 = y2
        if lower_y2 > lower_y1:
            regions['lower_body'] = cv2.resize(frame[lower_y1:lower_y2, x1:x2], (224, 224))
        
        return regions
    
    def extract_fashion_attributes(self, person_image: np.ndarray, region_name: str = "unknown") -> Dict[str, str]:
        """Extract fashion attributes using Fashionpedia-style analysis.
        
        Args:
            person_image: Cropped person image as numpy array
            region_name: Name of the body region being analyzed
            
        Returns:
            Dictionary of fashion attributes
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
            
            # Extract features using the model
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.logits
            
            # Analyze features to determine attributes with region context
            attributes = self._analyze_fashion_features(features, person_image, region_name)
            
            return attributes
            
        except Exception as e:
            logger.error(f"Error extracting fashion attributes: {e}")
            return {
                'category': 'unknown',
                'color': 'unknown',
                'pattern': 'solid',
                'sleeve_type': 'unknown',
                'neckline': 'unknown',
                'fit': 'regular',
                'material': 'unknown'
            }
    
    def _analyze_fashion_features(self, features: torch.Tensor, image: np.ndarray, region_name: str = "unknown") -> Dict[str, str]:
        """Analyze extracted features to determine fashion attributes.
        
        Args:
            features: Model output features
            image: Original image for additional analysis
            region_name: Name of the body region being analyzed
            
        Returns:
            Dictionary of fashion attributes
        """
        attributes = {}
        
        # Basic color analysis
        attributes['color'] = self._analyze_color(image)
        
        # Pattern analysis
        attributes['pattern'] = self._analyze_pattern(image)
        
        # Region-specific category detection
        if region_name == "head":
            # Head region - likely hat, cap, or hair
            attributes['category'] = self._detect_head_category(image)
        elif region_name == "upper_body":
            # Upper body - shirts, blouses, etc.
            attributes['category'] = self._detect_category(features, image)
            attributes['sleeve_type'] = self._detect_sleeve_type(features, image)
            attributes['neckline'] = self._detect_neckline(features, image)
        elif region_name == "lower_body":
            # Lower body - pants, skirts, etc.
            attributes['category'] = self._detect_lower_body_category(image)
        else:
            # Full body or unknown region
            attributes['category'] = self._detect_category(features, image)
            attributes['sleeve_type'] = self._detect_sleeve_type(features, image)
            attributes['neckline'] = self._detect_neckline(features, image)
        
        # Fit analysis
        attributes['fit'] = self._analyze_fit(features)
        
        # Material estimation (simplified)
        attributes['material'] = self._estimate_material(features)
        
        return attributes
    
    def _analyze_color(self, image: np.ndarray) -> str:
        """Analyze the dominant color in the image with improved accuracy for real-world lighting."""
        # Convert to multiple color spaces for better analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate color statistics first
        mean_hsv = np.mean(hsv, axis=(0, 1))
        std_hsv = np.std(hsv, axis=(0, 1))
        mean_rgb = np.mean(rgb, axis=(0, 1))
        std_rgb = np.std(rgb, axis=(0, 1))
        
        h, s, v = mean_hsv
        r, g, b = mean_rgb
        
        # Primary RGB-based color detection (more reliable for real-world lighting)
        # Check for dominant RGB channels with more lenient thresholds
        max_channel = max(r, g, b)
        min_channel = min(r, g, b)
        channel_diff = max_channel - min_channel
        
        # If there's a clear dominant color (difference > 10 - very lenient for real-world lighting)
        if channel_diff > 10:
            if r > g + 7 and r > b + 7:
                # Red dominant
                if g > b + 15:  # Red + Green = Orange/Yellow
                    if r > g + 25:
                        return "red"
                    else:
                        return "orange"
                elif b > g + 15:  # Red + Blue = Purple/Pink
                    if r > b + 25:
                        return "red"
                    else:
                        return "purple"
                else:
                    return "red"
            elif g > r + 7 and g > b + 7:
                # Green dominant
                if r > b + 7:  # Green + Red = Yellow
                    return "yellow"
                elif b > r + 7:  # Green + Blue = Teal/Cyan
                    return "green"  # Still call it green
                else:
                    return "green"
            elif b > r + 7 and b > g + 7:
                # Blue dominant
                if r > g + 7:  # Blue + Red = Purple
                    return "purple"
                else:
                    return "blue"
        
        # HSV-based detection for more subtle colors
        # Much more lenient thresholds for real-world lighting
        color_ranges = {
            'red': [
                ([0, 50, 50], [10, 255, 255]),       # Red lower - much more lenient
                ([170, 50, 50], [180, 255, 255])     # Red upper - much more lenient
            ],
            'orange': [([10, 50, 50], [25, 255, 255])],
            'yellow': [([25, 50, 50], [35, 255, 255])],
            'green': [([35, 50, 50], [85, 255, 255])],
            'blue': [([85, 50, 50], [130, 255, 255])],
            'purple': [([130, 50, 50], [170, 255, 255])],
            'pink': [([170, 50, 50], [180, 255, 255])],
            'brown': [([10, 30, 30], [20, 255, 200])],
            'white': [([0, 0, 180], [180, 50, 255])],
            'gray': [([0, 0, 80], [180, 50, 180])],
            'black': [([0, 0, 0], [180, 255, 80])]
        }
        
        max_val = 0
        dominant_color = "unknown"
        total_pixels = hsv.shape[0] * hsv.shape[1]
        
        for color_name, ranges in color_ranges.items():
            color_pixels = 0
            
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                color_pixels += cv2.countNonZero(mask)
            
            color_percentage = color_pixels / total_pixels
            
            # Much lower threshold for real-world detection
            if color_percentage > 0.03 and color_pixels > max_val:  # Reduced from 0.08 to 0.03
                max_val = color_pixels
                dominant_color = color_name
        
        # Enhanced fallback based on mean HSV values
        if dominant_color == "unknown":
            # Use mean HSV values for classification
            if v < 80:  # Very dark
                return "black"
            elif v > 200 and s < 50:  # Very bright, low saturation
                return "white"
            elif s < 50 and 80 < v < 180:  # Low saturation, medium brightness
                return "gray"
            elif h < 10 or h > 170:  # Red hues
                return "red"
            elif 10 <= h < 25:  # Orange hues
                return "orange"
            elif 25 <= h < 35:  # Yellow hues
                return "yellow"
            elif 35 <= h < 85:  # Green hues
                return "green"
            elif 85 <= h < 130:  # Blue hues
                return "blue"
            elif 130 <= h < 170:  # Purple hues
                return "purple"
            else:
                return "gray"
        
        return dominant_color
    
    def _analyze_pattern(self, image: np.ndarray) -> str:
        """Analyze the pattern in the image with improved accuracy."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with multiple thresholds
        edges_low = cv2.Canny(blurred, 30, 100)
        edges_high = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge density
        edge_density_low = np.sum(edges_low > 0) / (edges_low.shape[0] * edges_low.shape[1])
        edge_density_high = np.sum(edges_high > 0) / (edges_high.shape[0] * edges_high.shape[1])
        
        # Use the lower threshold for pattern detection
        edge_density = edge_density_low
        
        # Calculate texture variance
        texture_variance = np.var(blurred)
        
        # Pattern classification with improved thresholds for real-world patterns
        if edge_density < 0.02:
            return "solid"
        elif edge_density < 0.06:
            # Check for subtle patterns like bee designs
            if texture_variance > 800:
                # Look for small circular patterns (bees, polka dots, etc.)
                circles = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT, 1, 15,
                    param1=40, param2=25, minRadius=3, maxRadius=20
                )
                
                if circles is not None and len(circles[0]) > 2:
                    return "polka dot"  # Could be bees or similar small patterns
                else:
                    return "textured"
            else:
                return "solid"
        elif edge_density < 0.12:
            # Check for stripes
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(edges_low, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges_low, cv2.MORPH_OPEN, vertical_kernel)
            
            horizontal_density = np.sum(horizontal_lines) / (edges_low.shape[0] * edges_low.shape[1])
            vertical_density = np.sum(vertical_lines) / (edges_low.shape[0] * edges_low.shape[1])
            
            # Check for dominant direction
            if horizontal_density > 0.003 and horizontal_density > vertical_density * 1.5:
                return "striped"
            elif vertical_density > 0.003 and vertical_density > horizontal_density * 1.5:
                return "striped"
            else:
                return "patterned"
        else:
            # High edge density - check for specific patterns
            # Check for polka dots or small patterns
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=5, maxRadius=30
            )
            
            if circles is not None and len(circles[0]) > 3:
                return "polka dot"
            
            # Check for plaid/checkered patterns
            if texture_variance > 1000 and edge_density > 0.2:
                return "plaid"
            
            return "patterned"
    
    def _detect_category(self, features: torch.Tensor, image: np.ndarray) -> str:
        """Detect the clothing category based on features and image analysis."""
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        # Calculate additional features
        area = h * w
        perimeter = 2 * (h + w)
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
        
        # Analyze color distribution
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        value = np.mean(hsv[:, :, 2])
        
        # Region-specific category detection
        if aspect_ratio > 1.8:  # Very wide - likely pants/skirt
            if saturation > 100:  # High saturation suggests jeans
                return "jeans"
            else:
                return "pants"
        elif aspect_ratio > 1.3:  # Wide - likely pants/shorts
            if h < w * 0.6:  # Short height suggests shorts
                return "shorts"
            else:
                return "pants"
        elif aspect_ratio < 0.6:  # Very tall - likely dress
            return "dress"
        elif aspect_ratio < 0.8:  # Tall - likely dress or long shirt
            if value < 100:  # Dark suggests formal wear
                return "dress"
            else:
                return "shirt"
        elif 0.8 <= aspect_ratio <= 1.2:  # Square-ish - likely shirt/top
            if saturation > 120:  # High saturation suggests t-shirt
                return "t-shirt"
            elif compactness > 20:  # High compactness suggests formal shirt
                return "shirt"
            else:
                return "blouse"
        else:  # Default case
            return "shirt"
    
    def _analyze_fit(self, features: torch.Tensor) -> str:
        """Analyze the fit of the clothing item."""
        # Simplified analysis - would need trained model
        return "regular"
    
    def _estimate_material(self, features: torch.Tensor) -> str:
        """Estimate the material of the clothing item."""
        # Simplified estimation - would need trained model
        return "cotton"
    
    def _detect_sleeve_type(self, features: torch.Tensor, image: np.ndarray) -> str:
        """Detect the sleeve type for upper body items."""
        h, w = image.shape[:2]
        
        # Analyze the top portion of the image for sleeve detection
        top_portion = image[:h//3, :]
        
        # Convert to grayscale for edge analysis
        gray = cv2.cvtColor(top_portion, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density in different regions
        left_region = edges[:, :w//3]
        right_region = edges[:, 2*w//3:]
        center_region = edges[:, w//3:2*w//3]
        
        left_density = np.sum(left_region > 0) / left_region.size
        right_density = np.sum(right_region > 0) / right_region.size
        center_density = np.sum(center_region > 0) / center_region.size
        
        # Sleeve type classification based on edge patterns
        if left_density > 0.1 and right_density > 0.1:
            # High edge density on sides suggests sleeves
            if center_density < 0.05:
                return "long sleeve"
            else:
                return "short sleeve"
        elif left_density < 0.05 and right_density < 0.05:
            # Low edge density on sides suggests sleeveless
            return "sleeveless"
        else:
            # Default to short sleeve
            return "short sleeve"
    
    def _detect_neckline(self, features: torch.Tensor, image: np.ndarray) -> str:
        """Detect the neckline type for upper body items."""
        h, w = image.shape[:2]
        
        # Focus on the top portion for neckline analysis
        top_portion = image[:h//4, w//4:3*w//4]
        
        # Convert to grayscale
        gray = cv2.cvtColor(top_portion, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 30, 100)
        
        # Look for horizontal lines (necklines)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Count horizontal lines at different heights
        line_counts = []
        for y in range(0, gray.shape[0], 5):
            line_count = np.sum(horizontal_lines[y:y+5, :] > 0)
            line_counts.append(line_count)
        
        # Analyze line distribution
        if len(line_counts) > 0:
            max_lines = max(line_counts)
            avg_lines = np.mean(line_counts)
            
            if max_lines > 50:  # Strong horizontal line suggests v-neck
                return "v neck"
            elif avg_lines > 20:  # Moderate lines suggest round neck
                return "round neck"
            else:
                return "round neck"  # Default
        else:
            return "round neck"
    
    def _detect_head_category(self, image: np.ndarray) -> str:
        """Detect head accessories like hats, caps, etc."""
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        # Analyze color and pattern
        color = self._analyze_color(image)
        pattern = self._analyze_pattern(image)
        
        # Head accessory classification
        if aspect_ratio > 1.5:  # Wide suggests hat
            if color in ['black', 'gray', 'brown']:
                return "hat"
            else:
                return "cap"
        else:
            return "hat"
    
    def _detect_lower_body_category(self, image: np.ndarray) -> str:
        """Detect lower body clothing like pants, skirts, etc."""
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        # Analyze color and texture
        color = self._analyze_color(image)
        pattern = self._analyze_pattern(image)
        
        # Lower body classification
        if aspect_ratio > 1.8:  # Very wide suggests pants
            if color == 'blue' and pattern == 'solid':
                return "jeans"
            else:
                return "pants"
        elif aspect_ratio > 1.2:  # Wide suggests pants or skirt
            if pattern in ['plaid', 'checkered']:
                return "skirt"
            else:
                return "pants"
        else:
            return "pants"  # Default
    
    def generate_fashionpedia_description(self, attributes: Dict[str, str]) -> str:
        """Generate a Fashionpedia-style description from attributes.
        
        Args:
            attributes: Dictionary of fashion attributes
            
        Returns:
            Fashionpedia-style clothing description
        """
        description_parts = []
        
        # Add color (be more lenient)
        if attributes.get('color') and attributes['color'] != 'unknown':
            description_parts.append(attributes['color'])
        
        # Add pattern (only if not solid)
        if attributes.get('pattern') and attributes['pattern'] != 'solid':
            description_parts.append(attributes['pattern'])
        
        # Add sleeve type for upper body items (only if not unknown)
        if attributes.get('category') in ['shirt', 'blouse', 't-shirt', 'polo']:
            if attributes.get('sleeve_type') and attributes['sleeve_type'] != 'unknown':
                description_parts.append(attributes['sleeve_type'])
        
        # Add neckline for upper body items (only if not round neck - most common)
        if attributes.get('category') in ['shirt', 'blouse', 't-shirt', 'polo']:
            if attributes.get('neckline') and attributes['neckline'] not in ['round neck', 'unknown']:
                description_parts.append(attributes['neckline'])
        
        # Add category
        if attributes.get('category') and attributes['category'] != 'unknown':
            description_parts.append(attributes['category'])
        
        # Add material only if it's distinctive (not cotton)
        if attributes.get('material') and attributes['material'] not in ['cotton', 'unknown']:
            description_parts.append(attributes['material'])
        
        # If we have no description parts, provide a basic fallback
        if not description_parts:
            if attributes.get('color') and attributes['color'] != 'unknown':
                return f"{attributes['color']} clothing"
            else:
                return "clothing"
        
        return " ".join(description_parts)
    
    def process_detections(self, frame: np.ndarray, detections: List[Tuple[Tuple[int, int, int, int], str]]) -> List[Dict[str, Any]]:
        """Process all person detections and generate Fashionpedia-style descriptions.
        
        Args:
            frame: Input frame
            detections: List of (bbox, class_name) tuples from YOLO
            
        Returns:
            List of detection dictionaries with Fashionpedia-style descriptions
        """
        results = []
        
        for i, (bbox, class_name) in enumerate(detections):
            if class_name.lower() == "person":
                # Segment person into body regions
                regions = self.segment_person_regions(frame, bbox)
                
                # Generate detailed fashion descriptions for each region
                clothing_items = {}
                
                for region_name, region_image in regions.items():
                    # Extract Fashionpedia attributes
                    attributes = self.extract_fashion_attributes(region_image, region_name)
                    
                    # Generate description from attributes
                    description = self.generate_fashionpedia_description(attributes)
                    
                    if description and description != "clothing":
                        clothing_items[region_name] = description
                
                # Combine clothing items into a structured description
                clothing_desc = self._combine_clothing_descriptions(clothing_items)
                
                # Create detection result with detailed fashion information
                detection_result = {
                    "id": i + 1,
                    "bbox": list(bbox),
                    "class": class_name,
                    "description": clothing_desc,
                    "clothing_items": clothing_items,  # Detailed breakdown
                    "timestamp": time.time()
                }
                
                results.append(detection_result)
        
        return results
    
    def _combine_clothing_descriptions(self, clothing_items: Dict[str, str]) -> str:
        """Combine individual clothing item descriptions into a coherent description.
        
        Args:
            clothing_items: Dictionary of clothing descriptions by body region
            
        Returns:
            Combined clothing description
        """
        if not clothing_items:
            return "clothing not visible"
        
        # Build description from different regions, avoiding repetition
        description_parts = []
        seen_descriptions = set()
        
        # Priority order: upper body, lower body, head
        regions_order = ['upper_body', 'lower_body', 'head']
        
        for region in regions_order:
            if region in clothing_items:
                desc = clothing_items[region]
                # Only add if we haven't seen this exact description and it's not just "clothing"
                if desc not in seen_descriptions and desc != "clothing" and desc != "clothing not visible":
                    description_parts.append(desc)
                    seen_descriptions.add(desc)
        
        # If we have specific regions, use them; otherwise use full body
        if description_parts:
            return ", ".join(description_parts)
        elif 'full_body' in clothing_items:
            full_body_desc = clothing_items['full_body']
            if full_body_desc != "clothing" and full_body_desc != "clothing not visible":
                return full_body_desc
        
        # Final fallback - try to extract any useful information
        for region, desc in clothing_items.items():
            if desc and desc != "clothing" and desc != "clothing not visible":
                return desc
        
        return "clothing not visible"
