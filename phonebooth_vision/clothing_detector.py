"""Clothing description module using vision-language models.

This module provides functionality to generate descriptive clothing labels
for detected persons using image captioning models.
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

# Optional imports for advanced clothing detection
try:
    from transformers import (
        AutoProcessor, 
        AutoModelForVision2Seq,
        BlipProcessor,
        BlipForConditionalGeneration
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Advanced clothing detection will not work.")


class ClothingDetector:
    """Generates descriptive clothing labels for detected persons."""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: Optional[str] = None, config: Optional[dict] = None):
        # Store configuration for BLIP generation parameters
        self.config = config or {}
        
        # Clothing item to body area mapping
        self.clothing_lookup = {
            # Head items
            'head': [
                'hat', 'cap', 'beanie', 'headband', 'scarf', 'bandana', 'helmet', 'crown', 'tiara',
                'headphones', 'earmuffs', 'sunglasses', 'glasses', 'mask', 'face mask', 'balaclava',
                'hood', 'hair', 'wig', 'turban', 'hijab', 'veil', 'bonnet', 'beret',
                'fedora', 'baseball cap', 'cowboy hat', 'top hat', 'beanie hat', 'winter hat'
            ],
            # Upper body items
            'upper_body': [
                'shirt', 't-shirt', 'tshirt', 'blouse', 'sweater', 'jacket', 'coat', 'hoodie',
                'sweatshirt', 'cardigan', 'vest', 'tank top', 'polo', 'dress shirt', 'button down',
                'flannel', 'denim jacket', 'leather jacket', 'blazer', 'suit jacket', 'windbreaker',
                'rain jacket', 'winter coat', 'puffer jacket', 'bomber jacket', 'turtleneck',
                'long sleeve', 'short sleeve', 'sleeveless', 'crop top', 'tube top', 'halter top',
                'bra', 'undershirt', 'thermal', 'fleece', 'pullover', 'jersey', 'uniform'
            ],
            # Lower body items
            'lower_body': [
                'pants', 'jeans', 'trousers', 'shorts', 'skirt', 'dress', 'leggings', 'tights',
                'sweatpants', 'joggers', 'khakis', 'chinos', 'slacks', 'cargo pants', 'overalls',
                'jumpsuit', 'romper', 'culottes', 'capris', 'bermuda shorts', 'athletic shorts',
                'basketball shorts', 'swimming trunks', 'board shorts', 'dress pants', 'suit pants',
                'tuxedo pants', 'formal pants', 'casual pants', 'work pants', 'utility pants'
            ],
            # Footwear
            'feet': [
                'shoes', 'sneakers', 'boots', 'sandals', 'flip flops', 'heels', 'pumps', 'loafers',
                'oxfords', 'mules', 'clogs', 'espadrilles', 'moccasins', 'espadrilles', 'slides',
                'athletic shoes', 'running shoes', 'tennis shoes', 'basketball shoes', 'cleats',
                'dress shoes', 'formal shoes', 'casual shoes', 'work boots', 'hiking boots',
                'winter boots', 'rain boots', 'ankle boots', 'knee high boots', 'thigh high boots'
            ]
        }
        """Initialize the clothing detector with a vision-language model.
        
        Args:
            model_name: Name of the vision-language model to use
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for advanced clothing detection. Install with: pip install transformers")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and processor
        try:
            if "blip" in model_name.lower():
                self.processor = BlipProcessor.from_pretrained(model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
            else:
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(self.device)
            
            logger.info(f"Loaded clothing detection model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
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
        """Segment a person into different body regions for detailed clothing analysis.
        
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
    
    def generate_clothing_description(self, person_image: np.ndarray) -> str:
        """Generate a clothing description for a person image.
        
        Args:
            person_image: Cropped person image as numpy array
            
        Returns:
            Descriptive clothing label
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
            
            # Prepare inputs for the model
            if "blip" in self.model_name.lower():
                # For BLIP, use basic image captioning (works much better than conditional generation)
                inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
                
                # Generate caption without text input - this actually works!
                # Use configuration parameters if available, otherwise use defaults
                max_length = self.config.get('max_length', 100)
                num_beams = self.config.get('num_beams', 5)
                temperature = self.config.get('temperature', 0.7)
                do_sample = self.config.get('do_sample', True)
                top_p = self.config.get('top_p', 0.9)
                
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p
                )
                
                caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            else:
                # Use fashion-specific prompts for better clothing detection
                fashion_prompts = [
                    "What clothing items is this person wearing?",
                    "Describe the person's outfit in detail:",
                    "What clothes can you see in this image?",
                    "List the clothing items visible:"
                ]
                
                # Try multiple prompts and pick the best result
                best_caption = ""
                max_length = 0
                
                for prompt in fashion_prompts:
                    try:
                        inputs = self.processor(pil_image, text=prompt, return_tensors="pt").to(self.device)
                        
                        outputs = self.model.generate(
                            **inputs,
                            max_length=75,
                            num_beams=5,
                            early_stopping=True,
                            do_sample=False,
                            temperature=0.7
                        )
                        
                        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                        
                        # Remove the prompt from the response
                        if prompt.lower() in caption.lower():
                            caption = caption.replace(prompt, "").strip()
                        
                        # Prefer longer, more detailed descriptions
                        if len(caption.split()) > max_length:
                            best_caption = caption
                            max_length = len(caption.split())
                            
                    except Exception as e:
                        logger.debug(f"Prompt failed: {prompt}, error: {e}")
                        continue
                
                caption = best_caption if best_caption else "clothing not visible"
            
            # Clean and enhance the description
            description = self._enhance_clothing_description(caption)
            return description
            
        except Exception as e:
            logger.error(f"Error generating clothing description: {e}")
            return "clothing not visible"
    
    def generate_detailed_clothing_description(self, regions: Dict[str, np.ndarray]) -> Dict[str, str]:
        """Generate detailed clothing descriptions for each body region.
        
        Args:
            regions: Dictionary of body region crops
            
        Returns:
            Dictionary of clothing descriptions for each region
        """
        clothing_items = {}
        
        try:
            # Analyze upper body (shirts, jackets, etc.)
            if 'upper_body' in regions:
                upper_desc = self.generate_clothing_description(regions['upper_body'])
                if upper_desc and upper_desc != "clothing not visible":
                    clothing_items['upper_body'] = upper_desc
            
            # Analyze lower body (pants, skirts, etc.)
            if 'lower_body' in regions:
                lower_desc = self.generate_clothing_description(regions['lower_body'])
                if lower_desc and lower_desc != "clothing not visible":
                    clothing_items['lower_body'] = lower_desc
            
            # Analyze head region (hats, accessories)
            if 'head' in regions:
                head_desc = self.generate_clothing_description(regions['head'])
                if head_desc and head_desc != "clothing not visible":
                    clothing_items['head'] = head_desc
            
            # If no specific regions detected, use full body
            if not clothing_items and 'full_body' in regions:
                full_desc = self.generate_clothing_description(regions['full_body'])
                if full_desc and full_desc != "clothing not visible":
                    clothing_items['full_body'] = full_desc
            
        except Exception as e:
            logger.error(f"Error generating detailed clothing descriptions: {e}")
            # Fallback to full body description
            if 'full_body' in regions:
                clothing_items['full_body'] = self.generate_clothing_description(regions['full_body'])
        
        return clothing_items
    
    def _enhance_clothing_description(self, caption: str) -> str:
        """Enhance and clean the clothing description.
        
        Args:
            caption: Raw caption from the model (e.g., "a man wearing a green hat")
            
        Returns:
            Enhanced clothing description focusing on the clothing items
        """
        # Keep the original caption mostly intact, just do minimal cleaning
        description = caption.strip()
        
        # Only do basic standardization, don't remove content
        fashion_replacements = {
            "tshirt": "t-shirt",
            "t shirt": "t-shirt", 
            "trousers": "pants",
            "cap": "hat",
            "sneakers": "shoes",
            "footwear": "shoes",
        }
        
        for old_term, new_term in fashion_replacements.items():
            description = description.replace(old_term, new_term)
        
        # If no meaningful clothing description, return a fallback
        if not description or len(description.strip()) < 3:
            return "clothing item"
        
        return description.strip()
    
    def _categorize_clothing_items(self, description: str) -> Dict[str, List[str]]:
        """Categorize clothing items from BLIP description into body areas.
        
        Args:
            description: Full BLIP description of clothing
            
        Returns:
            Dictionary mapping body areas to lists of clothing items
        """
        categorized_items = {
            'head': [],
            'upper_body': [],
            'lower_body': [],
            'feet': []
        }
        description_lower = description.lower()
        
        # Extract clothing items from the description
        # Split by common separators and clean up
        import re
        
        # More comprehensive splitting to catch all clothing items
        # Split by commas, semicolons, "and", "with", "wearing", etc.
        items = re.split(r'[,;]|\sand\s|\swith\s|\swearing\s|\shas\s|\salso\s|\sbut\s|\sor\s', description_lower)
        items = [item.strip() for item in items if item.strip()]
        
        # If splitting didn't work well, try to extract clothing items more directly
        if len(items) <= 1:
            # Look for clothing patterns in the full description
            clothing_patterns = [
                r'wearing\s+([^,]+)',  # "wearing a green hat and blue shirt"
                r'has\s+([^,]+)',      # "has a red jacket"
                r'with\s+([^,]+)',     # "with black pants"
            ]
            for pattern in clothing_patterns:
                matches = re.findall(pattern, description_lower)
                if matches:
                    items.extend(matches)
        
        # Process each item and categorize it
        for item in items:
            item = item.strip()
            if not item or len(item) < 3:
                continue
                
            # Find which body area this item belongs to
            for body_area, clothing_list in self.clothing_lookup.items():
                for clothing_item in clothing_list:
                    # Use flexible matching to catch clothing items
                    # Try exact word boundary match first
                    pattern = r'\b' + re.escape(clothing_item) + r'\b'
                    if re.search(pattern, item):
                        # Add item to the list for this body area (don't overwrite)
                        if item not in categorized_items[body_area]:
                            categorized_items[body_area].append(item)
                        break
                    # If no exact match, try partial match for compound terms
                    elif clothing_item in item and len(clothing_item) > 3:
                        # Add item to the list for this body area (don't overwrite)
                        if item not in categorized_items[body_area]:
                            categorized_items[body_area].append(item)
                        break
                else:
                    continue  # Only break inner loop if item was found
                break  # Break outer loop if item was found
        
        # Remove empty lists
        return {k: v for k, v in categorized_items.items() if v}
    
    def process_detections(self, frame: np.ndarray, detections: List[Tuple[Tuple[int, int, int, int], str]]) -> List[Dict[str, Any]]:
        """Process all person detections and generate detailed clothing descriptions.
        
        Args:
            frame: Input frame
            detections: List of (bbox, class_name) tuples from YOLO
            
        Returns:
            List of detection dictionaries with detailed clothing descriptions
        """
        results = []
        
        for i, (bbox, class_name) in enumerate(detections):
            if class_name.lower() == "person":
                # Generate comprehensive clothing description using BLIP on full person area
                full_body_crop = self.crop_person(frame, bbox)
                full_description = self.generate_clothing_description(full_body_crop)
                
                # Categorize ALL clothing items from the BLIP description into body areas
                categorized_items = self._categorize_clothing_items(full_description)
                
                # Create detection result with full description and categorized items
                detection_result = {
                    "id": i + 1,
                    "bbox": list(bbox),
                    "class": class_name,
                    "description": full_description,  # Keep full BLIP description unchanged
                    "clothing_items": categorized_items,  # Categorized by body area (lists of items)
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
        
        # Build description from different regions
        description_parts = []
        
        # Add upper body items
        if 'upper_body' in clothing_items:
            description_parts.append(clothing_items['upper_body'])
        
        # Add lower body items
        if 'lower_body' in clothing_items:
            description_parts.append(clothing_items['lower_body'])
        
        # Add head accessories
        if 'head' in clothing_items:
            description_parts.append(clothing_items['head'])
        
        # If we have specific regions, use them; otherwise use full body
        if description_parts:
            return ", ".join(description_parts)
        elif 'full_body' in clothing_items:
            return clothing_items['full_body']
        else:
            return "clothing not visible"


class SimpleClothingDetector:
    """Simplified clothing detector using basic color and pattern analysis."""
    
    def __init__(self):
        """Initialize the simple clothing detector."""
        # Improved HSV color ranges with better separation
        self.color_names = {
            'red': ([0, 100, 100], [10, 255, 255]),      # Lower saturation threshold
            'orange': ([10, 100, 100], [25, 255, 255]),
            'yellow': ([25, 100, 100], [35, 255, 255]),
            'green': ([35, 100, 100], [85, 255, 255]),
            'blue': ([85, 100, 100], [130, 255, 255]),
            'purple': ([130, 100, 100], [170, 255, 255]),
            'pink': ([170, 100, 100], [180, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'gray': ([0, 0, 100], [180, 30, 200]),
            'black': ([0, 0, 0], [180, 255, 30]),
            'brown': ([10, 100, 50], [20, 255, 200])     # Better brown detection
        }
    
    def get_dominant_color(self, image: np.ndarray) -> str:
        """Get the dominant color in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dominant color name
        """
        # Ensure image is large enough for accurate color detection
        if image.shape[0] < 100 or image.shape[1] < 100:
            # Resize to minimum size for better color detection
            image = cv2.resize(image, (224, 224))
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apply some preprocessing to improve color detection
        # Blur slightly to reduce noise
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # Find dominant color
        max_val = 0
        dominant_color = "unknown"
        color_counts = {}
        
        for color_name, (lower, upper) in self.color_names.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            color_pixels = cv2.countNonZero(mask)
            color_counts[color_name] = color_pixels
            
            if color_pixels > max_val:
                max_val = color_pixels
                dominant_color = color_name
        
        # Only return a color if it has a significant presence
        total_pixels = image.shape[0] * image.shape[1]
        color_percentage = max_val / total_pixels
        
        if color_percentage < 0.1:  # Less than 10% of pixels
            dominant_color = "unknown"
            logger.debug(f"Color percentage too low: {color_percentage:.3f}")
        
        # Debug logging
        logger.debug(f"Color detection results: {color_counts}")
        logger.debug(f"Dominant color: {dominant_color} with {max_val} pixels ({color_percentage:.3f})")
        
        return dominant_color
    
    def detect_patterns(self, image: np.ndarray) -> List[str]:
        """Detect basic patterns in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for pattern analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # More conservative threshold for pattern detection
        if edge_density > 0.15:  # Increased from 0.1
            patterns.append("patterned")
            logger.debug(f"Pattern detected: edge_density={edge_density:.3f}")
        
        # Check for stripes (horizontal lines) - more conservative
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_line_density = np.sum(horizontal_lines) / (edges.shape[0] * edges.shape[1])
        
        if horizontal_line_density > 0.02:  # More conservative threshold
            patterns.append("striped")
            logger.debug(f"Stripes detected: horizontal_line_density={horizontal_line_density:.3f}")
        
        return patterns
    
    def generate_simple_description(self, person_image: np.ndarray) -> str:
        """Generate a simple clothing description based on color and patterns.
        
        Args:
            person_image: Cropped person image
            
        Returns:
            Simple clothing description
        """
        try:
            # Get dominant color
            color = self.get_dominant_color(person_image)
            logger.debug(f"Detected color: {color}")
            
            # Detect patterns
            patterns = self.detect_patterns(person_image)
            logger.debug(f"Detected patterns: {patterns}")
            
            # Build description
            description_parts = []
            
            if patterns:
                description_parts.extend(patterns)
            
            if color != "unknown":
                description_parts.append(color)
            
            description_parts.append("clothing")
            
            final_description = " ".join(description_parts)
            logger.debug(f"Final description: {final_description}")
            
            return final_description
            
        except Exception as e:
            logger.error(f"Error in simple clothing detection: {e}")
            return "casual clothing"
    
    def process_detections(self, frame: np.ndarray, detections: List[Tuple[Tuple[int, int, int, int], str]]) -> List[Dict[str, Any]]:
        """Process all person detections with simple clothing analysis.
        
        Args:
            frame: Input frame
            detections: List of (bbox, class_name) tuples from YOLO
            
        Returns:
            List of detection dictionaries with clothing descriptions
        """
        results = []
        
        for i, (bbox, class_name) in enumerate(detections):
            if class_name.lower() == "person":
                # Segment person into body regions
                regions = self.segment_person_regions(frame, bbox)
                
                # Generate detailed clothing descriptions for each region
                clothing_items = self.generate_detailed_simple_description(regions)
                
                # Combine clothing items into a structured description
                clothing_desc = self._combine_clothing_descriptions(clothing_items)
                
                # Create detection result with detailed clothing information
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
    
    def segment_person_regions(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, np.ndarray]:
        """Segment a person into different body regions for detailed clothing analysis.
        
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
    
    def generate_detailed_simple_description(self, regions: Dict[str, np.ndarray]) -> Dict[str, str]:
        """Generate detailed clothing descriptions for each body region using simple analysis.
        
        Args:
            regions: Dictionary of body region crops
            
        Returns:
            Dictionary of clothing descriptions for each region
        """
        clothing_items = {}
        
        try:
            # Analyze upper body (shirts, jackets, etc.)
            if 'upper_body' in regions:
                upper_desc = self.generate_simple_description(regions['upper_body'])
                if upper_desc and upper_desc != "casual clothing":
                    clothing_items['upper_body'] = upper_desc
            
            # Analyze lower body (pants, skirts, etc.)
            if 'lower_body' in regions:
                lower_desc = self.generate_simple_description(regions['lower_body'])
                if lower_desc and lower_desc != "casual clothing":
                    clothing_items['lower_body'] = lower_desc
            
            # Analyze head region (hats, accessories)
            if 'head' in regions:
                head_desc = self.generate_simple_description(regions['head'])
                if head_desc and head_desc != "casual clothing":
                    clothing_items['head'] = head_desc
            
            # If no specific regions detected, use full body
            if not clothing_items and 'full_body' in regions:
                full_desc = self.generate_simple_description(regions['full_body'])
                if full_desc and full_desc != "casual clothing":
                    clothing_items['full_body'] = full_desc
            
        except Exception as e:
            logger.error(f"Error generating detailed simple clothing descriptions: {e}")
            # Fallback to full body description
            if 'full_body' in regions:
                clothing_items['full_body'] = self.generate_simple_description(regions['full_body'])
        
        return clothing_items
    
    def _combine_clothing_descriptions(self, clothing_items: Dict[str, str]) -> str:
        """Combine individual clothing item descriptions into a coherent description.
        
        Args:
            clothing_items: Dictionary of clothing descriptions by body region
            
        Returns:
            Combined clothing description
        """
        if not clothing_items:
            return "casual clothing"
        
        # Build description from different regions
        description_parts = []
        
        # Add upper body items
        if 'upper_body' in clothing_items:
            description_parts.append(clothing_items['upper_body'])
        
        # Add lower body items
        if 'lower_body' in clothing_items:
            description_parts.append(clothing_items['lower_body'])
        
        # Add head accessories
        if 'head' in clothing_items:
            description_parts.append(clothing_items['head'])
        
        # If we have specific regions, use them; otherwise use full body
        if description_parts:
            return ", ".join(description_parts)
        elif 'full_body' in clothing_items:
            return clothing_items['full_body']
        else:
            return "casual clothing"
