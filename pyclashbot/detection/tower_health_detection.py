"""Tower health detection using OCR and template matching."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract


@dataclass
class TowerHealth:
    """Represents tower health information."""
    left_tower_health: Optional[int] = None
    right_tower_health: Optional[int] = None
    enemy_left_tower_health: Optional[int] = None
    enemy_right_tower_health: Optional[int] = None
    confidence: float = 0.0
    timestamp: float = 0.0


class TowerHealthDetector:
    """Detects tower health using OCR and template matching."""
    
    def __init__(self):
        """Initialize tower health detector."""
        # Tower health regions (normalized coordinates for 419x633 screen)
        self.tower_regions = {
            'own_left': (0.1, 0.4, 0.2, 0.1),    # x, y, width, height
            'own_right': (0.7, 0.4, 0.2, 0.1),
            'enemy_left': (0.1, 0.1, 0.2, 0.1),
            'enemy_right': (0.7, 0.1, 0.2, 0.1)
        }
        
        # OCR configuration
        self.ocr_config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
        
        # Health bar detection parameters
        self.health_bar_color_ranges = {
            'green': ([40, 50, 50], [80, 255, 255]),    # HSV ranges
            'yellow': ([20, 50, 50], [40, 255, 255]),
            'red': ([0, 50, 50], [20, 255, 255])
        }
        
        # Template matching for health bars
        self.health_bar_templates = self._create_health_bar_templates()
    
    def _create_health_bar_templates(self) -> Dict[str, np.ndarray]:
        """Create templates for health bar detection."""
        templates = {}
        
        # Create simple health bar templates
        for health in [100, 75, 50, 25, 0]:
            # Green health bar
            green_bar = np.zeros((10, 100), dtype=np.uint8)
            green_bar[:, :int(health)] = [0, 255, 0]  # Green
            templates[f'green_{health}'] = green_bar
            
            # Red health bar
            red_bar = np.zeros((10, 100), dtype=np.uint8)
            red_bar[:, :int(health)] = [0, 0, 255]  # Red
            templates[f'red_{health}'] = red_bar
        
        return templates
    
    def extract_tower_region(self, frame: np.ndarray, region_name: str) -> np.ndarray:
        """
        Extract tower health region from frame.
        
        Args:
            frame: Input frame
            region_name: Name of the tower region
            
        Returns:
            Extracted region
        """
        if region_name not in self.tower_regions:
            return np.array([])
        
        x, y, w, h = self.tower_regions[region_name]
        height, width = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(x * width)
        y1 = int(y * height)
        x2 = int((x + w) * width)
        y2 = int((y + h) * height)
        
        return frame[y1:y2, x1:x2]
    
    def detect_health_bar(self, region: np.ndarray) -> Optional[float]:
        """
        Detect health bar percentage using color analysis.
        
        Args:
            region: Tower region image
            
        Returns:
            Health percentage (0-100) or None if not detected
        """
        if region.size == 0:
            return None
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Find health bar pixels
        health_pixels = 0
        total_pixels = 0
        
        for color_name, (lower, upper) in self.health_bar_color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            
            mask = cv2.inRange(hsv, lower, upper)
            pixels = cv2.countNonZero(mask)
            
            if color_name == 'green':
                health_pixels += pixels
            elif color_name == 'yellow':
                health_pixels += pixels * 0.5  # Yellow is half health
            elif color_name == 'red':
                health_pixels += pixels * 0.25  # Red is low health
            
            total_pixels += pixels
        
        if total_pixels == 0:
            return None
        
        health_percentage = (health_pixels / total_pixels) * 100
        return min(100.0, max(0.0, health_percentage))
    
    def extract_health_text(self, region: np.ndarray) -> Optional[int]:
        """
        Extract health value using OCR.
        
        Args:
            region: Tower region image
            
        Returns:
            Health value or None if not detected
        """
        if region.size == 0:
            return None
        
        # Preprocess image for OCR
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (white text on dark background)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        try:
            # Extract text using OCR
            text = pytesseract.image_to_string(binary, config=self.ocr_config).strip()
            
            # Extract numbers from text
            numbers = re.findall(r'\d+', text)
            if numbers:
                return int(numbers[0])
        except Exception:
            pass
        
        return None
    
    def detect_tower_health_combined(self, region: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Combine health bar detection and OCR for robust health detection.
        
        Args:
            region: Tower region image
            
        Returns:
            Tuple of (health_value, confidence)
        """
        # Try OCR first
        ocr_health = self.extract_health_text(region)
        
        # Try health bar detection
        bar_health = self.detect_health_bar(region)
        
        # Combine results
        if ocr_health is not None and bar_health is not None:
            # Both methods detected something
            if abs(ocr_health - bar_health) < 20:  # Close values
                return ocr_health, 0.9
            else:
                # Values differ, prefer OCR
                return ocr_health, 0.6
        elif ocr_health is not None:
            return ocr_health, 0.7
        elif bar_health is not None:
            return int(bar_health), 0.5
        else:
            return None, 0.0
    
    def detect_all_tower_health(self, frame: np.ndarray) -> TowerHealth:
        """
        Detect health for all towers in the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            TowerHealth object with detected values
        """
        tower_health = TowerHealth()
        total_confidence = 0.0
        detected_count = 0
        
        # Detect each tower
        for region_name in self.tower_regions:
            region = self.extract_tower_region(frame, region_name)
            health_value, confidence = self.detect_tower_health_combined(region)
            
            if health_value is not None:
                if region_name == 'own_left':
                    tower_health.left_tower_health = health_value
                elif region_name == 'own_right':
                    tower_health.right_tower_health = health_value
                elif region_name == 'enemy_left':
                    tower_health.enemy_left_tower_health = health_value
                elif region_name == 'enemy_right':
                    tower_health.enemy_right_tower_health = health_value
                
                total_confidence += confidence
                detected_count += 1
        
        # Calculate overall confidence
        if detected_count > 0:
            tower_health.confidence = total_confidence / detected_count
        
        tower_health.timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        
        return tower_health
    
    def create_health_visualization(self, frame: np.ndarray, tower_health: TowerHealth) -> np.ndarray:
        """
        Create visualization of detected tower health.
        
        Args:
            frame: Original frame
            tower_health: Detected tower health data
            
        Returns:
            Frame with health overlays
        """
        vis_frame = frame.copy()
        
        # Draw health values on frame
        health_values = [
            (tower_health.left_tower_health, 'own_left', (0, 255, 0)),
            (tower_health.right_tower_health, 'own_right', (0, 255, 0)),
            (tower_health.enemy_left_tower_health, 'enemy_left', (0, 0, 255)),
            (tower_health.enemy_right_tower_health, 'enemy_right', (0, 0, 255))
        ]
        
        for health_value, region_name, color in health_values:
            if health_value is not None:
                x, y, w, h = self.tower_regions[region_name]
                height, width = frame.shape[:2]
                
                # Convert to pixel coordinates
                x1 = int(x * width)
                y1 = int(y * height)
                
                # Draw health text
                text = f"{health_value}%"
                cv2.putText(vis_frame, text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_frame
    
    def update_tower_regions(self, screen_width: int, screen_height: int):
        """
        Update tower regions based on screen dimensions.
        
        Args:
            screen_width: Screen width
            screen_height: Screen height
        """
        # Adjust regions based on screen size
        # This could be made more sophisticated with actual tower detection
        pass


class TowerHealthDetectorConfig:
    """Configuration class for tower health detector."""
    
    def __init__(self):
        self.enable_ocr = True
        self.enable_health_bar_detection = True
        self.ocr_confidence_threshold = 0.5
        self.health_bar_confidence_threshold = 0.3
        self.enable_visualization = True
        
    def update_from_dict(self, config_dict: Dict):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
