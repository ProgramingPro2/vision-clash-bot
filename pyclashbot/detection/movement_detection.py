"""Movement-based unit detection module using frame differencing and connected components."""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class UnitBlob:
    """Represents a detected unit blob with tracking information."""
    centroid: Tuple[int, int]
    area: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    unit_id: Optional[int] = None
    speed_vector: Tuple[float, float] = (0.0, 0.0)
    confidence: float = 1.0
    last_seen: float = 0.0
    movement_history: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.movement_history is None:
            self.movement_history = deque(maxlen=10)
        self.last_seen = time.time()


class MovementDetector:
    """Detects moving units using negative greyscale frame differencing."""
    
    def __init__(self, 
                 min_area: int = 50,
                 max_area: int = 5000,
                 threshold: int = 30,
                 history_length: int = 3):
        """
        Initialize movement detector.
        
        Args:
            min_area: Minimum blob area to consider as a unit
            max_area: Maximum blob area to consider as a unit
            threshold: Threshold for frame differencing
            history_length: Number of frames to keep for differencing
        """
        self.min_area = min_area
        self.max_area = max_area
        self.threshold = threshold
        self.history_length = history_length
        
        # Frame history for differencing
        self.frame_history: deque = deque(maxlen=history_length)
        
        # Connected components parameters
        self.connectivity = 8
        
        # Morphological operations kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Background subtractor for more robust detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for movement detection."""
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def detect_movement_negative_diff(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect movement using negative greyscale frame differencing.
        
        Args:
            frame: Current frame (grayscale)
            
        Returns:
            Binary mask of moving regions
        """
        processed_frame = self.preprocess_frame(frame)
        
        # Add to history
        self.frame_history.append(processed_frame)
        
        if len(self.frame_history) < 2:
            return np.zeros_like(processed_frame)
        
        # Calculate difference between current and previous frame
        diff = cv2.absdiff(self.frame_history[-1], self.frame_history[-2])
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        
        return thresh
    
    def detect_movement_bg_subtractor(self, frame: np.ndarray) -> np.ndarray:
        """
        Alternative movement detection using background subtractor.
        
        Args:
            frame: Current frame
            
        Returns:
            Binary mask of moving regions
        """
        processed_frame = self.preprocess_frame(frame)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(processed_frame)
        
        # Apply morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        
        return fg_mask
    
    def find_connected_components(self, binary_mask: np.ndarray) -> List[UnitBlob]:
        """
        Find connected components and convert to UnitBlob objects.
        
        Args:
            binary_mask: Binary mask of moving regions
            
        Returns:
            List of detected unit blobs
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=self.connectivity
        )
        
        blobs = []
        
        # Process each component (skip background label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by area
            if self.min_area <= area <= self.max_area:
                centroid = (int(centroids[i][0]), int(centroids[i][1]))
                bbox = (
                    stats[i, cv2.CC_STAT_LEFT],
                    stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH],
                    stats[i, cv2.CC_STAT_HEIGHT]
                )
                
                blob = UnitBlob(
                    centroid=centroid,
                    area=area,
                    bbox=bbox
                )
                blobs.append(blob)
        
        return blobs
    
    def detect_units(self, frame: np.ndarray, use_bg_subtractor: bool = False) -> List[UnitBlob]:
        """
        Main method to detect moving units in a frame.
        
        Args:
            frame: Input frame
            use_bg_subtractor: Whether to use background subtractor instead of frame differencing
            
        Returns:
            List of detected unit blobs
        """
        if use_bg_subtractor:
            binary_mask = self.detect_movement_bg_subtractor(frame)
        else:
            binary_mask = self.detect_movement_negative_diff(frame)
        
        blobs = self.find_connected_components(binary_mask)
        
        return blobs
    
    def classify_unit_by_size(self, blob: UnitBlob) -> str:
        """
        Classify unit type based on blob size/area.
        
        Args:
            blob: Unit blob to classify
            
        Returns:
            Unit type classification
        """
        area = blob.area
        
        if area < 100:
            return "small_unit"  # Skeletons, goblins
        elif area < 300:
            return "medium_unit"  # Archers, wizards
        elif area < 800:
            return "large_unit"  # Giants, knights
        else:
            return "building"  # Towers, buildings
    
    def get_detection_visualization(self, frame: np.ndarray, blobs: List[UnitBlob]) -> np.ndarray:
        """
        Create visualization of detected units.
        
        Args:
            frame: Original frame
            blobs: List of detected blobs
            
        Returns:
            Frame with detection overlays
        """
        vis_frame = frame.copy()
        
        for blob in blobs:
            # Draw bounding box
            x, y, w, h = blob.bbox
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw centroid
            cv2.circle(vis_frame, blob.centroid, 3, (0, 0, 255), -1)
            
            # Draw unit ID if available
            if blob.unit_id is not None:
                cv2.putText(vis_frame, f"ID:{blob.unit_id}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw unit classification
            unit_type = self.classify_unit_by_size(blob)
            cv2.putText(vis_frame, unit_type, 
                       (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return vis_frame


class MovementDetectorConfig:
    """Configuration class for movement detector parameters."""
    
    def __init__(self):
        self.min_area = 50
        self.max_area = 5000
        self.threshold = 30
        self.history_length = 3
        self.use_bg_subtractor = False
        self.enable_visualization = True
        
    def update_from_dict(self, config_dict: Dict):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
