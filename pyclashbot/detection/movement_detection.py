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
        
        # Bridge mask to ignore scenery movement
        self.bridge_mask = None
        self.bridge_mask_created = False
        
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
    
    def create_bridge_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a mask to ignore bridge tiles and scenery movement.
        Based on screenshot analysis of 253x384px image.
        
        Args:
            frame: Input frame
            
        Returns:
            Binary mask where 0 = ignore (bridge/scenery), 255 = detect movement
        """
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)  # Start with all ignored
        
        # Reference screenshot dimensions: 253x384px
        ref_width, ref_height = 253, 384
        
        # Calculate scaling factors
        scale_x = width / ref_width
        scale_y = height / ref_height
        
        # Main detection area: 35,30 to 225,280 (normalized from screenshot)
        # This is the only area we want to use for movement detection
        main_x1 = int(35 * scale_x)
        main_y1 = int(30 * scale_y)
        main_x2 = int(225 * scale_x)
        main_y2 = int(280 * scale_y)
        
        # Set main detection area to 255 (detect movement)
        mask[main_y1:main_y2, main_x1:main_x2] = 255
        
        # Areas to ignore (set to 0):
        # 1. 195,145 to 170,250 (bridge area - note: x1 > x2, so this is a vertical strip)
        ignore1_x1 = int(170 * scale_x)
        ignore1_x2 = int(195 * scale_x)
        ignore1_y1 = int(145 * scale_y)
        ignore1_y2 = int(250 * scale_y)
        mask[ignore1_y1:ignore1_y2, ignore1_x1:ignore1_x2] = 0
        
        # 2. 80,155 to 175,170 (horizontal strip)
        ignore2_x1 = int(80 * scale_x)
        ignore2_x2 = int(175 * scale_x)
        ignore2_y1 = int(155 * scale_y)
        ignore2_y2 = int(170 * scale_y)
        mask[ignore2_y1:ignore2_y2, ignore2_x1:ignore2_x2] = 0
        
        # 3. 0,145 to 65,170 (left edge)
        ignore3_x1 = int(0 * scale_x)
        ignore3_x2 = int(65 * scale_x)
        ignore3_y1 = int(145 * scale_y)
        ignore3_y2 = int(170 * scale_y)
        mask[ignore3_y1:ignore3_y2, ignore3_x1:ignore3_x2] = 0
        
        return mask
    
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
        
        # Apply bridge mask to ignore scenery movement
        if not self.bridge_mask_created:
            self.bridge_mask = self.create_bridge_mask(frame)
            self.bridge_mask_created = True
        
        # Apply bridge mask
        thresh = cv2.bitwise_and(thresh, self.bridge_mask)
        
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
        
        # Apply bridge mask to ignore scenery movement
        if not self.bridge_mask_created:
            self.bridge_mask = self.create_bridge_mask(frame)
            self.bridge_mask_created = True
        
        # Apply bridge mask
        fg_mask = cv2.bitwise_and(fg_mask, self.bridge_mask)
        
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
