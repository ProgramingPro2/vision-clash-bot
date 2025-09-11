"""Unit tracking system with centroid-based tracking and persistent unit IDs."""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .movement_detection import UnitBlob


@dataclass
class TrackedUnit:
    """Represents a tracked unit with persistent ID and movement history."""
    unit_id: int
    centroid: Tuple[int, int]
    area: int
    bbox: Tuple[int, int, int, int]
    speed_vector: Tuple[float, float] = (0.0, 0.0)
    confidence: float = 1.0
    last_seen: float = field(default_factory=time.time)
    movement_history: deque = field(default_factory=lambda: deque(maxlen=20))
    occlusion_count: int = 0
    max_occlusion_frames: int = 5
    unit_type: str = "unknown"
    
    def update(self, blob: UnitBlob):
        """Update tracked unit with new blob data."""
        # Calculate speed vector
        if len(self.movement_history) > 0:
            prev_centroid = self.movement_history[-1]
            dx = blob.centroid[0] - prev_centroid[0]
            dy = blob.centroid[1] - prev_centroid[1]
            self.speed_vector = (dx, dy)
        
        # Update properties
        self.centroid = blob.centroid
        self.area = blob.area
        self.bbox = blob.bbox
        self.confidence = blob.confidence
        self.last_seen = time.time()
        self.occlusion_count = 0
        
        # Add to movement history
        self.movement_history.append(blob.centroid)
    
    def predict_next_position(self) -> Tuple[int, int]:
        """Predict next position based on speed vector."""
        if len(self.movement_history) < 2:
            return self.centroid
        
        # Simple linear prediction
        predicted_x = int(self.centroid[0] + self.speed_vector[0])
        predicted_y = int(self.centroid[1] + self.speed_vector[1])
        
        return (predicted_x, predicted_y)
    
    def is_occluded(self) -> bool:
        """Check if unit is currently occluded."""
        return self.occlusion_count > 0
    
    def increment_occlusion(self):
        """Increment occlusion counter."""
        self.occlusion_count += 1
    
    def is_lost(self) -> bool:
        """Check if unit should be considered lost."""
        return self.occlusion_count >= self.max_occlusion_frames


class UnitTracker:
    """Tracks units across frames using centroid-based distance matching."""
    
    def __init__(self, 
                 max_distance: float = 50.0,
                 max_occlusion_frames: int = 5,
                 min_track_length: int = 3):
        """
        Initialize unit tracker.
        
        Args:
            max_distance: Maximum distance for matching blobs to tracks
            max_occlusion_frames: Maximum frames a unit can be occluded
            min_track_length: Minimum track length before considering valid
        """
        self.max_distance = max_distance
        self.max_occlusion_frames = max_occlusion_frames
        self.min_track_length = min_track_length
        
        # Track management
        self.tracked_units: Dict[int, TrackedUnit] = {}
        self.next_unit_id = 1
        self.lost_units: List[TrackedUnit] = []
        
        # Performance tracking
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_lost = 0
    
    def calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def match_blobs_to_tracks(self, blobs: List[UnitBlob]) -> Tuple[Dict[int, UnitBlob], List[UnitBlob]]:
        """
        Match detected blobs to existing tracks using distance-based assignment.
        
        Args:
            blobs: List of detected blobs
            
        Returns:
            Tuple of (matched_tracks, unmatched_blobs)
        """
        matched_tracks = {}
        unmatched_blobs = blobs.copy()
        
        # Create distance matrix
        if not self.tracked_units or not blobs:
            return matched_tracks, unmatched_blobs
        
        # Calculate distances between all tracks and blobs
        distances = []
        for track_id, track in self.tracked_units.items():
            if track.is_lost():
                continue
                
            predicted_pos = track.predict_next_position()
            for i, blob in enumerate(blobs):
                distance = self.calculate_distance(predicted_pos, blob.centroid)
                distances.append((distance, track_id, i))
        
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        
        # Assign matches greedily
        used_tracks = set()
        used_blobs = set()
        
        for distance, track_id, blob_idx in distances:
            if distance > self.max_distance:
                break
                
            if track_id not in used_tracks and blob_idx not in used_blobs:
                matched_tracks[track_id] = blobs[blob_idx]
                used_tracks.add(track_id)
                used_blobs.add(blob_idx)
        
        # Remove matched blobs from unmatched list
        unmatched_blobs = [blob for i, blob in enumerate(blobs) if i not in used_blobs]
        
        return matched_tracks, unmatched_blobs
    
    def update_tracks(self, matched_tracks: Dict[int, UnitBlob]):
        """Update existing tracks with matched blobs."""
        for track_id, blob in matched_tracks.items():
            if track_id in self.tracked_units:
                self.tracked_units[track_id].update(blob)
    
    def create_new_tracks(self, unmatched_blobs: List[UnitBlob]):
        """Create new tracks for unmatched blobs."""
        for blob in unmatched_blobs:
            new_track = TrackedUnit(
                unit_id=self.next_unit_id,
                centroid=blob.centroid,
                area=blob.area,
                bbox=blob.bbox,
                max_occlusion_frames=self.max_occlusion_frames
            )
            new_track.update(blob)
            
            self.tracked_units[self.next_unit_id] = new_track
            self.next_unit_id += 1
            self.total_tracks_created += 1
    
    def handle_occlusions(self):
        """Handle occluded tracks and remove lost ones."""
        tracks_to_remove = []
        
        for track_id, track in self.tracked_units.items():
            if track_id not in [t.unit_id for t in self.tracked_units.values()]:
                continue
                
            # Increment occlusion for unmatched tracks
            if not any(track_id == t.unit_id for t in self.tracked_units.values()):
                track.increment_occlusion()
            
            # Remove lost tracks
            if track.is_lost():
                if len(track.movement_history) >= self.min_track_length:
                    self.lost_units.append(track)
                tracks_to_remove.append(track_id)
                self.total_tracks_lost += 1
        
        # Remove lost tracks
        for track_id in tracks_to_remove:
            del self.tracked_units[track_id]
    
    def track_units(self, blobs: List[UnitBlob]) -> List[TrackedUnit]:
        """
        Main tracking method that processes new blobs and updates tracks.
        
        Args:
            blobs: List of detected blobs from current frame
            
        Returns:
            List of currently tracked units
        """
        self.frame_count += 1
        
        # Match blobs to existing tracks
        matched_tracks, unmatched_blobs = self.match_blobs_to_tracks(blobs)
        
        # Update matched tracks
        self.update_tracks(matched_tracks)
        
        # Create new tracks for unmatched blobs
        self.create_new_tracks(unmatched_blobs)
        
        # Handle occlusions
        self.handle_occlusions()
        
        # Return active tracks
        return list(self.tracked_units.values())
    
    def get_tracking_stats(self) -> Dict:
        """Get tracking performance statistics."""
        return {
            "frame_count": self.frame_count,
            "active_tracks": len(self.tracked_units),
            "total_tracks_created": self.total_tracks_created,
            "total_tracks_lost": self.total_tracks_lost,
            "lost_units": len(self.lost_units)
        }
    
    def get_visualization_data(self, tracked_units: List[TrackedUnit]) -> Dict:
        """
        Get data for visualization of tracked units.
        
        Args:
            tracked_units: List of tracked units
            
        Returns:
            Dictionary with visualization data
        """
        vis_data = {
            "bounding_boxes": [],
            "centroids": [],
            "unit_ids": [],
            "speed_vectors": [],
            "unit_types": [],
            "occlusion_status": []
        }
        
        for unit in tracked_units:
            vis_data["bounding_boxes"].append(unit.bbox)
            vis_data["centroids"].append(unit.centroid)
            vis_data["unit_ids"].append(unit.unit_id)
            vis_data["speed_vectors"].append(unit.speed_vector)
            vis_data["unit_types"].append(unit.unit_type)
            vis_data["occlusion_status"].append(unit.is_occluded())
        
        return vis_data
    
    def reset(self):
        """Reset the tracker state."""
        self.tracked_units.clear()
        self.lost_units.clear()
        self.next_unit_id = 1
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_lost = 0


class UnitTrackerConfig:
    """Configuration class for unit tracker parameters."""
    
    def __init__(self):
        self.max_distance = 50.0
        self.max_occlusion_frames = 5
        self.min_track_length = 3
        self.enable_prediction = True
        self.enable_visualization = True
        
    def update_from_dict(self, config_dict: Dict):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
