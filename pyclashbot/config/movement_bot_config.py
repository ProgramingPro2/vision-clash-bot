"""Configuration settings for movement-based bot."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict

from ..utils.caching import FileCache


@dataclass
class MovementBotSettings:
    """Settings for movement-based bot configuration."""
    
    # File paths
    model_save_path: str = "models/dqn_model.pth"
    training_data_path: str = "data/training_data.pkl"
    config_save_path: str = "config/movement_bot_config.json"
    
    # Movement detection settings
    movement_detection_enabled: bool = True
    min_area: int = 50
    max_area: int = 5000
    threshold: int = 30
    history_length: int = 3
    use_bg_subtractor: bool = False
    enable_movement_visualization: bool = True
    
    # Unit tracking settings
    unit_tracking_enabled: bool = True
    max_distance: float = 50.0
    max_occlusion_frames: int = 5
    min_track_length: int = 3
    enable_prediction: bool = True
    enable_tracking_visualization: bool = True
    
    # Tower health detection settings
    tower_health_enabled: bool = True
    enable_ocr: bool = True
    enable_health_bar_detection: bool = True
    ocr_confidence_threshold: float = 0.5
    health_bar_confidence_threshold: float = 0.3
    enable_tower_health_visualization: bool = True
    
    # DQN settings
    dqn_enabled: bool = True
    state_size: int = 200
    action_size: int = 12
    hidden_sizes: list = field(default_factory=lambda: [512, 256, 128])
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    
    # Performance settings
    target_fps: int = 30
    enable_visualization: bool = True
    save_training_data: bool = True
    auto_save_model: bool = True
    auto_save_interval: int = 1000  # Save every 1000 frames
    
    # Visualization settings
    show_bounding_boxes: bool = True
    show_unit_ids: bool = True
    show_speed_vectors: bool = True
    show_tower_health: bool = True
    show_action_info: bool = True
    show_performance_stats: bool = True
    
    # Training settings
    enable_training: bool = True
    training_frequency: int = 1  # Train every N frames
    reward_scale: float = 1.0
    experience_replay: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'model_save_path': self.model_save_path,
            'training_data_path': self.training_data_path,
            'config_save_path': self.config_save_path,
            'movement_detection_enabled': self.movement_detection_enabled,
            'min_area': self.min_area,
            'max_area': self.max_area,
            'threshold': self.threshold,
            'history_length': self.history_length,
            'use_bg_subtractor': self.use_bg_subtractor,
            'enable_movement_visualization': self.enable_movement_visualization,
            'unit_tracking_enabled': self.unit_tracking_enabled,
            'max_distance': self.max_distance,
            'max_occlusion_frames': self.max_occlusion_frames,
            'min_track_length': self.min_track_length,
            'enable_prediction': self.enable_prediction,
            'enable_tracking_visualization': self.enable_tracking_visualization,
            'tower_health_enabled': self.tower_health_enabled,
            'enable_ocr': self.enable_ocr,
            'enable_health_bar_detection': self.enable_health_bar_detection,
            'ocr_confidence_threshold': self.ocr_confidence_threshold,
            'health_bar_confidence_threshold': self.health_bar_confidence_threshold,
            'enable_tower_health_visualization': self.enable_tower_health_visualization,
            'dqn_enabled': self.dqn_enabled,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_sizes': self.hidden_sizes,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'target_fps': self.target_fps,
            'enable_visualization': self.enable_visualization,
            'save_training_data': self.save_training_data,
            'auto_save_model': self.auto_save_model,
            'auto_save_interval': self.auto_save_interval,
            'show_bounding_boxes': self.show_bounding_boxes,
            'show_unit_ids': self.show_unit_ids,
            'show_speed_vectors': self.show_speed_vectors,
            'show_tower_health': self.show_tower_health,
            'show_action_info': self.show_action_info,
            'show_performance_stats': self.show_performance_stats,
            'enable_training': self.enable_training,
            'training_frequency': self.training_frequency,
            'reward_scale': self.reward_scale,
            'experience_replay': self.experience_replay
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MovementBotSettings':
        """Create settings from dictionary."""
        settings = cls()
        for key, value in data.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        return settings
    
    def save_to_file(self, filepath: str = None):
        """Save settings to JSON file using project's caching system."""
        if filepath is None:
            filepath = self.config_save_path
        
        # Use project's caching system
        cache = FileCache("movement_bot_config.json")
        cache.cache_data(self.to_dict())
    
    @classmethod
    def load_from_file(cls, filepath: str = None) -> 'MovementBotSettings':
        """Load settings from JSON file using project's caching system."""
        if filepath is None:
            filepath = cls().config_save_path
        
        # Use project's caching system
        cache = FileCache("movement_bot_config.json")
        data = cache.load_data()
        
        if data:
            return cls.from_dict(data)
        else:
            # Return default settings if no data exists
            return cls()
    
    def update_from_dict(self, data: Dict[str, Any]):
        """Update settings from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Default configuration instance
DEFAULT_MOVEMENT_BOT_CONFIG = MovementBotSettings()


def get_movement_bot_config() -> MovementBotSettings:
    """Get movement bot configuration, loading from file if available."""
    return MovementBotSettings.load_from_file()


def save_movement_bot_config(config: MovementBotSettings):
    """Save movement bot configuration to file."""
    config.save_to_file()
