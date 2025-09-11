"""Main integration module for movement-based unit tracking and DQN decision making."""

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..detection.movement_detection import MovementDetector, MovementDetectorConfig
from ..detection.tower_health_detection import TowerHealthDetector, TowerHealthDetectorConfig
from ..detection.unit_tracking import TrackedUnit, UnitTracker, UnitTrackerConfig
from ..utils.logger import Logger
from .dqn_agent import DQNAgent, GameAction, GameReward, GameState, GameStateProcessor


@dataclass
class BotConfig:
    """Configuration for the movement-based bot."""
    # Movement detection
    movement_detection_enabled: bool = True
    movement_detection_config: MovementDetectorConfig = None
    
    # Unit tracking
    unit_tracking_enabled: bool = True
    unit_tracking_config: UnitTrackerConfig = None
    
    # Tower health detection
    tower_health_enabled: bool = True
    tower_health_config: TowerHealthDetectorConfig = None
    
    # DQN agent
    dqn_enabled: bool = True
    dqn_model_path: str = "models/dqn_model.pth"
    dqn_state_size: int = 200  # Will be calculated dynamically
    dqn_action_size: int = 12  # 4 cards * 3 positions each
    
    # Performance
    target_fps: int = 30
    enable_visualization: bool = True
    save_training_data: bool = True
    
    def __post_init__(self):
        if self.movement_detection_config is None:
            self.movement_detection_config = MovementDetectorConfig()
        if self.unit_tracking_config is None:
            self.unit_tracking_config = UnitTrackerConfig()
        if self.tower_health_config is None:
            self.tower_health_config = TowerHealthDetectorConfig()


class MovementBasedBot:
    """Main bot class integrating movement detection, unit tracking, and DQN decision making."""
    
    def __init__(self, config: BotConfig, logger: Logger):
        """
        Initialize movement-based bot.
        
        Args:
            config: Bot configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.movement_detector = None
        self.unit_tracker = None
        self.tower_health_detector = None
        self.dqn_agent = None
        self.state_processor = None
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        self.last_frame_time = time.time()
        
        # Game state tracking
        self.current_game_state = None
        self.previous_game_state = None
        self.last_action = None
        self.last_action_time = 0
        
        # Training data collection
        self.training_data = []
        self.episode_data = []
        
        # Threading for real-time processing
        self.processing_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.running = False
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all bot components."""
        try:
            # Movement detector
            if self.config.movement_detection_enabled:
                self.movement_detector = MovementDetector(
                    min_area=self.config.movement_detection_config.min_area,
                    max_area=self.config.movement_detection_config.max_area,
                    threshold=self.config.movement_detection_config.threshold,
                    history_length=self.config.movement_detection_config.history_length
                )
                self.logger.log("Movement detector initialized")
            
            # Unit tracker
            if self.config.unit_tracking_enabled:
                self.unit_tracker = UnitTracker(
                    max_distance=self.config.unit_tracking_config.max_distance,
                    max_occlusion_frames=self.config.unit_tracking_config.max_occlusion_frames,
                    min_track_length=self.config.unit_tracking_config.min_track_length
                )
                self.logger.log("Unit tracker initialized")
            
            # Tower health detector
            if self.config.tower_health_enabled:
                self.tower_health_detector = TowerHealthDetector()
                self.logger.log("Tower health detector initialized")
            
            # DQN agent
            if self.config.dqn_enabled:
                self.state_processor = GameStateProcessor()
                self.dqn_agent = DQNAgent(
                    state_size=self.config.dqn_state_size,
                    action_size=self.config.dqn_action_size
                )
                
                # Load existing model if available
                if os.path.exists(self.config.dqn_model_path):
                    self.dqn_agent.load_model(self.config.dqn_model_path)
                    self.logger.log(f"Loaded DQN model from {self.config.dqn_model_path}")
                else:
                    self.logger.log("Starting with new DQN model")
            
            self.logger.log("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            print(f"[ERROR] Failed to initialize movement-based bot components: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame and return bot decisions.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with processing results and decisions
        """
        start_time = time.time()
        self.frame_count += 1
        
        results = {
            'frame_count': self.frame_count,
            'processing_time': 0,
            'detected_units': [],
            'tracked_units': [],
            'tower_health': None,
            'game_state': None,
            'action': None,
            'visualization': None
        }
        
        try:
            # Movement detection
            detected_blobs = []
            if self.movement_detector:
                detected_blobs = self.movement_detector.detect_units(frame)
                results['detected_units'] = detected_blobs
            
            # Unit tracking
            tracked_units = []
            if self.unit_tracker and detected_blobs:
                tracked_units = self.unit_tracker.track_units(detected_blobs)
                results['tracked_units'] = tracked_units
            
            # Tower health detection
            tower_health = None
            if self.tower_health_detector:
                tower_health = self.tower_health_detector.detect_all_tower_health(frame)
                results['tower_health'] = tower_health
            
            # Game state processing
            game_state = None
            if self.dqn_agent and tracked_units and tower_health:
                # Get elixir count (placeholder - would need actual detection)
                elixir_count = 5.0  # Placeholder
                
                # Get time remaining (placeholder)
                time_remaining = 120.0  # Placeholder
                
                # Get card availability (placeholder)
                card_availability = [True, True, True, True]  # Placeholder
                
                # Process game state
                game_state = self.state_processor.process_game_state(
                    tracked_units=tracked_units,
                    elixir_count=elixir_count,
                    tower_health=[
                        tower_health.left_tower_health or 100,
                        tower_health.right_tower_health or 100,
                        tower_health.enemy_left_tower_health or 100,
                        tower_health.enemy_right_tower_health or 100
                    ],
                    time_remaining=time_remaining,
                    card_availability=card_availability
                )
                results['game_state'] = game_state
            
            # DQN decision making
            action = None
            if self.dqn_agent and game_state:
                # Select action
                action_index = self.dqn_agent.select_action(game_state)
                
                # Convert action index to card and position
                card_index = action_index % 4
                position_index = action_index // 4
                
                # Map position index to actual coordinates
                positions = [(0.2, 0.5), (0.5, 0.5), (0.8, 0.5)]  # Left, center, right
                if position_index < len(positions):
                    position = positions[position_index]
                else:
                    position = (0.5, 0.5)  # Default center
                
                action = {
                    'card_index': card_index,
                    'position': position,
                    'action_index': action_index,
                    'timestamp': time.time()
                }
                results['action'] = action
                
                # Store for training
                if self.config.save_training_data:
                    self._store_training_data(game_state, action)
            
            # Create visualization
            if self.config.enable_visualization:
                vis_frame = self._create_visualization(frame, results)
                results['visualization'] = vis_frame
            
            # Update previous state
            self.previous_game_state = self.current_game_state
            self.current_game_state = game_state
            self.last_action = action
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            print(f"[ERROR] Error processing frame in movement-based bot: {e}")
            results['error'] = str(e)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        self.processing_times.append(processing_time)
        
        # Keep only recent processing times
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
        
        return results
    
    def _store_training_data(self, game_state: GameState, action: Dict):
        """Store training data for DQN."""
        if self.previous_game_state is not None and self.last_action is not None:
            # Calculate reward (placeholder - would need actual game outcome tracking)
            reward = self._calculate_reward(game_state, self.previous_game_state)
            
            # Create training data
            training_data = {
                'state': self.previous_game_state,
                'action': GameAction(
                    card_index=self.last_action['card_index'],
                    position=self.last_action['position'],
                    timestamp=self.last_action['timestamp']
                ),
                'reward': GameReward(
                    immediate_reward=reward,
                    game_outcome=None,  # Would be set at game end
                    timestamp=time.time()
                ),
                'next_state': game_state,
                'done': False  # Would be True at game end
            }
            
            self.training_data.append(training_data)
            
            # Train DQN
            if len(self.training_data) >= 32:  # Batch size
                self._train_dqn()
    
    def _calculate_reward(self, current_state: GameState, previous_state: GameState) -> float:
        """Calculate reward based on state changes."""
        # Placeholder reward calculation
        # In practice, this would consider:
        # - Damage dealt to enemy towers
        # - Units killed
        # - Elixir efficiency
        # - Tower health changes
        
        reward = 0.0
        
        # Simple reward based on tower health changes
        if current_state.tower_health and previous_state.tower_health:
            for i in range(len(current_state.tower_health)):
                if i < len(previous_state.tower_health):
                    health_change = previous_state.tower_health[i] - current_state.tower_health[i]
                    if i < 2:  # Own towers
                        reward -= health_change * 0.1  # Penalty for losing health
                    else:  # Enemy towers
                        reward += health_change * 0.1  # Reward for dealing damage
        
        return reward
    
    def _train_dqn(self):
        """Train the DQN agent."""
        if not self.dqn_agent or len(self.training_data) < 32:
            return
        
        try:
            # Sample batch from training data
            batch = self.training_data[-32:]  # Use last 32 experiences
            
            # Train on batch
            for data in batch:
                self.dqn_agent.remember(
                    data['state'],
                    data['action'],
                    data['reward'],
                    data['next_state'],
                    data['done']
                )
            
            # Perform training step
            loss = self.dqn_agent.replay()
            
            if self.frame_count % 100 == 0:  # Log every 100 frames
                stats = self.dqn_agent.get_training_stats()
                self.logger.log(f"DQN Training - Loss: {loss:.4f}, Epsilon: {stats['epsilon']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error training DQN: {e}")
    
    def _create_visualization(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Create visualization of bot processing results."""
        vis_frame = frame.copy()
        
        try:
            # Draw detected units
            if self.movement_detector and results['detected_units']:
                vis_frame = self.movement_detector.get_detection_visualization(
                    vis_frame, results['detected_units']
                )
            
            # Draw tracked units with IDs and speed vectors
            if results['tracked_units']:
                for unit in results['tracked_units']:
                    # Draw bounding box
                    x, y, w, h = unit.bbox
                    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw unit ID
                    cv2.putText(vis_frame, f"ID:{unit.unit_id}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw speed vector
                    if unit.speed_vector != (0.0, 0.0):
                        end_x = int(unit.centroid[0] + unit.speed_vector[0] * 5)
                        end_y = int(unit.centroid[1] + unit.speed_vector[1] * 5)
                        cv2.arrowedLine(vis_frame, unit.centroid, (end_x, end_y), (255, 0, 0), 2)
            
            # Draw tower health
            if self.tower_health_detector and results['tower_health']:
                vis_frame = self.tower_health_detector.create_health_visualization(
                    vis_frame, results['tower_health']
                )
            
            # Draw action information
            if results['action']:
                action = results['action']
                text = f"Action: Card {action['card_index']} at {action['position']}"
                cv2.putText(vis_frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw performance info
            fps = 1.0 / results['processing_time'] if results['processing_time'] > 0 else 0
            perf_text = f"FPS: {fps:.1f}, Frame: {results['frame_count']}"
            cv2.putText(vis_frame, perf_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
        
        return vis_frame
    
    def save_model(self, filepath: str = None):
        """Save the DQN model."""
        if not self.dqn_agent:
            return
        
        if filepath is None:
            filepath = self.config.dqn_model_path
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.dqn_agent.save_model(filepath)
            self.logger.log(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str = None):
        """Load the DQN model."""
        if not self.dqn_agent:
            return
        
        if filepath is None:
            filepath = self.config.dqn_model_path
        
        try:
            if os.path.exists(filepath):
                self.dqn_agent.load_model(filepath)
                self.logger.log(f"Model loaded from {filepath}")
            else:
                self.logger.log(f"Model file not found: {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
    
    def reset_training(self):
        """Reset DQN training state."""
        if self.dqn_agent:
            self.dqn_agent.reset_training()
            self.training_data.clear()
            self.logger.log("Training state reset")
    
    def get_performance_stats(self) -> Dict:
        """Get bot performance statistics."""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        stats = {
            'frame_count': self.frame_count,
            'avg_processing_time': avg_processing_time,
            'fps': fps,
            'target_fps': self.config.target_fps,
            'fps_performance': fps / self.config.target_fps if self.config.target_fps > 0 else 0
        }
        
        # Add component-specific stats
        if self.unit_tracker:
            stats.update(self.unit_tracker.get_tracking_stats())
        
        if self.dqn_agent:
            stats.update(self.dqn_agent.get_training_stats())
        
        return stats
    
    def update_config(self, new_config: Dict):
        """Update bot configuration."""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.logger.log("Configuration updated")
    
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Save model before cleanup
        self.save_model()
        
        self.logger.log("Bot cleanup completed")
