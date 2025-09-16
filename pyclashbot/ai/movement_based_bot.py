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
from ..bot.card_detection import check_which_cards_are_available, identify_hand_cards


@dataclass
class BotConfig:
    """Configuration for the movement-based bot - Training Mode Only."""
    # Training Mode - Always enabled
    training_mode: bool = True
    
    # Movement detection - Required for training
    movement_detection_enabled: bool = True
    movement_detection_config: MovementDetectorConfig = None
    
    # Unit tracking - Required for training
    unit_tracking_enabled: bool = True
    unit_tracking_config: UnitTrackerConfig = None
    
    # Tower health detection - Required for training
    tower_health_enabled: bool = True
    tower_health_config: TowerHealthDetectorConfig = None
    
    # DQN agent - Training focused
    dqn_enabled: bool = True
    dqn_model_path: str = "models/dqn_model.pth"
    dqn_backup_path: str = "models/dqn_model_backup.pth"
    dqn_state_size: int = 200  # Will be calculated dynamically
    dqn_action_size: int = 7  # 1 wait + 4 cards + 2 position outputs (x, y)
    
    # Auto-save settings
    auto_save_interval: int = 10  # Save every N battles
    auto_save_on_exit: bool = True
    max_backup_models: int = 5  # Keep N backup models
    
    # Training settings
    target_fps: int = 30
    enable_visualization: bool = True
    save_training_data: bool = True
    collect_training_data: bool = True  # Always collect data for training
    
    def __post_init__(self):
        if self.movement_detection_config is None:
            self.movement_detection_config = MovementDetectorConfig()
        if self.unit_tracking_config is None:
            self.unit_tracking_config = UnitTrackerConfig()
        if self.tower_health_config is None:
            self.tower_health_config = TowerHealthDetectorConfig()


class MovementBasedBot:
    """Main bot class integrating movement detection, unit tracking, and DQN decision making."""
    
    def __init__(self, config: BotConfig, logger: Logger, emulator=None):
        """
        Initialize movement-based bot.
        
        Args:
            config: Bot configuration
            logger: Logger instance
            emulator: Emulator instance for card detection
        """
        self.config = config
        self.logger = logger
        self.emulator = emulator
        
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
        self.battle_start_time = time.time()
        
        # Game state tracking
        self.current_game_state = None
        self.previous_game_state = None
        self.last_action = None
        self.last_action_time = 0
        
        # Training data collection
        self.training_data = []
        self.episode_data = []
        
        # Auto-save tracking
        self.battle_count = 0
        self.last_save_time = time.time()
        self.model_save_history = []
        
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
            else:
                self.logger.log("Movement detection disabled")
            
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
                    action_size=self.config.dqn_action_size,
                    logger=self.logger
                )
                
                # Auto-load best available model
                self.auto_load_model()
            
            self.logger.log("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            # Don't raise - allow bot to continue with limited functionality
            self.logger.log("Continuing with limited functionality")
    
    def _detect_elixir_from_frame(self, frame: np.ndarray) -> float:
        """
        Detect elixir count from the game frame using the EXACT same method as the original bot.
        Test each elixir amount from 1-10 until we find the exact amount.
        
        Args:
            frame: Game frame
            
        Returns:
            Detected elixir count (0-10)
        """
        try:
            # Import the original bot's elixir detection constants and functions
            from pyclashbot.detection.image_rec import pixel_is_equal
            
            # EXACT same constants from the original bot
            ELIXIR_COORDS = [
                [613, 149],
                [613, 165],
                [613, 188],
                [613, 212],
                [613, 240],
                [613, 262],
                [613, 287],
                [613, 314],
                [613, 339],
                [613, 364],
            ]
            ELIXIR_COLOR = [240, 137, 244]
            
            # Calculate scaled coordinates based on actual frame size
            original_width, original_height = 633, 419  # Original bot's expected resolution
            actual_width, actual_height = frame.shape[1], frame.shape[0]
            scale_x = actual_width / original_width
            scale_y = actual_height / original_height
            
            # Test each elixir amount from 1-10 using EXACT same logic as original bot
            elixir_count = 0
            for test_amount in range(1, 11):  # Test 1 through 10
                # Use EXACT same coordinate access as original bot: ELIXIR_COORDS[elixer_count - 1][0], ELIXIR_COORDS[elixer_count - 1][1]
                x = ELIXIR_COORDS[test_amount - 1][0]  # x coordinate
                y = ELIXIR_COORDS[test_amount - 1][1]  # y coordinate
                
                # Check if coordinates are within frame bounds (no scaling needed - original bot works!)
                if y < frame.shape[0] and x < frame.shape[1]:
                    # Use EXACT same pixel comparison as original bot's count_elixer()
                    if pixel_is_equal(frame[y, x], ELIXIR_COLOR, tol=65):
                        elixir_count = test_amount  # This amount is available
                        if self.frame_count % 10 == 0:  # Log successful detections
                            self.logger.log(f"  - SUCCESS: Test {test_amount} elixir at ({x},{y}) is purple!")
                    else:
                        # If this elixir dot is not visible, we've found the max
                        if self.frame_count % 10 == 0:  # Log failed detections
                            pixel = frame[y, x]
                            self.logger.log(f"  - FAILED: Test {test_amount} elixir at ({x},{y}) is {pixel.tolist()}, not purple")
                        break
                else:
                    # If coordinates are out of bounds, we've found the max
                    if self.frame_count % 10 == 0:  # Log out of bounds
                        self.logger.log(f"  - OUT OF BOUNDS: Test {test_amount} elixir at ({x},{y}) is outside frame {frame.shape}")
                    break
            
            if self.frame_count % 10 == 0:  # Log every 10 frames for debugging
                self.logger.log(f"Frame elixir detection: found {elixir_count} elixir using original bot method")
                self.logger.log(f"Frame shape: {frame.shape}")
                self.logger.log(f"ELIXIR_COLOR: {ELIXIR_COLOR}")
                
                # Calculate scaled coordinates based on actual frame size
                # Original coordinates are for a specific resolution, need to scale them
                original_width, original_height = 633, 419  # Original bot's expected resolution
                actual_width, actual_height = frame.shape[1], frame.shape[0]
                scale_x = actual_width / original_width
                scale_y = actual_height / original_height
                
                self.logger.log(f"Original resolution: {original_width}x{original_height}")
                self.logger.log(f"Actual resolution: {actual_width}x{actual_height}")
                self.logger.log(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
                
                # Log ALL test results for debugging
                for test_amount in range(1, 11):  # Test all 10 elixir amounts
                    coord = ELIXIR_COORDS[test_amount - 1]
                    x, y = coord[0], coord[1]
                    
                    # Scale coordinates to match actual frame size
                    scaled_x = int(x * scale_x)
                    scaled_y = int(y * scale_y)
                    
                    if scaled_y < frame.shape[0] and scaled_x < frame.shape[1]:
                        pixel = frame[scaled_y, scaled_x]
                        is_elixir = pixel_is_equal(pixel, ELIXIR_COLOR, tol=65)
                        self.logger.log(f"  - Test {test_amount} elixir at ({x},{y}) -> scaled ({scaled_x},{scaled_y}): {pixel.tolist()} -> {is_elixir}")
                    else:
                        self.logger.log(f"  - Test {test_amount} elixir at ({x},{y}) -> scaled ({scaled_x},{scaled_y}): OUT OF BOUNDS (frame: {frame.shape})")
            
            return float(elixir_count)
            
        except Exception as e:
            self.logger.error(f"Error detecting elixir: {e}")
            # Fallback to a reasonable default
            return 5.0
    
    def _detect_elixir_from_emulator(self) -> float:
        """
        Detect elixir count using the EXACT same method as the original bot.
        Test each elixir amount from 1-10 until we find the exact amount.
        
        Returns:
            Detected elixir count (0-10)
        """
        try:
            # Import the original bot's elixir detection constants and functions
            from pyclashbot.detection.image_rec import pixel_is_equal
            
            # EXACT same constants from the original bot
            ELIXIR_COORDS = [
                [613, 149],
                [613, 165],
                [613, 188],
                [613, 212],
                [613, 240],
                [613, 262],
                [613, 287],
                [613, 314],
                [613, 339],
                [613, 364],
            ]
            ELIXIR_COLOR = [240, 137, 244]
            
            # Get fresh screenshot from emulator (EXACT same as original bot)
            iar = self.emulator.screenshot()
            if iar is None:
                self.logger.error("Emulator screenshot returned None")
                return 5.0
            
            # Test each elixir amount from 1-10 using EXACT same logic as original bot
            elixir_count = 0
            for test_amount in range(1, 11):  # Test 1 through 10
                # Use EXACT same coordinate access as original bot: ELIXIR_COORDS[elixer_count - 1][0], ELIXIR_COORDS[elixer_count - 1][1]
                x = ELIXIR_COORDS[test_amount - 1][0]  # x coordinate
                y = ELIXIR_COORDS[test_amount - 1][1]  # y coordinate
                
                # Check if coordinates are within frame bounds
                if y < iar.shape[0] and x < iar.shape[1]:
                    # Use EXACT same pixel comparison as original bot's count_elixer()
                    if pixel_is_equal(iar[y, x], ELIXIR_COLOR, tol=65):
                        elixir_count = test_amount  # This amount is available
                        if self.frame_count % 10 == 0:  # Log successful detections
                            self.logger.log(f"  - SUCCESS: Test {test_amount} elixir at ({x},{y}) is purple!")
                    else:
                        # If this elixir dot is not visible, we've found the max
                        if self.frame_count % 10 == 0:  # Log failed detections
                            pixel = iar[y, x]
                            self.logger.log(f"  - FAILED: Test {test_amount} elixir at ({x},{y}) is {pixel.tolist()}, not purple")
                        break
                else:
                    # If coordinates are out of bounds, we've found the max
                    if self.frame_count % 10 == 0:  # Log out of bounds
                        self.logger.log(f"  - OUT OF BOUNDS: Test {test_amount} elixir at ({x},{y}) is outside frame {iar.shape}")
                    break
            
            if self.frame_count % 10 == 0:  # Log every 10 frames for debugging
                self.logger.log(f"Emulator elixir detection: found {elixir_count} elixir using original bot method")
                self.logger.log(f"Emulator frame shape: {iar.shape}")
                self.logger.log(f"ELIXIR_COLOR: {ELIXIR_COLOR}")
            
            return float(elixir_count)
            
        except Exception as e:
            self.logger.error(f"Error detecting elixir from emulator: {e}")
            # Fallback to a reasonable default
            return 5.0
    
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
        
        if self.frame_count % 5 == 0:  # Log every 5 frames for detailed debugging
            self.logger.log("=" * 100)
            self.logger.log(f"MOVEMENT BOT FRAME PROCESSING #{self.frame_count}")
            self.logger.log("=" * 100)
            self.logger.log(f"Frame details:")
            self.logger.log(f"  - Shape: {frame.shape}")
            self.logger.log(f"  - Dtype: {frame.dtype}")
            self.logger.log(f"  - Min/Max values: {frame.min()}/{frame.max()}")
            self.logger.log(f"  - Memory usage: {frame.nbytes} bytes")
            self.logger.log(f"  - Processing start time: {start_time:.6f}")
        
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
            if self.frame_count % 5 == 0:
                self.logger.log("STEP 1: MOVEMENT DETECTION")
            
            if self.movement_detector:
                if self.frame_count % 5 == 0:
                    self.logger.log("  - Movement detector available, detecting units...")
                detection_start = time.time()
                detected_blobs = self.movement_detector.detect_units(frame)
                detection_time = time.time() - detection_start
                results['detected_units'] = detected_blobs
                
                if self.frame_count % 5 == 0:
                    self.logger.log(f"  - Detection completed in {detection_time:.6f}s")
                    self.logger.log(f"  - Detected {len(detected_blobs)} units")
                    for i, blob in enumerate(detected_blobs):
                        self.logger.log(f"    Unit {i}: area={blob.area}, centroid={blob.centroid}")
            else:
                if self.frame_count % 5 == 0:
                    self.logger.log("  - Movement detector NOT available")
            
            # Unit tracking
            tracked_units = []
            if self.frame_count % 5 == 0:
                self.logger.log("STEP 2: UNIT TRACKING")
            
            if self.unit_tracker and detected_blobs:
                if self.frame_count % 5 == 0:
                    self.logger.log(f"  - Unit tracker available, tracking {len(detected_blobs)} blobs...")
                tracking_start = time.time()
                tracked_units = self.unit_tracker.track_units(detected_blobs)
                tracking_time = time.time() - tracking_start
                results['tracked_units'] = tracked_units
                
                if self.frame_count % 5 == 0:
                    self.logger.log(f"  - Tracking completed in {tracking_time:.6f}s")
                    self.logger.log(f"  - Tracked {len(tracked_units)} units")
                    for i, unit in enumerate(tracked_units):
                        self.logger.log(f"    Track {i}: ID={unit.unit_id}, centroid={unit.centroid}, speed={unit.speed_vector}")
            else:
                if self.frame_count % 5 == 0:
                    missing = []
                    if not self.unit_tracker: missing.append("unit_tracker")
                    if not detected_blobs: missing.append("detected_blobs")
                    self.logger.log(f"  - Unit tracking skipped, missing: {missing}")
            
            # Tower health detection
            tower_health = None
            if self.frame_count % 5 == 0:
                self.logger.log("STEP 3: TOWER HEALTH DETECTION")
            
            if self.tower_health_detector:
                if self.frame_count % 5 == 0:
                    self.logger.log("  - Tower health detector available, detecting health...")
                health_start = time.time()
                tower_health = self.tower_health_detector.detect_all_tower_health(frame)
                health_time = time.time() - health_start
                results['tower_health'] = tower_health
                
                if self.frame_count % 5 == 0:
                    self.logger.log(f"  - Health detection completed in {health_time:.6f}s")
                    if tower_health:
                        self.logger.log(f"  - Tower health: left={tower_health.left_tower_health}, right={tower_health.right_tower_health}")
                        self.logger.log(f"  - Enemy health: left={tower_health.enemy_left_tower_health}, right={tower_health.enemy_right_tower_health}")
                    else:
                        self.logger.log("  - No tower health detected")
            else:
                if self.frame_count % 5 == 0:
                    self.logger.log("  - Tower health detector NOT available")
            
            # Game state processing
            game_state = None
            if self.dqn_agent and tracked_units and tower_health:
                # Use emulator-based detection (like original bot) instead of frame-based
                elixir_count = self._detect_elixir_from_emulator()
                
                if self.frame_count % 5 == 0:
                    self.logger.log("STEP 4: GAME STATE CREATION")
                    self.logger.log(f"  - Tracked units: {len(tracked_units)}")
                    self.logger.log(f"  - Detected elixir: {elixir_count}")
                    self.logger.log(f"  - Tower health available: {tower_health is not None}")
                
                # Get time remaining (placeholder)
                time_remaining = 120.0  # Placeholder
                
                # Get card availability (placeholder)
                card_availability = [True, True, True, True]  # Placeholder
                
                if self.frame_count % 30 == 0:
                    self.logger.log(f"Creating game state: elixir={elixir_count:.1f}, units={len(tracked_units)}, time={time_remaining}")
                
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
                
                if self.frame_count % 30 == 0:
                    self.logger.log(f"Game state created: {game_state}")
            else:
                if self.frame_count % 30 == 0:
                    missing = []
                    if not self.dqn_agent: missing.append("dqn_agent")
                    if not tracked_units: missing.append("tracked_units")
                    if not tower_health: missing.append("tower_health")
                    self.logger.log(f"Cannot create game state, missing: {missing}")
            
            # DQN decision making
            action = None
            if self.dqn_agent and game_state and self.emulator:
                if self.frame_count % 30 == 0:
                    self.logger.log("DQN decision making: Getting available cards")
                
                # Get available cards from hand
                available_cards = check_which_cards_are_available(self.emulator)
                
                # Get elixir costs for cards (placeholder - would need actual detection)
                card_elixir_costs = [3.0, 3.0, 3.0, 3.0]  # Default costs
                
                if self.frame_count % 30 == 0:
                    self.logger.log(f"Available cards: {available_cards}, costs: {card_elixir_costs}")
                
                # Select action using new method
                action = self.dqn_agent.select_action(game_state, available_cards, card_elixir_costs)
                
                if self.frame_count % 30 == 0:
                    self.logger.log(f"DQN selected action: {action.action_type} - {action.card_index} at {action.position}")
                
                # Identify the card if playing a card
                if action.action_type == "play_card" and action.card_index is not None:
                    try:
                        action.card_identity = identify_hand_cards(self.emulator, action.card_index)
                        if self.frame_count % 30 == 0:
                            self.logger.log(f"Identified card: {action.card_identity}")
                    except Exception as e:
                        self.logger.error(f"Failed to identify card {action.card_index}: {e}")
                        action.card_identity = "unknown"
                
                results['action'] = action
                
                # Store for training
                if self.config.save_training_data:
                    self._store_training_data(game_state, action)
            else:
                # Fallback: Simple random action if DQN not available
                if self.emulator and self.frame_count % 60 == 0:  # Every 2 seconds at 30fps
                    try:
                        if self.frame_count % 60 == 0:
                            self.logger.log("Using fallback action system")
                        available_cards = check_which_cards_are_available(self.emulator)
                        if available_cards:
                            import random
                            from .dqn_agent import GameAction
                            card_index = random.choice(available_cards)
                            position = (random.uniform(0.2, 0.8), random.uniform(0.3, 0.7))
                            action = GameAction(
                                action_type="play_card",
                                card_index=card_index,
                                position=position,
                                timestamp=time.time(),
                                elixir_cost=3.0
                            )
                            results['action'] = action
                            self.logger.log(f"Fallback action: Playing card {card_index} at {position}")
                        else:
                            if self.frame_count % 60 == 0:
                                self.logger.log("Fallback: No available cards")
                    except Exception as e:
                        self.logger.error(f"Fallback action failed: {e}")
                else:
                    if self.frame_count % 60 == 0:
                        missing = []
                        if not self.dqn_agent: missing.append("dqn_agent")
                        if not game_state: missing.append("game_state")
                        if not self.emulator: missing.append("emulator")
                        self.logger.log(f"DQN decision making skipped, missing: {missing}")
            
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
            results['error'] = str(e)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        self.processing_times.append(processing_time)
        
        # Keep only recent processing times
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
        
        return results
    
    def execute_action(self, action: GameAction) -> GameAction:
        """
        Execute an action (wait or play card) and check for success.
        
        Args:
            action: GameAction object with action type and details
            
        Returns:
            Updated GameAction with success flags
        """
        if not self.emulator or not action:
            return action
        
        if action.action_type == "wait":
            # For wait action, just mark as successful and return
            action.placement_success = True  # Wait is always "successful"
            action.detection_success = False  # No detection for wait
            return action
        
        elif action.action_type == "play_card":
            try:
                # Convert normalized position to screen coordinates
                screen_width, screen_height = 419, 633  # Clash Royale screen dimensions
                x = int(action.position[0] * screen_width)
                y = int(action.position[1] * screen_height)
                
                # Click the card
                card_x = 133 + (action.card_index * 66)  # Card positions
                card_y = 582
                self.emulator.click(card_x, card_y)
                
                # Small delay for card selection
                time.sleep(0.1)
                
                # Click the placement position
                self.emulator.click(x, y)
                
                # Mark placement as successful
                action.placement_success = True
                
                # Check for unit detection in next few frames
                self._check_detection_success(action)
                
            except Exception as e:
                self.logger.error(f"Failed to execute card action: {e}")
                action.placement_success = False
        
        return action
    
    def _check_detection_success(self, action: GameAction):
        """
        Check if a unit was successfully detected after placement.
        
        Args:
            action: GameAction object to update with detection success
        """
        if not self.movement_detector or not self.unit_tracker:
            return
        
        try:
            # Wait a bit for unit to appear
            time.sleep(0.5)
            
            # Get current frame
            frame = self.emulator.screenshot()
            if frame is None:
                return
            
            # Detect units
            blobs = self.movement_detector.detect_units(frame)
            tracked_units = self.unit_tracker.track_units(blobs)
            
            # Check if any new units were detected near the placement position
            placement_x = action['position'][0] * 419  # Convert back to screen coords
            placement_y = action['position'][1] * 633
            
            detection_threshold = 50  # Pixels
            
            for unit in tracked_units:
                distance = ((unit.centroid[0] - placement_x) ** 2 + 
                           (unit.centroid[1] - placement_y) ** 2) ** 0.5
                
                if distance < detection_threshold:
                    action['detection_success'] = True
                    self.logger.log(f"Successfully detected unit at {unit.centroid} after placement")
                    break
            
        except Exception as e:
            self.logger.error(f"Error checking detection success: {e}")
    
    def _store_training_data(self, game_state: GameState, action: GameAction):
        """Store training data for DQN."""
        if self.previous_game_state is not None and self.last_action is not None:
            # Calculate reward with placement and detection components
            reward = self._calculate_reward(game_state, self.previous_game_state, self.last_action)
            
            # Create training data
            training_data = {
                'state': self.previous_game_state,
                'action': self.last_action,  # Already a GameAction object
                'reward': reward,
                'next_state': game_state,
                'done': False  # Would be True at game end
            }
            
            self.training_data.append(training_data)
            
            # Train DQN
            if len(self.training_data) >= 32:  # Batch size
                self._train_dqn()
    
    def _calculate_reward(self, current_state: GameState, previous_state: GameState, action: GameAction = None) -> GameReward:
        """Calculate reward based on state changes and action success."""
        # Base reward calculation
        immediate_reward = 0.0
        
        # Simple reward based on tower health changes
        if current_state.tower_health and previous_state.tower_health:
            for i in range(len(current_state.tower_health)):
                if i < len(previous_state.tower_health):
                    health_change = previous_state.tower_health[i] - current_state.tower_health[i]
                    if i < 2:  # Own towers
                        immediate_reward -= health_change * 0.1  # Penalty for losing health
                    else:  # Enemy towers
                        immediate_reward += health_change * 0.1  # Reward for dealing damage
        
        # Calculate all reward components
        placement_reward = 0.0
        detection_reward = 0.0
        elixir_efficiency_reward = 0.0
        wait_reward = 0.0
        
        if action and self.dqn_agent:
            # Calculate placement reward
            placement_reward = self.dqn_agent.calculate_placement_reward(action)
            
            # Calculate detection reward
            detection_reward = self.dqn_agent.calculate_detection_reward(action)
            
            # Calculate elixir efficiency reward
            elixir_efficiency_reward = self.dqn_agent.calculate_elixir_efficiency_reward(
                action, previous_state.elixir_count
            )
            
            # Calculate wait reward
            wait_reward = self.dqn_agent.calculate_wait_reward(action, previous_state)
        
        return GameReward(
            immediate_reward=immediate_reward,
            placement_reward=placement_reward,
            detection_reward=detection_reward,
            elixir_efficiency_reward=elixir_efficiency_reward,
            wait_reward=wait_reward,
            timestamp=time.time()
        )
    
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
    
    def auto_save_model(self):
        """Automatically save the model with backup management."""
        if not self.dqn_agent:
            return
        
        try:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.dqn_model_path), exist_ok=True)
            
            # Save backup of current model
            if os.path.exists(self.config.dqn_model_path):
                backup_path = f"{self.config.dqn_model_path}.backup_{int(time.time())}"
                import shutil
                shutil.copy2(self.config.dqn_model_path, backup_path)
                self.model_save_history.append(backup_path)
                
                # Clean up old backups
                if len(self.model_save_history) > self.config.max_backup_models:
                    old_backup = self.model_save_history.pop(0)
                    if os.path.exists(old_backup):
                        os.remove(old_backup)
            
            # Save current model
            self.dqn_agent.save_model(self.config.dqn_model_path)
            self.last_save_time = time.time()
            
            self.logger.log(f"Model auto-saved to {self.config.dqn_model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to auto-save model: {e}")
    
    def auto_load_model(self):
        """Automatically load the best available model."""
        if not self.dqn_agent:
            return
        
        try:
            # Try to load main model first
            if os.path.exists(self.config.dqn_model_path):
                self.dqn_agent.load_model(self.config.dqn_model_path)
                self.logger.log(f"Loaded main model from {self.config.dqn_model_path}")
                return
            
            # Try to load backup model
            if os.path.exists(self.config.dqn_backup_path):
                self.dqn_agent.load_model(self.config.dqn_backup_path)
                self.logger.log(f"Loaded backup model from {self.config.dqn_backup_path}")
                return
            
            # Try to load any available backup
            backup_files = [f for f in os.listdir("models/") if f.startswith("dqn_model.pth.backup_")]
            if backup_files:
                latest_backup = sorted(backup_files)[-1]
                backup_path = os.path.join("models", latest_backup)
                self.dqn_agent.load_model(backup_path)
                self.logger.log(f"Loaded latest backup model from {backup_path}")
                return
            
            self.logger.log("No existing model found, starting with new model")
            
        except Exception as e:
            self.logger.error(f"Failed to auto-load model: {e}")
    
    def on_battle_end(self, battle_result: str):
        """Called when a battle ends - handles auto-save logic."""
        self.battle_count += 1
        
        # Auto-save based on interval
        if self.battle_count % self.config.auto_save_interval == 0:
            self.auto_save_model()
        
        # Save training data
        if self.config.save_training_data and self.episode_data:
            self._save_episode_data(battle_result)
    
    def _save_episode_data(self, battle_result: str):
        """Save episode training data."""
        try:
            episode_data = {
                'battle_result': battle_result,
                'episode_data': self.episode_data.copy(),
                'timestamp': time.time(),
                'battle_count': self.battle_count
            }
            
            # Save to training data file
            training_file = "data/training_episodes.json"
            os.makedirs(os.path.dirname(training_file), exist_ok=True)
            
            # Load existing data
            if os.path.exists(training_file):
                with open(training_file, 'r') as f:
                    all_data = json.load(f)
            else:
                all_data = []
            
            all_data.append(episode_data)
            
            # Save updated data
            with open(training_file, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            # Clear episode data for next battle
            self.episode_data.clear()
            
            self.logger.log(f"Saved episode data for battle {self.battle_count}")
            
        except Exception as e:
            self.logger.error(f"Failed to save episode data: {e}")
    
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        
        # Auto-save on exit if enabled
        if self.config.auto_save_on_exit and self.dqn_agent:
            self.auto_save_model()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        self.logger.log("Bot cleanup completed")
