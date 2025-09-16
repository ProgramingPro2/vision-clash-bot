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
from ..bot.card_detection import check_which_cards_are_available, identify_hand_cards, get_card_group

# Emote coordinates (from original bot)
EMOTE_BUTTON_COORD = (67, 521)
EMOTE_ICON_COORDS = [
    (124, 419),
    (182, 420),
    (255, 411),
    (312, 423),
    (133, 471),
    (188, 472),
    (243, 469),
    (308, 470),
]


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
    auto_save_interval: int = 1  # Save every N battles
    auto_save_on_exit: bool = True
    max_backup_models: int = 5  # Keep N backup models
    
    # Training settings
    target_fps: int = 30
    enable_visualization: bool = True
    save_training_data: bool = True
    collect_training_data: bool = True  # Always collect data for training
    
    # Emote settings (like original bot)
    enable_emotes: bool = True
    emote_chance: float = 0.1  # 10% chance to emote after playing a card
    
    def __post_init__(self):
        if self.movement_detection_config is None:
            self.movement_detection_config = MovementDetectorConfig()
        if self.unit_tracking_config is None:
            self.unit_tracking_config = UnitTrackerConfig()
        if self.tower_health_config is None:
            self.tower_health_config = TowerHealthDetectorConfig()


# Card cost lookup based on card groups/classes (Updated August 2025)
CARD_GROUP_COSTS = {
    # Spells (typically 1-6 elixir)
    "zap": 2.0,           # Zap, Goblin Curse, Void
    "arrows": 3.0,        # Arrows
    "snowball": 2.0,      # Snowball, Giant Snowball
    "log": 2.0,           # Log, Barbarian Barrel
    "fireball": 4.0,      # Fireball
    "poison": 4.0,        # Poison
    "freeze": 4.0,        # Freeze
    "earthquake": 3.0,    # Earthquake
    "rocket": 6.0,        # Rocket
    "lightning": 6.0,     # Lightning
    "tornado": 3.0,       # Tornado
    
    # Win conditions (typically 3-6 elixir)
    "hog": 4.0,           # Hog, Battle Ram, Ram Rider, etc.
    "miner": 3.0,         # Miner
    "goblin_barrel": 3.0, # Goblin Barrel
    "goblin_drill": 4.0,  # Goblin Drill
    "graveyard": 5.0,     # Graveyard
    "xbow": 6.0,          # X-Bow, Mortar
    
    # Buildings (typically 3-6 elixir)
    "turret": 4.0,        # Cannon, Tesla, Bomb Tower, etc.
    "spawner": 5.0,       # Furnace, Goblin Hut, Barbarian Hut, etc.
    
    # Support troops (typically 3-5 elixir) - Updated for August 2025
    "long_range": 4.0,    # Witch, Night Witch (Wizard moved to 4 elixir)
    "princess": 3.0,      # Princess
    "bigboi": 7.0,        # Mega Knight, Royal Delivery, Mighty Miner
    
    # Default fallback
    "No group": 4.0,      # Unknown cards
}

# Individual card costs for more precise detection (Updated August 2025)
INDIVIDUAL_CARD_COSTS = {
    # 1 elixir cards
    "skeletons": 1.0,
    "ice_spirit": 1.0,
    "fire_spirit": 1.0,
    "electro_spirit": 1.0,
    "heal_spirit": 1.0,
    
    # 2 elixir cards
    "zap": 2.0,
    "gob_curse": 2.0,
    "void": 2.0,
    "snowball": 2.0,
    "log": 2.0,
    "ice_golem": 2.0,
    "goblins": 2.0,
    "spear_goblins": 2.0,
    "bats": 2.0,
    "fire_cracker": 2.0,
    "evo_fire_cracker": 2.0,
    "wall_breakers": 2.0,
    "barbarian_barrel": 2.0,
    "giant_snowball": 2.0,
    
    # 3 elixir cards
    "miner": 3.0,
    "goblin_barrel": 3.0,
    "evo_goblin_barrel": 3.0,
    "princess": 3.0,
    "knight": 3.0,
    "archers": 3.0,
    "dart_goblin": 3.0,
    "earthquake": 3.0,
    "tornado": 3.0,
    "cannon": 3.0,
    "tesla": 3.0,
    "goblin_cage": 3.0,
    "tombstone": 3.0,
    "bomber": 3.0,
    "minions": 3.0,
    "skeleton_army": 3.0,
    "clone": 3.0,
    "ice_wizard": 3.0,
    "mega_minion": 3.0,
    "arrows": 3.0,
    "bandit": 3.0,
    "goblin_gang": 3.0,
    "guards": 3.0,
    "heal": 3.0,
    
    # 4 elixir cards (Updated August 2025 - Wizard reduced from 5 to 4)
    "hog": 4.0,
    "battle_ram": 4.0,
    "evo_battle_ram": 4.0,
    "ram_rider": 4.0,
    "fireball": 4.0,
    "poison": 4.0,
    "freeze": 4.0,
    "bomb_tower": 4.0,
    "inferno_tower": 4.0,
    "wizard": 4.0,  # UPDATED: Reduced from 5 to 4 in August 2025
    "flying_machine": 4.0,
    "magic_archer": 4.0,
    "valkyrie": 4.0,
    "musketeer": 4.0,
    "goblin_drill": 4.0,
    "skeleton_barrel": 4.0,
    "royal_hogs": 4.0,
    "baby_dragon": 4.0,
    "mini_pekka": 4.0,
    "dark_prince": 4.0,
    "lumberjack": 4.0,
    "furnace": 4.0,
    "mortar": 4.0,
    "night_witch": 4.0,
    "electro_wizard": 4.0,
    "inferno_dragon": 4.0,
    
    # 5 elixir cards
    "graveyard": 5.0,
    "witch": 5.0,
    "barbarians": 5.0,
    "minion_horde": 5.0,
    "balloon": 5.0,
    "giant": 5.0,
    "prince": 5.0,
    "executioner": 5.0,
    "cannon_cart": 5.0,
    "goblin_hut": 5.0,
    "barbarian_hut": 5.0,
    "bowler": 5.0,
    
    # 6 elixir cards
    "rocket": 6.0,
    "lightning": 6.0,
    "xbow": 6.0,
    "sparky": 6.0,
    "elite_barbarians": 6.0,
    "royal_giant": 6.0,
    "giant_skeleton": 6.0,
    "elixir_collector": 6.0,
    
    # 7 elixir cards
    "mega_knight": 7.0,
    "royal_delivery": 7.0,
    "mighty_miner": 7.0,
    "pekka": 7.0,
    "lava_hound": 7.0,
    
    # 8 elixir cards
    "golem": 8.0,
    
    # 9 elixir cards
    "three_musketeers": 9.0,
}


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
        
        # Emote cooldown to prevent conflicts with card placement
        self.emote_cooldown_end = 0
        
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
        Falls back to game mechanics estimation if detection fails.
        
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
                coord_x = ELIXIR_COORDS[test_amount - 1][0]  # First coordinate from original bot
                coord_y = ELIXIR_COORDS[test_amount - 1][1]  # Second coordinate from original bot
                
                # Try the EXACT same access as original bot: iar[coord_x, coord_y]
                # But check bounds first to avoid crashes
                if coord_x < iar.shape[0] and coord_y < iar.shape[1]:
                    # Use EXACT same pixel comparison as original bot's count_elixer()
                    if pixel_is_equal(iar[coord_x, coord_y], ELIXIR_COLOR, tol=65):
                        elixir_count = test_amount  # This amount is available
                        if self.frame_count % 10 == 0:  # Log successful detections
                            self.logger.log(f"  - SUCCESS: Test {test_amount} elixir at iar[{coord_x},{coord_y}] is purple!")
                    else:
                        # If this elixir dot is not visible, we've found the max
                        if self.frame_count % 10 == 0:  # Log failed detections
                            pixel = iar[coord_x, coord_y]
                            self.logger.log(f"  - FAILED: Test {test_amount} elixir at iar[{coord_x},{coord_y}] is {pixel.tolist()}, not purple")
                        break
                else:
                    # If coordinates are out of bounds, we've found the max
                    if self.frame_count % 10 == 0:  # Log out of bounds
                        self.logger.log(f"  - OUT OF BOUNDS: Test {test_amount} elixir at iar[{coord_x},{coord_y}] is outside frame {iar.shape}")
                    break
            
            if self.frame_count % 10 == 0:  # Log every 10 frames for debugging
                self.logger.log(f"Emulator elixir detection: found {elixir_count} elixir using original bot method")
                self.logger.log(f"Emulator frame shape: {iar.shape}")
                self.logger.log(f"ELIXIR_COLOR: {ELIXIR_COLOR}")
            
            return float(elixir_count)
            
        except Exception as e:
            self.logger.error(f"Error detecting elixir from emulator: {e}")
            # Fallback to game mechanics estimation
            return self._estimate_elixir_from_game_mechanics()
    
    def _estimate_elixir_from_game_mechanics(self) -> float:
        """
        Backup elixir estimation based on Clash Royale game mechanics.
        
        Game mechanics (Updated August 2025):
        - Players start with 5 elixir
        - Elixir regenerates at 1 elixir every 2.8 seconds (normal)
        - Elixir regenerates at 1 elixir every 1.4 seconds (double elixir, after 2 minutes)
        - Elixir regenerates at 1 elixir every 0.93 seconds (triple elixir, overtime)
        - Maximum elixir is 10
        
        Returns:
            Estimated elixir count (0-10)
        """
        try:
            # Get battle elapsed time
            elapsed_time = time.time() - self.battle_start_time
            
            # Start with 5 elixir (game mechanics)
            base_elixir = 5.0
            
            # Calculate elixir regeneration based on time
            if elapsed_time < 120:  # First 2 minutes - normal elixir (2.8s per elixir)
                regenerated_elixir = elapsed_time / 2.8
            elif elapsed_time < 180:  # 2-3 minutes - double elixir (1.4s per elixir)
                # First 2 minutes at normal rate, then double rate
                normal_elixir = 120 / 2.8
                double_elixir = (elapsed_time - 120) / 1.4
                regenerated_elixir = normal_elixir + double_elixir
            else:  # 3+ minutes - triple elixir (0.93s per elixir)
                # First 2 minutes normal, next minute double, then triple
                normal_elixir = 120 / 2.8
                double_elixir = 60 / 1.4
                triple_elixir = (elapsed_time - 180) / 0.93
                regenerated_elixir = normal_elixir + double_elixir + triple_elixir
            
            # Total elixir = starting + regenerated, capped at 10
            total_elixir = min(base_elixir + regenerated_elixir, 10.0)
            
            if self.frame_count % 30 == 0:  # Log every 30 frames
                self.logger.log(f"BACKUP: Estimated elixir from game mechanics: {total_elixir:.1f} (elapsed: {elapsed_time:.1f}s)")
            
            return total_elixir
            
        except Exception as e:
            self.logger.error(f"Error in backup elixir estimation: {e}")
            # Final fallback
            return 5.0
    
    def send_emote(self):
        """Send a random emote (like the original bot)."""
        if not self.config.enable_emotes or not self.emulator:
            return
        
        try:
            self.logger.log("Hitting an emote")
            
            # Set emote cooldown (2 seconds to prevent conflicts)
            self.emote_cooldown_end = time.time() + 2.0
            
            # Click emote button
            self.emulator.click(EMOTE_BUTTON_COORD[0], EMOTE_BUTTON_COORD[1])
            time.sleep(0.33)
            
            # Click random emote icon
            import random
            emote_coord = random.choice(EMOTE_ICON_COORDS)
            self.emulator.click(emote_coord[0], emote_coord[1])
            
            self.logger.log("Emote sent successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to send emote: {e}")
    
    def is_emote_cooldown_active(self) -> bool:
        """Check if emote cooldown is currently active."""
        return time.time() < self.emote_cooldown_end
    
    def _get_card_elixir_costs(self, available_cards: List[int]) -> List[float]:
        """
        Get elixir costs for available cards by identifying them and looking up costs.
        
        Args:
            available_cards: List of card indices (0-3)
            
        Returns:
            List of elixir costs for each card
        """
        if not self.emulator or not available_cards:
            return [4.0] * 4  # Default fallback
        
        card_costs = []
        
        for card_index in range(4):  # Always check all 4 card slots
            if card_index in available_cards:
                try:
                    # Identify the card
                    card_identity = identify_hand_cards(self.emulator, card_index)
                    
                    # Try individual card lookup first
                    if card_identity in INDIVIDUAL_CARD_COSTS:
                        cost = INDIVIDUAL_CARD_COSTS[card_identity]
                        if self.frame_count % 30 == 0:
                            self.logger.log(f"  - Card {card_index} ({card_identity}): {cost} elixir (individual lookup)")
                    else:
                        # Fall back to group-based lookup
                        card_group = get_card_group(card_identity)
                        cost = CARD_GROUP_COSTS.get(card_group, 4.0)  # Default to 4 elixir
                        if self.frame_count % 30 == 0:
                            self.logger.log(f"  - Card {card_index} ({card_identity}, group: {card_group}): {cost} elixir (group lookup)")
                    
                    card_costs.append(cost)
                    
                except Exception as e:
                    # Fallback to default cost
                    cost = 4.0
                    card_costs.append(cost)
                    if self.frame_count % 30 == 0:
                        self.logger.log(f"  - Card {card_index}: {cost} elixir (fallback due to error: {e})")
            else:
                # Card not available, use default cost
                card_costs.append(4.0)
        
        if self.frame_count % 30 == 0:
            self.logger.log(f"Final card costs: {card_costs}")
        
        return card_costs
    
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
                
                # Skip movement detection during emote cooldown to avoid conflicts
                if self.is_emote_cooldown_active():
                    if self.frame_count % 5 == 0:
                        self.logger.log("  - Skipping movement detection - emote cooldown active")
                    detected_blobs = []
                    detection_time = 0.0
                else:
                    detection_start = time.time()
                    detected_blobs = self.movement_detector.detect_units(frame)
                    detection_time = time.time() - detection_start
                
                results['detected_units'] = detected_blobs
                
                if self.frame_count % 5 == 0:
                    if detection_time > 0:
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
                
                # Get elixir costs for cards using card identification
                card_elixir_costs = self._get_card_elixir_costs(available_cards)
                
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
            # Check if emote cooldown is active - don't place cards during emote
            if self.is_emote_cooldown_active():
                self.logger.log("Skipping card placement - emote cooldown active")
                action.placement_success = False
                action.detection_success = False
                return action
            
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
                
                # Send emote with configured chance (like original bot)
                if self.config.enable_emotes:
                    import random
                    if random.random() < self.config.emote_chance:
                        self.send_emote()
                
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
