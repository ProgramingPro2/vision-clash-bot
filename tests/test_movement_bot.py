"""Test script for movement-based bot components."""

import os
import sys
import time
from typing import Any, Dict

import cv2
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyclashbot.detection.movement_detection import MovementDetector, MovementDetectorConfig
from pyclashbot.detection.unit_tracking import UnitTracker, UnitTrackerConfig
from pyclashbot.detection.tower_health_detection import TowerHealthDetector, TowerHealthDetectorConfig
from pyclashbot.ai.dqn_agent import DQNAgent, GameStateProcessor, GameState
from pyclashbot.ai.movement_based_bot import MovementBasedBot, BotConfig
from pyclashbot.config.movement_bot_config import MovementBotSettings


class MockEmulator:
    """Mock emulator for testing."""
    
    def __init__(self):
        self.frame_count = 0
    
    def screenshot(self):
        """Generate a mock screenshot."""
        # Create a simple test frame with some moving objects
        frame = np.zeros((633, 419, 3), dtype=np.uint8)
        
        # Add some moving rectangles to simulate units
        time_factor = self.frame_count * 0.1
        for i in range(3):
            x = int(100 + i * 100 + 50 * np.sin(time_factor + i))
            y = int(200 + 30 * np.cos(time_factor + i))
            cv2.rectangle(frame, (x-10, y-10), (x+10, y+10), (0, 255, 0), -1)
        
        # Add some static elements
        cv2.rectangle(frame, (50, 50), (150, 100), (255, 0, 0), -1)  # Tower
        cv2.rectangle(frame, (250, 50), (350, 100), (255, 0, 0), -1)  # Tower
        
        self.frame_count += 1
        return frame


class MockLogger:
    """Mock logger for testing."""
    
    def __init__(self):
        self.messages = []
    
    def log(self, message: str):
        print(f"[LOG] {message}")
        self.messages.append(message)
    
    def error(self, message: str):
        print(f"[ERROR] {message}")
        self.messages.append(f"ERROR: {message}")
    
    def change_status(self, message: str):
        print(f"[STATUS] {message}")
        self.messages.append(f"STATUS: {message}")


def test_movement_detection():
    """Test movement detection module."""
    print("\n=== Testing Movement Detection ===")
    
    config = MovementDetectorConfig()
    detector = MovementDetector(
        min_area=config.min_area,
        max_area=config.max_area,
        threshold=config.threshold,
        history_length=config.history_length
    )
    
    emulator = MockEmulator()
    
    # Test with multiple frames
    for i in range(10):
        frame = emulator.screenshot()
        blobs = detector.detect_units(frame)
        
        print(f"Frame {i}: Detected {len(blobs)} units")
        for j, blob in enumerate(blobs):
            print(f"  Unit {j}: Area={blob.area}, Centroid={blob.centroid}")
    
    print("Movement detection test completed ✓")


def test_unit_tracking():
    """Test unit tracking module."""
    print("\n=== Testing Unit Tracking ===")
    
    config = UnitTrackerConfig()
    tracker = UnitTracker(
        max_distance=config.max_distance,
        max_occlusion_frames=config.max_occlusion_frames,
        min_track_length=config.min_track_length
    )
    
    detector = MovementDetector()
    emulator = MockEmulator()
    
    # Test tracking over multiple frames
    for i in range(15):
        frame = emulator.screenshot()
        blobs = detector.detect_units(frame)
        tracked_units = tracker.track_units(blobs)
        
        print(f"Frame {i}: {len(tracked_units)} tracked units")
        for unit in tracked_units:
            print(f"  Track {unit.unit_id}: Pos={unit.centroid}, Speed={unit.speed_vector}")
    
    # Get tracking stats
    stats = tracker.get_tracking_stats()
    print(f"Tracking stats: {stats}")
    
    print("Unit tracking test completed ✓")


def test_tower_health_detection():
    """Test tower health detection module."""
    print("\n=== Testing Tower Health Detection ===")
    
    detector = TowerHealthDetector()
    emulator = MockEmulator()
    
    # Test with multiple frames
    for i in range(5):
        frame = emulator.screenshot()
        tower_health = detector.detect_all_tower_health(frame)
        
        print(f"Frame {i}: Tower health detected")
        print(f"  Own Left: {tower_health.left_tower_health}")
        print(f"  Own Right: {tower_health.right_tower_health}")
        print(f"  Enemy Left: {tower_health.enemy_left_tower_health}")
        print(f"  Enemy Right: {tower_health.enemy_right_tower_health}")
        print(f"  Confidence: {tower_health.confidence}")
    
    print("Tower health detection test completed ✓")


def test_dqn_agent():
    """Test DQN agent module."""
    print("\n=== Testing DQN Agent ===")
    
    # Create a simple DQN agent
    agent = DQNAgent(
        state_size=50,
        action_size=12,
        hidden_sizes=[64, 32],
        learning_rate=0.001
    )
    
    # Create a mock game state
    state_processor = GameStateProcessor()
    
    # Mock tracked units
    from pyclashbot.detection.unit_tracking import TrackedUnit
    from collections import deque
    
    mock_units = []
    for i in range(3):
        unit = TrackedUnit(
            unit_id=i,
            centroid=(100 + i * 50, 200),
            area=100 + i * 50,
            bbox=(90 + i * 50, 190, 20, 20),
            movement_history=deque([(100 + i * 50, 200)], maxlen=20)
        )
        mock_units.append(unit)
    
    # Create game state
    game_state = state_processor.process_game_state(
        tracked_units=mock_units,
        elixir_count=5.0,
        tower_health=[100, 100, 100, 100],
        time_remaining=120.0,
        card_availability=[True, True, True, True]
    )
    
    # Test action selection
    action = agent.select_action(game_state)
    print(f"Selected action: {action}")
    
    # Test training
    for i in range(10):
        next_state = game_state  # Same state for simplicity
        reward = GameReward(immediate_reward=0.1, game_outcome=None, timestamp=time.time())
        action_obj = GameAction(card_index=0, position=(0.5, 0.5), timestamp=time.time())
        
        agent.remember(game_state, action_obj, reward, next_state, False)
    
    # Train the agent
    loss = agent.replay()
    print(f"Training loss: {loss}")
    
    # Get training stats
    stats = agent.get_training_stats()
    print(f"Training stats: {stats}")
    
    print("DQN agent test completed ✓")


def test_movement_based_bot():
    """Test the complete movement-based bot."""
    print("\n=== Testing Movement-Based Bot ===")
    
    config = BotConfig()
    logger = MockLogger()
    bot = MovementBasedBot(config, logger)
    
    emulator = MockEmulator()
    
    # Test processing multiple frames
    for i in range(10):
        frame = emulator.screenshot()
        results = bot.process_frame(frame)
        
        print(f"Frame {i}:")
        print(f"  Processing time: {results['processing_time']:.3f}s")
        print(f"  Detected units: {len(results['detected_units'])}")
        print(f"  Tracked units: {len(results['tracked_units'])}")
        print(f"  Action: {results['action']}")
    
    # Get performance stats
    stats = bot.get_performance_stats()
    print(f"Bot performance stats: {stats}")
    
    # Test model saving/loading
    test_model_path = "test_model.pth"
    bot.save_model(test_model_path)
    print(f"Model saved to {test_model_path}")
    
    # Clean up
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    
    bot.cleanup()
    print("Movement-based bot test completed ✓")


def test_configuration():
    """Test configuration management."""
    print("\n=== Testing Configuration ===")
    
    # Test default configuration
    config = MovementBotSettings()
    print(f"Default config: {config.to_dict()}")
    
    # Test configuration update
    config.min_area = 100
    config.max_area = 6000
    config.dqn_enabled = False
    
    print(f"Updated config: min_area={config.min_area}, max_area={config.max_area}, dqn_enabled={config.dqn_enabled}")
    
    # Test saving and loading
    test_config_path = "test_config.json"
    config.save_to_file(test_config_path)
    
    loaded_config = MovementBotSettings.load_from_file(test_config_path)
    print(f"Loaded config matches: {loaded_config.min_area == config.min_area}")
    
    # Clean up
    if os.path.exists(test_config_path):
        os.remove(test_config_path)
    
    print("Configuration test completed ✓")


def run_all_tests():
    """Run all tests."""
    print("Starting Movement-Based Bot Component Tests")
    print("=" * 50)
    
    try:
        test_movement_detection()
        test_unit_tracking()
        test_tower_health_detection()
        test_dqn_agent()
        test_movement_based_bot()
        test_configuration()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully! ✓")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
