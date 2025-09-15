"""Integration of movement-based bot with existing fight system."""

import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..ai.movement_based_bot import BotConfig, MovementBasedBot
from ..detection.tower_health_detection import TowerHealth
from ..detection.unit_tracking import TrackedUnit
from ..utils.logger import Logger
from .card_detection import check_which_cards_are_available
from .fight import (
    check_for_in_battle_with_delay,
    check_if_in_battle,
    create_default_bridge_iar,
    play_a_card,
    wait_for_elixer,
)


class MovementFightManager:
    """Manages movement-based fighting with DQN decision making."""
    
    def __init__(self, emulator, logger: Logger, config: BotConfig):
        """
        Initialize movement fight manager.
        
        Args:
            emulator: Emulator instance
            logger: Logger instance
            config: Bot configuration
        """
        self.emulator = emulator
        self.logger = logger
        self.config = config
        
        # Initialize movement-based bot with emulator
        self.movement_bot = MovementBasedBot(config, logger, emulator)
        
        # Fight state tracking
        self.in_battle = False
        self.battle_start_time = 0
        self.last_action_time = 0
        self.action_cooldown = 0.5  # Minimum time between actions (reduced for more responsiveness)
        
        # Performance tracking
        self.frames_processed = 0
        self.actions_taken = 0
        self.processing_times = []
        
        # Threading for real-time processing
        self.processing_thread = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.action_queue = queue.Queue(maxsize=10)
    
    def start_battle_processing(self):
        """Start real-time battle processing."""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        self.logger.log("Started movement-based battle processing")
    
    def stop_battle_processing(self):
        """Stop real-time battle processing."""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        self.logger.log("Stopped movement-based battle processing")
    
    def _processing_loop(self):
        """Main processing loop for real-time battle analysis."""
        self.logger.log("Processing loop started")
        while self.running and self.in_battle:
            try:
                # Get frame from emulator
                frame = self.emulator.screenshot()
                if frame is None:
                    self.logger.log("No frame captured, retrying...")
                    time.sleep(0.1)
                    continue
                
                if self.frames_processed % 30 == 0:  # Log every 30 frames
                    self.logger.log(f"Captured frame: {frame.shape if hasattr(frame, 'shape') else 'unknown'}")
                
                # Process frame with movement bot
                start_time = time.time()
                results = self.movement_bot.process_frame(frame)
                processing_time = time.time() - start_time
                
                self.frames_processed += 1
                self.processing_times.append(processing_time)
                
                # Keep only recent processing times
                if len(self.processing_times) > 100:
                    self.processing_times = self.processing_times[-100:]
                
                # Check if we should take an action
                if self._should_take_action(results):
                    action = results.get('action')
                    if action:
                        self.action_queue.put(action)
                        self.actions_taken += 1
                        self.logger.log(f"Queued action: {action.action_type} - Card {action.card_index} at {action.position}")
                    else:
                        self.logger.log("No action available from DQN agent")
                else:
                    # Log why we're not taking action (less frequently)
                    if self.frames_processed % 60 == 0:  # Log every 2 seconds
                        current_time = time.time()
                        if current_time - self.last_action_time < self.action_cooldown:
                            self.logger.log(f"Action on cooldown: {self.action_cooldown - (current_time - self.last_action_time):.2f}s remaining")
                        else:
                            self.logger.log("No valid action to take")
                
                # Log performance periodically
                if self.frames_processed % 100 == 0:
                    avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
                    fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                    self.logger.log(f"Movement Bot - FPS: {fps:.1f}, Actions: {self.actions_taken}")
                
                # Maintain target FPS
                target_frame_time = 1.0 / self.config.target_fps
                sleep_time = max(0, target_frame_time - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _should_take_action(self, results: Dict[str, Any]) -> bool:
        """Determine if we should take an action based on results."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_action_time < self.action_cooldown:
            if self.frames_processed % 60 == 0:
                self.logger.log(f"Action blocked by cooldown: {self.action_cooldown - (current_time - self.last_action_time):.2f}s remaining")
            return False
        
        # Check if we have a valid action
        action = results.get('action')
        if not action:
            if self.frames_processed % 60 == 0:
                self.logger.log("No action in results")
            return False
        
        if self.frames_processed % 60 == 0:
            self.logger.log(f"Action available: {action.action_type} - {action.card_index} at {action.position}")
        
        # Check elixir availability (placeholder - would need actual detection)
        # For now, assume we have enough elixir
        return True
    
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute an action in the game.
        
        Args:
            action: Action to execute
            
        Returns:
            True if action was executed successfully
        """
        try:
            self.logger.log(f"Executing action: {action}")
            
            card_index = action['card_index']
            position = action['position']
            
            # Check if card is available
            available_cards = check_which_cards_are_available(self.emulator)
            self.logger.log(f"Available cards: {available_cards}")
            
            if not available_cards[card_index]:
                self.logger.log(f"Card {card_index} not available")
                return False
            
            # Convert normalized position to screen coordinates
            screen_width, screen_height = 419, 633  # Standard emulator dimensions
            x = int(position[0] * screen_width)
            y = int(position[1] * screen_height)
            
            self.logger.log(f"Converted position ({position[0]:.3f}, {position[1]:.3f}) to screen coords ({x}, {y})")
            
            # Validate coordinates using existing validation system
            from .recorder import is_valid_play_input
            if not is_valid_play_input((x, y), card_index):
                self.logger.log(f"Invalid coordinates: ({x}, {y}) for card {card_index}")
                return False
            
            self.logger.log(f"Coordinates validated, playing card {card_index} at ({x}, {y})")
            
            # Play the card
            success = self._play_card_at_position(card_index, x, y)
            
            if success:
                self.last_action_time = time.time()
                self.logger.log(f"Successfully played card {card_index} at ({x}, {y})")
                
                # Store action for training
                self._store_action_for_training(action)
            else:
                self.logger.log(f"Failed to play card {card_index} at ({x}, {y})")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing action: {e}")
            return False
    
    def _play_card_at_position(self, card_index: int, x: int, y: int) -> bool:
        """
        Play a card at the specified position.
        
        Args:
            card_index: Index of card to play
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if card was played successfully
        """
        try:
            # Click on the card
            card_coords = [(142, 561), (210, 563), (272, 561), (341, 563)]
            if card_index < len(card_coords):
                card_x, card_y = card_coords[card_index]
                self.logger.log(f"Clicking card {card_index} at ({card_x}, {card_y})")
                self.emulator.click(card_x, card_y)
                time.sleep(0.1)
                
                # Click at the target position
                self.logger.log(f"Clicking target position at ({x}, {y})")
                self.emulator.click(x, y)
                time.sleep(0.1)
                
                self.logger.log(f"Card {card_index} play sequence completed")
                return True
            else:
                self.logger.error(f"Invalid card index: {card_index}")
            
        except Exception as e:
            self.logger.error(f"Error playing card: {e}")
        
        return False
    
    def _store_action_for_training(self, action: Dict[str, Any]):
        """Store action for DQN training."""
        # This would be integrated with the DQN training system
        # For now, just log the action
        self.logger.log(f"Stored action for training: {action}")
    
    def movement_fight_loop(self, recording_flag: bool = False) -> bool:
        """
        Main fight loop using movement-based decision making.
        
        Args:
            recording_flag: Whether to record the fight
            
        Returns:
            True if fight completed successfully
        """
        self.logger.change_status("Starting movement-based fight loop")
        
        # Initialize bridge
        create_default_bridge_iar(self.emulator)
        
        # Start battle processing
        self.start_battle_processing()
        self.in_battle = True
        self.battle_start_time = time.time()
        
        try:
            while check_for_in_battle_with_delay(self.emulator):
                if recording_flag:
                    # Save frame for recording
                    frame = self.emulator.screenshot()
                    if frame is not None:
                        # Save frame (placeholder - would use actual recording system)
                        pass
                
                # Process any pending actions
                self._process_pending_actions()
                
                # Check if still in battle
                if not check_if_in_battle(self.emulator):
                    self.logger.change_status("Not in battle anymore!")
                    break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
        
        except Exception as e:
            self.logger.error(f"Error in movement fight loop: {e}")
            return False
        
        finally:
            # Stop processing and cleanup
            self.stop_battle_processing()
            self.in_battle = False
            
            # Log fight statistics
            self._log_fight_statistics()
        
        self.logger.change_status("Movement-based fight completed")
        return True
    
    def _process_pending_actions(self):
        """Process any pending actions from the action queue."""
        try:
            while not self.action_queue.empty():
                action = self.action_queue.get_nowait()
                
                # Execute the action
                executed_action = self.movement_bot.execute_action(action)
                
                # Store action for training
                self._store_action_for_training(executed_action)
                
                self.actions_taken += 1
                self.last_action_time = time.time()
                
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing pending actions: {e}")
    
    def execute_action_immediately(self, action: Dict) -> Dict:
        """
        Execute an action immediately and return updated action with success flags.
        
        Args:
            action: Action dictionary with card_index and position
            
        Returns:
            Updated action dictionary with placement_success and detection_success flags
        """
        if not action:
            return action
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            return action
        
        # Execute the action
        executed_action = self.movement_bot.execute_action(action)
        
        # Store for training
        self._store_action_for_training(executed_action)
        
        self.actions_taken += 1
        self.last_action_time = current_time
        
        return executed_action
    
    def _log_fight_statistics(self):
        """Log fight statistics."""
        battle_duration = time.time() - self.battle_start_time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        self.logger.log(f"Fight Statistics:")
        self.logger.log(f"  Duration: {battle_duration:.1f}s")
        self.logger.log(f"  Frames Processed: {self.frames_processed}")
        self.logger.log(f"  Actions Taken: {self.actions_taken}")
        self.logger.log(f"  Average FPS: {fps:.1f}")
        self.logger.log(f"  Average Processing Time: {avg_processing_time:.3f}s")
        
        # Get movement bot stats
        bot_stats = self.movement_bot.get_performance_stats()
        self.logger.log(f"  Active Tracks: {bot_stats.get('active_tracks', 0)}")
        self.logger.log(f"  Total Tracks Created: {bot_stats.get('total_tracks_created', 0)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        stats = {
            'frames_processed': self.frames_processed,
            'actions_taken': self.actions_taken,
            'avg_processing_time': avg_processing_time,
            'fps': fps,
            'in_battle': self.in_battle,
            'battle_duration': time.time() - self.battle_start_time if self.in_battle else 0
        }
        
        # Add movement bot stats
        if self.movement_bot:
            stats.update(self.movement_bot.get_performance_stats())
        
        return stats
    
    def update_config(self, new_config: BotConfig):
        """Update bot configuration."""
        self.config = new_config
        if self.movement_bot:
            self.movement_bot.update_config(new_config.__dict__)
        self.logger.log("Movement fight manager configuration updated")
    
    def save_model(self, filepath: str = None):
        """Save the DQN model."""
        if self.movement_bot:
            self.movement_bot.save_model(filepath)
    
    def load_model(self, filepath: str = None):
        """Load the DQN model."""
        if self.movement_bot:
            self.movement_bot.load_model(filepath)
    
    def reset_training(self):
        """Reset DQN training."""
        if self.movement_bot:
            self.movement_bot.reset_training()
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_battle_processing()
        if self.movement_bot:
            self.movement_bot.cleanup()
        self.logger.log("Movement fight manager cleanup completed")
