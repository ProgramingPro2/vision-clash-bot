"""Deep Q-Learning agent for Clash Royale decision making."""

import os
import pickle
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..detection.unit_tracking import TrackedUnit


@dataclass
class GameState:
    """Represents the current game state for DQN input."""
    unit_positions: List[Tuple[float, float]]  # Normalized positions
    unit_sizes: List[float]  # Normalized areas
    unit_speeds: List[Tuple[float, float]]  # Normalized speed vectors
    unit_types: List[int]  # Encoded unit types
    elixir_count: float  # Current elixir (0-10)
    tower_health: List[float]  # Normalized tower health [own_left, own_right, enemy_left, enemy_right]
    time_remaining: float  # Normalized time remaining
    card_availability: List[float]  # Which cards are available (0 or 1)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert game state to tensor for neural network input."""
        # Flatten all features into a single vector
        features = []
        
        # Unit features (pad to fixed size)
        max_units = 20
        unit_features = []
        for i in range(max_units):
            if i < len(self.unit_positions):
                unit_features.extend([
                    self.unit_positions[i][0], self.unit_positions[i][1],
                    self.unit_sizes[i],
                    self.unit_speeds[i][0], self.unit_speeds[i][1],
                    self.unit_types[i]
                ])
            else:
                unit_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        features.extend(unit_features)
        features.append(self.elixir_count)
        features.extend(self.tower_health)
        features.append(self.time_remaining)
        features.extend(self.card_availability)
        
        return torch.FloatTensor(features)


@dataclass
class GameAction:
    """Represents a game action taken by the agent."""
    card_index: int  # Which card to play (0-3)
    position: Tuple[float, float]  # Where to play it (normalized coordinates)
    timestamp: float  # When the action was taken


@dataclass
class GameReward:
    """Represents a reward signal for the agent."""
    immediate_reward: float  # Immediate reward (damage dealt, units killed, etc.)
    game_outcome: Optional[float]  # Final game outcome (-1, 0, 1 for loss, draw, win)
    timestamp: float  # When the reward was received


class DQNNetwork(nn.Module):
    """Deep Q-Network for Clash Royale decision making."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        """
        Initialize DQN network.
        
        Args:
            input_size: Size of input state vector
            hidden_sizes: List of hidden layer sizes
            output_size: Number of possible actions
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: GameState, action: GameAction, reward: GameReward, 
             next_state: GameState, done: bool):
        """Add experience to buffer."""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Learning agent for Clash Royale."""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 hidden_sizes: List[int] = [512, 256, 128],
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            hidden_sizes: Hidden layer sizes for neural network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate of epsilon decay
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, hidden_sizes, action_size)
        self.target_network = DQNNetwork(state_size, hidden_sizes, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_history = []
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
    
    def select_action(self, state: GameState, available_actions: List[int] = None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            available_actions: List of available action indices
            
        Returns:
            Selected action index
        """
        if available_actions is None:
            available_actions = list(range(self.action_size))
        
        if random.random() < self.epsilon:
            # Random action
            return random.choice(available_actions)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                
                # Mask unavailable actions
                masked_q_values = q_values.clone()
                for i in range(self.action_size):
                    if i not in available_actions:
                        masked_q_values[0, i] = float('-inf')
                
                return masked_q_values.argmax().item()
    
    def remember(self, state: GameState, action: GameAction, reward: GameReward,
                 next_state: GameState, done: bool):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self) -> float:
        """
        Train the network on a batch of experiences.
        
        Returns:
            Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        
        # Prepare batch data
        states = torch.stack([exp[0].to_tensor() for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp[1].card_index for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2].immediate_reward for exp in batch]).to(self.device)
        next_states = torch.stack([exp[3].to_tensor() for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp[4] for exp in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.loss_history.append(loss.item())
        return loss.item()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'loss_history': self.loss_history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
            self.episode_rewards = checkpoint['episode_rewards']
            self.episode_lengths = checkpoint['episode_lengths']
            self.loss_history = checkpoint['loss_history']
    
    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'memory_size': len(self.memory),
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'total_episodes': len(self.episode_rewards)
        }
    
    def reset_training(self):
        """Reset training state."""
        self.epsilon = 1.0
        self.training_step = 0
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.loss_history.clear()
        self.memory.buffer.clear()
        
        # Reinitialize networks
        self.q_network = DQNNetwork(self.state_size, [512, 256, 128], self.action_size)
        self.target_network = DQNNetwork(self.state_size, [512, 256, 128], self.action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.q_network.to(self.device)
        self.target_network.to(self.device)


class GameStateProcessor:
    """Processes game data into DQN-compatible state vectors."""
    
    def __init__(self, screen_width: int = 419, screen_height: int = 633):
        """
        Initialize state processor.
        
        Args:
            screen_width: Screen width for normalization
            screen_height: Screen height for normalization
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.unit_type_encoding = {
            "small_unit": 0,
            "medium_unit": 1,
            "large_unit": 2,
            "building": 3,
            "unknown": 4
        }
    
    def normalize_position(self, x: int, y: int) -> Tuple[float, float]:
        """Normalize position to [0, 1] range."""
        return (x / self.screen_width, y / self.screen_height)
    
    def normalize_area(self, area: int) -> float:
        """Normalize area to [0, 1] range."""
        return min(area / 1000.0, 1.0)  # Cap at 1000 pixels
    
    def normalize_speed(self, speed: Tuple[float, float]) -> Tuple[float, float]:
        """Normalize speed vector to [-1, 1] range."""
        max_speed = 50.0  # Maximum expected speed
        return (
            max(-1.0, min(1.0, speed[0] / max_speed)),
            max(-1.0, min(1.0, speed[1] / max_speed))
        )
    
    def process_game_state(self, 
                          tracked_units: List[TrackedUnit],
                          elixir_count: float,
                          tower_health: List[float],
                          time_remaining: float,
                          card_availability: List[bool]) -> GameState:
        """
        Process game data into GameState object.
        
        Args:
            tracked_units: List of currently tracked units
            elixir_count: Current elixir count
            tower_health: Tower health values
            time_remaining: Time remaining in battle
            card_availability: Which cards are available
            
        Returns:
            Processed game state
        """
        # Process unit data
        unit_positions = []
        unit_sizes = []
        unit_speeds = []
        unit_types = []
        
        for unit in tracked_units:
            norm_pos = self.normalize_position(unit.centroid[0], unit.centroid[1])
            norm_area = self.normalize_area(unit.area)
            norm_speed = self.normalize_speed(unit.speed_vector)
            unit_type = self.unit_type_encoding.get(unit.unit_type, 4)
            
            unit_positions.append(norm_pos)
            unit_sizes.append(norm_area)
            unit_speeds.append(norm_speed)
            unit_types.append(unit_type)
        
        # Normalize other values
        norm_elixir = elixir_count / 10.0  # Max elixir is 10
        norm_tower_health = [h / 100.0 for h in tower_health]  # Assume max health is 100
        norm_time = time_remaining / 180.0  # 3 minutes = 180 seconds
        norm_card_availability = [1.0 if available else 0.0 for available in card_availability]
        
        return GameState(
            unit_positions=unit_positions,
            unit_sizes=unit_sizes,
            unit_speeds=unit_speeds,
            unit_types=unit_types,
            elixir_count=norm_elixir,
            tower_health=norm_tower_health,
            time_remaining=norm_time,
            card_availability=norm_card_availability
        )
