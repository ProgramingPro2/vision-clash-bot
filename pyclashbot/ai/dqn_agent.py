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
from ..utils.logger import Logger


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
    action_type: str  # "play_card" or "wait"
    card_index: Optional[int] = None  # Which card to play (0-3) if action_type is "play_card"
    position: Optional[Tuple[float, float]] = None  # Where to play it (normalized coordinates)
    timestamp: float = 0.0  # When the action was taken
    card_identity: Optional[str] = None  # Identified card name (e.g., "zap", "hog")
    placement_success: bool = False  # Whether placement was successful
    detection_success: bool = False  # Whether unit was detected after placement
    elixir_cost: float = 0.0  # Elixir cost of the action


@dataclass
class GameReward:
    """Represents a reward signal for the agent."""
    immediate_reward: float  # Immediate reward (damage dealt, units killed, etc.)
    placement_reward: float = 0.0  # Reward for successful card placement
    detection_reward: float = 0.0  # Reward for successful unit detection after placement
    elixir_efficiency_reward: float = 0.0  # Reward for good elixir management
    wait_reward: float = 0.0  # Reward for strategic waiting
    game_outcome: Optional[float] = None  # Final game outcome (-1, 0, 1 for loss, draw, win)
    timestamp: float = 0.0  # When the reward was received


class DQNNetwork(nn.Module):
    """Improved Deep Q-Network for Clash Royale decision making with attention mechanism."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        """
        Initialize improved DQN network.
        
        Args:
            input_size: Size of input state vector
            hidden_sizes: List of hidden layer sizes
            output_size: Number of possible actions
        """
        super(DQNNetwork, self).__init__()
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Main network layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1 if i < len(hidden_sizes) - 1 else 0.2)  # Less dropout in early layers
            ])
            prev_size = hidden_size
        
        self.main_network = nn.Sequential(*layers)
        
        # Separate heads for different action types
        self.action_head = nn.Linear(prev_size, 5)  # wait + 4 cards
        self.position_head = nn.Linear(prev_size, 2)  # x, y coordinates
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Normalize input
        x = self.input_norm(x)
        
        # Main network
        features = self.main_network(x)
        
        # Separate heads
        action_output = self.action_head(features)
        position_output = self.position_head(features)
        
        # Combine outputs: [action_scores, position_x, position_y]
        return torch.cat([action_output, position_output], dim=1)


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
                 logger: Optional[Logger] = None,
                 hidden_sizes: List[int] = [512, 256, 128],
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100,
                 placement_reward: float = 0.1,
                 detection_reward: float = 0.2,
                 elixir_efficiency_reward: float = 0.05,
                 wait_reward: float = 0.02):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            logger: Logger instance for logging
            hidden_sizes: Hidden layer sizes for neural network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate of epsilon decay
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            placement_reward: Reward for successful card placement
            detection_reward: Reward for successful unit detection
            elixir_efficiency_reward: Reward for good elixir management
            wait_reward: Reward for strategic waiting
        """
        self.state_size = state_size
        self.action_size = action_size
        self.logger = logger
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.placement_reward = placement_reward
        self.detection_reward = detection_reward
        self.elixir_efficiency_reward = elixir_efficiency_reward
        self.wait_reward = wait_reward
        
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
    
    def select_action(self, state: GameState, available_cards: List[int] = None, 
                     card_elixir_costs: List[float] = None) -> GameAction:
        """
        Select action using epsilon-greedy policy with wait option and continuous placement.
        
        Args:
            state: Current game state
            available_cards: List of available card indices (0-3)
            card_elixir_costs: List of elixir costs for each card
            
        Returns:
            GameAction object with action type, card, position, etc.
        """
        if self.logger:
            self.logger.log("=" * 80)
            self.logger.log("DQN SELECT_ACTION DEBUG START")
            self.logger.log("=" * 80)
            self.logger.log(f"Input state details:")
            self.logger.log(f"  - Elixir count: {state.elixir_count}")
            self.logger.log(f"  - Unit positions: {len(state.unit_positions)} units")
            self.logger.log(f"  - Unit sizes: {len(state.unit_sizes)} sizes")
            self.logger.log(f"  - Unit speeds: {len(state.unit_speeds)} speeds")
            self.logger.log(f"  - Unit types: {len(state.unit_types)} types")
            self.logger.log(f"  - Tower health: {state.tower_health}")
            self.logger.log(f"  - Time remaining: {state.time_remaining}")
            self.logger.log(f"  - Card availability: {state.card_availability}")
            self.logger.log(f"  - Epsilon: {self.epsilon:.6f}")
            self.logger.log(f"  - Training step: {self.training_step}")
            self.logger.log(f"  - Memory size: {len(self.memory)}")
        
        if available_cards is None:
            available_cards = list(range(4))
        if card_elixir_costs is None:
            card_elixir_costs = [3.0, 3.0, 3.0, 3.0]  # Default costs
        
        if self.logger:
            self.logger.log(f"Available cards: {available_cards}")
            self.logger.log(f"Card elixir costs: {card_elixir_costs}")
            self.logger.log(f"Device: {self.device}")
            self.logger.log(f"Q-network parameters: {sum(p.numel() for p in self.q_network.parameters())} total params")
        
        # Check which cards we can afford
        # Normalize card costs to match elixir normalization (0-1 range)
        affordable_cards = []
        if self.logger:
            self.logger.log("Checking card affordability:")
        for i, card_idx in enumerate(available_cards):
            cost = card_elixir_costs[i]
            # Normalize card cost to match elixir normalization (divide by 10)
            normalized_cost = cost / 10.0
            can_afford = normalized_cost <= state.elixir_count
            if self.logger:
                self.logger.log(f"  - Card {card_idx}: cost={cost}, normalized_cost={normalized_cost}, elixir={state.elixir_count}, affordable={can_afford}")
            if can_afford:
                affordable_cards.append(card_idx)
        
        if self.logger:
            self.logger.log(f"Affordable cards: {affordable_cards}")
            self.logger.log(f"Total affordable: {len(affordable_cards)}/{len(available_cards)}")
        
        # Always allow waiting
        available_actions = ["wait"]
        if affordable_cards:
            available_actions.extend([f"play_card_{i}" for i in affordable_cards])
        
        if self.logger:
            self.logger.log(f"Available actions: {available_actions}")
            self.logger.log(f"Total available actions: {len(available_actions)}")
        
        random_value = random.random()
        if self.logger:
            self.logger.log(f"Random value: {random_value:.6f}, Epsilon: {self.epsilon:.6f}")
        
        if random_value < self.epsilon:
            # Random action
            if self.logger:
                self.logger.log("=" * 40)
                self.logger.log("RANDOM ACTION PATH")
                self.logger.log("=" * 40)
                self.logger.log(f"Taking RANDOM action (random={random_value:.6f} < epsilon={self.epsilon:.6f})")
            
            wait_chance = random.random()
            if self.logger:
                self.logger.log(f"Wait chance: {wait_chance:.6f} (30% threshold)")
            
            if wait_chance < 0.3 and available_actions:  # 30% chance to wait
                if self.logger:
                    self.logger.log("Random action: WAIT (30% chance)")
                action = GameAction(
                    action_type="wait",
                    timestamp=time.time(),
                    elixir_cost=0.0
                )
                if self.logger:
                    self.logger.log(f"Generated wait action: {action}")
                return action
            elif affordable_cards:
                card_index = random.choice(affordable_cards)
                position = (random.random(), random.random())
                elixir_cost = card_elixir_costs[available_cards.index(card_index)] / 10.0  # Normalize to match elixir
                if self.logger:
                    self.logger.log(f"Random action: PLAY_CARD {card_index} at {position}")
                    self.logger.log(f"  - Card index: {card_index}")
                    self.logger.log(f"  - Position: {position}")
                    self.logger.log(f"  - Elixir cost: {elixir_cost}")
                action = GameAction(
                    action_type="play_card",
                    card_index=card_index,
                    position=position,
                    timestamp=time.time(),
                    elixir_cost=elixir_cost
                )
                if self.logger:
                    self.logger.log(f"Generated play_card action: {action}")
                return action
            else:
                if self.logger:
                    self.logger.log("Random action: WAIT (no affordable cards)")
                action = GameAction(
                    action_type="wait",
                    timestamp=time.time(),
                    elixir_cost=0.0
                )
                if self.logger:
                    self.logger.log(f"Generated fallback wait action: {action}")
                return action
        else:
            # Greedy action using neural network
            if self.logger:
                self.logger.log("=" * 40)
                self.logger.log("GREEDY ACTION PATH")
                self.logger.log("=" * 40)
                self.logger.log(f"Taking GREEDY action (random={random_value:.6f} >= epsilon={self.epsilon:.6f})")
            
            with torch.no_grad():
                if self.logger:
                    self.logger.log("Converting state to tensor...")
                state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
                if self.logger:
                    self.logger.log(f"State tensor shape: {state_tensor.shape}")
                    self.logger.log(f"State tensor device: {state_tensor.device}")
                    self.logger.log(f"State tensor dtype: {state_tensor.dtype}")
                    self.logger.log(f"State tensor min/max: {state_tensor.min().item():.6f}/{state_tensor.max().item():.6f}")
                
                if self.logger:
                    self.logger.log("Forward pass through Q-network...")
                q_values = self.q_network(state_tensor)
                
                if self.logger:
                    self.logger.log(f"Q-values shape: {q_values.shape}")
                    self.logger.log(f"Q-values device: {q_values.device}")
                    self.logger.log(f"Q-values dtype: {q_values.dtype}")
                    self.logger.log(f"Raw Q-values: {q_values[0].tolist()}")
                    self.logger.log(f"Action Q-values (first 5): {q_values[0, :5].tolist()}")
                    self.logger.log(f"Position Q-values (last 2): {q_values[0, 5:7].tolist()}")
                
                # Action space: [wait, card0, card1, card2, card3, pos_x, pos_y]
                # First 5 outputs: wait + 4 cards
                action_q_values = q_values[0, :5]
                
                if self.logger:
                    self.logger.log("Creating action masks...")
                # Mask unavailable/unaffordable actions
                masked_q_values = action_q_values.clone()
                masked_q_values[0] = float('-inf')  # Mask wait initially
                
                if self.logger:
                    self.logger.log(f"Initial masked values: {masked_q_values.tolist()}")
                
                # Check if we should wait (low elixir or no good plays)
                if state.elixir_count < 2.0 or not affordable_cards:
                    masked_q_values[0] = action_q_values[0]  # Allow waiting
                    if self.logger:
                        self.logger.log(f"Allowing wait due to low elixir ({state.elixir_count}) or no affordable cards")
                        self.logger.log(f"Updated masked values: {masked_q_values.tolist()}")
                else:
                    # Mask unaffordable cards
                    if self.logger:
                        self.logger.log("Masking unaffordable cards...")
                    for i, card_idx in enumerate(available_cards):
                        if card_idx not in affordable_cards:
                            masked_q_values[card_idx + 1] = float('-inf')
                            if self.logger:
                                self.logger.log(f"  - Masked card {card_idx} (index {card_idx + 1})")
                    if self.logger:
                        self.logger.log(f"Final masked values: {masked_q_values.tolist()}")
                
                # Select best action
                if self.logger:
                    self.logger.log("Selecting best action...")
                action_idx = masked_q_values.argmax().item()
                
                if self.logger:
                    self.logger.log(f"Selected action index: {action_idx}")
                    self.logger.log(f"Selected Q-value: {masked_q_values[action_idx].item():.6f}")
                
                if action_idx == 0:  # Wait action
                    if self.logger:
                        self.logger.log("Greedy action: WAIT")
                    return GameAction(
                        action_type="wait",
                        timestamp=time.time(),
                        elixir_cost=0.0
                    )
                else:  # Play card action
                    card_index = action_idx - 1
                    if card_index in affordable_cards:
                        # Get position from network outputs
                        pos_x = torch.sigmoid(q_values[0, 5]).item()  # Normalize to [0,1]
                        pos_y = torch.sigmoid(q_values[0, 6]).item()  # Normalize to [0,1]
                        
                        if self.logger:
                            self.logger.log(f"Greedy action: PLAY_CARD {card_index} at ({pos_x:.3f}, {pos_y:.3f})")
                        
                        return GameAction(
                            action_type="play_card",
                            card_index=card_index,
                            position=(pos_x, pos_y),
                            timestamp=time.time(),
                            elixir_cost=card_elixir_costs[available_cards.index(card_index)] / 10.0  # Normalize to match elixir
                        )
                    else:
                        # Fallback to wait if card not affordable
                        if self.logger:
                            self.logger.log(f"Greedy action: WAIT (card {card_index} not affordable)")
                        return GameAction(
                            action_type="wait",
                            timestamp=time.time(),
                            elixir_cost=0.0
                        )
    
    def remember(self, state: GameState, action: GameAction, reward: GameReward,
                 next_state: GameState, done: bool):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self) -> float:
        """
        Train the network on a batch of experiences with improved loss function.
        
        Returns:
            Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch with priority-based sampling
        batch = self.memory.sample(self.batch_size)
        
        # Prepare batch data
        states = torch.stack([exp[0].to_tensor() for exp in batch]).to(self.device)
        
        # Convert actions to action indices and positions
        action_indices = []
        target_positions = []
        for exp in batch:
            action = exp[1]
            if action.action_type == "wait":
                action_indices.append(0)  # Wait action is index 0
                target_positions.append([0.5, 0.5])  # Neutral position for wait
            else:  # play_card
                action_indices.append(action.card_index + 1)  # Card actions are indices 1-4
                target_positions.append([action.position[0], action.position[1]])
        
        actions = torch.LongTensor(action_indices).to(self.device)
        target_positions = torch.FloatTensor(target_positions).to(self.device)
        
        # Combine all reward components with importance weighting
        total_rewards = torch.FloatTensor([
            exp[2].immediate_reward * 2.0 +  # Higher weight for immediate rewards
            exp[2].placement_reward + 
            exp[2].detection_reward * 1.5 +  # Higher weight for detection
            exp[2].elixir_efficiency_reward + 
            exp[2].wait_reward
            for exp in batch
        ]).to(self.device)
        
        next_states = torch.stack([exp[3].to_tensor() for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp[4] for exp in batch]).to(self.device)
        
        # Current Q values
        current_outputs = self.q_network(states)
        current_action_q = current_outputs[:, :5].gather(1, actions.unsqueeze(1))
        current_positions = current_outputs[:, 5:7]
        
        # Next Q values from target network
        with torch.no_grad():
            next_outputs = self.target_network(next_states)
            next_q_values = next_outputs[:, :5].max(1)[0]
            target_q_values = total_rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute losses
        action_loss = F.mse_loss(current_action_q.squeeze(), target_q_values)
        position_loss = F.mse_loss(current_positions, target_positions)
        
        # Combined loss with weighting
        total_loss = action_loss + 0.1 * position_loss  # Position loss is less important
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network with soft update
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            # Soft update instead of hard copy
            tau = 0.005  # Soft update rate
            for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        
        # Adaptive epsilon decay
        if self.epsilon > self.epsilon_min:
            # Slower decay for better exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.loss_history.append(total_loss.item())
        
        # Log training progress periodically
        if self.training_step % 100 == 0 and self.logger:
            self.logger.log(f"DQN Training - Step: {self.training_step}, Total Loss: {total_loss.item():.4f}, "
                          f"Action Loss: {action_loss.item():.4f}, Position Loss: {position_loss.item():.4f}, "
                          f"Epsilon: {self.epsilon:.3f}")
        
        return total_loss.item()
    
    def calculate_placement_reward(self, action: GameAction) -> float:
        """Calculate reward for successful card placement."""
        if action.placement_success:
            return self.placement_reward
        return 0.0
    
    def calculate_detection_reward(self, action: GameAction) -> float:
        """Calculate reward for successful unit detection after placement."""
        if action.detection_success:
            return self.detection_reward
        return 0.0
    
    def calculate_elixir_efficiency_reward(self, action: GameAction, current_elixir: float) -> float:
        """Calculate reward for good elixir management."""
        if action.action_type == "wait":
            # Reward waiting when elixir is low
            if current_elixir < 3.0:
                return self.elixir_efficiency_reward
        else:  # play_card
            # Reward efficient elixir usage
            elixir_remaining = current_elixir - action.elixir_cost
            if elixir_remaining >= 2.0:  # Good elixir management
                return self.elixir_efficiency_reward * 0.5
        return 0.0
    
    def calculate_wait_reward(self, action: GameAction, game_state: GameState) -> float:
        """Calculate reward for strategic waiting."""
        if action.action_type == "wait":
            # Reward waiting in appropriate situations
            if game_state.elixir_count < 4.0:  # Low elixir
                return self.wait_reward
            elif len([h for h in game_state.tower_health if h < 0.3]) > 0:  # Critical health
                return self.wait_reward * 0.5  # Smaller reward for waiting when in danger
        return 0.0
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        try:
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
            if self.logger:
                self.logger.log(f"DQN model saved to {filepath}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save DQN model to {filepath}: {e}")
            raise
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        try:
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
                if self.logger:
                    self.logger.log(f"DQN model loaded from {filepath}")
            else:
                if self.logger:
                    self.logger.log(f"DQN model file not found: {filepath}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load DQN model from {filepath}: {e}")
            raise
    
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
