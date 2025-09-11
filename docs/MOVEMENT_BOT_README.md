# Movement-Based Unit Tracking Bot

This document describes the new movement-based unit tracking system with Deep Q-Learning integration for the py-clash-bot project.

## Overview

The movement-based bot replaces traditional vision-based card recognition with a hybrid system that:

1. **Detects moving units** using negative greyscale frame differencing
2. **Tracks units** across frames with persistent IDs and speed vectors
3. **Makes decisions** using Deep Q-Learning based on unit positions and game state
4. **Preserves existing card recognition** for hand management
5. **Provides real-time visualization** of tracked units and decisions

## Architecture

### Core Components

#### 1. Movement Detection (`pyclashbot/detection/movement_detection.py`)

- **Negative greyscale frame differencing** to detect moving objects
- **Connected components analysis** to group pixels into unit blobs
- **Size-based unit classification** (small, medium, large, building)
- **Configurable parameters** for sensitivity and filtering

#### 2. Unit Tracking (`pyclashbot/detection/unit_tracking.py`)

- **Centroid-based tracking** with distance matching
- **Persistent unit IDs** across frames and brief occlusions
- **Speed vector calculation** from centroid displacement
- **Occlusion handling** with configurable timeout

#### 3. Deep Q-Learning (`pyclashbot/ai/dqn_agent.py`)

- **Neural network** for decision making
- **Experience replay** for stable training
- **State representation** from unit positions, sizes, speeds, and game state
- **Reward system** based on game outcomes and unit interactions

#### 4. Tower Health Detection (`pyclashbot/detection/tower_health_detection.py`)

- **OCR-based health reading** using Tesseract
- **Health bar analysis** using color detection
- **Template matching** for robust detection
- **Confidence scoring** for reliability

#### 5. Integration Layer (`pyclashbot/ai/movement_based_bot.py`)

- **Main bot class** that coordinates all components
- **Real-time processing** with configurable FPS targets
- **Training data collection** and model management
- **Performance monitoring** and statistics

## Installation

### Dependencies

The following new dependencies have been added to `pyproject.toml`:

```toml
"torch>=2.0.0,<3.0.0",
"scikit-learn>=1.3.0,<2.0.0",
"scipy>=1.11.0,<2.0.0",
"matplotlib>=3.7.0,<4.0.0",
"pytesseract>=0.3.10,<1.0.0"
```

### Setup

1. Install dependencies:

```bash
pip install -e .
```

2. Install Tesseract OCR:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

3. Create model directory:

```bash
mkdir -p models
mkdir -p data
mkdir -p config
```

## Usage

### GUI Integration

The movement-based bot is integrated into the main GUI with a new "Movement Bot" tab containing:

- **Movement Detection Settings**: Configure detection parameters
- **Unit Tracking Settings**: Adjust tracking behavior
- **Deep Q-Learning Settings**: Manage DQN model and training
- **Visualization Settings**: Control real-time display options
- **Performance Monitoring**: View statistics and metrics

### Programmatic Usage

```python
from pyclashbot.ai.movement_based_bot import MovementBasedBot, BotConfig
from pyclashbot.utils.logger import Logger

# Create configuration
config = BotConfig()
config.movement_detection_enabled = True
config.unit_tracking_enabled = True
config.dqn_enabled = True

# Initialize bot
logger = Logger()
bot = MovementBasedBot(config, logger)

# Process frames
frame = emulator.screenshot()
results = bot.process_frame(frame)

# Get action decision
action = results['action']
if action:
    print(f"Play card {action['card_index']} at {action['position']}")

# Save model
bot.save_model("models/my_model.pth")
```

### Fight Integration

The movement-based bot can be used in fights by setting the `movement_bot_mode` parameter:

```python
from pyclashbot.bot.fight import do_fight_state

success = do_fight_state(
    emulator=emulator,
    logger=logger,
    random_fight_mode=False,
    fight_mode_choosed="Classic 1v1",
    movement_bot_mode=True  # Enable movement-based bot
)
```

## Configuration

### Movement Detection

```python
config.movement_detection_config.min_area = 50      # Minimum blob area
config.movement_detection_config.max_area = 5000    # Maximum blob area
config.movement_detection_config.threshold = 30     # Frame differencing threshold
config.movement_detection_config.use_bg_subtractor = False  # Use background subtractor
```

### Unit Tracking

```python
config.unit_tracking_config.max_distance = 50.0     # Max distance for matching
config.unit_tracking_config.max_occlusion_frames = 5  # Max frames for occlusion
config.unit_tracking_config.min_track_length = 3    # Min track length
```

### Deep Q-Learning

```python
config.dqn_config.learning_rate = 0.001
config.dqn_config.gamma = 0.95
config.dqn_config.epsilon = 1.0
config.dqn_config.epsilon_decay = 0.995
config.dqn_config.memory_size = 10000
config.dqn_config.batch_size = 32
```

## Training

### Automatic Training

The bot automatically trains during gameplay:

1. **Collects experiences** from each action taken
2. **Stores state transitions** in replay buffer
3. **Trains periodically** using experience replay
4. **Updates target network** for stable learning

### Manual Training

```python
# Reset training state
bot.reset_training()

# Load existing model
bot.load_model("models/pretrained.pth")

# Save current model
bot.save_model("models/current.pth")
```

### Training Data

Training data is automatically saved to `data/training_data.pkl` and includes:

- Game states (unit positions, sizes, speeds, tower health)
- Actions taken (card played, position)
- Rewards received (immediate and game outcome)
- State transitions

## Visualization

### Real-time Display

The bot provides real-time visualization showing:

- **Unit bounding boxes** with IDs
- **Speed vectors** as arrows
- **Tower health** overlays
- **Action information** (card and position)
- **Performance statistics** (FPS, processing time)

### Visualization Controls

```python
# Enable/disable visualization
config.enable_visualization = True

# Control specific elements
config.show_bounding_boxes = True
config.show_unit_ids = True
config.show_speed_vectors = True
config.show_tower_health = True
config.show_action_info = True
config.show_performance_stats = True
```

## Performance

### Target Performance

- **30+ FPS** real-time processing
- **<50ms** average processing time per frame
- **Real-time decision making** with minimal latency
- **Efficient memory usage** with configurable buffer sizes

### Optimization

The system is optimized for real-time performance:

1. **Efficient algorithms** for movement detection
2. **Optimized tracking** with distance-based matching
3. **GPU acceleration** for neural network inference
4. **Configurable processing** to balance accuracy and speed

## Testing

### Component Tests

Run the test suite to verify all components:

```bash
python tests/test_movement_bot.py
```

This tests:

- Movement detection accuracy
- Unit tracking persistence
- DQN training and inference
- Tower health detection
- Integration and performance

### Manual Testing

1. **Enable movement bot** in GUI
2. **Start a fight** with movement bot mode
3. **Monitor performance** in real-time
4. **Check visualization** for accuracy
5. **Review training progress** in logs

## Troubleshooting

### Common Issues

1. **Low FPS**: Reduce target FPS or disable visualization
2. **Poor tracking**: Adjust max_distance and occlusion parameters
3. **Training instability**: Reduce learning rate or increase batch size
4. **OCR errors**: Install Tesseract and check image quality

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Monitor performance statistics:

```python
stats = bot.get_performance_stats()
print(f"FPS: {stats['fps']:.1f}")
print(f"Processing time: {stats['avg_processing_time']:.3f}s")
print(f"Active tracks: {stats['active_tracks']}")
```

## Future Enhancements

### Planned Features

1. **Advanced unit classification** using CNN
2. **Multi-object tracking** with Kalman filters
3. **Reinforcement learning** with more sophisticated rewards
4. **Real-time model updates** during gameplay
5. **A/B testing framework** for different strategies

### Research Areas

1. **Computer vision** improvements for better detection
2. **Deep learning** architectures for decision making
3. **Game theory** for strategic play
4. **Transfer learning** between different game modes

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Code Style

Follow the existing code style:

- Use type hints
- Add docstrings
- Follow PEP 8
- Include tests for new features

### Testing

All new features must include:

- Unit tests for individual components
- Integration tests for the full system
- Performance benchmarks
- Documentation updates

## License

This project maintains the same license as the original py-clash-bot project.
