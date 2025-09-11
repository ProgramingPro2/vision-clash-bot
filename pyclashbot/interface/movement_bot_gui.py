"""GUI components for movement-based bot features."""

from typing import Any, Callable, Dict

import FreeSimpleGUI as sg

from ..ai.movement_based_bot import BotConfig


class MovementBotGUI:
    """GUI components for movement-based bot configuration and monitoring."""
    
    def __init__(self):
        """Initialize movement bot GUI components."""
        self.config = BotConfig()
    
    def create_movement_detection_tab(self) -> sg.Tab:
        """Create tab for movement detection configuration."""
        layout = [
            [sg.Text("Movement Detection Settings", font=("Arial", 12, "bold"))],
            [sg.HSeparator()],
            
            [sg.Checkbox("Enable Movement Detection", 
                        key="-MOVEMENT_DETECTION_ENABLED-", 
                        default=self.config.movement_detection_enabled)],
            
            [sg.Text("Min Area:"), 
             sg.InputText(str(self.config.movement_detection_config.min_area), 
                         key="-MIN_AREA-", size=(10, 1))],
            
            [sg.Text("Max Area:"), 
             sg.InputText(str(self.config.movement_detection_config.max_area), 
                         key="-MAX_AREA-", size=(10, 1))],
            
            [sg.Text("Threshold:"), 
             sg.InputText(str(self.config.movement_detection_config.threshold), 
                         key="-THRESHOLD-", size=(10, 1))],
            
            [sg.Checkbox("Use Background Subtractor", 
                        key="-USE_BG_SUBTRACTOR-", 
                        default=self.config.movement_detection_config.use_bg_subtractor)],
            
            [sg.Checkbox("Enable Visualization", 
                        key="-MOVEMENT_VISUALIZATION-", 
                        default=self.config.movement_detection_config.enable_visualization)],
        ]
        
        return sg.Tab("Movement Detection", layout, key="-MOVEMENT_TAB-")
    
    def create_unit_tracking_tab(self) -> sg.Tab:
        """Create tab for unit tracking configuration."""
        layout = [
            [sg.Text("Unit Tracking Settings", font=("Arial", 12, "bold"))],
            [sg.HSeparator()],
            
            [sg.Checkbox("Enable Unit Tracking", 
                        key="-UNIT_TRACKING_ENABLED-", 
                        default=self.config.unit_tracking_enabled)],
            
            [sg.Text("Max Distance:"), 
             sg.InputText(str(self.config.unit_tracking_config.max_distance), 
                         key="-MAX_DISTANCE-", size=(10, 1))],
            
            [sg.Text("Max Occlusion Frames:"), 
             sg.InputText(str(self.config.unit_tracking_config.max_occlusion_frames), 
                         key="-MAX_OCCLUSION-", size=(10, 1))],
            
            [sg.Text("Min Track Length:"), 
             sg.InputText(str(self.config.unit_tracking_config.min_track_length), 
                         key="-MIN_TRACK_LENGTH-", size=(10, 1))],
            
            [sg.Checkbox("Enable Prediction", 
                        key="-ENABLE_PREDICTION-", 
                        default=self.config.unit_tracking_config.enable_prediction)],
            
            [sg.Checkbox("Enable Tracking Visualization", 
                        key="-TRACKING_VISUALIZATION-", 
                        default=self.config.unit_tracking_config.enable_visualization)],
        ]
        
        return sg.Tab("Unit Tracking", layout, key="-TRACKING_TAB-")
    
    def create_dqn_tab(self) -> sg.Tab:
        """Create tab for DQN configuration and management."""
        layout = [
            [sg.Text("Deep Q-Learning Settings", font=("Arial", 12, "bold"))],
            [sg.HSeparator()],
            
            [sg.Checkbox("Enable DQN", 
                        key="-DQN_ENABLED-", 
                        default=self.config.dqn_enabled)],
            
            [sg.Text("Model Path:"), 
             sg.InputText(self.config.dqn_model_path, 
                         key="-MODEL_PATH-", size=(30, 1))],
            
            [sg.Text("Learning Rate:"), 
             sg.InputText("0.001", key="-LEARNING_RATE-", size=(10, 1))],
            
            [sg.Text("Epsilon:"), 
             sg.InputText("1.0", key="-EPSILON-", size=(10, 1))],
            
            [sg.Text("Gamma:"), 
             sg.InputText("0.95", key="-GAMMA-", size=(10, 1))],
            
            [sg.HSeparator()],
            [sg.Text("Model Management", font=("Arial", 10, "bold"))],
            
            [sg.Button("Save Model", key="-SAVE_MODEL-", size=(12, 1)),
             sg.Button("Load Model", key="-LOAD_MODEL-", size=(12, 1))],
            
            [sg.Button("Delete Current Model", key="-DELETE_MODEL-", size=(12, 1)),
             sg.Button("Reset Training", key="-RESET_TRAINING-", size=(12, 1))],
            
            [sg.HSeparator()],
            [sg.Text("Training Progress", font=("Arial", 10, "bold"))],
            
            [sg.Text("Epsilon:"), sg.Text("1.000", key="-CURRENT_EPSILON-", size=(8, 1))],
            [sg.Text("Training Step:"), sg.Text("0", key="-TRAINING_STEP-", size=(8, 1))],
            [sg.Text("Memory Size:"), sg.Text("0", key="-MEMORY_SIZE-", size=(8, 1))],
            [sg.Text("Avg Reward:"), sg.Text("0.000", key="-AVG_REWARD-", size=(8, 1))],
            [sg.Text("Avg Loss:"), sg.Text("0.000", key="-AVG_LOSS-", size=(8, 1))],
        ]
        
        return sg.Tab("Deep Q-Learning", layout, key="-DQN_TAB-")
    
    def create_visualization_tab(self) -> sg.Tab:
        """Create tab for visualization settings."""
        layout = [
            [sg.Text("Visualization Settings", font=("Arial", 12, "bold"))],
            [sg.HSeparator()],
            
            [sg.Checkbox("Enable Real-time Visualization", 
                        key="-REALTIME_VISUALIZATION-", 
                        default=self.config.enable_visualization)],
            
            [sg.Text("Target FPS:"), 
             sg.InputText(str(self.config.target_fps), 
                         key="-TARGET_FPS-", size=(10, 1))],
            
            [sg.HSeparator()],
            [sg.Text("Display Options", font=("Arial", 10, "bold"))],
            
            [sg.Checkbox("Show Unit Bounding Boxes", 
                        key="-SHOW_BOUNDING_BOXES-", default=True)],
            
            [sg.Checkbox("Show Unit IDs", 
                        key="-SHOW_UNIT_IDS-", default=True)],
            
            [sg.Checkbox("Show Speed Vectors", 
                        key="-SHOW_SPEED_VECTORS-", default=True)],
            
            [sg.Checkbox("Show Tower Health", 
                        key="-SHOW_TOWER_HEALTH-", default=True)],
            
            [sg.Checkbox("Show Action Information", 
                        key="-SHOW_ACTION_INFO-", default=True)],
            
            [sg.Checkbox("Show Performance Stats", 
                        key="-SHOW_PERFORMANCE-", default=True)],
            
            [sg.HSeparator()],
            [sg.Text("Visualization Window", font=("Arial", 10, "bold"))],
            
            [sg.Button("Open Visualization Window", key="-OPEN_VIS_WINDOW-", size=(20, 1))],
            [sg.Button("Close Visualization Window", key="-CLOSE_VIS_WINDOW-", size=(20, 1))],
        ]
        
        return sg.Tab("Visualization", layout, key="-VIS_TAB-")
    
    def create_performance_tab(self) -> sg.Tab:
        """Create tab for performance monitoring."""
        layout = [
            [sg.Text("Performance Monitoring", font=("Arial", 12, "bold"))],
            [sg.HSeparator()],
            
            [sg.Text("Frame Processing", font=("Arial", 10, "bold"))],
            [sg.Text("Frame Count:"), sg.Text("0", key="-FRAME_COUNT-", size=(10, 1))],
            [sg.Text("FPS:"), sg.Text("0.0", key="-CURRENT_FPS-", size=(10, 1))],
            [sg.Text("Avg Processing Time:"), sg.Text("0.000", key="-AVG_PROCESSING_TIME-", size=(10, 1))],
            [sg.Text("FPS Performance:"), sg.Text("0.0%", key="-FPS_PERFORMANCE-", size=(10, 1))],
            
            [sg.HSeparator()],
            [sg.Text("Unit Tracking", font=("Arial", 10, "bold"))],
            [sg.Text("Active Tracks:"), sg.Text("0", key="-ACTIVE_TRACKS-", size=(10, 1))],
            [sg.Text("Total Tracks Created:"), sg.Text("0", key="-TOTAL_TRACKS_CREATED-", size=(10, 1))],
            [sg.Text("Total Tracks Lost:"), sg.Text("0", key="-TOTAL_TRACKS_LOST-", size=(10, 1))],
            [sg.Text("Lost Units:"), sg.Text("0", key="-LOST_UNITS-", size=(10, 1))],
            
            [sg.HSeparator()],
            [sg.Text("DQN Training", font=("Arial", 10, "bold"))],
            [sg.Text("Total Episodes:"), sg.Text("0", key="-TOTAL_EPISODES-", size=(10, 1))],
            [sg.Text("Current Epsilon:"), sg.Text("1.000", key="-CURRENT_EPSILON-", size=(10, 1))],
            [sg.Text("Training Step:"), sg.Text("0", key="-TRAINING_STEP-", size=(10, 1))],
            [sg.Text("Memory Size:"), sg.Text("0", key="-MEMORY_SIZE-", size=(10, 1))],
            
            [sg.HSeparator()],
            [sg.Button("Refresh Stats", key="-REFRESH_STATS-", size=(15, 1))],
            [sg.Button("Export Stats", key="-EXPORT_STATS-", size=(15, 1))],
        ]
        
        return sg.Tab("Performance", layout, key="-PERFORMANCE_TAB-")
    
    def create_movement_bot_tab_group(self) -> sg.TabGroup:
        """Create the main tab group for movement-based bot features."""
        tabs = [
            self.create_movement_detection_tab(),
            self.create_unit_tracking_tab(),
            self.create_dqn_tab(),
            self.create_visualization_tab(),
            self.create_performance_tab()
        ]
        
        return sg.TabGroup([tabs], key="-MOVEMENT_BOT_TABS-", enable_events=True)
    
    def create_control_buttons(self) -> list:
        """Create control buttons for movement-based bot."""
        return [
            [sg.Button("Start Movement Bot", key="-START_MOVEMENT_BOT-", size=(15, 1)),
             sg.Button("Stop Movement Bot", key="-STOP_MOVEMENT_BOT-", size=(15, 1))],
            
            [sg.Button("Toggle Movement Detection", key="-TOGGLE_MOVEMENT-", size=(18, 1)),
             sg.Button("Toggle Unit Tracking", key="-TOGGLE_TRACKING-", size=(18, 1))],
            
            [sg.Button("Toggle DQN", key="-TOGGLE_DQN-", size=(15, 1)),
             sg.Button("Toggle Visualization", key="-TOGGLE_VISUALIZATION-", size=(18, 1))],
        ]
    
    def update_performance_stats(self, window: sg.Window, stats: Dict[str, Any]):
        """Update performance statistics in the GUI."""
        try:
            # Frame processing stats
            window["-FRAME_COUNT-"].update(str(stats.get('frame_count', 0)))
            window["-CURRENT_FPS-"].update(f"{stats.get('fps', 0.0):.1f}")
            window["-AVG_PROCESSING_TIME-"].update(f"{stats.get('avg_processing_time', 0.0):.3f}")
            window["-FPS_PERFORMANCE-"].update(f"{stats.get('fps_performance', 0.0)*100:.1f}%")
            
            # Unit tracking stats
            window["-ACTIVE_TRACKS-"].update(str(stats.get('active_tracks', 0)))
            window["-TOTAL_TRACKS_CREATED-"].update(str(stats.get('total_tracks_created', 0)))
            window["-TOTAL_TRACKS_LOST-"].update(str(stats.get('total_tracks_lost', 0)))
            window["-LOST_UNITS-"].update(str(stats.get('lost_units', 0)))
            
            # DQN training stats
            window["-TOTAL_EPISODES-"].update(str(stats.get('total_episodes', 0)))
            window["-CURRENT_EPSILON-"].update(f"{stats.get('epsilon', 1.0):.3f}")
            window["-TRAINING_STEP-"].update(str(stats.get('training_step', 0)))
            window["-MEMORY_SIZE-"].update(str(stats.get('memory_size', 0)))
            window["-AVG_REWARD-"].update(f"{stats.get('avg_reward', 0.0):.3f}")
            window["-AVG_LOSS-"].update(f"{stats.get('avg_loss', 0.0):.3f}")
            
        except Exception as e:
            print(f"Error updating performance stats: {e}")
    
    def get_config_from_gui(self, window: sg.Window) -> BotConfig:
        """Get configuration from GUI values."""
        config = BotConfig()
        
        try:
            # Movement detection config
            config.movement_detection_enabled = window["-MOVEMENT_DETECTION_ENABLED-"].get()
            config.movement_detection_config.min_area = int(window["-MIN_AREA-"].get())
            config.movement_detection_config.max_area = int(window["-MAX_AREA-"].get())
            config.movement_detection_config.threshold = int(window["-THRESHOLD-"].get())
            config.movement_detection_config.use_bg_subtractor = window["-USE_BG_SUBTRACTOR-"].get()
            config.movement_detection_config.enable_visualization = window["-MOVEMENT_VISUALIZATION-"].get()
            
            # Unit tracking config
            config.unit_tracking_enabled = window["-UNIT_TRACKING_ENABLED-"].get()
            config.unit_tracking_config.max_distance = float(window["-MAX_DISTANCE-"].get())
            config.unit_tracking_config.max_occlusion_frames = int(window["-MAX_OCCLUSION-"].get())
            config.unit_tracking_config.min_track_length = int(window["-MIN_TRACK_LENGTH-"].get())
            config.unit_tracking_config.enable_prediction = window["-ENABLE_PREDICTION-"].get()
            config.unit_tracking_config.enable_visualization = window["-TRACKING_VISUALIZATION-"].get()
            
            # DQN config
            config.dqn_enabled = window["-DQN_ENABLED-"].get()
            config.dqn_model_path = window["-MODEL_PATH-"].get()
            
            # Visualization config
            config.enable_visualization = window["-REALTIME_VISUALIZATION-"].get()
            config.target_fps = int(window["-TARGET_FPS-"].get())
            
        except Exception as e:
            print(f"Error getting config from GUI: {e}")
        
        return config
    
    def set_config_to_gui(self, window: sg.Window, config: BotConfig):
        """Set configuration values in the GUI."""
        try:
            # Movement detection config
            window["-MOVEMENT_DETECTION_ENABLED-"].update(config.movement_detection_enabled)
            window["-MIN_AREA-"].update(str(config.movement_detection_config.min_area))
            window["-MAX_AREA-"].update(str(config.movement_detection_config.max_area))
            window["-THRESHOLD-"].update(str(config.movement_detection_config.threshold))
            window["-USE_BG_SUBTRACTOR-"].update(config.movement_detection_config.use_bg_subtractor)
            window["-MOVEMENT_VISUALIZATION-"].update(config.movement_detection_config.enable_visualization)
            
            # Unit tracking config
            window["-UNIT_TRACKING_ENABLED-"].update(config.unit_tracking_enabled)
            window["-MAX_DISTANCE-"].update(str(config.unit_tracking_config.max_distance))
            window["-MAX_OCCLUSION-"].update(str(config.unit_tracking_config.max_occlusion_frames))
            window["-MIN_TRACK_LENGTH-"].update(str(config.unit_tracking_config.min_track_length))
            window["-ENABLE_PREDICTION-"].update(config.unit_tracking_config.enable_prediction)
            window["-TRACKING_VISUALIZATION-"].update(config.unit_tracking_config.enable_visualization)
            
            # DQN config
            window["-DQN_ENABLED-"].update(config.dqn_enabled)
            window["-MODEL_PATH-"].update(config.dqn_model_path)
            
            # Visualization config
            window["-REALTIME_VISUALIZATION-"].update(config.enable_visualization)
            window["-TARGET_FPS-"].update(str(config.target_fps))
            
        except Exception as e:
            print(f"Error setting config to GUI: {e}")


def create_visualization_window() -> sg.Window:
    """Create a separate window for real-time visualization."""
    layout = [
        [sg.Text("Movement-Based Bot Visualization", font=("Arial", 14, "bold"))],
        [sg.HSeparator()],
        [sg.Image(key="-VIS_IMAGE-", size=(800, 600))],
        [sg.HSeparator()],
        [sg.Text("Status:"), sg.Text("Ready", key="-VIS_STATUS-", size=(20, 1))],
        [sg.Button("Close", key="-CLOSE_VIS-", size=(10, 1))]
    ]
    
    return sg.Window("Bot Visualization", layout, finalize=True, resizable=True)
