"""Handlers for movement bot GUI controls integrated into main interface.

This module provides event handlers for the movement bot controls in the main GUI,
including toggles for bot vision and emotes, as well as model management functions.
The handlers work with a global registry system that allows settings to be applied
even when the movement bot is not currently running.
"""

import os
import tkinter as tk
from tkinter import filedialog
from typing import Optional, Dict, Any

import FreeSimpleGUI as sg

from ..ai.movement_bot_registry import set_movement_bot_setting, get_movement_bot_setting


class MovementBotHandlers:
    """Simple handlers for movement bot controls in the main GUI."""
    
    def __init__(self, bot_instance: Optional[Any] = None) -> None:
        """Initialize with optional bot instance.
        
        Args:
            bot_instance: Optional movement bot instance
        """
        self.bot = bot_instance
    
    def set_bot_instance(self, bot_instance: Optional[Any]) -> None:
        """Set the bot instance for handlers.
        
        Args:
            bot_instance: Movement bot instance to use for handlers
        """
        self.bot = bot_instance
    
    def sync_checkbox_state(self, window: sg.Window) -> str:
        """Sync checkbox state with stored settings.
        
        Args:
            window: FreeSimpleGUI window instance
            
        Returns:
            Status message about the sync operation
        """
        try:
            # Get current setting from registry
            enable_emotes = get_movement_bot_setting('enable_emotes', True)
            window["-EMOTES_CHECKBOX-"].update(enable_emotes)
            
            status = "ON" if enable_emotes else "OFF"
            window["-BOT_STATUS-"].update(f"Emotes: {status}")
            return f"Checkbox synced - Emotes: {status}"
        except Exception as e:
            window["-BOT_STATUS-"].update(f"Error: {e}")
            return f"Error syncing checkbox: {e}"
    
    def handle_toggle_bot_vision(self, window: sg.Window) -> str:
        """Handle bot vision toggle.
        
        Args:
            window: FreeSimpleGUI window instance
            
        Returns:
            Status message about the toggle operation
        """
        try:
            # Toggle the setting in the registry
            current_value = get_movement_bot_setting('enable_visualization', True)
            new_value = not current_value
            set_movement_bot_setting('enable_visualization', new_value)
            
            status = "ON" if new_value else "OFF"
            window["-BOT_STATUS-"].update(f"Bot vision: {status}")
            
            if self.bot:
                return f"Bot vision overlay: {status} (Applied to running bot)"
            else:
                return f"Bot vision overlay: {status} (Will be applied when bot starts)"
                
        except Exception as e:
            window["-BOT_STATUS-"].update(f"Error: {e}")
            return f"Error toggling bot vision: {e}"
    
    def handle_emotes_checkbox(self, window: sg.Window, values: Dict[str, Any]) -> str:
        """Handle emote checkbox change.
        
        Args:
            window: FreeSimpleGUI window instance
            values: Dictionary of window values
            
        Returns:
            Status message about the checkbox change
        """
        try:
            # Update the setting in the registry
            new_value = values["-EMOTES_CHECKBOX-"]
            set_movement_bot_setting('enable_emotes', new_value)
            
            status = "ON" if new_value else "OFF"
            window["-BOT_STATUS-"].update(f"Emotes: {status}")
            
            if self.bot:
                return f"Emotes: {status} (Applied to running bot)"
            else:
                return f"Emotes: {status} (Will be applied when bot starts)"
                
        except Exception as e:
            window["-BOT_STATUS-"].update(f"Error: {e}")
            return f"Error updating emotes: {e}"
    
    def handle_delete_model(self, window):
        """Handle model deletion with confirmation."""
        try:
            # Show confirmation dialog
            layout = [
                [sg.Text("Are you sure you want to delete the current model?")],
                [sg.Text("This action cannot be undone!")],
                [sg.Button("Yes, Delete", key="-CONFIRM_DELETE-", button_color=("white", "red")),
                 sg.Button("Cancel", key="-CANCEL_DELETE-")]
            ]
            
            confirm_window = sg.Window("Confirm Model Deletion", layout, modal=True)
            
            while True:
                event, values = confirm_window.read()
                if event in (sg.WINDOW_CLOSED, "-CANCEL_DELETE-"):
                    confirm_window.close()
                    window["-BOT_STATUS-"].update("Model deletion cancelled")
                    return "Model deletion cancelled"
                elif event == "-CONFIRM_DELETE-":
                    confirm_window.close()
                    break
            
            # Delete model files
            model_paths = [
                "models/dqn_model.pth",
                "models/dqn_model_backup.pth"
            ]
            
            deleted_files = []
            for model_path in model_paths:
                if os.path.exists(model_path):
                    os.remove(model_path)
                    deleted_files.append(model_path)
            
            # Delete backup models
            if os.path.exists("models/"):
                for file in os.listdir("models/"):
                    if file.startswith("dqn_model.pth.backup_"):
                        os.remove(os.path.join("models", file))
                        deleted_files.append(file)
            
            if deleted_files:
                window["-BOT_STATUS-"].update("Model deleted successfully")
                return f"Model deleted successfully. Removed: {', '.join(deleted_files)}"
            else:
                window["-BOT_STATUS-"].update("No model files found")
                return "No model files found to delete"
                
        except Exception as e:
            window["-BOT_STATUS-"].update(f"Error: {e}")
            return f"Error deleting model: {e}"
    
    def handle_reset_model(self, window):
        """Handle model reset."""
        try:
            if self.bot:
                self.bot.reset_training()
                window["-BOT_STATUS-"].update("Model reset successfully")
                return "Model reset successfully"
            else:
                window["-BOT_STATUS-"].update("Movement bot not running - Start bot first")
                return "Movement bot not running. Please start the bot with movement bot mode enabled first."
        except Exception as e:
            window["-BOT_STATUS-"].update(f"Error: {e}")
            return f"Error resetting model: {e}"
    
    def handle_load_model(self, window):
        """Handle model loading from file."""
        try:
            # Create a temporary root window for file dialog
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Load DQN Model",
                filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")],
                initialdir="models/"
            )
            
            root.destroy()
            
            if file_path and self.bot:
                # Load the model
                self.bot.load_model(file_path)
                window["-BOT_STATUS-"].update("Model loaded successfully")
                return f"Model loaded from: {file_path}"
            elif file_path:
                window["-BOT_STATUS-"].update("Movement bot not running - Start bot first")
                return f"Model file selected: {file_path} (Movement bot not running. Please start the bot with movement bot mode enabled first.)"
            else:
                window["-BOT_STATUS-"].update("Model loading cancelled")
                return "Model loading cancelled"
                
        except Exception as e:
            window["-BOT_STATUS-"].update(f"Error: {e}")
            return f"Error loading model: {e}"
    
    def handle_save_model(self, window):
        """Handle model saving."""
        try:
            if self.bot:
                self.bot.auto_save_model()
                window["-BOT_STATUS-"].update("Model saved successfully")
                return "Model saved successfully"
            else:
                window["-BOT_STATUS-"].update("Movement bot not running - Start bot first")
                return "Movement bot not running. Please start the bot with movement bot mode enabled first."
        except Exception as e:
            window["-BOT_STATUS-"].update(f"Error: {e}")
            return f"Error saving model: {e}"


# Global handler instance
movement_bot_handlers = MovementBotHandlers()


def handle_movement_bot_event(
    event: str, 
    window: sg.Window, 
    values: Optional[Dict[str, Any]] = None, 
    bot_instance: Optional[Any] = None
) -> Optional[str]:
    """Handle movement bot GUI events.
    
    Args:
        event: The GUI event that occurred
        window: FreeSimpleGUI window instance
        values: Optional dictionary of window values
        bot_instance: Optional movement bot instance
        
    Returns:
        Status message if event was handled, None otherwise
    """
    # Update bot instance if provided
    if bot_instance:
        movement_bot_handlers.set_bot_instance(bot_instance)
    
    # Route events to appropriate handlers
    if event == "-TOGGLE_BOT_VISION-":
        return movement_bot_handlers.handle_toggle_bot_vision(window)
    elif event == "-EMOTES_CHECKBOX-":
        return movement_bot_handlers.handle_emotes_checkbox(window, values)
    elif event == "-DELETE_MODEL-":
        return movement_bot_handlers.handle_delete_model(window)
    elif event == "-RESET_MODEL-":
        return movement_bot_handlers.handle_reset_model(window)
    elif event == "-LOAD_MODEL-":
        return movement_bot_handlers.handle_load_model(window)
    elif event == "-SAVE_MODEL-":
        return movement_bot_handlers.handle_save_model(window)
    
    return None


def sync_movement_bot_ui(window: sg.Window, bot_instance: Optional[Any] = None) -> str:
    """Sync UI elements with bot configuration.
    
    Args:
        window: FreeSimpleGUI window instance
        bot_instance: Optional movement bot instance
        
    Returns:
        Status message about the sync operation
    """
    if bot_instance:
        movement_bot_handlers.set_bot_instance(bot_instance)
        return movement_bot_handlers.sync_checkbox_state(window)
    return "Bot instance not available"
