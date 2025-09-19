"""Simple handlers for movement bot GUI controls integrated into main interface."""

import os
import tkinter as tk
from tkinter import filedialog
import FreeSimpleGUI as sg


class MovementBotHandlers:
    """Simple handlers for movement bot controls in the main GUI."""
    
    def __init__(self, bot_instance=None):
        """Initialize with optional bot instance."""
        self.bot = bot_instance
    
    def set_bot_instance(self, bot_instance):
        """Set the bot instance for handlers."""
        self.bot = bot_instance
    
    def handle_toggle_bot_vision(self, window):
        """Handle bot vision toggle."""
        try:
            if self.bot:
                # Toggle visualization in bot
                self.bot.config.enable_visualization = not self.bot.config.enable_visualization
                status = "ON" if self.bot.config.enable_visualization else "OFF"
                window["-BOT_STATUS-"].update(f"Bot vision: {status}")
                return f"Bot vision overlay: {status}"
            else:
                window["-BOT_STATUS-"].update("Bot not initialized")
                return "Bot instance not available"
        except Exception as e:
            window["-BOT_STATUS-"].update(f"Error: {e}")
            return f"Error toggling bot vision: {e}"
    
    def handle_toggle_emotes(self, window):
        """Handle emote toggle."""
        try:
            if self.bot:
                # Toggle emotes in bot
                self.bot.config.enable_emotes = not self.bot.config.enable_emotes
                status = "ON" if self.bot.config.enable_emotes else "OFF"
                window["-BOT_STATUS-"].update(f"Emotes: {status}")
                return f"Emotes: {status}"
            else:
                window["-BOT_STATUS-"].update("Bot not initialized")
                return "Bot instance not available"
        except Exception as e:
            window["-BOT_STATUS-"].update(f"Error: {e}")
            return f"Error toggling emotes: {e}"
    
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
                window["-BOT_STATUS-"].update("Bot not initialized")
                return "Bot instance not available"
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
                window["-BOT_STATUS-"].update("Model file selected")
                return f"Model file selected: {file_path} (Bot instance not available)"
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
                window["-BOT_STATUS-"].update("Bot not initialized")
                return "Bot instance not available"
        except Exception as e:
            window["-BOT_STATUS-"].update(f"Error: {e}")
            return f"Error saving model: {e}"


# Global handler instance
movement_bot_handlers = MovementBotHandlers()


def handle_movement_bot_event(event, window, bot_instance=None):
    """Handle movement bot GUI events."""
    # Update bot instance if provided
    if bot_instance:
        movement_bot_handlers.set_bot_instance(bot_instance)
    
    # Route events to appropriate handlers
    if event == "-TOGGLE_BOT_VISION-":
        return movement_bot_handlers.handle_toggle_bot_vision(window)
    elif event == "-TOGGLE_EMOTES-":
        return movement_bot_handlers.handle_toggle_emotes(window)
    elif event == "-DELETE_MODEL-":
        return movement_bot_handlers.handle_delete_model(window)
    elif event == "-RESET_MODEL-":
        return movement_bot_handlers.handle_reset_model(window)
    elif event == "-LOAD_MODEL-":
        return movement_bot_handlers.handle_load_model(window)
    elif event == "-SAVE_MODEL-":
        return movement_bot_handlers.handle_save_model(window)
    
    return None
