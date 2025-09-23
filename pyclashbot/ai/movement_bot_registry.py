"""Global registry for movement bot instances to enable UI access.

This module provides a centralized registry for movement bot instances and their
settings, allowing the UI to interact with the bot even when it's not currently
running. Settings are stored and applied when the bot is created or when they
are changed while the bot is running.
"""

from typing import Optional, Dict, Any
from .movement_based_bot import MovementBasedBot


class MovementBotRegistry:
    """Global registry for movement bot instances and settings."""
    
    _instance: Optional[MovementBasedBot] = None
    _settings: Dict[str, Any] = {
        'enable_visualization': True
    }
    
    @classmethod
    def set_bot(cls, bot: MovementBasedBot):
        """Set the current movement bot instance and apply stored settings."""
        cls._instance = bot
        # Apply stored settings to the bot
        if bot and hasattr(bot, 'config'):
            bot.config.enable_visualization = cls._settings.get('enable_visualization', True)
    
    @classmethod
    def get_bot(cls) -> Optional[MovementBasedBot]:
        """Get the current movement bot instance."""
        return cls._instance
    
    @classmethod
    def clear_bot(cls):
        """Clear the current movement bot instance."""
        cls._instance = None
    
    @classmethod
    def set_setting(cls, key: str, value: Any):
        """Set a setting that will be applied when bot is created."""
        cls._settings[key] = value
        # If bot is already running, apply the setting immediately
        if cls._instance and hasattr(cls._instance, 'config'):
            if key == 'enable_visualization':
                cls._instance.config.enable_visualization = value
    
    @classmethod
    def get_setting(cls, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        return cls._settings.get(key, default)


# Global functions for easy access
def register_movement_bot(bot: MovementBasedBot):
    """Register a movement bot instance globally."""
    MovementBotRegistry.set_bot(bot)


def get_movement_bot() -> Optional[MovementBasedBot]:
    """Get the globally registered movement bot instance."""
    return MovementBotRegistry.get_bot()


def clear_movement_bot():
    """Clear the globally registered movement bot instance."""
    MovementBotRegistry.clear_bot()


def set_movement_bot_setting(key: str, value: Any):
    """Set a movement bot setting."""
    MovementBotRegistry.set_setting(key, value)


def get_movement_bot_setting(key: str, default: Any = None) -> Any:
    """Get a movement bot setting."""
    return MovementBotRegistry.get_setting(key, default)
