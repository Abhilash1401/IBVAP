"""Plugin interface and lifecycle manager for IBVAP V1 Core."""

from __future__ import annotations

import abc
from enum import Enum
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PluginStatus(str, Enum):
    """Lifecycle statuses for IBVAP plugins."""
    UNLOADED = "UNLOADED"
    DISCOVERED = "DISCOVERED"
    LOADED = "LOADED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    PROCESSING = "PROCESSING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


class BasePlugin(abc.ABC):
    """Abstract base class that all IBVAP plugins (Engine 1, Engine 2, etc.) must implement.
    
    Plugins must remain isolated and never invoke other plugins directly.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self._status: PluginStatus = PluginStatus.LOADED

    @property
    def status(self) -> PluginStatus:
        """Current lifecycle status of the plugin."""
        return self._status

    @abc.abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize plugin resources, models, and parameters.
        
        Returns:
            True if initialization was successful, False otherwise.
        """
        pass

    @abc.abstractmethod
    def process(self, input_data: Any) -> Any:
        """Execute processing on the input data and return result.
        
        For Engine 1: receives FramePacket, returns MotionEvent.
        """
        pass

    def get_status(self) -> PluginStatus:
        """Query the operational state of the plugin."""
        return self._status

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release all allocated models and resources cleanly."""
        pass


class PluginManager:
    """Core component responsible for managing plugin lifecycles and registration."""

    def __init__(self) -> None:
        self._plugins: Dict[str, BasePlugin] = {}

    def register_plugin(self, plugin: BasePlugin) -> None:
        """Register a plugin instance with Core."""
        if plugin.name in self._plugins:
            logger.warning(f"Plugin '{plugin.name}' is already registered. Overwriting.")
        self._plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name} (Status: {plugin.status})")

    def initialize_plugin(self, name: str, config: Dict[str, Any]) -> bool:
        """Initialize a registered plugin with configuration."""
        plugin = self._plugins.get(name)
        if plugin is None:
            raise KeyError(f"Plugin '{name}' not found in registry.")

        logger.info(f"Initializing plugin '{name}'...")
        plugin._status = PluginStatus.INITIALIZING
        try:
            success = plugin.initialize(config)
            if success:
                plugin._status = PluginStatus.READY
                logger.info(f"Plugin '{name}' is READY.")
                return True
            else:
                plugin._status = PluginStatus.ERROR
                logger.error(f"Plugin '{name}' initialization failed.")
                return False
        except Exception as e:
            plugin._status = PluginStatus.ERROR
            logger.exception(f"Exception during initialization of plugin '{name}': {e}")
            return False

    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Retrieve a registered plugin by name."""
        return self._plugins.get(name)

    def shutdown_all(self) -> None:
        """Shutdown all registered plugins in an isolated manner."""
        for name, plugin in self._plugins.items():
            try:
                logger.info(f"Shutting down plugin: {name}")
                plugin.shutdown()
                plugin._status = PluginStatus.SHUTDOWN
            except Exception as e:
                logger.error(f"Error shutting down plugin '{name}': {e}")
