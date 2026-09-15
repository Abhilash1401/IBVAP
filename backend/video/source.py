"""Abstract base class for all video sources in IBVAP V1."""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np

from backend.contracts.frame import SourceConfig, SourceStatus

logger = logging.getLogger(__name__)


class VideoSource(abc.ABC):
    """Abstract base class defining the lifecycle and interface for video sources.
    
    All video sources (files, RTSP streams, virtual devices) must adhere to
    this contract to maintain full source abstraction.
    """

    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self._status: SourceStatus = SourceStatus.DISCONNECTED

    @property
    def source_id(self) -> str:
        """Originating source ID."""
        return self.config.source_id

    @property
    def status(self) -> SourceStatus:
        """Current operational status of the source."""
        return self._status

    @abc.abstractmethod
    def open(self) -> bool:
        """Open the video source and prepare for frame reading.
        
        Returns:
            True if opened successfully, False otherwise.
        """
        pass

    @abc.abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the video source.
        
        Returns:
            A tuple of (success, frame). When success is False and stream ended,
            frame is None.
        """
        pass

    @abc.abstractmethod
    def is_opened(self) -> bool:
        """Check if the underlying capture resource is open and active."""
        pass

    @abc.abstractmethod
    def release(self) -> None:
        """Release underlying capture resources and reset state."""
        pass

    @abc.abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return video stream metadata (fps, width, height, frame count, etc.)."""
        pass

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
