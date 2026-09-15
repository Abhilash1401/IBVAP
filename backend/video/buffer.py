"""Bounded rolling frame buffer for pre/post-motion event clip context."""

from __future__ import annotations

import collections
import logging
import threading
from typing import List, Optional

from backend.contracts.frame import FramePacket
from backend.video.exceptions import BufferCapacityError

logger = logging.getLogger(__name__)


class RollingFrameBuffer:
    """Thread-safe circular ring buffer for FramePacket instances.
    
    Maintains a bounded window of recent frames in memory to supply
    pre-motion context to event clip generation without keeping the entire
    video stream in memory.
    """

    def __init__(self, capacity: int = 150) -> None:
        """Initialize buffer with fixed capacity (e.g. 150 frames = 5s @ 30fps).
        
        Args:
            capacity: Maximum number of FramePackets retained. Must be positive.
        """
        if capacity <= 0:
            raise BufferCapacityError(f"Buffer capacity must be > 0, got {capacity}")

        self._capacity: int = capacity
        self._buffer: collections.deque[FramePacket] = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        """Maximum frame capacity of the buffer."""
        return self._capacity

    def append(self, packet: FramePacket) -> None:
        """Add a new FramePacket to the buffer, automatically evicting oldest if full."""
        if not isinstance(packet, FramePacket):
            raise TypeError(f"Expected FramePacket, got {type(packet).__name__}")

        with self._lock:
            self._buffer.append(packet)

    def get_pre_event_frames(self, count: Optional[int] = None) -> List[FramePacket]:
        """Retrieve recent frames leading up to an event.
        
        Args:
            count: Number of recent frames requested. If None, returns all available frames.
            
        Returns:
            A list of FramePackets in chronological order.
        """
        with self._lock:
            if not self._buffer:
                return []
            
            if count is None or count >= len(self._buffer):
                return list(self._buffer)
            
            # Slice the last `count` items from the deque
            return list(self._buffer)[-count:]

    def get_all_frames(self) -> List[FramePacket]:
        """Retrieve a copy of all current frames in chronological order."""
        return self.get_pre_event_frames(count=None)

    def clear(self) -> None:
        """Clear all frames from the buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        """Return the current number of frames stored in the buffer."""
        with self._lock:
            return len(self._buffer)

    def __repr__(self) -> str:
        return f"RollingFrameBuffer(len={len(self)}, capacity={self._capacity})"
