"""High-level VideoReader engine coordinating video sources, buffer, and FramePackets."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Iterator, Optional

from backend.contracts.frame import FramePacket
from backend.video.buffer import RollingFrameBuffer
from backend.video.source import VideoSource

logger = logging.getLogger(__name__)


class VideoReader:
    """Orchestrator for reading frames from a VideoSource and emitting FramePackets.
    
    Manages sequence numbering, timestamping, FramePacket encapsulation,
    and rolling buffer updates.
    """

    def __init__(
        self,
        source: VideoSource,
        buffer: Optional[RollingFrameBuffer] = None,
    ) -> None:
        self.source = source
        self.buffer = buffer
        self._sequence_number: int = 0
        self._is_running: bool = False

    @property
    def sequence_number(self) -> int:
        """Current monotonic sequence number."""
        return self._sequence_number

    @property
    def source_id(self) -> str:
        """Source ID from the underlying video source."""
        return self.source.source_id

    def open(self) -> bool:
        """Open the underlying video source."""
        self._sequence_number = 0
        self._is_running = True
        return self.source.open()

    def read_packet(self) -> Optional[FramePacket]:
        """Read the next frame, package it into a FramePacket, and buffer it.
        
        Returns:
            A new FramePacket if read successfully, or None if EOS or error.
        """
        if not self.source.is_opened():
            if not self.open():
                return None

        success, raw_frame = self.source.read_frame()
        if not success or raw_frame is None:
            self._is_running = False
            return None

        now = time.time()
        packet = FramePacket.create(
            source_id=self.source_id,
            timestamp=now,
            frame=raw_frame,
            sequence_number=self._sequence_number,
            frame_id=str(uuid.uuid4()),
        )

        self._sequence_number += 1

        if self.buffer is not None:
            self.buffer.append(packet)

        return packet

    def stream_packets(self) -> Iterator[FramePacket]:
        """Generator that continuously yields FramePackets until EOS or stopped."""
        self._is_running = True
        try:
            while self._is_running:
                packet = self.read_packet()
                if packet is None:
                    break
                yield packet
        finally:
            self.close()

    def close(self) -> None:
        """Stop streaming and release the underlying video source."""
        self._is_running = False
        self.source.release()
        logger.info(f"VideoReader closed for source: {self.source_id}")

    def __enter__(self) -> VideoReader:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
