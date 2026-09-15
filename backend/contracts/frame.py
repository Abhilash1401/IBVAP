"""FramePacket and video ingestion contracts for IBVAP V1."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceType(str, Enum):
    """Supported video source types in IBVAP V1."""
    FILE = "FILE"
    RTSP = "RTSP"


class SourceStatus(str, Enum):
    """Operational statuses for video sources."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"
    EOS = "EOS"


class SourceConfig(BaseModel):
    """Configuration input for a video source."""
    model_config = ConfigDict(frozen=True)

    source_id: str = Field(..., description="Unique identifier for the video source")
    source_type: SourceType = Field(..., description="Type of video source (FILE or RTSP)")
    source_uri: str = Field(..., description="File path or RTSP URL")
    target_fps: Optional[float] = Field(default=None, description="Optional target/limiting frame rate")
    reconnect_attempts: int = Field(default=3, description="Number of reconnect attempts for network streams")
    reconnect_delay_seconds: float = Field(default=2.0, description="Delay between reconnect attempts in seconds")


class FramePacket(BaseModel):
    """Canonical frame container passed from Video Integration to Core / Engine 1.
    
    Attributes:
        frame_id: Unique identifier for the processed frame.
        source_id: Identifier of the originating video source.
        timestamp: POSIX epoch timestamp of acquisition.
        frame: OpenCV / NumPy image array (BGR).
        frame_width: Pixel width of the frame.
        frame_height: Pixel height of the frame.
        sequence_number: Monotonically increasing sequence number for this source.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique frame UUID")
    source_id: str = Field(..., description="Originating source ID")
    timestamp: float = Field(..., description="Acquisition epoch timestamp")
    frame: np.ndarray = Field(..., description="NumPy array containing BGR image data")
    frame_width: int = Field(..., description="Pixel width of the image")
    frame_height: int = Field(..., description="Pixel height of the image")
    sequence_number: int = Field(..., ge=0, description="0-indexed sequence number")

    @field_validator("frame")
    @classmethod
    def validate_numpy_frame(cls, v: Any) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise TypeError(f"frame must be a numpy.ndarray, got {type(v).__name__}")
        if v.ndim not in (2, 3):
            raise ValueError(f"frame must have 2 or 3 dimensions, got {v.ndim}")
        return v

    @model_validator(mode="after")
    def validate_dimensions(self) -> FramePacket:
        h, w = self.frame.shape[:2]
        if self.frame_width != w:
            raise ValueError(f"frame_width ({self.frame_width}) does not match frame.shape[1] ({w})")
        if self.frame_height != h:
            raise ValueError(f"frame_height ({self.frame_height}) does not match frame.shape[0] ({h})")
        return self

    @classmethod
    def create(
        cls,
        source_id: str,
        timestamp: float,
        frame: np.ndarray,
        sequence_number: int,
        frame_id: Optional[str] = None,
    ) -> FramePacket:
        """Convenience factory method to create a FramePacket directly from a numpy frame."""
        h, w = frame.shape[:2]
        return cls(
            frame_id=frame_id or str(uuid.uuid4()),
            source_id=source_id,
            timestamp=timestamp,
            frame=frame,
            frame_width=w,
            frame_height=h,
            sequence_number=sequence_number,
        )
