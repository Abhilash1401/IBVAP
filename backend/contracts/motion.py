"""MotionEvent contract for IBVAP V1 Engine 1 motion detection."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class MotionEventType(str, Enum):
    """Event types produced by Engine 1."""
    NO_MOTION = "NO_MOTION"
    MOTION_DETECTED = "MOTION_DETECTED"


class MotionRegion(BaseModel):
    """Bounding box representing the primary region of motion in the frame."""
    x: int = Field(..., ge=0, description="X coordinate of top-left corner")
    y: int = Field(..., ge=0, description="Y coordinate of top-left corner")
    width: int = Field(..., gt=0, description="Width of motion bounding box")
    height: int = Field(..., gt=0, description="Height of motion bounding box")

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


class MotionEvent(BaseModel):
    """Standardized output produced by Engine 1 for each evaluated FramePacket.
    
    Attributes:
        event_type: NO_MOTION or MOTION_DETECTED.
        source_id: Originating video source.
        timestamp: Acquisition epoch timestamp.
        frame_id: Unique correlation identifier for the frame.
        motion_score: Normalized or pixel-area motion magnitude.
        motion_region: Optional bounding box of detected motion.
        selected_frame_or_reference: The in-memory NumPy frame or reference for Engine 2.
        metadata: Additional diagnostic or algorithmic metadata.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_type: MotionEventType = Field(..., description="Classification of movement")
    source_id: str = Field(..., description="Originating video source ID")
    timestamp: float = Field(..., description="Capture timestamp")
    frame_id: str = Field(..., description="Correlation identifier for the frame")
    motion_score: float = Field(..., ge=0.0, description="Magnitude of detected motion")
    motion_region: Optional[MotionRegion] = Field(default=None, description="Bounding region of movement")
    selected_frame_or_reference: Optional[np.ndarray] = Field(
        default=None, description="In-memory frame passed downstream when motion is detected"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Algorithmic metadata")
