"""ClassificationRequest contract for IBVAP V1 downstream handoff to Engine 2."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from backend.contracts.motion import MotionRegion


class ClassificationRequest(BaseModel):
    """Standardized request routed from Core to Engine 2 when motion is detected.
    
    Attributes:
        request_id: Unique identifier for this inference request.
        source_id: Originating video source identifier.
        frame_id: Frame correlation identifier.
        timestamp: Acquisition timestamp of the triggering frame.
        frame: The in-memory NumPy frame to run YOLO inference on.
        motion_region: Optional bounding box where motion was localized.
        metadata: Pipeline metadata (e.g. motion score, pre-event context links).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request UUID")
    source_id: str = Field(..., description="Originating source ID")
    frame_id: str = Field(..., description="Frame correlation identifier")
    timestamp: float = Field(..., description="Acquisition epoch timestamp")
    frame: np.ndarray = Field(..., description="NumPy array containing BGR image data")
    motion_region: Optional[MotionRegion] = Field(default=None, description="Localized motion region")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Pipeline contextual metadata")
