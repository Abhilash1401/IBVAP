"""ClassificationResult contract for IBVAP V1 Engine 2 inference output."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DetectedClass(str, Enum):
    """V1 classification categories produced by Engine 2."""
    HUMAN = "HUMAN"
    VEHICLE = "VEHICLE"
    NON_TARGET = "NON_TARGET"


class BoundingBox(BaseModel):
    """Axis-aligned bounding box for a detected object."""
    x: int = Field(..., ge=0, description="X coordinate of top-left corner")
    y: int = Field(..., ge=0, description="Y coordinate of top-left corner")
    width: int = Field(..., gt=0, description="Width of bounding box")
    height: int = Field(..., gt=0, description="Height of bounding box")


class Detection(BaseModel):
    """A single object detection produced by Engine 2.

    Attributes:
        detected_class: HUMAN, VEHICLE, or NON_TARGET.
        confidence: Model confidence score in [0.0, 1.0].
        bounding_box: Pixel-space bounding box of the detected object.
    """
    detected_class: DetectedClass = Field(..., description="Classification category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bounding_box: BoundingBox = Field(..., description="Object bounding box")


class ClassificationResult(BaseModel):
    """Standardized inference result produced by Engine 2.

    Contains all detection data needed by the API/notification layer
    to decide whether to raise a V1 alert.

    Attributes:
        request_id: Correlation ID matching the originating ClassificationRequest.
        source_id: Originating video source identifier.
        frame_id: Frame correlation identifier.
        timestamp: Acquisition epoch timestamp of the original frame.
        detections: List of detected objects (may be empty).
        processing_time_ms: Wall-clock inference time in milliseconds.
        model_name: Model identifier used for this inference.
        metadata: Additional diagnostic metadata.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(..., description="Correlation ID from ClassificationRequest")
    source_id: str = Field(..., description="Originating source ID")
    frame_id: str = Field(..., description="Frame correlation identifier")
    timestamp: float = Field(..., description="Acquisition epoch timestamp")
    detections: List[Detection] = Field(default_factory=list, description="Detected objects")
    processing_time_ms: float = Field(..., ge=0.0, description="Inference wall-clock time in ms")
    model_name: str = Field(..., description="Model identifier used for inference")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic metadata")

    @property
    def has_human(self) -> bool:
        """True if at least one HUMAN detection is present."""
        return any(d.detected_class == DetectedClass.HUMAN for d in self.detections)

    @property
    def has_vehicle(self) -> bool:
        """True if at least one VEHICLE detection is present."""
        return any(d.detected_class == DetectedClass.VEHICLE for d in self.detections)

    @property
    def has_target(self) -> bool:
        """True if at least one HUMAN or VEHICLE detection is present."""
        return self.has_human or self.has_vehicle
