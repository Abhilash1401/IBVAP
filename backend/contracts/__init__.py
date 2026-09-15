"""IBVAP Shared Data Contracts.

This package defines canonical data structures and interfaces exchanged across
system boundaries according to the IBVAP V1 specification.
"""

from backend.contracts.classification import ClassificationRequest
from backend.contracts.classification_result import (
    BoundingBox,
    ClassificationResult,
    DetectedClass,
    Detection,
)
from backend.contracts.frame import (
    FramePacket,
    SourceConfig,
    SourceStatus,
    SourceType,
)
from backend.contracts.motion import (
    MotionEvent,
    MotionEventType,
    MotionRegion,
)

__all__ = [
    "BoundingBox",
    "ClassificationRequest",
    "ClassificationResult",
    "DetectedClass",
    "Detection",
    "FramePacket",
    "MotionEvent",
    "MotionEventType",
    "MotionRegion",
    "SourceConfig",
    "SourceStatus",
    "SourceType",
]
