"""Engine 2 - Human / Vehicle Detection Plugin for IBVAP V1.

Uses Ultralytics YOLOv8 for local inference on frames forwarded by Engine 1
through the Core Orchestrator.  Engine 2 MUST NOT independently read the
continuous CCTV stream — it only processes ClassificationRequests.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np

from backend.contracts.classification import ClassificationRequest
from backend.contracts.classification_result import (
    BoundingBox,
    ClassificationResult,
    DetectedClass,
    Detection,
)
from backend.core.plugin import BasePlugin, PluginStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# COCO class-ID → IBVAP class mapping
# ---------------------------------------------------------------------------
# Standard COCO-trained YOLOv8 class indices:
#   0  = person
#   1  = bicycle,  2 = car,  3 = motorcycle,  5 = bus,  7 = truck
# All other classes are mapped to NON_TARGET.
# ---------------------------------------------------------------------------
HUMAN_COCO_IDS: Set[int] = {0}
VEHICLE_COCO_IDS: Set[int] = {1, 2, 3, 5, 7}


def _map_coco_to_ibvap(coco_class_id: int) -> DetectedClass:
    """Map a COCO class index to an IBVAP V1 DetectedClass."""
    if coco_class_id in HUMAN_COCO_IDS:
        return DetectedClass.HUMAN
    elif coco_class_id in VEHICLE_COCO_IDS:
        return DetectedClass.VEHICLE
    return DetectedClass.NON_TARGET


class Engine2Config:
    """Configuration parameters for Engine 2 YOLO inference.

    Attributes:
        model_path: Path to YOLOv8 weights file (default: yolov8n.pt).
        confidence_threshold: Minimum confidence to keep a detection.
        input_size: YOLO inference input resolution (pixels).
        device: Inference device — "cpu", "cuda", "cuda:0", or "auto".
        filter_non_targets: If True, NON_TARGET detections are excluded
            from the result (default True for V1).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        input_size: int = 640,
        device: str = "auto",
        filter_non_targets: bool = True,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.device = device
        self.filter_non_targets = filter_non_targets


class Engine2HumanVehicleDetector(BasePlugin):
    """YOLO-based Human / Vehicle Detection Gate (Engine 2).

    Receives ClassificationRequest objects routed by the Core Orchestrator
    when Engine 1 detects meaningful movement.  Returns ClassificationResult
    containing HUMAN / VEHICLE / NON_TARGET detections with bounding boxes
    and confidence scores.

    Architecture rule: Engine 2 MUST NOT independently read the continuous
    video stream.  It only processes pre-filtered frames from Engine 1.
    """

    def __init__(self, name: str = "engine2_human_vehicle_detector") -> None:
        super().__init__(name=name)
        self.config = Engine2Config()
        self._model = None
        self._model_name: str = ""

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Load YOLOv8 model and prepare for inference.

        Args:
            config: Dictionary with optional keys matching Engine2Config fields.

        Returns:
            True if model was loaded successfully.
        """
        self._status = PluginStatus.INITIALIZING
        self.config = Engine2Config(
            model_path=str(config.get("model_path", "yolov8n.pt")),
            confidence_threshold=float(config.get("confidence_threshold", 0.40)),
            input_size=int(config.get("input_size", 640)),
            device=str(config.get("device", "auto")),
            filter_non_targets=bool(config.get("filter_non_targets", True)),
        )

        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]

            self._model = YOLO(self.config.model_path)
            self._model_name = self.config.model_path

            # Resolve device
            if self.config.device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device

            logger.info(
                f"Engine 2 initialized: model={self.config.model_path}, "
                f"device={device}, conf_thresh={self.config.confidence_threshold}, "
                f"input_size={self.config.input_size}"
            )
            self._status = PluginStatus.READY
            return True

        except Exception as e:
            logger.error(f"Engine 2 initialization failed: {e}")
            self._status = PluginStatus.ERROR
            return False

    def process(self, input_data: ClassificationRequest) -> ClassificationResult:
        """Run YOLO inference on a ClassificationRequest frame.

        Args:
            input_data: ClassificationRequest from Core Orchestrator.

        Returns:
            ClassificationResult with HUMAN/VEHICLE/NON_TARGET detections.

        Raises:
            TypeError: If input is not a ClassificationRequest.
            RuntimeError: If model is not loaded.
        """
        if not isinstance(input_data, ClassificationRequest):
            raise TypeError(
                f"Engine 2 expects ClassificationRequest, got {type(input_data).__name__}"
            )

        if self._model is None:
            raise RuntimeError("Engine 2 model is not loaded. Call initialize() first.")

        self._status = PluginStatus.PROCESSING
        start_time = time.perf_counter()

        try:
            frame = input_data.frame

            # Validate frame
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                logger.warning(f"Engine 2 received invalid frame for request {input_data.request_id}")
                self._status = PluginStatus.READY
                return self._empty_result(input_data, start_time)

            # Optional ROI crop when motion_region is provided
            inference_frame = frame
            roi_offset_x, roi_offset_y = 0, 0

            if input_data.motion_region is not None:
                r = input_data.motion_region
                h, w = frame.shape[:2]
                # Add padding around motion region for context
                pad = 30
                x1 = max(0, r.x - pad)
                y1 = max(0, r.y - pad)
                x2 = min(w, r.x + r.width + pad)
                y2 = min(h, r.y + r.height + pad)

                # Only crop if the ROI is significantly smaller than the full frame
                roi_area = (x2 - x1) * (y2 - y1)
                frame_area = w * h
                if roi_area < frame_area * 0.70:
                    inference_frame = frame[y1:y2, x1:x2]
                    roi_offset_x, roi_offset_y = x1, y1

            # Resolve device
            if self.config.device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device

            # Run YOLO inference
            results = self._model.predict(
                source=inference_frame,
                conf=self.config.confidence_threshold,
                imgsz=self.config.input_size,
                device=device,
                verbose=False,
            )

            # Parse detections
            detections = self._parse_results(results, roi_offset_x, roi_offset_y)

            # Filter NON_TARGET if configured
            if self.config.filter_non_targets:
                detections = [d for d in detections if d.detected_class != DetectedClass.NON_TARGET]

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            self._status = PluginStatus.READY

            result = ClassificationResult(
                request_id=input_data.request_id,
                source_id=input_data.source_id,
                frame_id=input_data.frame_id,
                timestamp=input_data.timestamp,
                detections=detections,
                processing_time_ms=round(elapsed_ms, 2),
                model_name=self._model_name,
                metadata={
                    "confidence_threshold": self.config.confidence_threshold,
                    "input_size": self.config.input_size,
                    "device": device,
                    "total_raw_detections": len(results[0].boxes) if results else 0,
                    "filtered_detections": len(detections),
                },
            )

            if detections:
                classes_found = [d.detected_class.value for d in detections]
                logger.info(
                    f"Engine 2 detected {classes_found} on source '{input_data.source_id}' "
                    f"(frame {input_data.frame_id[:8]}, {elapsed_ms:.1f}ms)"
                )

            return result

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Engine 2 inference error: {e}")
            self._status = PluginStatus.READY
            return ClassificationResult(
                request_id=input_data.request_id,
                source_id=input_data.source_id,
                frame_id=input_data.frame_id,
                timestamp=input_data.timestamp,
                detections=[],
                processing_time_ms=round(elapsed_ms, 2),
                model_name=self._model_name,
                metadata={"error": str(e)},
            )

    def shutdown(self) -> None:
        """Release model and allocated resources."""
        self._model = None
        self._model_name = ""
        self._status = PluginStatus.SHUTDOWN
        logger.info("Engine 2 Human/Vehicle Detector shut down cleanly.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_results(
        self,
        results: list,
        roi_offset_x: int = 0,
        roi_offset_y: int = 0,
    ) -> List[Detection]:
        """Convert Ultralytics YOLO results to IBVAP Detection objects.

        Args:
            results: List of Ultralytics Results objects.
            roi_offset_x: X offset to add if inference was on a cropped ROI.
            roi_offset_y: Y offset to add if inference was on a cropped ROI.

        Returns:
            List of Detection objects.
        """
        detections: List[Detection] = []

        if not results:
            return detections

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                # Extract box coordinates (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                # Apply ROI offset to map back to full-frame coordinates
                x1 += roi_offset_x
                y1 += roi_offset_y
                x2 += roi_offset_x
                y2 += roi_offset_y

                width = max(1, x2 - x1)
                height = max(1, y2 - y1)

                # Extract class and confidence
                cls_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())

                ibvap_class = _map_coco_to_ibvap(cls_id)

                detections.append(
                    Detection(
                        detected_class=ibvap_class,
                        confidence=round(confidence, 4),
                        bounding_box=BoundingBox(
                            x=max(0, x1),
                            y=max(0, y1),
                            width=width,
                            height=height,
                        ),
                    )
                )

        return detections

    def _empty_result(
        self,
        request: ClassificationRequest,
        start_time: float,
    ) -> ClassificationResult:
        """Build an empty ClassificationResult for invalid/error frames."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return ClassificationResult(
            request_id=request.request_id,
            source_id=request.source_id,
            frame_id=request.frame_id,
            timestamp=request.timestamp,
            detections=[],
            processing_time_ms=round(elapsed_ms, 2),
            model_name=self._model_name,
            metadata={"reason": "invalid_frame"},
        )
