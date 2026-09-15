"""Engine 1 - Lightweight Motion Detection Plugin for IBVAP V1."""

from __future__ import annotations

import logging

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from backend.contracts.frame import FramePacket
from backend.contracts.motion import MotionEvent, MotionEventType, MotionRegion
from backend.core.plugin import BasePlugin, PluginStatus

logger = logging.getLogger(__name__)


def merge_bounding_boxes(boxes: List[Dict[str, int]], proximity: int = 40) -> List[Dict[str, int]]:
    """Cluster and merge fragmented bounding boxes from the same moving entity.
    
    If two boxes overlap or are within `proximity` pixels of each other,
    they are merged into a single cohesive bounding rectangle.
    """
    if not boxes:
        return []

    rects = [[b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]] for b in boxes]
    merged = True

    while merged:
        merged = False
        new_rects = []
        skip = set()

        for i in range(len(rects)):
            if i in skip:
                continue
            x1, y1, x2, y2 = rects[i]

            for j in range(i + 1, len(rects)):
                if j in skip:
                    continue
                jx1, jy1, jx2, jy2 = rects[j]

                # Check proximity / overlap in both dimensions
                if not (x1 > jx2 + proximity or x2 + proximity < jx1 or
                        y1 > jy2 + proximity or y2 + proximity < jy1):
                    x1 = min(x1, jx1)
                    y1 = min(y1, jy1)
                    x2 = max(x2, jx2)
                    y2 = max(y2, jy2)
                    skip.add(j)
                    merged = True

            new_rects.append([x1, y1, x2, y2])
        rects = new_rects

    return [{"x": r[0], "y": r[1], "width": r[2] - r[0], "height": r[3] - r[1]} for r in rects]


class MotionDetectorConfig:
    """Configuration parameters for Engine 1 motion detection."""

    def __init__(
        self,
        diff_threshold: int = 25,
        min_motion_area: int = 400,
        max_motion_area_ratio: float = 0.60,
        motion_score_threshold: float = 0.002,
        blur_kernel_size: int = 21,
        morph_iterations: int = 2,
        learning_rate: float = 0.05,
        cluster_proximity: int = 40,
    ) -> None:
        self.diff_threshold = diff_threshold
        self.min_motion_area = min_motion_area
        self.max_motion_area_ratio = max_motion_area_ratio
        self.motion_score_threshold = motion_score_threshold
        self.blur_kernel_size = blur_kernel_size
        self.morph_iterations = morph_iterations
        self.learning_rate = learning_rate
        self.cluster_proximity = cluster_proximity


class Engine1MotionDetector(BasePlugin):
    """Lightweight OpenCV-based Motion Detection Gate (Engine 1).
    
    Evaluates each FramePacket cheaply. Unifies fragmented contour movements
    into a single cohesive target bounding box per moving object.
    """

    def __init__(self, name: str = "engine1_motion_detector") -> None:
        super().__init__(name=name)
        self.config = MotionDetectorConfig()
        self._source_references: Dict[str, np.ndarray] = {}

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Configure Engine 1 parameters from configuration dict."""
        self.config = MotionDetectorConfig(
            diff_threshold=int(config.get("diff_threshold", 25)),
            min_motion_area=int(config.get("min_motion_area", 400)),
            max_motion_area_ratio=float(config.get("max_motion_area_ratio", 0.60)),
            motion_score_threshold=float(config.get("motion_score_threshold", 0.002)),
            blur_kernel_size=int(config.get("blur_kernel_size", 21)),
            morph_iterations=int(config.get("morph_iterations", 2)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            cluster_proximity=int(config.get("cluster_proximity", 40)),
        )
        self._source_references.clear()
        self._status = PluginStatus.READY
        logger.info(
            f"Engine 1 initialized with: diff_thresh={self.config.diff_threshold}, "
            f"min_area={self.config.min_motion_area}, cluster_prox={self.config.cluster_proximity}"
        )
        return True

    def process(self, input_data: FramePacket) -> MotionEvent:
        """Perform lightweight motion analysis on a FramePacket.
        
        Args:
            input_data: FramePacket from Video Integration.
            
        Returns:
            MotionEvent (NO_MOTION or MOTION_DETECTED).
        """
        if not isinstance(input_data, FramePacket):
            raise TypeError(f"Engine 1 expects FramePacket, got {type(input_data).__name__}")

        self._status = PluginStatus.PROCESSING
        source_id = input_data.source_id
        raw_frame = input_data.frame

        # Step 1: Grayscale conversion + Gaussian blur
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        ksize = self.config.blur_kernel_size
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Retrieve background reference
        ref_float = self._source_references.get(source_id)

        # Initial frame initializes the reference background
        if ref_float is None:
            self._source_references[source_id] = blurred.astype(np.float32)
            self._status = PluginStatus.READY
            return MotionEvent(
                event_type=MotionEventType.NO_MOTION,
                source_id=source_id,
                timestamp=input_data.timestamp,
                frame_id=input_data.frame_id,
                motion_score=0.0,
                motion_region=None,
                selected_frame_or_reference=None,
                metadata={"reason": "initial_reference_frame"},
            )

        # Convert float32 reference to uint8 for difference calculation
        bg_uint8 = cv2.convertScaleAbs(ref_float)

        # Step 2: Absolute difference between current frame and background reference
        frame_diff = cv2.absdiff(bg_uint8, blurred)

        # Step 3: Intensity thresholding
        _, thresh = cv2.threshold(
            frame_diff, self.config.diff_threshold, 255, cv2.THRESH_BINARY
        )

        # Step 4: Morphological dilation to filter small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=self.config.morph_iterations)

        # Step 5: Contour detection & raw box extraction
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_w = input_data.frame_width
        frame_h = input_data.frame_height
        frame_area = float(frame_w * frame_h)
        max_allowed_contour_area = frame_area * self.config.max_motion_area_ratio

        total_motion_area = 0
        raw_boxes: List[Dict[str, int]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.config.min_motion_area <= area <= max_allowed_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Ignore box if it covers almost the whole camera (>80% width AND >80% height)
                if w >= int(frame_w * 0.80) and h >= int(frame_h * 0.80):
                    continue

                total_motion_area += area
                raw_boxes.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})


        # Step 6: Cluster & merge nearby boxes belonging to the SAME moving object
        unified_boxes = merge_bounding_boxes(raw_boxes, proximity=self.config.cluster_proximity)


        # Identify primary (largest) unified motion region
        primary_region: Optional[MotionRegion] = None
        max_box_area = 0

        for b in unified_boxes:
            box_area = b["width"] * b["height"]
            if box_area > max_box_area:
                max_box_area = box_area
                primary_region = MotionRegion(x=b["x"], y=b["y"], width=b["width"], height=b["height"])

        # Compute normalized motion score
        motion_score = (total_motion_area / frame_area) if frame_area > 0 else 0.0

        # Detect global lighting/auto-exposure shift
        is_global_shift = total_motion_area > (frame_area * 0.65) or (
            len(contours) > 0 and any(cv2.contourArea(c) > max_allowed_contour_area for c in contours)
        )

        has_motion = (
            not is_global_shift
            and motion_score >= self.config.motion_score_threshold
            and primary_region is not None
        )

        # Background adaptation
        if is_global_shift:
            cv2.accumulateWeighted(blurred.astype(np.float32), ref_float, 0.25)
        else:
            cv2.accumulateWeighted(blurred.astype(np.float32), ref_float, self.config.learning_rate)

        self._status = PluginStatus.READY

        if has_motion:
            return MotionEvent(
                event_type=MotionEventType.MOTION_DETECTED,
                source_id=source_id,
                timestamp=input_data.timestamp,
                frame_id=input_data.frame_id,
                motion_score=round(motion_score, 5),
                motion_region=primary_region,
                selected_frame_or_reference=raw_frame,
                metadata={
                    "total_motion_area": int(total_motion_area),
                    "sequence_number": input_data.sequence_number,
                    "all_motion_regions": unified_boxes,
                },
            )
        else:
            return MotionEvent(
                event_type=MotionEventType.NO_MOTION,
                source_id=source_id,
                timestamp=input_data.timestamp,
                frame_id=input_data.frame_id,
                motion_score=round(motion_score, 5),
                motion_region=None,
                selected_frame_or_reference=None,
                metadata={
                    "total_motion_area": int(total_motion_area),
                    "sequence_number": input_data.sequence_number,
                },
            )

    def shutdown(self) -> None:
        """Clean up references on shutdown."""
        self._source_references.clear()
        self._status = PluginStatus.SHUTDOWN
        logger.info("Engine 1 Motion Detector shut down cleanly.")
