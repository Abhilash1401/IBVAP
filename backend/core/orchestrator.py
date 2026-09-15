"""Core Pipeline Orchestration Layer for IBVAP V1."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.contracts.classification import ClassificationRequest
from backend.contracts.frame import FramePacket
from backend.contracts.motion import MotionEvent, MotionEventType
from backend.core.plugin import BasePlugin, PluginManager
from backend.video.reader import VideoReader

logger = logging.getLogger(__name__)

# Downstream Engine 2 handler callback type: Callable[[ClassificationRequest], Any]
ClassificationHandler = Callable[[ClassificationRequest], Any]


class CoreOrchestrator:
    """Central orchestrator for the IBVAP V1 progressive processing pipeline.
    
    Coordinates:
        VideoReader (Member 1) -> Engine 1 (Motion Gate) -> [Core Routing] -> Engine 2 (YOLO AI)
    
    Ensures plugins remain strictly isolated and never call each other directly.
    """

    def __init__(
        self,
        engine1_plugin: BasePlugin,
        engine2_handler: Optional[ClassificationHandler] = None,
        plugin_manager: Optional[PluginManager] = None,
    ) -> None:
        self.engine1 = engine1_plugin
        self.engine2_handler = engine2_handler
        self.plugin_manager = plugin_manager or PluginManager()

        # Pipeline runtime statistics
        self.total_frames: int = 0
        self.suppressed_frames: int = 0
        self.motion_frames: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @property
    def suppression_ratio(self) -> float:
        """Percentage of frames where expensive AI was avoided."""
        if self.total_frames == 0:
            return 0.0
        return round((self.suppressed_frames / self.total_frames) * 100.0, 2)

    def process_frame_packet(
        self, packet: FramePacket, pre_event_frames: Optional[List[FramePacket]] = None
    ) -> Optional[ClassificationRequest]:
        """Process a single FramePacket through Engine 1 and route to Engine 2 if motion detected.
        
        Args:
            packet: Standardized FramePacket from VideoReader.
            pre_event_frames: Optional list of recent context frames from RollingFrameBuffer.
            
        Returns:
            ClassificationRequest if motion was detected and routed, None if suppressed.
        """
        request, _ = self.process_frame_packet_with_event(
            packet, pre_event_frames=pre_event_frames
        )
        return request

    def process_frame_packet_with_event(
        self, packet: FramePacket, pre_event_frames: Optional[List[FramePacket]] = None
    ) -> Tuple[Optional[ClassificationRequest], MotionEvent]:
        """Process packet once; return (request, motion_event) sharing one Engine 1 call."""
        self.total_frames += 1

        # Single Engine 1 evaluation per packet.
        motion_event: MotionEvent = self.engine1.process(packet)

        if motion_event.event_type == MotionEventType.NO_MOTION:
            self.suppressed_frames += 1
            return None, motion_event

        # Motion Detected path: Core builds standardized ClassificationRequest
        self.motion_frames += 1
        logger.info(
            f"Motion detected on source '{packet.source_id}' "
            f"(Score: {motion_event.motion_score}, Frame ID: {packet.frame_id[:8]})"
        )

        request = ClassificationRequest(
            request_id=str(uuid.uuid4()),
            source_id=packet.source_id,
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
            frame=packet.frame,
            motion_region=motion_event.motion_region,
            metadata={
                "motion_score": motion_event.motion_score,
                "sequence_number": packet.sequence_number,
                "pre_event_frames_count": len(pre_event_frames) if pre_event_frames else 0,
                "all_motion_regions": motion_event.metadata.get("all_motion_regions", []),
            },
        )

        # Route to downstream Engine 2 if registered
        if self.engine2_handler is not None:
            self.engine2_handler(request)

        return request, motion_event

    def run_pipeline(
        self,
        reader: VideoReader,
        max_frames: Optional[int] = None,
        on_motion_callback: Optional[Callable[[ClassificationRequest, MotionEvent], None]] = None,
    ) -> Dict[str, Any]:
        """Run the complete pipeline reading from VideoReader until EOS or max_frames.
        
        Args:
            reader: VideoReader instance from Member 1 (Video Integration).
            max_frames: Optional frame limit.
            on_motion_callback: Optional callback invoked when motion is detected.
            
        Returns:
            Summary dictionary containing pipeline performance metrics.
        """
        self.total_frames = 0
        self.suppressed_frames = 0
        self.motion_frames = 0
        self.start_time = time.time()

        logger.info(f"Starting Core pipeline execution for source: {reader.source_id}")

        for packet in reader.stream_packets():
            # Retrieve pre-event frames from the rolling buffer if available
            pre_event_frames = None
            if reader.buffer is not None:
                pre_event_frames = reader.buffer.get_pre_event_frames()

            # Single Engine 1 evaluation per packet; reuse MotionEvent for callback.
            request, motion_event = self.process_frame_packet_with_event(
                packet, pre_event_frames=pre_event_frames
            )

            if request is not None and on_motion_callback is not None:
                on_motion_callback(request, motion_event)

            if max_frames and self.total_frames >= max_frames:
                break

        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        fps = (self.total_frames / elapsed) if elapsed > 0 else 0.0

        summary = {
            "source_id": reader.source_id,
            "total_frames": self.total_frames,
            "motion_frames": self.motion_frames,
            "suppressed_frames": self.suppressed_frames,
            "suppression_ratio_pct": self.suppression_ratio,
            "elapsed_seconds": round(elapsed, 3),
            "processing_fps": round(fps, 1),
        }

        logger.info(f"Pipeline execution completed: {summary}")
        return summary
