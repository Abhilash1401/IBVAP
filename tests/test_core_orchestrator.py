"""Unit and integration tests for CoreOrchestrator."""

import os
import unittest
from unittest.mock import MagicMock
import numpy as np

from backend.contracts.classification import ClassificationRequest
from backend.contracts.frame import FramePacket, SourceConfig, SourceType
from backend.contracts.motion import MotionEvent, MotionEventType, MotionRegion
from backend.core.orchestrator import CoreOrchestrator
from backend.core.plugin import BasePlugin, PluginStatus
from backend.engine1.motion_detector import Engine1MotionDetector
from backend.video.buffer import RollingFrameBuffer
from backend.video.file_source import FileSource
from backend.video.reader import VideoReader
from tests.helpers import create_synthetic_video


class MockEngine1(BasePlugin):
    """Mock Engine 1 plugin for isolated testing of routing decisions."""

    def __init__(self, should_detect_motion: bool = False):
        super().__init__(name="mock_engine1")
        self.should_detect_motion = should_detect_motion

    def initialize(self, config):
        return True

    def process(self, packet: FramePacket) -> MotionEvent:
        if self.should_detect_motion:
            return MotionEvent(
                event_type=MotionEventType.MOTION_DETECTED,
                source_id=packet.source_id,
                timestamp=packet.timestamp,
                frame_id=packet.frame_id,
                motion_score=0.15,
                motion_region=MotionRegion(x=10, y=10, width=50, height=50),
                selected_frame_or_reference=packet.frame,
            )
        else:
            return MotionEvent(
                event_type=MotionEventType.NO_MOTION,
                source_id=packet.source_id,
                timestamp=packet.timestamp,
                frame_id=packet.frame_id,
                motion_score=0.0,
            )

    def shutdown(self):
        pass


class TestCoreOrchestrator(unittest.TestCase):
    """Test suite for CoreOrchestrator."""

    def _create_packet(self) -> FramePacket:
        return FramePacket.create(
            source_id="CAM_CORE_TEST",
            timestamp=1700000000.0,
            frame=np.zeros((100, 100, 3), dtype=np.uint8),
            sequence_number=1,
        )

    def test_routing_suppression_on_no_motion(self):
        mock_e1 = MockEngine1(should_detect_motion=False)
        mock_e2 = MagicMock()

        orchestrator = CoreOrchestrator(engine1_plugin=mock_e1, engine2_handler=mock_e2)
        packet = self._create_packet()

        result = orchestrator.process_frame_packet(packet)

        # Must suppress downstream call
        self.assertIsNone(result)
        mock_e2.assert_not_called()
        self.assertEqual(orchestrator.total_frames, 1)
        self.assertEqual(orchestrator.suppressed_frames, 1)
        self.assertEqual(orchestrator.motion_frames, 0)
        self.assertEqual(orchestrator.suppression_ratio, 100.0)

    def test_routing_delegation_on_motion_detected(self):
        mock_e1 = MockEngine1(should_detect_motion=True)
        mock_e2 = MagicMock()

        orchestrator = CoreOrchestrator(engine1_plugin=mock_e1, engine2_handler=mock_e2)
        packet = self._create_packet()

        result = orchestrator.process_frame_packet(packet)

        # Must trigger downstream Engine 2 call with ClassificationRequest
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ClassificationRequest)
        self.assertEqual(result.source_id, "CAM_CORE_TEST")
        self.assertEqual(result.frame_id, packet.frame_id)
        self.assertIsNotNone(result.motion_region)
        mock_e2.assert_called_once_with(result)

        self.assertEqual(orchestrator.total_frames, 1)
        self.assertEqual(orchestrator.suppressed_frames, 0)
        self.assertEqual(orchestrator.motion_frames, 1)
        self.assertEqual(orchestrator.suppression_ratio, 0.0)

    def test_end_to_end_pipeline_with_video_reader(self):
        # Create a synthetic video with moving elements
        video_path = create_synthetic_video(num_frames=15, width=320, height=240, fps=10)

        try:
            # 1. Member 1: Video Integration (UNTOUCHED)
            config = SourceConfig(
                source_id="INTEG_CAM",
                source_type=SourceType.FILE,
                source_uri=video_path,
            )
            source = FileSource(config)
            buffer = RollingFrameBuffer(capacity=10)
            reader = VideoReader(source=source, buffer=buffer)

            # 2. Member 2: Engine 1 Motion Detector + Core Orchestrator
            detector = Engine1MotionDetector()
            detector.initialize({"min_motion_area": 200})

            captured_requests = []
            def dummy_engine2(request: ClassificationRequest):
                captured_requests.append(request)

            orchestrator = CoreOrchestrator(
                engine1_plugin=detector,
                engine2_handler=dummy_engine2,
            )

            # 3. Run Pipeline
            stats = orchestrator.run_pipeline(reader)

            self.assertEqual(stats["total_frames"], 15)
            self.assertGreater(stats["processing_fps"], 0)
            self.assertEqual(stats["total_frames"], stats["motion_frames"] + stats["suppressed_frames"])
            # In our synthetic video, after the initial frame, subsequent frames have changed pixels
            self.assertGreater(stats["motion_frames"], 0)
            self.assertEqual(len(captured_requests), stats["motion_frames"])

        finally:
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except OSError:
                    pass

    def test_engine1_called_exactly_once_per_frame_in_run_pipeline(self):
        """Regression: run_pipeline() must evaluate Engine 1 exactly once per frame."""
        from unittest.mock import MagicMock

        # Wrap a real detector so behavior is realistic, but count process() calls.
        detector = Engine1MotionDetector()
        detector.initialize({"min_motion_area": 200})
        call_counter = MagicMock(wraps=detector.process)
        # Patch at instance level to count without changing behavior.
        detector.process = call_counter  # type: ignore[method-assign]

        video_path = create_synthetic_video(num_frames=8, width=320, height=240, fps=10)
        try:
            config = SourceConfig(
                source_id="COUNT_CAM",
                source_type=SourceType.FILE,
                source_uri=video_path,
            )
            reader = VideoReader(
                source=FileSource(config),
                buffer=RollingFrameBuffer(capacity=8),
            )
            callback_pairs = []

            def on_motion(request: ClassificationRequest, event: MotionEvent) -> None:
                callback_pairs.append((request, event))

            orchestrator = CoreOrchestrator(engine1_plugin=detector)
            stats = orchestrator.run_pipeline(reader, on_motion_callback=on_motion)

            # Exactly one Engine 1 evaluation per ingested frame.
            self.assertEqual(stats["total_frames"], 8)
            self.assertEqual(call_counter.call_count, 8)

            # Every motion callback received a valid MotionEvent produced by the
            # single routing evaluation (no second evaluation exists anymore).
            self.assertEqual(len(callback_pairs), stats["motion_frames"])
            for request, event in callback_pairs:
                self.assertIsInstance(event, MotionEvent)
                self.assertEqual(event.event_type, MotionEventType.MOTION_DETECTED)
                self.assertEqual(request.frame_id, event.frame_id)
                self.assertIs(request.motion_region, event.motion_region)
        finally:
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except OSError:
                    pass

    def test_process_frame_packet_single_evaluation_with_callback_event(self):
        """process_frame_packet_with_event returns the routing MotionEvent itself."""
        from unittest.mock import MagicMock

        mock_e1 = MockEngine1(should_detect_motion=True)
        spy = MagicMock(wraps=mock_e1.process)
        mock_e1.process = spy  # type: ignore[method-assign]

        orchestrator = CoreOrchestrator(engine1_plugin=mock_e1)
        packet = self._create_packet()

        request, event = orchestrator.process_frame_packet_with_event(packet)

        self.assertEqual(spy.call_count, 1)
        self.assertIsNotNone(request)
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, MotionEventType.MOTION_DETECTED)
        self.assertIs(request.motion_region, event.motion_region)
        self.assertEqual(request.frame_id, packet.frame_id)


if __name__ == "__main__":
    unittest.main()
