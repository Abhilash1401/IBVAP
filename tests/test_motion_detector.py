"""Unit tests for Engine 1 Motion Detection Plugin."""

import unittest
import cv2
import numpy as np

from backend.contracts.frame import FramePacket
from backend.contracts.motion import MotionEventType
from backend.core.plugin import PluginStatus
from backend.engine1.motion_detector import Engine1MotionDetector


class TestEngine1MotionDetector(unittest.TestCase):
    """Test suite for Engine1MotionDetector."""

    def setUp(self):
        self.detector = Engine1MotionDetector()
        self.detector.initialize({
            "diff_threshold": 25,
            "min_motion_area": 300,
            "motion_score_threshold": 0.002,
            "blur_kernel_size": 21,
            "morph_iterations": 2,
        })

    def tearDown(self):
        self.detector.shutdown()

    def _create_packet(self, frame: np.ndarray, seq: int, source_id: str = "CAM_TEST") -> FramePacket:
        return FramePacket.create(
            source_id=source_id,
            timestamp=1700000000.0 + seq,
            frame=frame,
            sequence_number=seq,
        )

    def test_plugin_lifecycle(self):
        self.assertEqual(self.detector.get_status(), PluginStatus.READY)
        self.detector.shutdown()
        self.assertEqual(self.detector.get_status(), PluginStatus.SHUTDOWN)

    def test_static_frames_yield_no_motion(self):
        # Generate constant static background (e.g. gray image)
        static_frame = np.full((240, 320, 3), fill_value=128, dtype=np.uint8)

        # Feed initial frame (establishes background reference)
        event0 = self.detector.process(self._create_packet(static_frame, 0))
        self.assertEqual(event0.event_type, MotionEventType.NO_MOTION)

        # Feed 5 subsequent identical frames
        for i in range(1, 6):
            event = self.detector.process(self._create_packet(static_frame, i))
            self.assertEqual(event.event_type, MotionEventType.NO_MOTION)
            self.assertEqual(event.motion_score, 0.0)
            self.assertIsNone(event.motion_region)

    def test_moving_object_triggers_motion_detected(self):
        # Frame 0: Dark background
        frame0 = np.zeros((240, 320, 3), dtype=np.uint8)
        self.detector.process(self._create_packet(frame0, 0))

        # Frame 1: Large bright white rectangle placed in center (significant motion)
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (100, 80), (220, 180), (255, 255, 255), -1)

        event1 = self.detector.process(self._create_packet(frame1, 1))

        self.assertEqual(event1.event_type, MotionEventType.MOTION_DETECTED)
        self.assertGreater(event1.motion_score, 0.01)
        self.assertIsNotNone(event1.motion_region)
        self.assertIsNotNone(event1.selected_frame_or_reference)
        # Bounding box should surround the rectangle
        self.assertAlmostEqual(event1.motion_region.x, 100, delta=15)
        self.assertAlmostEqual(event1.motion_region.y, 80, delta=15)

    def test_small_noise_filtered_out(self):
        # Frame 0: Dark background
        frame0 = np.zeros((240, 320, 3), dtype=np.uint8)
        self.detector.process(self._create_packet(frame0, 0))

        # Frame 1: Tiny 3x3 pixel noise spot (area = 9 px < min_motion_area 300)
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (50, 50), (53, 53), (255, 255, 255), -1)

        event1 = self.detector.process(self._create_packet(frame1, 1))
        self.assertEqual(event1.event_type, MotionEventType.NO_MOTION)

    def test_multiple_sources_isolated_state(self):
        # Ensure CAM_A and CAM_B maintain distinct background references
        frame_a = np.zeros((240, 320, 3), dtype=np.uint8)
        frame_b = np.full((240, 320, 3), fill_value=200, dtype=np.uint8)

        # Initialize both sources
        self.detector.process(self._create_packet(frame_a, 0, source_id="CAM_A"))
        self.detector.process(self._create_packet(frame_b, 0, source_id="CAM_B"))

        # Second frames identical to their own backgrounds must NOT trigger motion
        event_a = self.detector.process(self._create_packet(frame_a, 1, source_id="CAM_A"))
        event_b = self.detector.process(self._create_packet(frame_b, 1, source_id="CAM_B"))

        self.assertEqual(event_a.event_type, MotionEventType.NO_MOTION)
        self.assertEqual(event_b.event_type, MotionEventType.NO_MOTION)


if __name__ == "__main__":
    unittest.main()
