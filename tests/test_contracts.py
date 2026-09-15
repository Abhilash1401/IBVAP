"""Tests for IBVAP FramePacket and SourceConfig contracts."""

import unittest
import numpy as np
from pydantic import ValidationError

from backend.contracts.frame import (
    FramePacket,
    SourceConfig,
    SourceStatus,
    SourceType,
)


class TestContracts(unittest.TestCase):
    """Test suite for shared contracts."""

    def test_source_config_valid(self):
        config = SourceConfig(
            source_id="CAM_01",
            source_type=SourceType.FILE,
            source_uri="videos/border_east.mp4",
            target_fps=25.0,
            reconnect_attempts=5,
            reconnect_delay_seconds=1.5,
        )
        self.assertEqual(config.source_id, "CAM_01")
        self.assertEqual(config.source_type, SourceType.FILE)
        self.assertEqual(config.source_uri, "videos/border_east.mp4")
        self.assertEqual(config.target_fps, 25.0)

    def test_frame_packet_valid_creation(self):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        packet = FramePacket.create(
            source_id="CAM_TEST",
            timestamp=1700000000.0,
            frame=fake_frame,
            sequence_number=0,
        )
        self.assertEqual(packet.source_id, "CAM_TEST")
        self.assertEqual(packet.sequence_number, 0)
        self.assertEqual(packet.frame_width, 640)
        self.assertEqual(packet.frame_height, 480)
        self.assertIsNotNone(packet.frame_id)
        self.assertIsInstance(packet.frame, np.ndarray)

    def test_frame_packet_dimension_mismatch(self):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            FramePacket(
                frame_id="test-id",
                source_id="CAM_TEST",
                timestamp=1700000000.0,
                frame=fake_frame,
                frame_width=320,  # Mismatched width!
                frame_height=480,
                sequence_number=0,
            )

    def test_frame_packet_non_numpy_rejected(self):
        with self.assertRaises(ValidationError):
            FramePacket(
                frame_id="test-id",
                source_id="CAM_TEST",
                timestamp=1700000000.0,
                frame="not_an_image",  # Invalid type!
                frame_width=640,
                frame_height=480,
                sequence_number=0,
            )

    def test_frame_packet_negative_sequence_rejected(self):
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValidationError):
            FramePacket(
                source_id="CAM_TEST",
                timestamp=1700000000.0,
                frame=fake_frame,
                frame_width=100,
                frame_height=100,
                sequence_number=-1,  # Must be >= 0
            )


if __name__ == "__main__":
    unittest.main()
