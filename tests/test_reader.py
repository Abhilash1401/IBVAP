"""Integration tests for VideoReader coordinator."""

import os
import unittest
import numpy as np

from backend.contracts.frame import FramePacket, SourceConfig, SourceType
from backend.video.buffer import RollingFrameBuffer
from backend.video.file_source import FileSource
from backend.video.reader import VideoReader
from tests.helpers import create_synthetic_video


class TestVideoReader(unittest.TestCase):
    """Integration test suite for VideoReader."""

    @classmethod
    def setUpClass(cls):
        cls.test_video_path = create_synthetic_video(
            num_frames=12, width=320, height=240, fps=10
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_video_path):
            try:
                os.remove(cls.test_video_path)
            except OSError:
                pass

    def test_stream_packets_end_to_end(self):
        config = SourceConfig(
            source_id="BORDER_CAM_01",
            source_type=SourceType.FILE,
            source_uri=self.test_video_path,
        )
        source = FileSource(config)
        buffer = RollingFrameBuffer(capacity=8)
        reader = VideoReader(source=source, buffer=buffer)

        packets = []
        for packet in reader.stream_packets():
            self.assertIsInstance(packet, FramePacket)
            self.assertEqual(packet.source_id, "BORDER_CAM_01")
            self.assertEqual(packet.frame_width, 320)
            self.assertEqual(packet.frame_height, 240)
            self.assertEqual(packet.frame.shape, (240, 320, 3))
            self.assertIsInstance(packet.frame, np.ndarray)
            self.assertGreater(packet.timestamp, 0)
            packets.append(packet)

        # Total frames received must match video
        self.assertGreaterEqual(len(packets), 12)

        # Monotonically increasing sequence numbers starting from 0
        expected_seqs = list(range(len(packets)))
        actual_seqs = [p.sequence_number for p in packets]
        self.assertEqual(actual_seqs, expected_seqs)

        # Verify buffer retained the last 8 frames
        self.assertEqual(len(buffer), 8)
        buffer_frames = buffer.get_all_frames()
        self.assertEqual(
            [p.sequence_number for p in buffer_frames],
            expected_seqs[-8:],
        )

        # Underlying source should be cleanly released after streaming
        self.assertFalse(source.is_opened())

    def test_reader_context_manager(self):
        config = SourceConfig(
            source_id="BORDER_CAM_CTX",
            source_type=SourceType.FILE,
            source_uri=self.test_video_path,
        )
        source = FileSource(config)
        with VideoReader(source=source) as reader:
            pkt1 = reader.read_packet()
            self.assertIsNotNone(pkt1)
            self.assertEqual(pkt1.sequence_number, 0)

            pkt2 = reader.read_packet()
            self.assertIsNotNone(pkt2)
            self.assertEqual(pkt2.sequence_number, 1)

        self.assertFalse(source.is_opened())


if __name__ == "__main__":
    unittest.main()
