"""Unit tests for RollingFrameBuffer."""

import threading
import unittest
import numpy as np

from backend.contracts.frame import FramePacket
from backend.video.buffer import RollingFrameBuffer
from backend.video.exceptions import BufferCapacityError


class TestRollingFrameBuffer(unittest.TestCase):
    """Test suite for RollingFrameBuffer."""

    def _create_packet(self, seq: int) -> FramePacket:
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        return FramePacket.create(
            source_id="CAM_BUF_TEST",
            timestamp=1000.0 + seq,
            frame=fake_frame,
            sequence_number=seq,
        )

    def test_invalid_capacity(self):
        with self.assertRaises(BufferCapacityError):
            RollingFrameBuffer(capacity=0)
        with self.assertRaises(BufferCapacityError):
            RollingFrameBuffer(capacity=-10)

    def test_bounded_capacity_eviction(self):
        buf = RollingFrameBuffer(capacity=5)
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.capacity, 5)

        # Append 10 packets (sequences 0 to 9)
        for i in range(10):
            buf.append(self._create_packet(i))

        # Length must be bounded at capacity (5)
        self.assertEqual(len(buf), 5)

        # Oldest packets (0..4) should have been evicted; remaining should be 5..9
        frames = buf.get_all_frames()
        self.assertEqual(len(frames), 5)
        self.assertEqual([p.sequence_number for p in frames], [5, 6, 7, 8, 9])

    def test_get_pre_event_frames_subset(self):
        buf = RollingFrameBuffer(capacity=10)
        for i in range(8):
            buf.append(self._create_packet(i))

        # Request last 3 frames
        recent = buf.get_pre_event_frames(count=3)
        self.assertEqual(len(recent), 3)
        self.assertEqual([p.sequence_number for p in recent], [5, 6, 7])

        # Request more than available returns all
        recent_all = buf.get_pre_event_frames(count=20)
        self.assertEqual(len(recent_all), 8)

    def test_clear_buffer(self):
        buf = RollingFrameBuffer(capacity=5)
        buf.append(self._create_packet(1))
        buf.append(self._create_packet(2))
        self.assertEqual(len(buf), 2)

        buf.clear()
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.get_all_frames(), [])

    def test_type_enforcement(self):
        buf = RollingFrameBuffer(capacity=5)
        with self.assertRaises(TypeError):
            buf.append("not_a_frame_packet")

    def test_thread_safety(self):
        buf = RollingFrameBuffer(capacity=50)

        def worker(start_seq: int):
            for i in range(25):
                buf.append(self._create_packet(start_seq + i))

        t1 = threading.Thread(target=worker, args=(0,))
        t2 = threading.Thread(target=worker, args=(100,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(len(buf), 50)


if __name__ == "__main__":
    unittest.main()
