"""Unit tests for FileSource local video ingestion."""

import os
import unittest
import numpy as np

from backend.contracts.frame import SourceConfig, SourceStatus, SourceType
from backend.video.exceptions import SourceNotFoundError
from backend.video.file_source import FileSource
from tests.helpers import create_synthetic_video


class TestFileSource(unittest.TestCase):
    """Test suite for FileSource."""

    @classmethod
    def setUpClass(cls):
        cls.test_video_path = create_synthetic_video(
            num_frames=10, width=320, height=240, fps=10
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_video_path):
            try:
                os.remove(cls.test_video_path)
            except OSError:
                pass

    def test_open_valid_file(self):
        config = SourceConfig(
            source_id="TEST_FILE",
            source_type=SourceType.FILE,
            source_uri=self.test_video_path,
        )
        source = FileSource(config)
        self.assertFalse(source.is_opened())
        self.assertEqual(source.status, SourceStatus.DISCONNECTED)

        opened = source.open()
        self.assertTrue(opened)
        self.assertTrue(source.is_opened())
        self.assertEqual(source.status, SourceStatus.CONNECTED)

        meta = source.get_metadata()
        self.assertEqual(meta["width"], 320)
        self.assertEqual(meta["height"], 240)
        self.assertGreaterEqual(meta["total_frames"], 10)

        source.release()
        self.assertFalse(source.is_opened())
        self.assertEqual(source.status, SourceStatus.DISCONNECTED)

    def test_missing_file_raises_not_found(self):
        config = SourceConfig(
            source_id="MISSING",
            source_type=SourceType.FILE,
            source_uri="non_existent_file_12345.mp4",
        )
        source = FileSource(config)
        with self.assertRaises(SourceNotFoundError):
            source.open()
        self.assertEqual(source.status, SourceStatus.ERROR)

    def test_read_frames_until_eos(self):
        config = SourceConfig(
            source_id="TEST_READ",
            source_type=SourceType.FILE,
            source_uri=self.test_video_path,
        )
        source = FileSource(config)
        source.open()

        frames_read = 0
        while True:
            ret, frame = source.read_frame()
            if not ret:
                break
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (240, 320, 3))
            self.assertEqual(frame.dtype, np.uint8)
            frames_read += 1

        self.assertGreaterEqual(frames_read, 10)
        self.assertEqual(source.status, SourceStatus.EOS)

        # Subsequent reads after EOS should continue returning (False, None) safely
        ret, frame = source.read_frame()
        self.assertFalse(ret)
        self.assertIsNone(frame)

        source.release()

    def test_context_manager(self):
        config = SourceConfig(
            source_id="TEST_CTX",
            source_type=SourceType.FILE,
            source_uri=self.test_video_path,
        )
        with FileSource(config) as source:
            self.assertTrue(source.is_opened())
            ret, frame = source.read_frame()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)

        self.assertFalse(source.is_opened())


if __name__ == "__main__":
    unittest.main()
