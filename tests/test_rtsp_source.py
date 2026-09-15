"""Unit tests for RTSPSource with dependency injection and mock capture."""

import unittest
from unittest.mock import MagicMock
import numpy as np

from backend.contracts.frame import SourceConfig, SourceStatus, SourceType
from backend.video.exceptions import SourceConnectionError
from backend.video.rtsp_source import RTSPSource


class TestRTSPSource(unittest.TestCase):
    """Test suite for RTSPSource network stream ingestion."""

    def test_invalid_source_type(self):
        config = SourceConfig(
            source_id="CAM_RTSP",
            source_type=SourceType.FILE,  # Mismatch!
            source_uri="rtsp://192.168.1.100:554/live",
        )
        with self.assertRaises(ValueError):
            RTSPSource(config)

    def test_successful_connection_with_mock_capture(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            3: 1280.0,  # cv2.CAP_PROP_FRAME_WIDTH
            4: 720.0,   # cv2.CAP_PROP_FRAME_HEIGHT
            5: 30.0,    # cv2.CAP_PROP_FPS
        }.get(prop, 0.0)

        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)

        config = SourceConfig(
            source_id="CAM_RTSP_01",
            source_type=SourceType.RTSP,
            source_uri="rtsp://192.168.1.50:554/ch0",
            reconnect_attempts=2,
            reconnect_delay_seconds=0.01,
        )

        source = RTSPSource(config, capture_factory=lambda uri: mock_cap)
        self.assertEqual(source.status, SourceStatus.DISCONNECTED)

        opened = source.open()
        self.assertTrue(opened)
        self.assertEqual(source.status, SourceStatus.CONNECTED)

        meta = source.get_metadata()
        self.assertEqual(meta["width"], 1280)
        self.assertEqual(meta["height"], 720)
        self.assertEqual(meta["fps"], 30.0)

        # Test frame read
        ret, frame = source.read_frame()
        self.assertTrue(ret)
        self.assertEqual(frame.shape, (720, 1280, 3))

        source.release()
        self.assertEqual(source.status, SourceStatus.DISCONNECTED)
        mock_cap.release.assert_called_once()

    def test_failed_connection_retries_and_raises(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False  # Always fails to open

        config = SourceConfig(
            source_id="CAM_FAIL",
            source_type=SourceType.RTSP,
            source_uri="rtsp://invalid.ip/stream",
            reconnect_attempts=2,
            reconnect_delay_seconds=0.01,
        )

        source = RTSPSource(config, capture_factory=lambda uri: mock_cap)
        with self.assertRaises(SourceConnectionError):
            source.open()

        self.assertEqual(source.status, SourceStatus.ERROR)

    def test_read_failure_transitions_to_error(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # Network drop during stream

        config = SourceConfig(
            source_id="CAM_DROP",
            source_type=SourceType.RTSP,
            source_uri="rtsp://192.168.1.50:554/ch0",
        )

        source = RTSPSource(config, capture_factory=lambda uri: mock_cap)
        source.open()
        self.assertEqual(source.status, SourceStatus.CONNECTED)

        ret, frame = source.read_frame()
        self.assertFalse(ret)
        self.assertIsNone(frame)
        self.assertEqual(source.status, SourceStatus.ERROR)


if __name__ == "__main__":
    unittest.main()
