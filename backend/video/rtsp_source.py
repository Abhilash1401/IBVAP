"""RTSP / IP camera stream source implementation for IBVAP V1."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple
import cv2
import numpy as np

from backend.contracts.frame import SourceConfig, SourceStatus, SourceType
from backend.video.exceptions import SourceConnectionError
from backend.video.source import VideoSource

logger = logging.getLogger(__name__)

# Type alias for capture factory (useful for unit test dependency injection)
CaptureFactory = Callable[[str], cv2.VideoCapture]


class RTSPSource(VideoSource):
    """VideoSource implementation for network RTSP/IP camera streams."""

    def __init__(
        self,
        config: SourceConfig,
        capture_factory: Optional[CaptureFactory] = None,
    ) -> None:
        super().__init__(config)
        if config.source_type != SourceType.RTSP:
            raise ValueError(f"Expected source_type RTSP, got {config.source_type}")

        self.rtsp_uri = config.source_uri
        self._capture_factory: CaptureFactory = capture_factory or cv2.VideoCapture
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Dict[str, Any] = {}

    def open(self) -> bool:
        """Attempt to open connection to the RTSP stream with retry policy."""
        self._status = SourceStatus.CONNECTING
        logger.info(f"Connecting to RTSP stream: {self.source_id} at {self.rtsp_uri}")

        attempts = max(1, self.config.reconnect_attempts)
        for attempt in range(1, attempts + 1):
            try:
                self._cap = self._capture_factory(self.rtsp_uri)
                if self._cap is not None and self._cap.isOpened():
                    self._status = SourceStatus.CONNECTED
                    self._cache_metadata()
                    logger.info(f"Connected to RTSP stream successfully on attempt {attempt}")
                    return True
            except Exception as e:
                logger.warning(f"RTSP connection attempt {attempt}/{attempts} failed: {e}")

            if attempt < attempts:
                time.sleep(self.config.reconnect_delay_seconds)

        self._status = SourceStatus.ERROR
        self._cap = None
        raise SourceConnectionError(
            f"Failed to connect to RTSP stream '{self.rtsp_uri}' after {attempts} attempts."
        )

    def _cache_metadata(self) -> None:
        if self._cap is None:
            return

        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self._metadata = {
            "source_id": self.source_id,
            "source_type": SourceType.RTSP.value,
            "uri": self.rtsp_uri,
            "width": width,
            "height": height,
            "fps": fps,
        }

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from RTSP stream."""
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning(f"RTSP stream frame read failed for source: {self.source_id}")
            self._status = SourceStatus.ERROR
            return False, None

        return True, frame

    def is_opened(self) -> bool:
        """Check whether the RTSP stream capture is active."""
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Disconnect and release RTSP stream capture."""
        if self._cap is not None:
            logger.info(f"Releasing RTSP connection for source: {self.source_id}")
            self._cap.release()
            self._cap = None
        self._status = SourceStatus.DISCONNECTED

    def get_metadata(self) -> Dict[str, Any]:
        """Return RTSP stream metadata."""
        return self._metadata.copy()
