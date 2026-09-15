"""Local file video source implementation for IBVAP V1."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from backend.contracts.frame import SourceConfig, SourceStatus, SourceType
from backend.video.exceptions import (
    SourceConnectionError,
    SourceNotFoundError,
)
from backend.video.source import VideoSource

logger = logging.getLogger(__name__)


class FileSource(VideoSource):
    """VideoSource implementation for local media files (MP4, AVI, MOV, etc.)."""

    def __init__(self, config: SourceConfig) -> None:
        super().__init__(config)
        if config.source_type != SourceType.FILE:
            raise ValueError(f"Expected source_type FILE, got {config.source_type}")
        
        self.file_path = Path(config.source_uri).resolve()
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Dict[str, Any] = {}

    def open(self) -> bool:
        """Open the local video file and extract metadata."""
        if not self.file_path.is_file():
            self._status = SourceStatus.ERROR
            raise SourceNotFoundError(f"Video file not found: {self.file_path}")

        self._status = SourceStatus.CONNECTING
        logger.info(f"Opening video file: {self.file_path}")
        
        self._cap = cv2.VideoCapture(str(self.file_path))
        if not self._cap.isOpened():
            self._status = SourceStatus.ERROR
            self._cap = None
            raise SourceConnectionError(f"Failed to open video file with OpenCV: {self.file_path}")

        self._status = SourceStatus.CONNECTED
        self._cache_metadata()
        logger.info(f"Video file opened successfully: {self.config.source_id} ({self._metadata})")
        return True

    def _cache_metadata(self) -> None:
        if self._cap is None:
            return
        
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = (total_frames / fps) if fps > 0 else 0.0

        self._metadata = {
            "source_id": self.source_id,
            "source_type": SourceType.FILE.value,
            "path": str(self.file_path),
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": duration_sec,
        }

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read sequential frame from the local video file.
        
        Returns:
            (True, frame) if frame was read successfully.
            (False, None) if End of Stream (EOS) is reached or on read failure.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.info(f"End of stream reached for source: {self.source_id}")
            self._status = SourceStatus.EOS
            return False, None

        return True, frame

    def is_opened(self) -> bool:
        """Check whether the underlying OpenCV capture object is active."""
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Release the capture object and mark as DISCONNECTED."""
        if self._cap is not None:
            logger.info(f"Releasing capture resource for source: {self.source_id}")
            self._cap.release()
            self._cap = None
        if self._status != SourceStatus.EOS:
            self._status = SourceStatus.DISCONNECTED

    def get_metadata(self) -> Dict[str, Any]:
        """Return cached video metadata."""
        return self._metadata.copy()
