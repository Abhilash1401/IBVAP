"""IBVAP Video Ingestion & Integration Package.

Provides abstractions and readers for video files and RTSP camera streams,
bounded rolling buffers, and standardized FramePacket generation.
"""

from backend.video.buffer import RollingFrameBuffer
from backend.video.exceptions import (
    BufferCapacityError,
    EndOfStreamError,
    FrameDecodeError,
    SourceConnectionError,
    SourceNotFoundError,
    VideoIntegrationError,
)
from backend.video.file_source import FileSource
from backend.video.reader import VideoReader
from backend.video.rtsp_source import RTSPSource
from backend.video.source import VideoSource

__all__ = [
    "BufferCapacityError",
    "EndOfStreamError",
    "FileSource",
    "FrameDecodeError",
    "RollingFrameBuffer",
    "RTSPSource",
    "SourceConnectionError",
    "SourceNotFoundError",
    "VideoIntegrationError",
    "VideoReader",
    "VideoSource",
]
