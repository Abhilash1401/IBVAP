"""Exceptions for the IBVAP Video Integration / Ingestion module."""


class VideoIntegrationError(Exception):
    """Base exception for all video ingestion errors."""
    pass


class SourceNotFoundError(VideoIntegrationError):
    """Raised when a video file or device path cannot be found."""
    pass


class SourceConnectionError(VideoIntegrationError):
    """Raised when an RTSP stream or camera connection fails or drops."""
    pass


class FrameDecodeError(VideoIntegrationError):
    """Raised when a video frame fails to decode or contains corrupted data."""
    pass


class EndOfStreamError(VideoIntegrationError):
    """Raised or signaled when a video stream reaches its natural end."""
    pass


class BufferCapacityError(VideoIntegrationError):
    """Raised when buffer parameters or invariants are violated."""
    pass
