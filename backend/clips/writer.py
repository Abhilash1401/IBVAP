"""IBVAP V1 moving event clip writer.

Writes short MP4 clips from pre-event + trigger frames.
Only event clips are persisted (never the full stream).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2

from backend.contracts.frame import FramePacket

logger = logging.getLogger(__name__)


class ClipWriter:
    """Writes FramePacket frame lists to local MP4 files."""

    def __init__(self, clips_dir: str = "storage/event_clips",
                 fps: float = 20.0) -> None:
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps) if fps and fps > 0 else 20.0
        self._working_codec: Optional[str] = None

    def write_clip(self, event_id: str,
                   packets: List[FramePacket]) -> Optional[str]:
        """Write packets to MP4. Returns absolute path or None."""
        if not packets:
            return None
        safe = "".join(c for c in event_id if c.isalnum() or c in "-_") or "evt"
        out = self.clips_dir / f"{safe}.mp4"
        h, w = packets[0].frame.shape[:2]
        
        # Use cached working codec or discover one
        codecs_to_try = [self._working_codec] if self._working_codec else ["mp4v", "avc1", "H264"]
        writer = None
        for codec in codecs_to_try:
            if not codec:
                continue
            fourcc = cv2.VideoWriter_fourcc(*codec)
            candidate = cv2.VideoWriter(str(out), fourcc, self.fps, (w, h))
            if candidate.isOpened():
                writer = candidate
                self._working_codec = codec
                break
        if writer is None or not writer.isOpened():
            logger.error("ClipWriter: VideoWriter failed for %s", out)
            return None
        try:
            for p in packets:
                frame = p.frame
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
        finally:
            writer.release()
        logger.info("ClipWriter: wrote %d frames -> %s", len(packets), out)
        return str(out.resolve())

    def build_event_packets(
        self,
        pre_frames: List[FramePacket],
        trigger: FramePacket,
        post_frames: Optional[List[FramePacket]] = None,
        max_pre: int = 30,
        max_post: int = 30,
    ) -> List[FramePacket]:
        """Assemble pre + trigger + post packets, bounded."""
        pre = pre_frames[-max_pre:] if pre_frames else []
        post = (post_frames or [])[:max_post]
        # Avoid duplicating trigger if already last pre frame.
        if pre and pre[-1].frame_id == trigger.frame_id:
            return [*pre, *post]
        return [*pre, trigger, *post]
