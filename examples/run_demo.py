"""Demonstration of Member 1 Video Ingestion Pipeline.

Generates a sample video, initializes VideoReader with RollingFrameBuffer,
and prints the emitted FramePacket stream.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.contracts.frame import SourceConfig, SourceType
from backend.video.buffer import RollingFrameBuffer
from backend.video.file_source import FileSource
from backend.video.reader import VideoReader
from tests.helpers import create_synthetic_video


def main():
    print("=" * 60)
    print("IBVAP V1 - Member 1 Video Ingestion Demonstration")
    print("=" * 60)

    # 1. Generate synthetic test video
    print("\n[1] Generating 30-frame synthetic test video...")
    video_path = create_synthetic_video(num_frames=30, width=640, height=480, fps=15)
    print(f"    Video created at: {video_path}")

    try:
        # 2. Configure video source
        config = SourceConfig(
            source_id="BORDER_CAM_NORTH",
            source_type=SourceType.FILE,
            source_uri=video_path,
        )

        # 3. Initialize source, rolling buffer (capacity 10), and reader
        source = FileSource(config)
        buffer = RollingFrameBuffer(capacity=10)
        reader = VideoReader(source=source, buffer=buffer)

        print("\n[2] Ingesting frames through VideoReader...")
        start_time = time.time()
        packet_count = 0

        for packet in reader.stream_packets():
            packet_count += 1
            if packet.sequence_number % 5 == 0 or packet.sequence_number < 3:
                print(
                    f"    Emitted FramePacket: seq={packet.sequence_number:03d} | "
                    f"id={packet.frame_id[:8]}... | "
                    f"shape={packet.frame_width}x{packet.frame_height} | "
                    f"buffered={len(buffer)}/{buffer.capacity}"
                )

        elapsed = time.time() - start_time
        print(f"\n[3] Ingestion complete: {packet_count} frames processed in {elapsed:.3f}s")
        print(f"    Final source status: {source.status}")
        print(f"    Rolling buffer context frames retained: {len(buffer)}")

        # 4. Demonstrate pre-event frames retrieval
        pre_frames = buffer.get_pre_event_frames(count=5)
        print(f"\n[4] Retrieved {len(pre_frames)} pre-event context frames from buffer:")
        for pf in pre_frames:
            print(f"    - FramePacket seq={pf.sequence_number} (ts={pf.timestamp:.2f})")

        print("\nPipeline execution successful. Zero AI overhead in Video Integration.")
        print("=" * 60)

    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
