"""IBVAP Member 1 - Video Ingestion CLI Runner.

Usage:
    # Run with a local video file:
    python run.py --file path/to/video.mp4

    # Run with a local video file and live preview window:
    python run.py --file path/to/video.mp4 --display

    # Run with an RTSP camera stream:
    python run.py --rtsp rtsp://192.168.1.100:554/live

    # Run synthetic demo (no input file needed):
    python run.py --demo
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import cv2

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.contracts.frame import SourceConfig, SourceStatus, SourceType
from backend.video.buffer import RollingFrameBuffer
from backend.video.file_source import FileSource
from backend.video.reader import VideoReader
from backend.video.rtsp_source import RTSPSource
from backend.video.source import VideoSource
from tests.helpers import create_synthetic_video


class WebcamSource(VideoSource):
    """VideoSource for live USB/laptop webcams without modifying backend/video."""

    def __init__(self, config: SourceConfig, device_index: int = 0) -> None:
        super().__init__(config)
        self.device_index = device_index
        self._cap = None
        self._metadata = {}

    def open(self) -> bool:
        self._status = SourceStatus.CONNECTING
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            self._status = SourceStatus.ERROR
            return False
        self._status = SourceStatus.CONNECTED
        self._metadata = {
            "source_id": self.source_id,
            "device_index": self.device_index,
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480),
            "fps": float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0),
        }
        return True

    def read_frame(self):
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None
        return True, frame

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._status = SourceStatus.DISCONNECTED

    def get_metadata(self) -> dict:
        return self._metadata.copy()


def parse_args():
    parser = argparse.ArgumentParser(description="IBVAP Video Ingestion Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", type=str, help="Path to local video file (MP4, AVI, MOV)")
    group.add_argument("--rtsp", "-r", type=str, help="RTSP stream URL (e.g. rtsp://192.168.1.50:554/stream)")
    group.add_argument("--camera", "-c", type=int, nargs="?", const=0, help="Live webcam index (default: 0)")
    group.add_argument("--demo", "-d", action="store_true", help="Run with a generated synthetic test video")

    parser.add_argument("--source-id", type=str, default="CAM_01", help="Identifier for this camera/source")
    parser.add_argument("--buffer-capacity", type=int, default=90, help="Rolling buffer capacity (frames)")
    parser.add_argument("--display", action="store_true", help="Show live OpenCV display window")
    parser.add_argument("--fps-limit", type=float, default=None, help="Simulate real-time playback by sleeping to target FPS")
    return parser.parse_args()


def main():
    args = parse_args()
    cleanup_path = None

    if args.demo:
        print("[*] Generating synthetic test video...")
        video_uri = create_synthetic_video(num_frames=60, width=640, height=480, fps=15)
        source_type = SourceType.FILE
        cleanup_path = video_uri
        print(f"[*] Generated synthetic video at: {video_uri}")
    elif args.file:
        video_uri = args.file
        source_type = SourceType.FILE
    elif args.camera is not None:
        video_uri = f"webcam://{args.camera}"
        source_type = SourceType.FILE
        args.display = True
    else:
        video_uri = args.rtsp
        source_type = SourceType.RTSP

    config = SourceConfig(
        source_id=args.source_id,
        source_type=source_type,
        source_uri=video_uri,
        target_fps=args.fps_limit,
    )

    try:
        if args.camera is not None:
            source = WebcamSource(config, device_index=args.camera)
        elif source_type == SourceType.FILE:
            source = FileSource(config)
        else:
            source = RTSPSource(config)

        # Initialize rolling buffer and reader coordinator
        buffer = RollingFrameBuffer(capacity=args.buffer_capacity)
        reader = VideoReader(source=source, buffer=buffer)

        print(f"\n[*] Starting Ingestion for source: {args.source_id} ({source_type.value})")
        print(f"[*] Rolling buffer capacity: {args.buffer_capacity} frames")
        if args.display:
            print("[*] Display window enabled. Press 'q' in the window to stop.")

        start_time = time.time()
        frame_count = 0

        # Stream FramePackets through the pipeline
        for packet in reader.stream_packets():
            frame_count += 1

            # Log periodically to avoid terminal flooding
            if frame_count % 15 == 0 or frame_count <= 3:
                print(
                    f"  -> Packet #{packet.sequence_number:04d} | "
                    f"ID: {packet.frame_id[:8]}... | "
                    f"Size: {packet.frame_width}x{packet.frame_height} | "
                    f"Buffer: {len(buffer)}/{buffer.capacity}"
                )

            # Optional live preview
            if args.display:
                cv2.imshow(f"IBVAP Ingestion - {args.source_id}", packet.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[*] User requested stop via window.")
                    break

            # Optional FPS throttle for local file viewing
            if args.fps_limit and args.fps_limit > 0:
                time.sleep(1.0 / args.fps_limit)

        elapsed = time.time() - start_time
        fps = (frame_count / elapsed) if elapsed > 0 else 0
        print(f"\n[*] Ingestion complete:")
        print(f"    - Total frames ingested: {frame_count}")
        print(f"    - Elapsed time: {elapsed:.2f}s ({fps:.1f} FPS)")
        print(f"    - Final source status: {source.status.value}")
        print(f"    - Context frames in buffer: {len(buffer)}")

    finally:
        if args.display:
            cv2.destroyAllWindows()
        if cleanup_path and os.path.exists(cleanup_path):
            try:
                os.remove(cleanup_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
