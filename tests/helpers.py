"""Test fixtures and helpers for IBVAP tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator
import cv2
import numpy as np


def create_synthetic_video(
    num_frames: int = 15,
    width: int = 320,
    height: int = 240,
    fps: int = 10,
) -> str:
    """Create a temporary synthetic MP4 video file and return its path."""
    temp_dir = tempfile.mkdtemp(prefix="ibvap_test_video_")
    file_path = os.path.join(temp_dir, "test_input.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, float(fps), (width, height))

    if not out.isOpened():
        # Fallback to MJPG if mp4v is not supported by backend
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        file_path = os.path.join(temp_dir, "test_input.avi")
        out = cv2.VideoWriter(file_path, fourcc, float(fps), (width, height))

    for i in range(num_frames):
        # Static dark background
        frame = np.full((height, width, 3), fill_value=35, dtype=np.uint8)
        # Moving target rectangle (localized motion)
        x = 20 + (i * 14) % (width - 70)
        cv2.rectangle(frame, (x, 60), (x + 50, 140), (220, 220, 220), -1)
        cv2.putText(
            frame,
            f"F:{i}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        out.write(frame)

    out.release()
    return file_path
