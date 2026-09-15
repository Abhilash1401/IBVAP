"""IBVAP Pipeline Runner: Member 1 (Video Integration) + Member 2 (Core & Engine 1)
+ Member 3 (Engine 2 YOLO Human/Vehicle Detection).

Demonstrates the progressive delegation pipeline:
    Continuous Video -> VideoReader -> Core -> Engine 1 (Motion Gate)
    -> Engine 2 (YOLO: HUMAN / VEHICLE) -> ClassificationResult

Usage:
    # Run pipeline with a local video file (motion gating only, mock Engine 2):
    python run_pipeline.py --file path/to/video.mp4

    # Run with live visual motion display:
    python run_pipeline.py --file path/to/video.mp4 --display

    # Run synthetic demonstration:
    python run_pipeline.py --demo --display

    # Run full pipeline with REAL Engine 2 YOLO inference on motion frames:
    python run_pipeline.py --demo --engine2
    python run_pipeline.py --file path/to/video.mp4 --engine2 --display
"""

from __future__ import annotations

import argparse
from collections import deque
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.contracts.classification import ClassificationRequest
from backend.contracts.classification_result import ClassificationResult
from backend.contracts.frame import SourceConfig, SourceStatus, SourceType
from backend.contracts.motion import MotionEventType
from backend.core.orchestrator import CoreOrchestrator
from backend.engine1.motion_detector import Engine1MotionDetector
from backend.engine2.detector import Engine2HumanVehicleDetector
from backend.video.buffer import RollingFrameBuffer
from backend.video.exceptions import SourceNotFoundError
from backend.video.file_source import FileSource
from backend.video.reader import VideoReader
from backend.video.rtsp_source import RTSPSource
from backend.video.source import VideoSource
from backend.events.store import EventRecord, EventStore
from backend.clips.writer import ClipWriter


class WebcamSource(VideoSource):
    """VideoSource for live USB/laptop webcams without modifying backend/video."""

    def __init__(self, config: SourceConfig, device_index: int = 0) -> None:
        super().__init__(config)
        self.device_index = device_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: dict = {}

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

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
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


def create_demo_video_with_motion() -> str:
    """Create a temporary video with alternating static and moving scenes."""
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="ibvap_motion_demo_")
    file_path = os.path.join(temp_dir, "motion_test.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, 15.0, (640, 480))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        file_path = os.path.join(temp_dir, "motion_test.avi")
        out = cv2.VideoWriter(file_path, fourcc, 15.0, (640, 480))

    # Phase 1: 20 static frames (dark background)
    static_frame = np.full((480, 640, 3), fill_value=30, dtype=np.uint8)
    for _ in range(20):
        out.write(static_frame)

    # Phase 2: 25 frames with moving white rectangle (simulating person/vehicle walking across scene)
    for i in range(25):
        frame = static_frame.copy()
        x_pos = 50 + i * 18
        y_pos = 150 + int(np.sin(i / 3.0) * 10)
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 80, y_pos + 140), (255, 255, 255), -1)
        cv2.putText(frame, "MOVING TARGET", (x_pos - 10, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out.write(frame)

    # Phase 3: 15 static frames (scene goes quiet again)
    for _ in range(15):
        out.write(static_frame)

    out.release()
    return file_path


def parse_args():
    parser = argparse.ArgumentParser(description="IBVAP Video Ingestion + Engine 1 Motion Detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", type=str, help="Path to local video file")
    group.add_argument("--rtsp", "-r", type=str, help="RTSP stream URL (e.g. rtsp://192.168.1.100:554/live)")
    group.add_argument("--camera", "-c", type=int, nargs="?", const=0, help="Live webcam index (default: 0)")
    group.add_argument("--demo", "-d", action="store_true", help="Run with generated motion test video")

    parser.add_argument("--source-id", type=str, default="LIVE_CAM_01", help="Camera source identifier")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum motion pixel area to trigger Engine 2")
    parser.add_argument("--diff-thresh", type=int, default=25, help="Pixel intensity difference threshold")
    parser.add_argument("--display", action="store_true", help="Display live annotated OpenCV window")
    parser.add_argument("--fps-limit", type=float, default=None, help="Playback speed limiter for display")
    parser.add_argument("--engine2", action="store_true",
                        help="Run REAL Engine 2 YOLO inference on motion frames (default: mock sink)")
    parser.add_argument("--engine2-model", type=str, default="yolov8n.pt", help="YOLO weights path")
    parser.add_argument("--engine2-conf", type=float, default=0.40, help="YOLO confidence threshold")
    parser.add_argument("--engine2-imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--api", action="store_true", help="Serve REST API (FastAPI) in background thread")
    parser.add_argument("--api-port", type=int, default=8000, help="REST API port")
    parser.add_argument("--alert-cooldown", type=float, default=5.0, help="Alert dedup cooldown seconds")
    parser.add_argument("--event-db", type=str, default="storage/ibvap_events.db",
                        help="SQLite event DB path")
    parser.add_argument("--clips-dir", type=str, default="storage/event_clips",
                        help="Directory for moving event clips")
    parser.add_argument("--pre-frames", type=int, default=30, help="Pre-event frames per clip")
    parser.add_argument("--post-frames", type=int, default=20, help="Post-event frames per clip")
    parser.add_argument("--snapshot-every", type=int, default=3,
                        help="Encode live snapshot every N frames (0=off)")
    return parser.parse_args()


def main():
    args = parse_args()
    cleanup_path = None

    if args.demo:
        print("[*] Generating demonstration video with static & moving intervals...")
        video_uri = create_demo_video_with_motion()
        source_type = SourceType.FILE
        cleanup_path = video_uri
        print(f"[*] Generated video: {video_uri}")
    elif args.file:
        video_uri = args.file
        source_type = SourceType.FILE
    elif args.camera is not None:
        video_uri = f"webcam://{args.camera}"
        source_type = SourceType.FILE
        args.display = True  # Automatically enable display for live webcam
    else:
        video_uri = args.rtsp
        source_type = SourceType.RTSP

    # 1. Initialize Member 1 (Video Integration - UNTOUCHED)
    config = SourceConfig(
        source_id=args.source_id,
        source_type=source_type,
        source_uri=video_uri,
    )
    if args.camera is not None:
        source = WebcamSource(config, device_index=args.camera)
    elif source_type == SourceType.FILE:
        source = FileSource(config)
    else:
        source = RTSPSource(config)

    buffer = RollingFrameBuffer(capacity=90)
    reader = VideoReader(source=source, buffer=buffer)

    # 2. Initialize Member 2 (Engine 1 Motion Detector)
    detector = Engine1MotionDetector()
    detector.initialize({
        "diff_threshold": args.diff_thresh,
        "min_motion_area": args.min_area,
        "motion_score_threshold": 0.002,
        "blur_kernel_size": 21,
    })

    # 3. Engine 2: real YOLO detector (--engine2) or mock capture sink (default).
    #    + V1 Event persistence (SQLite) + moving clip writer.
    from backend.alerts.manager import AlertManager
    from backend.api import server as api_server
    alert_manager = AlertManager(cooldown_seconds=args.alert_cooldown)
    event_store = EventStore(db_path=args.event_db)
    clip_writer = ClipWriter(clips_dir=args.clips_dir, fps=20.0)
    captured_requests: List[ClassificationRequest] = []
    classification_results: List[ClassificationResult] = []
    pending_clips: deque = deque()  # {event_id, pre, post, need}
    engine2: Optional[Engine2HumanVehicleDetector] = None

    if args.engine2:
        print("[*] Initializing REAL Engine 2 (YOLO Human/Vehicle Detection)...")
        engine2 = Engine2HumanVehicleDetector()
        if not engine2.initialize({
            "model_path": args.engine2_model,
            "confidence_threshold": args.engine2_conf,
            "input_size": args.engine2_imgsz,
        }):
            print("[!] Engine 2 initialization failed. Falling back to mock sink.")
            engine2 = None
        else:
            print(f"[*] Engine 2 READY: model={args.engine2_model}, "
                  f"conf={args.engine2_conf}, imgsz={args.engine2_imgsz}")

    def _persist_motion_event(request: ClassificationRequest,
                              classification: str = "PENDING",
                              confidence: float = 0.0,
                              dets: Optional[List[Dict[str, Any]]] = None
                              ) -> Dict[str, Any]:
        """Persist motion event to SQLite + publish to SSE subscribers."""
        import uuid as _uuid
        event_id = str(_uuid.uuid4())
        rec = EventRecord(
            event_id=event_id,
            source_id=request.source_id,
            frame_id=request.frame_id,
            request_id=request.request_id,
            timestamp=request.timestamp,
            event_type="MOTION_DETECTED",
            motion_score=float(request.metadata.get("motion_score", 0.0)),
            classification=classification,
            confidence=confidence,
        )
        event_store.create_event(rec, detections=dets or [])
        api_server.STATE.events_total = event_store.count()
        payload = {"event_id": event_id, "source_id": rec.source_id,
                   "frame_id": rec.frame_id, "request_id": rec.request_id,
                   "timestamp": rec.timestamp, "created_at": rec.created_at,
                   "event_type": rec.event_type,
                   "motion_score": rec.motion_score,
                   "classification": classification, "confidence": confidence,
                   "detections": dets or [], "clip_path": "",
                   "notification_status": "NONE"}
        api_server.publish_event(payload)
        # Queue clip assembly: pre-event context + trigger frame.
        try:
            pre = buffer.get_pre_event_frames(count=args.pre_frames)
        except Exception:
            pre = []
        try:
            trigger = next((p for p in reversed(pre)
                            if p.frame_id == request.frame_id), None)
        except Exception:
            trigger = None
        from backend.contracts.frame import FramePacket as _FP
        if trigger is None:
            trigger = _FP.create(source_id=request.source_id,
                                 timestamp=request.timestamp,
                                 frame=request.frame,
                                 sequence_number=int(
                                     request.metadata.get(
                                         "sequence_number", 0)))
        pending_clips.append({"event_id": event_id, "pre": list(pre),
                              "trigger": trigger, "post": [],
                              "need": max(0, int(args.post_frames))})
        return payload

    def mock_engine2_receiver(request: ClassificationRequest):
        captured_requests.append(request)
        _persist_motion_event(request)

    def real_engine2_receiver(request: ClassificationRequest):
        captured_requests.append(request)
        assert engine2 is not None
        result = engine2.process(request)
        classification_results.append(result)
        api_server.record_result(result)
        dets = [{"class": d.detected_class.value,
                 "confidence": d.confidence,
                 "bbox": d.bounding_box.model_dump()}
                for d in result.detections]
        classes = sorted({d["class"] for d in dets
                          if d["class"] in ("HUMAN", "VEHICLE")})
        top = max([d["confidence"] for d in dets], default=0.0)
        payload = _persist_motion_event(
            request,
            classification="+".join(classes) if classes else (
                "TARGET" if result.has_target else "NO_TARGET"),
            confidence=float(top), dets=dets)
        event_store.update_classification(payload["event_id"],
                                          payload["classification"],
                                          float(top), dets)
        alert = alert_manager.evaluate(result, event_id=payload["event_id"])
        if alert is not None:
            event_store.mark_notified(payload["event_id"], "NOTIFIED")
            notif_data = {
                "type": "notification",
                "alert_id": alert.alert_id,
                "event_id": payload["event_id"],
                "source_id": alert.source_id,
                "timestamp": alert.timestamp,
                "severity": alert.severity,
                "detected_classes": alert.detected_classes,
                "confidence": float(top),
                "num_detections": alert.num_detections,
                "detections": dets,
                "clip_url": f"/events/{payload['event_id']}/clip",
                "created_at": alert.created_at,
                "acknowledged": False,
            }
            api_server.publish_event(notif_data)
            print(f"      [NOTIFICATION {alert.severity}] {alert.detected_classes} "
                  f"(conf={top:.2f}, {alert.num_detections} dets, id={alert.alert_id[:8]})")
        elif result.has_target:
            classes = [d.detected_class.value for d in result.detections]
            print(f"      [E2 YOLO] {classes} "
                  f"({len(result.detections)} detections, {result.processing_time_ms:.1f}ms)")

    engine2_handler = real_engine2_receiver if engine2 is not None else mock_engine2_receiver

    # 4. Initialize Core Orchestrator
    orchestrator = CoreOrchestrator(
        engine1_plugin=detector,
        engine2_handler=engine2_handler,
    )

    e2_mode = "REAL YOLO" if engine2 is not None else "MOCK sink"

    print("\n" + "=" * 65)
    print("  IBVAP PIPELINE: Video Integration (M1) -> Core -> Engine 1 (M2)")
    print("=" * 65)
    print(f"[*] Camera Source: {args.source_id} ({source_type.value})")
    print(f"[*] Engine 1 Settings: min_area={args.min_area}px, diff_thresh={args.diff_thresh}")
    print(f"[*] Engine 2 Mode: {e2_mode}")
    print(f"[*] Alerts: V1 rule HUMAN/VEHICLE, cooldown {args.alert_cooldown}s")
    api_thread = None
    if args.api:
        import threading
        import uvicorn
        app = api_server.create_app(alert_manager=alert_manager,
                                    event_store=event_store)
        api_server.STATE.source_id = args.source_id
        api_server.STATE.running = True
        api_server.STATE.started_at = time.time()
        api_thread = threading.Thread(
            target=uvicorn.run,
            kwargs={"app": app, "host": "127.0.0.1", "port": args.api_port, "log_level": "warning"},
            daemon=True)
        api_thread.start()
        print(f"[*] REST API serving on http://127.0.0.1:{args.api_port} "
              f"(/health /stats /status /events /alerts + dashboard /)")
    if args.display:
        print("[*] Display window enabled. Press 'q' in the window to stop.")

    try:
        reader.open()
    except SourceNotFoundError as e:
        print(f"\n[!] {e}")
        print("[!] The file 'video.mp4' was not found in the project folder.")
        print("    Fix options:")
        print('      python run_pipeline.py --file "C:\\full\\path\\to\\your\\video.mp4" --engine2 --display --api')
        print("      python run_pipeline.py --demo --engine2 --display --api   # no file needed")
        print("      python run_pipeline.py --camera --engine2 --display       # use webcam")
        return
    start_time = time.time()

    try:
        for packet in reader.stream_packets():
            # Core evaluates packet through Engine 1
            request = orchestrator.process_frame_packet(packet)

            # Mirror pipeline counters into API STATE.
            api_server.STATE.total_frames = orchestrator.total_frames
            api_server.STATE.suppressed_frames = orchestrator.suppressed_frames
            api_server.STATE.motion_frames = orchestrator.motion_frames
            api_server.STATE.source_id = args.source_id
            api_server.STATE.source_status = "CONNECTED"
            api_server.STATE.engine1_status = "READY"
            api_server.STATE.engine2_status = (
                "READY" if engine2 is not None else "MOCK")

            # Collect post-event frames for pending clips; finalize ready ones.
            if pending_clips:
                done = []
                for pend in list(pending_clips):
                    if pend["need"] > 0:
                        pend["post"].append(packet)
                        pend["need"] -= 1
                    if pend["need"] <= 0:
                        done.append(pend)
                for pend in done:
                    try:
                        packets = clip_writer.build_event_packets(
                            pend["pre"], pend["trigger"], pend["post"],
                            max_pre=args.pre_frames,
                            max_post=args.post_frames)
                        clip_path = clip_writer.write_clip(
                            pend["event_id"], packets)
                        if clip_path:
                            event_store.update_clip(
                                pend["event_id"], clip_path)
                            api_server.publish_event({
                                "type": "clip_ready",
                                "event_id": pend["event_id"],
                                "clip_url": (
                                    "/events/" + pend["event_id"] + "/clip"),
                                "frames": len(packets)})
                            print(f"      [CLIP] {len(packets)} frames -> "
                                  f"{clip_path}")
                    except Exception as e:
                        print(f"      [CLIP-ERR] {e}")
                    finally:
                        try:
                            pending_clips.remove(pend)
                        except ValueError:
                            pass

            # Live snapshot for dashboard (base64 JPEG, throttled).
            try:
                if args.snapshot_every and args.snapshot_every > 0:
                    if packet.sequence_number % args.snapshot_every == 0:
                        ok, jpg = cv2.imencode(
                            ".jpg", packet.frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                        if ok:
                            import base64 as _b64
                            jpg_bytes = jpg.tobytes()
                            api_server.STATE.last_frame_jpeg = jpg_bytes
                            api_server.STATE.snapshot_b64 = _b64.b64encode(
                                jpg_bytes).decode("ascii")
                            api_server.STATE.snapshot_seq = (
                                packet.sequence_number)
            except Exception:
                pass

            # Terminal progress (with live Engine 2 result when enabled)
            if engine2 is not None and classification_results:
                last = classification_results[-1] if request else None
            else:
                last = None
            if request and last is not None and last.detections:
                e2_tag = f" + E2:{','.join(d.detected_class.value for d in last.detections)}"
            elif request and last is not None:
                e2_tag = " + E2:no-target"
            else:
                e2_tag = ""
            status_tag = f"[MOTION -> ENGINE 2{e2_tag}]" if request else "[SUPPRESSED]"
            if packet.sequence_number % 5 == 0 or request is not None:
                print(
                    f"  Frame #{packet.sequence_number:03d} | "
                    f"Status: {status_tag:<22} | "
                    f"Buffer: {len(buffer):02d}/{buffer.capacity} | "
                    f"Suppression: {orchestrator.suppression_ratio:.1f}%"
                )

            # Live preview display with Member 1 & Member 2 HUD
            if args.display:
                display_frame = packet.frame.copy()
                h, w = display_frame.shape[:2]

                # Draw translucent top HUD banner
                hud_overlay = display_frame.copy()
                cv2.rectangle(hud_overlay, (0, 0), (w, 75), (20, 20, 20), -1)
                cv2.addWeighted(hud_overlay, 0.75, display_frame, 0.25, 0, display_frame)

                # Member 1 Status (Top line)
                m1_text = f"[M1 Ingestion] Frame #{packet.sequence_number:04d} | Buffer: {len(buffer):02d}/{buffer.capacity} | Source: {args.source_id}"
                cv2.putText(display_frame, m1_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

                # Member 2 Status (Second line)
                if request and request.motion_region:
                    m2_text = f"[M2 Engine 1] MOTION DETECTED (Score: {request.metadata.get('motion_score', 0):.4f}) -> Route to Engine 2"
                    m2_color = (0, 255, 0)
                    # Draw green bounding boxes specifically on the moving objects
                    all_regions = request.metadata.get("all_motion_regions", [])
                    if all_regions:
                        for reg in all_regions:
                            rx, ry, rw, rh = reg["x"], reg["y"], reg["width"], reg["height"]
                            cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"MOVING TARGET ({rw}x{rh})", (rx, max(25, ry - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        r = request.motion_region
                        cv2.rectangle(display_frame, (r.x, r.y), (r.x + r.width, r.y + r.height), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"MOVING TARGET ({r.width}x{r.height})", (r.x, max(25, r.y - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    m2_text = f"[M2 Engine 1] NO MOTION (Suppressed: {orchestrator.suppression_ratio:.1f}% AI Workload Saved)"
                    m2_color = (180, 180, 180)

                cv2.putText(display_frame, m2_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, m2_color, 2, cv2.LINE_AA)

                # Member 3 Status (Engine 2 YOLO boxes, blue) — only with --engine2
                if engine2 is not None and classification_results and request is not None:
                    latest = classification_results[-1]
                    if latest.frame_id == request.frame_id and latest.detections:
                        for det in latest.detections:
                            bb = det.bounding_box
                            cv2.rectangle(display_frame, (bb.x, bb.y),
                                          (bb.x + bb.width, bb.y + bb.height), (255, 0, 0), 2)
                            cv2.putText(display_frame,
                                        f"{det.detected_class.value} {det.confidence:.2f}",
                                        (bb.x, max(85, bb.y - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

                # Bottom control hint
                cv2.putText(display_frame, "Press 'q' to exit", (w - 140, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

                cv2.imshow(f"IBVAP V1 Live Pipeline - {args.source_id}", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[*] User stopped live camera feed.")
                    break

            if args.fps_limit and args.fps_limit > 0:
                time.sleep(1.0 / args.fps_limit)

    finally:
        elapsed = time.time() - start_time
        if args.display:
            cv2.destroyAllWindows()
        if engine2 is not None:
            engine2.shutdown()
        if cleanup_path and os.path.exists(cleanup_path):
            try:
                os.remove(cleanup_path)
            except OSError:
                pass

    print("\n" + "=" * 65)
    print("  PIPELINE PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"  Total Ingested Frames:    {orchestrator.total_frames}")
    print(f"  Suppressed Frames:        {orchestrator.suppressed_frames} (No motion, AI avoided)")
    print(f"  Motion Frames Routed:     {orchestrator.motion_frames} (Sent to Engine 2)")
    print(f"  AI Workload Reduction:    {orchestrator.suppression_ratio:.1f}% SAVED")
    print(f"  Processing Duration:      {elapsed:.2f}s ({orchestrator.total_frames / max(0.001, elapsed):.1f} FPS)")
    print(f"  Alerts Raised:          {alert_manager.stats()['total_alerts']} "
          f"(evaluated {alert_manager.stats()['total_evaluated']}, dupes suppressed "
          f"{alert_manager.stats()['total_suppressed_dupes']})")
    if engine2 is not None:
        n_targets = sum(1 for r in classification_results if r.has_target)
        avg_ms = (sum(r.processing_time_ms for r in classification_results)
                  / max(1, len(classification_results)))
        print(f"  Engine 2 Frames:          {len(classification_results)} (YOLO inferences)")
        print(f"  Engine 2 Target Frames:   {n_targets} (HUMAN/VEHICLE found)")
        print(f"  Engine 2 Avg Inference:   {avg_ms:.1f}ms/frame")
    print("=" * 65)


if __name__ == "__main__":
    main()
