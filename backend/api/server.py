"""IBVAP V1 REST API (FastAPI).

Read-only monitoring + control surface over the V1 pipeline.
Owns no video/AI logic: it only exposes orchestrator stats,
Engine 2 results, and AlertManager state managed by the runner.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    CORSMiddleware = None  # type: ignore[assignment]
    FileResponse = None  # type: ignore[assignment]
    StreamingResponse = None  # type: ignore[assignment]
    StaticFiles = None  # type: ignore[assignment]

    class BaseModel:  # type: ignore[no-redef]
        pass
    _HAS_FASTAPI = False


class PipelineState:
    """Shared mutable state written by the runner, read by the API."""

    def __init__(self) -> None:
        self.source_id = "LIVE_CAM_01"
        self.source_status = "CONNECTED"
        self.engine1_status = "READY"
        self.engine2_status = "READY"
        self.engine2_model = "yolov8n.pt"
        self.last_inference_ms: float = 0.0
        self.total_inference_ms: float = 0.0
        self.avg_inference_ms: float = 0.0
        self.running = False
        self.started_at: Optional[float] = None
        self.total_frames = 0
        self.suppressed_frames = 0
        self.motion_frames = 0
        self.engine2_frames = 0
        self.engine2_targets = 0
        self.events_total = 0
        self.last_result: Optional[Dict[str, Any]] = None
        self.recent_results: List[Dict[str, Any]] = []
        self.last_frame_jpeg: Optional[bytes] = None
        self.snapshot_b64: Optional[str] = None
        self.snapshot_seq: int = -1

    def summary(self) -> Dict[str, Any]:
        total = self.total_frames
        return {
            "source_id": self.source_id,
            "source_status": self.source_status,
            "engine1_status": self.engine1_status,
            "engine2_status": self.engine2_status,
            "engine2_model": self.engine2_model,
            "last_inference_ms": round(self.last_inference_ms, 1),
            "avg_inference_ms": round(self.avg_inference_ms, 1),
            "running": self.running,
            "uptime_seconds": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
            "total_frames": self.total_frames,
            "suppressed_frames": self.suppressed_frames,
            "motion_frames": self.motion_frames,
            "suppression_ratio_pct": round((self.suppressed_frames / total) * 100, 2) if total else 0.0,
            "engine2_frames": self.engine2_frames,
            "engine2_targets": self.engine2_targets,
            "events_total": self.events_total,
        }


STATE = PipelineState()
ALERTS_REF: Dict[str, Any] = {}
EVENTS_REF: Dict[str, Any] = {}
_SUBSCRIBERS: List[Any] = []


def publish_event(payload: Dict[str, Any]) -> None:
    """Push event payload to all SSE subscribers (thread-safe)."""
    for q in list(_SUBSCRIBERS):
        try:
            q.put_nowait(payload)
        except Exception:
            pass


def create_app(alert_manager=None, event_store=None) -> Any:
    """Build the FastAPI app. alert_manager: AlertManager instance."""
    if not _HAS_FASTAPI:
        raise RuntimeError("fastapi is not installed. pip install fastapi uvicorn")
    if alert_manager is not None:
        ALERTS_REF["manager"] = alert_manager
    if event_store is not None:
        EVENTS_REF["store"] = event_store
    app = FastAPI(title="IBVAP V1 API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "service": "ibvap-v1", "time": time.time()}

    @app.get("/stats")
    def stats() -> Dict[str, Any]:
        return STATE.summary()

    @app.get("/status")
    def status() -> Dict[str, Any]:
        s = STATE.summary()
        mgr = ALERTS_REF.get("manager")
        return {
            "camera": {"source_id": s["source_id"], "status": s["source_status"]},
            "engines": {
                "video": s["source_status"],
                "engine1": s["engine1_status"],
                "engine2": s["engine2_status"],
                "api": "READY"
            },
            "engine_details": {
                "engine1_role": "Lightweight Motion Gate (OpenCV)",
                "engine2_role": "Human / Vehicle Detection (YOLOv8)",
                "model": s["engine2_model"],
                "last_inference_ms": s["last_inference_ms"],
                "avg_inference_ms": s["avg_inference_ms"],
                "targets_detected": s["engine2_targets"],
            },
            "pipeline": s,
            "alerts": mgr.stats() if mgr else {},
        }

    @app.get("/detections/latest")
    def latest() -> Dict[str, Any]:
        return {"latest": STATE.last_result}

    @app.get("/detections/recent")
    def recent(limit: int = 20) -> Dict[str, Any]:
        return {"results": STATE.recent_results[-limit:][::-1]}

    @app.get("/alerts")
    def alerts(limit: int = 50) -> Dict[str, Any]:
        mgr = ALERTS_REF.get("manager")
        if mgr is None:
            return {"alerts": []}
        return {"alerts": [a.__dict__ for a in mgr.list_alerts(limit=limit)]}

    @app.get("/alerts/stats")
    def alert_stats() -> Dict[str, Any]:
        mgr = ALERTS_REF.get("manager")
        if mgr is None:
            return {"total_evaluated": 0, "total_alerts": 0,
                    "total_suppressed_dupes": 0, "stored": 0}
        return mgr.stats()

    @app.get("/events")
    def events(limit: int = 50) -> Dict[str, Any]:
        store = EVENTS_REF.get("store")
        if store is None:
            return {"events": []}
        return {"events": store.list_events(limit=limit)}

    @app.get("/events/{event_id}")
    def event_detail(event_id: str) -> Dict[str, Any]:
        from fastapi import HTTPException
        store = EVENTS_REF.get("store")
        if store is None:
            raise HTTPException(status_code=404, detail="no event store")
        ev = store.get_event(event_id)
        if ev is None:
            raise HTTPException(status_code=404, detail="event not found")
        return {"event": ev}

    @app.get("/events/{event_id}/clip")
    def event_clip(event_id: str) -> Any:
        from fastapi import HTTPException
        store = EVENTS_REF.get("store")
        if store is None:
            raise HTTPException(status_code=404, detail="no event store")
        ev = store.get_event(event_id)
        if ev is None or not ev.get("clip_path"):
            raise HTTPException(status_code=404, detail="clip not found")
        path = Path(ev["clip_path"])
        # Stored clip_path may be absolute; also try relative to project root.
        if not path.exists():
            raise HTTPException(status_code=404, detail="clip file missing")
        return FileResponse(str(path), media_type="video/mp4",
                            filename=f"{event_id}.mp4")

    @app.get("/notifications")
    def notifications(limit: int = 50) -> Dict[str, Any]:
        mgr = ALERTS_REF.get("manager")
        if mgr is None:
            return {"notifications": []}
        return {"notifications": [a.__dict__ for a in mgr.list_alerts(limit=limit)]}

    @app.post("/notifications/{alert_id}/ack")
    def acknowledge_notification(alert_id: str) -> Dict[str, Any]:
        from fastapi import HTTPException
        mgr = ALERTS_REF.get("manager")
        if mgr is None:
            raise HTTPException(status_code=404, detail="no alert manager")
        ok = mgr.acknowledge(alert_id)
        if not ok:
            raise HTTPException(status_code=404, detail="alert not found")
        return {"status": "acknowledged", "alert_id": alert_id}

    @app.get("/snapshot.jpg")
    def snapshot() -> Any:
        from fastapi import HTTPException, Response
        jpeg = STATE.last_frame_jpeg
        if jpeg is None and STATE.snapshot_b64:
            import base64 as _b64
            try:
                jpeg = _b64.b64decode(STATE.snapshot_b64)
            except Exception:
                jpeg = None
        if jpeg is None:
            raise HTTPException(status_code=404, detail="no frame yet")
        return Response(content=jpeg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store, must-revalidate"})

    @app.get("/events/stream")
    async def event_stream() -> Any:
        async def gen():
            q: asyncio.Queue = asyncio.Queue()
            _SUBSCRIBERS.append(q)
            try:
                yield "retry: 3000\n\n"
                # Replay recent events so a fresh dashboard is not empty.
                store = EVENTS_REF.get("store")
                if store is not None:
                    try:
                        for ev in store.list_events(limit=10)[::-1]:
                            import json as _json
                            yield f"data: {_json.dumps(ev)}\n\n"
                    except Exception:
                        pass
                # Replay recent notifications
                mgr = ALERTS_REF.get("manager")
                if mgr is not None:
                    try:
                        for a in mgr.list_alerts(limit=5)[::-1]:
                            import json as _json
                            notif_payload = {
                                "type": "notification",
                                **a.__dict__
                            }
                            yield f"data: {_json.dumps(notif_payload)}\n\n"
                    except Exception:
                        pass
                while True:
                    try:
                        payload = await asyncio.wait_for(q.get(), timeout=25.0)
                        import json as _json
                        yield f"data: {_json.dumps(payload)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                try:
                    _SUBSCRIBERS.remove(q)
                except ValueError:
                    pass
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # Serve the Neumorphic dashboard (frontend/dist) at / when built.
    try:
        dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
        if dist.exists():
            app.mount("/", StaticFiles(directory=str(dist), html=True),
                      name="dashboard")
    except Exception:
        pass

    return app


def record_result(result) -> None:
    """Called by run_pipeline per Engine 2 result; updates shared STATE."""
    STATE.engine2_frames += 1
    if result.has_target:
        STATE.engine2_targets += 1
    STATE.last_inference_ms = float(result.processing_time_ms)
    STATE.total_inference_ms += float(result.processing_time_ms)
    STATE.avg_inference_ms = STATE.total_inference_ms / max(1, STATE.engine2_frames)
    if result.model_name:
        STATE.engine2_model = result.model_name
    payload = {"request_id": result.request_id, "source_id": result.source_id,
               "frame_id": result.frame_id, "timestamp": result.timestamp,
               "detections": [{"class": d.detected_class.value,
                               "confidence": d.confidence,
                               "bbox": d.bounding_box.model_dump()}
                              for d in result.detections],
               "has_target": result.has_target,
               "processing_time_ms": result.processing_time_ms,
               "model_name": result.model_name}
    STATE.last_result = payload
    STATE.recent_results.append(payload)
    STATE.recent_results = STATE.recent_results[-100:]
