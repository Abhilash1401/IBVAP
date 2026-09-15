"""IBVAP V1 Event Store — SQLite persistence for events/detections.

SQLite stores ONLY event metadata (never continuous frames).
Clips live on filesystem; DB holds clip_path.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EventRecord:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    frame_id: str = ""
    request_id: str = ""
    timestamp: float = 0.0
    created_at: float = field(default_factory=time.time)
    event_type: str = "MOTION_DETECTED"
    motion_score: float = 0.0
    classification: str = ""
    confidence: float = 0.0
    clip_path: str = ""
    notification_status: str = "NONE"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    frame_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    created_at REAL NOT NULL,
    event_type TEXT NOT NULL,
    motion_score REAL NOT NULL DEFAULT 0,
    classification TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0,
    clip_path TEXT NOT NULL DEFAULT '',
    notification_status TEXT NOT NULL DEFAULT 'NONE'
);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    detected_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    x INTEGER NOT NULL, y INTEGER NOT NULL,
    width INTEGER NOT NULL, height INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    last_seen REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'CONNECTED'
);
"""


class EventStore:
    """Thread-safe SQLite event store."""

    def __init__(self, db_path: str = "storage/ibvap_events.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA)
                conn.commit()
            finally:
                conn.close()

    def create_event(self, record: EventRecord,
                     detections: Optional[List[Dict[str, Any]]] = None) -> str:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO events "
                    "(event_id,source_id,frame_id,request_id,timestamp,"
                    "created_at,event_type,motion_score,classification,"
                    "confidence,clip_path,notification_status) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (record.event_id, record.source_id, record.frame_id,
                     record.request_id, record.timestamp, record.created_at,
                     record.event_type, record.motion_score,
                     record.classification, record.confidence,
                     record.clip_path, record.notification_status))
                if detections:
                    conn.execute("DELETE FROM detections WHERE event_id=?",
                                 (record.event_id,))
                    for d in detections:
                        bb = d.get("bbox", {}) or {}
                        conn.execute(
                            "INSERT INTO detections (event_id,detected_class,"
                            "confidence,x,y,width,height)"
                            " VALUES (?,?,?,?,?,?,?)",
                            (record.event_id, d.get("class", ""),
                             float(d.get("confidence", 0.0)),
                             int(bb.get("x", 0)), int(bb.get("y", 0)),
                             int(bb.get("width", 1)),
                             int(bb.get("height", 1))))
                conn.execute(
                    "INSERT OR REPLACE INTO sources (source_id,last_seen,"
                    "status) VALUES (?,?,?)",
                    (record.source_id, time.time(), "CONNECTED"))
                conn.commit()
                return record.event_id
            finally:
                conn.close()

    def update_classification(self, event_id: str, classification: str,
                              confidence: float,
                              detections=None) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE events SET classification=?, confidence=?"
                    " WHERE event_id=?",
                    (classification, confidence, event_id))
                if detections:
                    conn.execute("DELETE FROM detections WHERE event_id=?",
                                 (event_id,))
                    for d in detections:
                        bb = d.get("bbox", {}) or {}
                        conn.execute(
                            "INSERT INTO detections (event_id,detected_class,"
                            "confidence,x,y,width,height)"
                            " VALUES (?,?,?,?,?,?,?)",
                            (event_id, d.get("class", ""),
                             float(d.get("confidence", 0.0)),
                             int(bb.get("x", 0)), int(bb.get("y", 0)),
                             int(bb.get("width", 1)),
                             int(bb.get("height", 1))))
                conn.commit()
            finally:
                conn.close()

    def update_clip(self, event_id: str, clip_path: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("UPDATE events SET clip_path=? WHERE event_id=?",
                             (clip_path, event_id))
                conn.commit()
            finally:
                conn.close()

    def mark_notified(self, event_id: str,
                      status: str = "NOTIFIED") -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE events SET notification_status=? WHERE event_id=?",
                    (status, event_id))
                conn.commit()
            finally:
                conn.close()

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM events WHERE event_id=?",
                                   (event_id,)).fetchone()
                if row is None:
                    return None
                ev = dict(row)
                dets = conn.execute(
                    "SELECT detected_class,confidence,x,y,width,height"
                    " FROM detections WHERE event_id=?",
                    (event_id,)).fetchall()
                ev["detections"] = [
                    {"class": d["detected_class"],
                     "confidence": d["confidence"],
                     "bbox": {"x": d["x"], "y": d["y"],
                              "width": d["width"], "height": d["height"]}}
                    for d in dets]
                return ev
            finally:
                conn.close()

    def list_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY created_at DESC LIMIT ?",
                    (limit,)).fetchall()
                out = []
                for r in rows:
                    ev = dict(r)
                    dets = conn.execute(
                        "SELECT detected_class,confidence,x,y,width,height"
                        " FROM detections WHERE event_id=?",
                        (ev["event_id"],)).fetchall()
                    ev["detections"] = [
                        {"class": d["detected_class"],
                         "confidence": d["confidence"],
                         "bbox": {"x": d["x"], "y": d["y"],
                                  "width": d["width"], "height": d["height"]}}
                        for d in dets]
                    out.append(ev)
                return out
            finally:
                conn.close()

    def update_classification(self, event_id: str, classification: str,
                              confidence: float,
                              detections: list | None = None) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE events SET classification=?, confidence=?"
                    " WHERE event_id=?",
                    (classification, confidence, event_id))
                if detections:
                    conn.execute("DELETE FROM detections WHERE event_id=?",
                                 (event_id,))
                    for d in detections:
                        bb = d.get("bbox", {}) or {}
                        conn.execute(
                            "INSERT INTO detections (event_id,detected_class,"
                            "confidence,x,y,width,height)"
                            " VALUES (?,?,?,?,?,?,?)",
                            (event_id, d.get("class", ""),
                             float(d.get("confidence", 0.0)),
                             int(bb.get("x", 0)), int(bb.get("y", 0)),
                             int(bb.get("width", 1)),
                             int(bb.get("height", 1))))
                conn.commit()
            finally:
                conn.close()

    def count(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
                return int(row["n"])
            finally:
                conn.close()

