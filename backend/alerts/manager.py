"""IBVAP Alert Manager — V1 in-memory alert store.

Receives ClassificationResults from Engine 2 (via Core), applies the V1
alert rule (HUMAN or VEHICLE present), deduplicates per source+class within
a cooldown window, and retains recent alerts in memory for the API layer.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.contracts.classification_result import ClassificationResult


@dataclass
class Alert:
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    frame_id: str = ""
    request_id: str = ""
    event_id: str = ""
    timestamp: float = 0.0
    detected_classes: List[str] = field(default_factory=list)
    num_detections: int = 0
    detections: List[Dict[str, Any]] = field(default_factory=list)
    severity: str = "HIGH"
    acknowledged: bool = False
    model_name: str = ""
    created_at: float = field(default_factory=time.time)


class AlertManager:
    """Thread-safe in-memory alert store with cooldown dedup."""

    def __init__(self, cooldown_seconds: float = 5.0, max_alerts: int = 500) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.max_alerts = max_alerts
        self._lock = threading.Lock()
        self._alerts: List[Alert] = []
        self._last_emit: Dict[str, float] = {}
        self.total_evaluated = 0
        self.total_alerts = 0
        self.total_suppressed_dupes = 0

    def evaluate(self, result: ClassificationResult, event_id: str = "") -> Optional[Alert]:
        """Apply V1 rule; return Alert if raised, None if suppressed."""
        self.total_evaluated += 1
        if not result.has_target:
            return None
        classes = sorted({d.detected_class.value for d in result.detections
                          if d.detected_class.value in ("HUMAN", "VEHICLE")})
        if not classes:
            return None
        key = f"{result.source_id}:{'+'.join(classes)}"
        now = time.time()
        with self._lock:
            last = self._last_emit.get(key, 0.0)
            if now - last < self.cooldown_seconds:
                self.total_suppressed_dupes += 1
                return None
            self._last_emit[key] = now
            severity = "CRITICAL" if "HUMAN" in classes else "MEDIUM"
            alert = Alert(
                source_id=result.source_id, frame_id=result.frame_id,
                request_id=result.request_id, event_id=event_id,
                timestamp=result.timestamp,
                detected_classes=classes, num_detections=len(result.detections),
                detections=[{"class": d.detected_class.value,
                             "confidence": d.confidence,
                             "bbox": d.bounding_box.model_dump()}
                            for d in result.detections],
                severity=severity,
                model_name=result.model_name)
            self._alerts.append(alert)
            if len(self._alerts) > self.max_alerts:
                self._alerts = self._alerts[-self.max_alerts:]
            self.total_alerts += 1
            return alert

    def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged by an operator."""
        with self._lock:
            for a in self._alerts:
                if a.alert_id == alert_id:
                    a.acknowledged = True
                    return True
        return False

    def list_alerts(self, limit: int = 50) -> List[Alert]:
        with self._lock:
            return list(reversed(self._alerts[-limit:]))

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"total_evaluated": self.total_evaluated,
                    "total_alerts": self.total_alerts,
                    "total_suppressed_dupes": self.total_suppressed_dupes,
                    "stored": len(self._alerts)}
