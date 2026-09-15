"""IBVAP Events Package — SQLite-backed V1 event/detection persistence."""
from backend.events.store import EventRecord, EventStore

__all__ = ["EventRecord", "EventStore"]
