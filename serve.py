"""IBVAP Standalone Dashboard Server.

Starts the FastAPI REST API + Dashboard on localhost without running the
full video-processing pipeline. Useful when you just want to open the
dashboard, browse events that were already stored, or keep the server
running persistently.

Usage:
    python serve.py                      # serve on http://localhost:8000
    python serve.py --port 8080          # custom port
    python serve.py --host 0.0.0.0      # expose on LAN
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IBVAP Dashboard — standalone API server"
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--event-db", default="storage/ibvap_events.db",
                   help="Path to SQLite events database")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from backend.alerts.manager import AlertManager
    from backend.events.store import EventStore
    from backend.api import server as api_server

    alert_manager = AlertManager(cooldown_seconds=5.0)
    event_store = EventStore(db_path=args.event_db)

    # Seed STATE with reasonable defaults so the dashboard isn't blank
    api_server.STATE.source_id = "LIVE_CAM_01"
    api_server.STATE.running = True
    api_server.STATE.started_at = time.time()
    api_server.STATE.events_total = event_store.count()

    app = api_server.create_app(
        alert_manager=alert_manager,
        event_store=event_store,
    )

    print()
    print("=" * 65)
    print("  IBVAP Dashboard Server")
    print("=" * 65)
    print(f"  Dashboard  →  http://{args.host}:{args.port}/")
    print(f"  Health     →  http://{args.host}:{args.port}/health")
    print(f"  Events API →  http://{args.host}:{args.port}/events")
    print(f"  Alerts API →  http://{args.host}:{args.port}/alerts")
    print(f"  Status API →  http://{args.host}:{args.port}/status")
    print("=" * 65)
    print("  Press Ctrl+C to stop.")
    print()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
