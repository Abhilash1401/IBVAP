"""Alert manager + API server tests (no network, no YOLO weights)."""
from __future__ import annotations

import unittest

from backend.alerts.manager import AlertManager
from backend.api.server import STATE, create_app, record_result
from backend.contracts.classification_result import (
    BoundingBox,
    ClassificationResult,
    DetectedClass,
    Detection,
)


def _result(classes, source="CAM_A"):
    dets = [Detection(detected_class=c, confidence=0.9,
                      bounding_box=BoundingBox(x=1, y=1, width=5, height=5))
            for c in classes]
    return ClassificationResult(
        request_id="r1", source_id=source, frame_id="f1", timestamp=1.0,
        detections=dets, processing_time_ms=2.0, model_name="m")


class TestAlertManager(unittest.TestCase):
    def test_target_raises_alert(self):
        mgr = AlertManager(cooldown_seconds=60)
        a = mgr.evaluate(_result([DetectedClass.HUMAN]))
        self.assertIsNotNone(a)
        self.assertEqual(a.detected_classes, ["HUMAN"])
        self.assertEqual(mgr.stats()["total_alerts"], 1)

    def test_nontarget_suppressed(self):
        mgr = AlertManager()
        self.assertIsNone(mgr.evaluate(_result([DetectedClass.NON_TARGET])))
        self.assertIsNone(mgr.evaluate(_result([])))
        self.assertEqual(mgr.stats()["total_alerts"], 0)

    def test_cooldown_dedup(self):
        mgr = AlertManager(cooldown_seconds=60)
        self.assertIsNotNone(mgr.evaluate(_result([DetectedClass.HUMAN])))
        self.assertIsNone(mgr.evaluate(_result([DetectedClass.HUMAN])))
        self.assertEqual(mgr.stats()["total_suppressed_dupes"], 1)

    def test_bounded_store(self):
        mgr = AlertManager(cooldown_seconds=0, max_alerts=3)
        for i in range(5):
            mgr.evaluate(_result([DetectedClass.VEHICLE], source=f"C{i}"))
        self.assertEqual(len(mgr.list_alerts(limit=10)), 3)


    def test_alert_severity_and_event_id(self):
        mgr = AlertManager(cooldown_seconds=0)
        a_human = mgr.evaluate(_result([DetectedClass.HUMAN]), event_id="evt-101")
        self.assertIsNotNone(a_human)
        self.assertEqual(a_human.severity, "CRITICAL")
        self.assertEqual(a_human.event_id, "evt-101")
        self.assertFalse(a_human.acknowledged)

        a_veh = mgr.evaluate(_result([DetectedClass.VEHICLE], source="CAM_B"), event_id="evt-102")
        self.assertIsNotNone(a_veh)
        self.assertEqual(a_veh.severity, "MEDIUM")
        self.assertEqual(a_veh.event_id, "evt-102")

    def test_acknowledge_alert(self):
        mgr = AlertManager(cooldown_seconds=0)
        a = mgr.evaluate(_result([DetectedClass.HUMAN]))
        self.assertFalse(a.acknowledged)
        ok = mgr.acknowledge(a.alert_id)
        self.assertTrue(ok)
        self.assertTrue(a.acknowledged)
        # Nonexistent alert returns False
        self.assertFalse(mgr.acknowledge("fake-id"))


class TestAPIServer(unittest.TestCase):
    def test_endpoints(self):
        from fastapi.testclient import TestClient
        mgr = AlertManager(cooldown_seconds=0)
        app = create_app(alert_manager=mgr)
        client = TestClient(app)
        self.assertEqual(client.get("/health").status_code, 200)
        self.assertIn("total_frames", client.get("/stats").json())
        mgr.evaluate(_result([DetectedClass.HUMAN]))
        alerts = client.get("/alerts").json()["alerts"]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["detected_classes"], ["HUMAN"])
        record_result(_result([DetectedClass.VEHICLE]))
        self.assertEqual(client.get("/detections/latest").json()["latest"]["has_target"], True)

    def test_notifications_and_acknowledgement(self):
        from fastapi.testclient import TestClient
        mgr = AlertManager(cooldown_seconds=0)
        app = create_app(alert_manager=mgr)
        client = TestClient(app)
        alert = mgr.evaluate(_result([DetectedClass.HUMAN]))
        
        # Test GET /notifications
        res = client.get("/notifications")
        self.assertEqual(res.status_code, 200)
        notifs = res.json()["notifications"]
        self.assertEqual(len(notifs), 1)
        self.assertEqual(notifs[0]["alert_id"], alert.alert_id)
        self.assertFalse(notifs[0]["acknowledged"])

        # Test POST /notifications/{alert_id}/ack
        ack_res = client.post(f"/notifications/{alert.alert_id}/ack")
        self.assertEqual(ack_res.status_code, 200)
        self.assertEqual(ack_res.json()["status"], "acknowledged")
        self.assertTrue(alert.acknowledged)

        # Test 404 for invalid alert
        self.assertEqual(client.post("/notifications/invalid-id/ack").status_code, 404)

    def test_snapshot_endpoint(self):
        from fastapi.testclient import TestClient
        import base64
        app = create_app()
        client = TestClient(app)

        # 404 when no frame
        STATE.last_frame_jpeg = None
        STATE.snapshot_b64 = None
        self.assertEqual(client.get("/snapshot.jpg").status_code, 404)

        # 200 with raw bytes
        dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        STATE.last_frame_jpeg = dummy_jpeg
        res = client.get("/snapshot.jpg")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "image/jpeg")
        self.assertEqual(res.content, dummy_jpeg)

        # 200 with base64 fallback
        STATE.last_frame_jpeg = None
        STATE.snapshot_b64 = base64.b64encode(dummy_jpeg).decode("ascii")
        res_b64 = client.get("/snapshot.jpg")
        self.assertEqual(res_b64.status_code, 200)
        self.assertEqual(res_b64.content, dummy_jpeg)

    def test_no_manager_returns_empty(self):
        from fastapi.testclient import TestClient
        from backend.api import server
        server.ALERTS_REF.clear()
        client = TestClient(create_app())
        self.assertEqual(client.get("/alerts").json(), {"alerts": []})
        self.assertEqual(client.get("/notifications").json(), {"notifications": []})


if __name__ == "__main__":
    unittest.main()
