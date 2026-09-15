"""Engine 2 unit tests — mock-based (no YOLO weights needed)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import numpy as np

from backend.contracts.classification import ClassificationRequest
from backend.contracts.classification_result import DetectedClass
from backend.contracts.frame import FramePacket
from backend.contracts.motion import MotionEvent, MotionEventType, MotionRegion
from backend.core.orchestrator import CoreOrchestrator
from backend.core.plugin import PluginStatus
from backend.engine2.detector import (
    Engine2Config,
    Engine2HumanVehicleDetector,
    _map_coco_to_ibvap,
)


def _make_request(frame=None, with_region=True) -> ClassificationRequest:
    if frame is None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # Region chosen so padded ROI (x1=70,y1=70,x2=210,y2=210) is <70% of
    # a 320x240 frame, forcing the crop path with offset (70,70).
    region = MotionRegion(x=100, y=100, width=80, height=110) if with_region else None
    return ClassificationRequest(
        source_id="CAM_E2", frame_id="frame-1", timestamp=1700000000.0,
        frame=frame, motion_region=region)


class _FakeTensor:
    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [_FakeTensor(np.array(b, dtype=np.float32)) for b in xyxy]
        self.cls = [_FakeTensor(np.float32(c)) for c in cls]
        self.conf = [_FakeTensor(np.float32(c)) for c in conf]

    def __len__(self):
        return len(self.xyxy)


class _FakeResult:
    def __init__(self, xyxy, cls, conf):
        self.boxes = _FakeBoxes(xyxy, cls, conf)


def _detector_with_fake_model(results, **cfg_overrides):
    det = Engine2HumanVehicleDetector()
    det.config = Engine2Config()
    for k, v in cfg_overrides.items():
        setattr(det.config, k, v)
    det._model = MagicMock()
    det._model.predict.return_value = results
    det._model_name = "fake-yolov8n.pt"
    det._status = PluginStatus.READY
    return det


class TestCocoMapping(unittest.TestCase):
    def test_human_vehicle_nontarget(self):
        self.assertEqual(_map_coco_to_ibvap(0), DetectedClass.HUMAN)
        for cid in (1, 2, 3, 5, 7):
            self.assertEqual(_map_coco_to_ibvap(cid), DetectedClass.VEHICLE)
        for cid in (16, 9, 45, 79):
            self.assertEqual(_map_coco_to_ibvap(cid), DetectedClass.NON_TARGET)


class TestEngine2Lifecycle(unittest.TestCase):
    def test_rejects_wrong_input_type(self):
        det = Engine2HumanVehicleDetector()
        pkt = FramePacket.create(
            source_id="X", timestamp=1.0,
            frame=np.zeros((10, 10, 3), dtype=np.uint8), sequence_number=0)
        with self.assertRaises(TypeError):
            det.process(pkt)  # type: ignore[arg-type]

    def test_requires_initialize(self):
        det = Engine2HumanVehicleDetector()
        with self.assertRaises(RuntimeError):
            det.process(_make_request())

    def test_shutdown(self):
        det = _detector_with_fake_model([_FakeResult([], [], [])])
        det.shutdown()
        self.assertEqual(det.status, PluginStatus.SHUTDOWN)


class TestEngine2Process(unittest.TestCase):
    def test_empty_detections(self):
        det = _detector_with_fake_model([_FakeResult([], [], [])])
        res = det.process(_make_request(with_region=False))
        self.assertEqual(res.detections, [])
        self.assertFalse(res.has_target)
        self.assertGreaterEqual(res.processing_time_ms, 0.0)
        self.assertEqual(res.source_id, "CAM_E2")
        self.assertEqual(det.status, PluginStatus.READY)

    def test_human_detection_and_roi_offset(self):
        # ROI-local box (5,5)-(35,45) with forced crop offset (70,70)
        # -> full-frame (75,75,30x40).
        det = _detector_with_fake_model(
            [_FakeResult(xyxy=[[5, 5, 35, 45]], cls=[0], conf=[0.9])])
        res = det.process(_make_request())
        self.assertEqual(len(res.detections), 1)
        d = res.detections[0]
        self.assertEqual(d.detected_class, DetectedClass.HUMAN)
        self.assertTrue(res.has_human and res.has_target)
        self.assertEqual(
            (d.bounding_box.x, d.bounding_box.y,
             d.bounding_box.width, d.bounding_box.height), (75, 75, 30, 40))

    def test_filter_non_targets(self):
        results = [_FakeResult(xyxy=[[0, 0, 10, 10], [20, 20, 40, 40]],
                               cls=[0, 16], conf=[0.9, 0.95])]
        det = _detector_with_fake_model(results, filter_non_targets=True)
        res = det.process(_make_request(with_region=False))
        self.assertEqual(len(res.detections), 1)
        self.assertEqual(res.detections[0].detected_class, DetectedClass.HUMAN)

    def test_keep_non_targets_when_disabled(self):
        results = [_FakeResult(xyxy=[[0, 0, 10, 10]], cls=[16], conf=[0.95])]
        det = _detector_with_fake_model(results, filter_non_targets=False)
        res = det.process(_make_request(with_region=False))
        self.assertEqual(len(res.detections), 1)
        self.assertEqual(res.detections[0].detected_class, DetectedClass.NON_TARGET)
        self.assertFalse(res.has_target)

    def test_inference_exception_returns_error_result(self):
        det = _detector_with_fake_model([_FakeResult([], [], [])])
        det._model.predict.side_effect = RuntimeError("boom")
        res = det.process(_make_request(with_region=False))
        self.assertEqual(res.detections, [])
        self.assertIn("boom", res.metadata.get("error", ""))


class TestCoreToEngine2Wiring(unittest.TestCase):
    def test_orchestrator_routes_request_object_to_engine2(self):
        received = []

        class FakeE1:
            def process(self, packet):
                return MotionEvent(
                    event_type=MotionEventType.MOTION_DETECTED,
                    source_id=packet.source_id, timestamp=packet.timestamp,
                    frame_id=packet.frame_id, motion_score=0.2,
                    motion_region=MotionRegion(x=1, y=1, width=10, height=10),
                    selected_frame_or_reference=packet.frame)

        orch = CoreOrchestrator(engine1_plugin=FakeE1(),  # type: ignore[arg-type]
                                engine2_handler=received.append)
        pkt = FramePacket.create(source_id="C", timestamp=1.0,
                                 frame=np.zeros((20, 20, 3), dtype=np.uint8),
                                 sequence_number=0)
        req = orch.process_frame_packet(pkt)
        self.assertIsNotNone(req)
        self.assertIsInstance(req, ClassificationRequest)
        self.assertEqual(len(received), 1)
        self.assertIs(received[0], req)


if __name__ == "__main__":
    unittest.main()

