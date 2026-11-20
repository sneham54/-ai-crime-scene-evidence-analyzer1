import unittest
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from flask import Flask

from app.services.yolo_service import YOLOService
from app.config import Config


class FakeBox:
    cls = np.array([8])
    conf = np.array([0.48])
    xyxy = np.array([[10.0, 20.0, 50.0, 80.0]])


class FakeResult:
    boxes = [FakeBox()]

    @staticmethod
    def plot(line_width=2):
        return np.zeros((100, 100, 3), dtype=np.uint8)


class FakeUltralyticsModel:
    names = {
        0: "gun", 1: "hammer", 2: "helmet", 3: "knife", 4: "mask",
        5: "person", 6: "pistol", 7: "rifle", 8: "weapon",
    }

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.calls = []

    def __call__(self, frame, **kwargs):
        self.calls.append((frame, kwargs))
        return [FakeResult()]

    def track(self, frame, **kwargs):
        self.track_call = (frame, kwargs)
        box = SimpleNamespace(cls=np.array([3]), conf=np.array([.73]), xyxy=np.array([[2.0, 3.0, 30.0, 40.0]]))
        boxes = type("Boxes", (list,), {})([box])
        boxes.id = np.array([17])
        return [SimpleNamespace(boxes=boxes)]


class YOLOVideoCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config.update({
            "YOLO_MODEL_PATH": "configured-candidate.pt",
            "YOLO_CONFIDENCE": 0.25,
            "YOLO_IMAGE_SIZE": 832,
            "VIDEO_TRACKER": "bytetrack.yaml",
            "VIDEO_CONFIDENCE_THRESHOLD": 0.2,
            "VIDEO_PERSON_ROI_ENABLED": False,
        })
        YOLOService._model = None
        YOLOService._model_path_used = None
        YOLOService._load_error = None

    def tearDown(self):
        YOLOService._model = None
        YOLOService._model_path_used = None
        YOLOService._load_error = None

    def test_video_detector_loads_nine_class_checkpoint_and_preserves_classes(self):
        model = FakeUltralyticsModel("backend/models/best.pt")
        loader = YOLOService.__new__(YOLOService)
        with self.app.app_context(), \
             patch("ultralytics.YOLO", return_value=model) as yolo_loader, \
             patch("os.path.isfile", return_value=True):
            loader._load_model()
            detections, annotated = loader.detect_frame(np.zeros((100, 100, 3), dtype=np.uint8))

        yolo_loader.assert_called_once()
        self.assertTrue(YOLOService._model_path_used.endswith(str(Path("backend") / "models" / "best.pt")))
        self.assertEqual(model.calls[0][1]["conf"], 0.25)
        self.assertEqual(model.calls[0][1]["imgsz"], 832)
        self.assertEqual(detections[0]["class_id"], 8)
        self.assertEqual(detections[0]["class"], "weapon")
        self.assertEqual(detections[0]["class_name"], "weapon")
        self.assertEqual(detections[0]["confidence"], 0.48)
        self.assertEqual(detections[0]["bbox"], [10.0, 20.0, 50.0, 80.0])
        self.assertEqual(detections[0]["category"], "Weapons")
        self.assertEqual(annotated.shape, (100, 100, 3))

    def test_tracking_uses_bytetrack_and_emits_checkpoint_class_and_identity(self):
        model = FakeUltralyticsModel("best.pt")
        YOLOService._model = model
        service = YOLOService.__new__(YOLOService)
        with self.app.app_context():
            detections, annotated = service.track_video_frame(
                np.zeros((100, 100, 3), dtype=np.uint8), 2.5, 75
            )
        options = model.track_call[1]
        self.assertTrue(options["persist"])
        self.assertEqual(options["tracker"], "bytetrack.yaml")
        self.assertNotIn("classes", options)
        self.assertEqual(detections[0]["class_id"], 3)
        self.assertEqual(detections[0]["class"], "knife")
        self.assertEqual(detections[0]["track_id"], 17)
        self.assertEqual(detections[0]["timestamp_sec"], 2.5)
        self.assertEqual(detections[0]["frame_number"], 75)
        self.assertEqual(annotated.shape, (100, 100, 3))
        with self.app.app_context():
            untimed_detections, _ = service.track_video_frame(
                np.zeros((100, 100, 3), dtype=np.uint8), None, 76
            )
        self.assertIsNone(untimed_detections[0]["timestamp_sec"])

    def test_configurable_class_confidence_threshold_suppresses_below_cutoff(self):
        from app.services.yolo_service import _class_confidence_threshold
        with self.app.app_context():
            # Default thresholds
            self.assertEqual(_class_confidence_threshold(0), 0.30)
            self.assertEqual(_class_confidence_threshold(1), 0.25)
            self.assertEqual(_class_confidence_threshold(2), 0.25)
            self.assertEqual(_class_confidence_threshold(3), 0.25)
            self.assertEqual(_class_confidence_threshold(4), 0.25)
            self.assertEqual(_class_confidence_threshold(5), 0.25)
            self.assertEqual(_class_confidence_threshold(6), 0.25)
            self.assertEqual(_class_confidence_threshold(7), 0.25)
            self.assertEqual(_class_confidence_threshold(8), 0.25)

            # Override threshold via config
            self.app.config["VIDEO_CLASS_CONFIDENCE_GUN"] = 0.50
            self.assertEqual(_class_confidence_threshold(0), 0.50)

            # Override via dictionary mapping
            self.app.config["VIDEO_CLASS_CONFIDENCE_THRESHOLDS"] = {0: 0.60, 3: 0.45}
            self.assertEqual(_class_confidence_threshold(0), 0.60)
            self.assertEqual(_class_confidence_threshold(3), 0.45)

    def test_person_roi_configuration_defaults_off_and_reads_environment(self):
        self.assertFalse(Config.VIDEO_PERSON_ROI_ENABLED)
        import app.config as config_module
        old_value = os.environ.get("VIDEO_PERSON_ROI_ENABLED")
        try:
            with patch.dict(os.environ, {"VIDEO_PERSON_ROI_ENABLED": "true"}):
                config_module = importlib.reload(config_module)
                self.assertTrue(config_module.Config.VIDEO_PERSON_ROI_ENABLED)
        finally:
            if old_value is None:
                os.environ.pop("VIDEO_PERSON_ROI_ENABLED", None)
            else:
                os.environ["VIDEO_PERSON_ROI_ENABLED"] = old_value
            importlib.reload(config_module)

    def test_person_roi_disabled_by_default_avoids_injected_boundary_weapons(self):
        class PersonBox:
            cls = np.array([5])  # person
            conf = np.array([0.70])
            xyxy = np.array([[260.0, 110.0, 320.0, 210.0]])

        class PersonResult:
            boxes = [PersonBox()]
            @staticmethod
            def plot(line_width=2):
                return np.zeros((240, 320, 3), dtype=np.uint8)

        class PersonModel(FakeUltralyticsModel):
            def __call__(self, frame, **kwargs):
                return [PersonResult()]

        model = PersonModel("best.pt")
        YOLOService._model = model
        service = YOLOService.__new__(YOLOService)
        with self.app.app_context():
            self.assertFalse(self.app.config.get("VIDEO_PERSON_ROI_ENABLED", False))
            detections, _ = service.detect_frame(np.zeros((240, 320, 3), dtype=np.uint8))
            # Only the person detection should be returned, no hallucinated ROI weapon
            self.assertEqual(len(detections), 1)
            self.assertEqual(detections[0]["class"], "person")
            self.assertEqual(detections[0]["class_id"], 5)

    def test_person_roi_rejects_edge_boxes_and_caps_resolution_to_crop(self):
        class PersonBox:
            cls = np.array([5])
            conf = np.array([0.70])
            xyxy = np.array([[50.0, 30.0, 150.0, 180.0]])

        class PersonResult:
            boxes = [PersonBox()]
            @staticmethod
            def plot(line_width=2):
                return np.zeros((240, 320, 3), dtype=np.uint8)

        class RecordingModel(FakeUltralyticsModel):
            def __call__(self, frame, **kwargs):
                self.calls.append((frame, kwargs))
                return [PersonResult()]

        model = RecordingModel("best.pt")
        YOLOService._model = model
        service = YOLOService.__new__(YOLOService)
        import app.config as config_module
        with patch.dict(os.environ, {"VIDEO_PERSON_ROI_ENABLED": "true"}):
            config_module = importlib.reload(config_module)
            self.app.config.update({
                "VIDEO_PERSON_ROI_ENABLED": config_module.Config.VIDEO_PERSON_ROI_ENABLED,
                "VIDEO_ROI_IMAGE_SIZE": 1280,
            })
            with self.app.app_context():
                service.detect_frame(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertEqual(len(model.calls), 2)
        crop, options = model.calls[1]
        self.assertGreaterEqual(crop.shape[0], 40)
        self.assertGreaterEqual(crop.shape[1], 40)
        self.assertLessEqual(options["imgsz"], 640)
        self.assertLessEqual(options["imgsz"], 2 * max(crop.shape[:2]))

        class EdgeBox(PersonBox):
            xyxy = np.array([[0.0, 30.0, 150.0, 180.0]])
        class EdgeResult(PersonResult):
            boxes = [EdgeBox()]
        class EdgeModel(RecordingModel):
            def __call__(self, frame, **kwargs):
                self.calls.append((frame, kwargs))
                return [EdgeResult()]
        edge_model = EdgeModel("best.pt")
        YOLOService._model = edge_model
        with self.app.app_context():
            edge_detections, _ = service.detect_frame(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertEqual(len(edge_model.calls), 1)
        self.assertEqual([item["class"] for item in edge_detections], ["person"])


if __name__ == "__main__":
    unittest.main()
