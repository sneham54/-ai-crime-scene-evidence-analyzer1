import os
import unittest
from datetime import timedelta
from io import BytesIO
from unittest.mock import patch
from unittest.mock import MagicMock

import cv2
import numpy as np

from app import create_app
from app.services.hybrid_detection_service import HybridDetectionService
from flask_jwt_extended import create_access_token


class HybridDetectionTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config.update(TESTING=True, UPLOAD_FOLDER=os.path.dirname(__file__),
                               CNN_IMAGE_SIZE=224, CNN_MEAN=(.485, .456, .406),
                               CNN_STD=(.229, .224, .225), CNN_PREPROCESSING_SOURCE="test")
        self.image_path = __file__
        with self.app.app_context():
            self.token = create_access_token(identity="isolated-hybrid-test-user")
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.app.db = None
        self.image_patches = [
            patch("app.services.hybrid_detection_service.cv2.imread", return_value=np.zeros((120, 160, 3), dtype=np.uint8)),
            patch("app.services.hybrid_detection_service.cv2.imwrite", return_value=True),
            patch("app.services.hybrid_detection_service.Path.mkdir"),
        ]
        for mocked in self.image_patches:
            mocked.start()

    def tearDown(self):
        for mocked in self.image_patches:
            mocked.stop()
        HybridDetectionService._model = None
        HybridDetectionService._classes = None
        HybridDetectionService._load_error = None

    def test_unsupported_yolo_class_preserved_without_cnn_call(self):
        detection = {"class_id": 2, "class": "helmet", "class_name": "helmet",
                     "confidence": .8, "bbox": [10, 12, 40, 50], "category": "Others"}
        with self.app.app_context(), \
             patch("app.services.hybrid_detection_service.YOLOService") as yolo_cls, \
             patch.object(HybridDetectionService, "load_model"), \
             patch.object(HybridDetectionService, "_classify") as classify:
            yolo_cls.return_value.detect.return_value = ([detection], "")
            yolo_cls.return_value.model_info = {"class_id_names": [(2, "helmet")]}
            HybridDetectionService._model = object()
            got = HybridDetectionService.analyze(self.image_path)
        classify.assert_not_called()
        self.assertEqual(got["detections"][0]["bbox"], detection["bbox"])
        self.assertEqual(got["detections"][0]["classification"], "unsupported")

    def test_cnn_mapping_is_by_class_name_and_yolo_coordinates_preserved(self):
        detection = {"class_id": 0, "class": "gun", "class_name": "gun",
                     "confidence": .8, "bbox": [10, 12, 40, 50], "category": "Weapons"}
        with self.app.app_context(), \
             patch("app.services.hybrid_detection_service.YOLOService") as yolo_cls, \
             patch.object(HybridDetectionService, "load_model"), \
             patch.object(HybridDetectionService, "_classify", return_value=("rifle", .92)):
            yolo_cls.return_value.detect.return_value = ([detection], "")
            # CNN index 0 is rifle, while the YOLO id for rifle is 7.
            yolo_cls.return_value.model_info = {"class_id_names": [(0, "gun"), (7, "rifle")]}
            HybridDetectionService._model = object()
            HybridDetectionService._classes = ["rifle", "gun", "knife", "person", "pistol"]
            got = HybridDetectionService.analyze(self.image_path)
        result = got["detections"][0]
        self.assertEqual(result["mapped_yolo_class_id"], 7)
        self.assertEqual(result["bbox"], detection["bbox"])
        self.assertEqual(result["yolo_class_id"], 0)

    def test_no_detections_returns_empty_lists(self):
        with self.app.app_context(), patch("app.services.hybrid_detection_service.YOLOService") as yolo_cls, \
             patch.object(HybridDetectionService, "load_model"):
            yolo_cls.return_value.detect.return_value = ([], "")
            HybridDetectionService._model = None
            HybridDetectionService._load_error = "checkpoint missing"
            got = HybridDetectionService.analyze(self.image_path)
        self.assertEqual(got["detections"], [])
        self.assertEqual(got["yolo_detections"], [])

    def test_invalid_image_is_rejected_by_analysis_route(self):
        with self.app.test_client() as client:
            response = client.post("/api/detection/analyze", json={"filepath": __file__}, headers=self.headers)
        self.assertEqual(response.status_code, 400)

    def test_analysis_route_returns_hybrid_and_yolo_comparison(self):
        detections = [{"class": "rifle", "class_name": "rifle", "confidence": .9,
                       "bbox": [1, 2, 30, 40], "category": "Weapons"}]
        hybrid_result = {"detections": detections, "yolo_detections": detections,
                         "annotated_path": os.path.join(os.path.dirname(__file__), "annotated.jpg"),
                         "cnn_available": True, "cnn_error": None,
                         "preprocessing": {"image_size": 224, "mean": [.485, .456, .406],
                                           "std": [.229, .224, .225], "source": "test"}}
        opened_image = MagicMock()
        opened_image.__enter__.return_value = opened_image
        self.app.db = None
        with patch("app.routes.detection.Image.open", return_value=opened_image), \
             patch("app.routes.detection.YOLOService") as yolo_cls, \
             patch("app.routes.detection.HybridDetectionService.analyze", return_value=hybrid_result):
            yolo_cls.return_value.model_info = {"path": "best.pt", "classes": []}
            with self.app.test_client() as client:
                response = client.post("/api/detection/analyze", json={"filepath": __file__, "compare": True}, headers=self.headers)
        payload = response.get_json()
        self.assertEqual(response.status_code, 200, payload)
        self.assertEqual(payload["pipeline"], "hybrid")
        self.assertEqual(payload["yolo_detections"], detections)
        self.assertEqual(payload["detections"][0]["bbox"], [1, 2, 30, 40])

    def test_image_routes_reject_missing_token(self):
        with self.app.test_client() as client:
            upload = client.post("/api/upload/image", data={})
            analysis = client.post("/api/detection/analyze", json={"filepath": __file__})
        self.assertEqual(upload.status_code, 401)
        self.assertEqual(analysis.status_code, 401)
        self.assertEqual(analysis.get_json()["code"], "authentication_required")

    def test_expired_token_has_clear_401_response(self):
        with self.app.app_context():
            expired = create_access_token(identity="demo-investigator", expires_delta=timedelta(seconds=-1))
        with self.app.test_client() as client:
            response = client.post("/api/detection/analyze", json={"filepath": __file__},
                                   headers={"Authorization": f"Bearer {expired}"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json()["code"], "token_expired")

    def test_authenticated_image_upload_accepts_file_field(self):
        with patch("app.routes.upload.save_uploaded_file", return_value=(__file__, "scene.jpg")):
            with self.app.test_client() as client:
                response = client.post("/api/upload/image", data={
                    "file": (BytesIO(b"mock-image-bytes"), "scene.jpg")
                }, headers=self.headers, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["original_name"], "scene.jpg")


if __name__ == "__main__":
    unittest.main()
