import tempfile
import unittest
from pathlib import Path
import shutil
import uuid
from unittest.mock import patch

import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights
from flask import Flask
from flask_jwt_extended import JWTManager

from app.routes.video import bp as video_blueprint
from app.services.bigru_mil_model import SEGMENT_COUNT
from app.services.r3d18_feature_service import (
    InvalidExperimentalVideoError,
    R3D18FeatureService,
    R3D18UnavailableError,
)


class FakeCapture:
    def __init__(self, _path):
        self.frame = np.zeros((24, 32, 3), dtype=np.uint8)
        self.position = 0

    def isOpened(self):
        return True

    def get(self, _prop):
        # CAP_PROP_FRAME_COUNT=7 and CAP_PROP_FPS=5 in OpenCV.
        return 320 if _prop == 7 else 32.0

    def set(self, _prop, position):
        self.position = position
        return True

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass


class FakeR3D(torch.nn.Module):
    def forward(self, _clip):
        return torch.ones((1, 512), dtype=torch.float32)


class R3D18FeatureServiceTests(unittest.TestCase):
    def test_segment_timestamp_mapping_uses_actual_duration_midpoints(self):
        timestamps = R3D18FeatureService.map_segment_timestamps(96.0)
        self.assertEqual(len(timestamps), 32)
        self.assertAlmostEqual(timestamps[0], 1.5)
        self.assertAlmostEqual(timestamps[-1], 94.5)
        self.assertTrue(all(a < b for a, b in zip(timestamps, timestamps[1:])))

    def test_official_kinetics_transform_produces_r3d_clip_layout(self):
        transform = R3D_18_Weights.KINETICS400_V1.transforms()
        clip = transform(torch.zeros((16, 3, 112, 112), dtype=torch.uint8))
        self.assertEqual(tuple(clip.shape), (3, 16, 112, 112))
        self.assertTrue(torch.isfinite(clip).all())

    def test_video_duration_limit_is_enforced(self):
        with patch("app.services.r3d18_feature_service.cv2.VideoCapture", FakeCapture):
            with self.assertRaisesRegex(InvalidExperimentalVideoError, "exceeds"):
                R3D18FeatureService.inspect_video("unused.mp4", 5)

    def test_extract_returns_finite_32_by_512_features(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            video.write(b"video")
            video.flush()
            transform = lambda frames: torch.zeros((3, 16, 112, 112), dtype=torch.float32)
            with patch.object(R3D18FeatureService, "_load_model", return_value=(FakeR3D(), transform)), \
                 patch("app.services.r3d18_feature_service.cv2.VideoCapture", FakeCapture):
                features, timestamps, duration, frame_count, fps = R3D18FeatureService().extract(video.name, 20)
        self.assertEqual(features.shape, (32, 512))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(len(timestamps), 32)
        self.assertEqual((duration, frame_count, fps), (10.0, 320, 32.0))

    def test_invalid_duration_and_empty_frames_are_rejected(self):
        with self.assertRaises(InvalidExperimentalVideoError):
            R3D18FeatureService.map_segment_timestamps(0)
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            video.write(b"video")
            video.flush()
            with patch.object(R3D18FeatureService, "inspect_video", return_value=(320, 32.0, 10.0)), \
                 patch.object(R3D18FeatureService, "_load_model", return_value=(FakeR3D(), lambda frames: frames)), \
                 patch("app.services.r3d18_feature_service.cv2.VideoCapture", EmptyCapture):
                with self.assertRaisesRegex(InvalidExperimentalVideoError, "No frames"):
                    R3D18FeatureService().extract(video.name, 20)


class EmptyCapture(FakeCapture):
    def read(self):
        return False, None


class ExperimentalVideoRouteTests(unittest.TestCase):
    def setUp(self):
        self.uploads = Path(__file__).resolve().parents[1] / "uploads" / f"test_r3d18_{uuid.uuid4().hex}"
        videos = self.uploads / "videos"
        videos.mkdir(parents=True)
        self.video = videos / "cctv.mp4"
        self.video.write_bytes(b"fixture")
        app = Flask(__name__)
        app.config.update(TESTING=True, JWT_SECRET_KEY="test", UPLOAD_FOLDER=str(self.uploads),
                          R3D18_MAX_DURATION_SEC=120, VIDEO_MAX_DURATION_SEC=120,
                          ALLOWED_VIDEO_EXTENSIONS={"mp4"}, ANOMALY_MODEL_PATH="unused")
        app.db = None
        JWTManager(app)
        app.register_blueprint(video_blueprint, url_prefix="/api/video")
        self.client = app.test_client()

    def tearDown(self):
        shutil.rmtree(self.uploads)

    def test_rejects_path_outside_configured_video_upload_directory(self):
        response = self.client.post("/api/video/analyze-experimental", json={"filepath": str(Path(__file__).resolve())})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "invalid_video_path")

    def test_returns_raw_logits_and_actual_duration_segment_mapping(self):
        features = np.zeros((32, 512), dtype=np.float32)
        timestamps = R3D18FeatureService.map_segment_timestamps(64)
        with patch("app.routes.video.R3D18FeatureService.extract", return_value=(features, timestamps, 64.0, 640, 10.0)), \
             patch("app.routes.video.BigruMILModel.get") as get_model:
            get_model.return_value.infer.return_value = [float(index) for index in range(32)]
            response = self.client.post("/api/video/analyze-experimental", json={"filepath": str(self.video)})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(payload["score_type"], "raw_logit_not_probability")
        self.assertEqual(payload["feature_shape"], [32, 512])
        self.assertEqual(payload["segment_scores"][0]["raw_logit"], 0.0)
        self.assertEqual(payload["segment_scores"][0]["timestamp_sec"], 1.0)
        self.assertIn("not an event boundary", payload["segment_mapping"])

    def test_reports_pretrained_weights_and_inference_errors(self):
        with patch("app.routes.video.R3D18FeatureService.extract", side_effect=R3D18UnavailableError("weights unavailable")):
            unavailable = self.client.post("/api/video/analyze-experimental", json={"filepath": str(self.video)})
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(unavailable.get_json()["code"], "r3d18_unavailable")
        with patch("app.routes.video.R3D18FeatureService.extract", side_effect=RuntimeError("model error")):
            failed = self.client.post("/api/video/analyze-experimental", json={"filepath": str(self.video)})
        self.assertEqual(failed.status_code, 500)
        self.assertEqual(failed.get_json()["code"], "experimental_video_inference_failed")


if __name__ == "__main__":
    unittest.main()
