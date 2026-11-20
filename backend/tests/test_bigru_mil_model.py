import unittest
from pathlib import Path

import numpy as np
import torch
from flask import Flask
from flask_jwt_extended import JWTManager
from torch import nn

from app.routes.video import bp as video_blueprint
from app.services.bigru_mil_model import BigruMILModel, ExperimentalModelUnavailableError


BACKEND_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT = BACKEND_DIR / "models" / "bigru_mil_best.pt"


class BigruMILModelTests(unittest.TestCase):
    def setUp(self):
        self.model = BigruMILModel(CHECKPOINT)

    def test_checkpoint_safely_loads_and_returns_32_finite_raw_logits(self):
        scores = self.model.infer(np.zeros((32, 512), dtype=np.float32))
        self.assertEqual(len(scores), 32)
        self.assertTrue(np.isfinite(scores).all())

    def test_invalid_feature_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"shape \(32, 512\)"):
            self.model.infer(np.zeros((31, 512), dtype=np.float32))

    def test_non_numeric_and_non_finite_features_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "numeric values"):
            self.model.infer([["x"] * 512 for _ in range(32)])
        features = np.zeros((32, 512), dtype=np.float32)
        features[4, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model.infer(features)

    def test_non_finite_model_outputs_are_rejected(self):
        class NonFiniteModel(nn.Module):
            def forward(self, _features):
                return torch.full((1, 32), float("nan"))

        self.model.model = NonFiniteModel()
        with self.assertRaisesRegex(RuntimeError, "non-finite raw logits"):
            self.model.infer(np.zeros((32, 512), dtype=np.float32))

    def test_missing_checkpoint_is_reported(self):
        with self.assertRaises(ExperimentalModelUnavailableError):
            BigruMILModel(CHECKPOINT.with_name("missing_bigru_checkpoint.pt"))


class BigruMILRouteTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.config.update(
            TESTING=True,
            JWT_SECRET_KEY="experimental-test-key",
            ANOMALY_MODEL_PATH=str(CHECKPOINT),
        )
        app.db = None
        JWTManager(app)
        app.register_blueprint(video_blueprint, url_prefix="/api/video")
        self.client = app.test_client()

    def test_endpoint_returns_unverified_raw_segment_logits_and_optional_timestamps(self):
        features = np.zeros((32, 512), dtype=np.float32).tolist()
        times = [index * 0.5 for index in range(32)]
        response = self.client.post("/api/video/analyze-features", json={
            "features": features,
            "timestamps_sec": times,
        })
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(len(payload["segment_scores"]), 32)
        self.assertTrue(all(item["timestamp_sec"] is not None for item in payload["segment_scores"]))
        self.assertEqual(payload["score_type"], "raw_logit_not_probability")
        self.assertEqual(payload["verification_status"], "unverified")
        self.assertTrue(payload["requires_human_review"])

    def test_endpoint_rejects_bad_shape_and_timestamps(self):
        bad_shape = self.client.post("/api/video/analyze-features", json={"features": [[0.0] * 512]})
        self.assertEqual(bad_shape.status_code, 400)
        bad_times = self.client.post("/api/video/analyze-features", json={
            "features": np.zeros((32, 512), dtype=np.float32).tolist(),
            "timestamps_sec": [float("nan")] * 32,
        })
        self.assertEqual(bad_times.status_code, 400)

    def test_endpoint_leaves_timestamps_empty_when_they_are_not_supplied(self):
        response = self.client.post("/api/video/analyze-features", json={
            "features": np.zeros((32, 512), dtype=np.float32).tolist(),
        })
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(all(item["timestamp_sec"] is None for item in payload["segment_scores"]))


if __name__ == "__main__":
    unittest.main()
