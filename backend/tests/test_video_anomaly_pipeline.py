import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from flask import Flask
from flask_jwt_extended import JWTManager

from training.video_anomaly import ConvAutoencoder, binary_metrics, group_anomaly_events, validate_split_manifest
from app.services.video_anomaly_service import VideoAnomalyScorer
from app.routes.video import bp as video_blueprint


class VideoAnomalyPipelineTests(unittest.TestCase):
    def test_autoencoder_reconstructs_same_shape(self):
        model = ConvAutoencoder().eval()
        frames = torch.rand(2, 3, 64, 64)
        with torch.inference_mode():
            reconstruction = model(frames)
        self.assertEqual(tuple(reconstruction.shape), tuple(frames.shape))
        self.assertTrue(torch.isfinite(reconstruction).all())

    def test_binary_metrics_distinguish_positive_class_and_ties(self):
        result = binary_metrics([False, False, True, True], [False, True, True, True], [0.1, 0.5, 0.5, 0.9])
        self.assertEqual(result["positive_class"], "anomalous")
        self.assertEqual(result["confusion_matrix"], {"tp": 2, "tn": 1, "fp": 1, "fn": 0})
        self.assertAlmostEqual(result["roc_auc"], 0.875)
        tied = binary_metrics([False, True], [False, False], [0.5, 0.5])
        self.assertEqual(tied["roc_auc"], 0.5)
        self.assertEqual(tied["pr_auc_average_precision"], 0.5)

    def test_single_ground_truth_class_has_no_auc(self):
        result = binary_metrics([False, False], [False, True], [0.1, 0.9])
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["pr_auc_average_precision"])

    def test_anomaly_events_merge_by_time_and_include_only_observed_outputs(self):
        frames = [
            {"frame_number": 3, "timestamp_sec": 1.0, "potential_anomaly": True,
             "anomaly_score": .2, "anomaly_threshold": .1, "evidence_image": "frame3.jpg", "detections": []},
            {"frame_number": 6, "timestamp_sec": 2.0, "potential_anomaly": True,
             "anomaly_score": .3, "anomaly_threshold": .1, "evidence_image": "frame6.jpg",
             "detections": [{"class": "gun", "confidence": .7, "bbox": [1, 2, 3, 4], "frame_number": 6}]},
            {"frame_number": 30, "timestamp_sec": 10.0, "potential_anomaly": True,
             "anomaly_score": .4, "anomaly_threshold": .1, "evidence_image": "frame30.jpg", "detections": []},
        ]
        events = group_anomaly_events(frames, max_gap_sec=2.0)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event_category"], "combined_event")
        self.assertEqual(events[0]["detected_classes"], ["gun"])
        self.assertEqual(events[0]["anomaly_score"], .3)
        self.assertIn("not a confirmed crime", events[0]["description"])
        self.assertEqual(events[1]["event_category"], "potential_anomaly")

    def test_manifest_requires_video_group_isolation(self):
        root = Path(__file__).resolve().parents[2]
        video = "samples/00_10_frame.mp4"
        manifest = {
            "class_names": ["Assault", "Robbery", "Shooting", "Normal"],
            "splits": {
                "train": [{"video": video, "class": "Normal", "source_group": "train"}],
                "validation": [{"video": video, "class": "Normal", "source_group": "validation"}],
                "test": [{"video": video, "class": "Assault", "source_group": "test"}],
            },
        }
        splits = validate_split_manifest(root, manifest)
        self.assertEqual([len(splits[key]) for key in ("train", "validation", "test")], [1, 1, 1])
        manifest["splits"]["test"][0]["source_group"] = "train"
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_split_manifest(root, manifest)

    def test_scorer_loads_calibrated_checkpoint_and_scores_rgb_preprocessed_frame(self):
        checkpoint = {"architecture": "conv_autoencoder_v1", "state_dict": ConvAutoencoder().state_dict(),
                      "image_size": 64, "input_range": [0.0, 1.0], "color_order": "RGB"}
        config_payload = {"status": "trained_thresholds_calibrated_test_not_evaluated",
                          "frame_anomaly_threshold": .01, "video_anomaly_threshold": .02}
        with patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value=json.dumps(config_payload)), \
             patch.object(torch, "load", return_value=checkpoint):
            scorer = VideoAnomalyScorer("model.pth", "video_anomaly_config.json")
        result = scorer.score_frame(np.zeros((48, 64, 3), dtype=np.uint8))
        self.assertGreaterEqual(result["anomaly_score"], 0)
        self.assertEqual(result["anomaly_threshold"], .01)
        self.assertEqual(result["score_type"], "frame_mean_squared_reconstruction_error")

    def test_removed_evaluation_page_endpoint_is_not_registered(self):
        app = Flask(__name__)
        app.config.update(JWT_SECRET_KEY="test")
        JWTManager(app)
        app.register_blueprint(video_blueprint, url_prefix="/api/video")
        response = app.test_client().get("/api/video/evaluation")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
