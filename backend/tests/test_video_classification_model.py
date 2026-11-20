import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

from app.services.video_classification_model import (
    CNNLSTM,
    IMAGE_MEAN,
    IMAGE_STD,
    VIDEO_CLASS_NAMES,
    VideoClassificationModel,
    VideoModelUnavailableError,
    decode_video_sequence,
    merge_temporal_window_events,
)


BACKEND_DIR = Path(__file__).resolve().parents[1]


class FakeCapture:
    def __init__(self, _path, frames=4, fps=2):
        self.position = 0
        self.released = False
        self.frames = frames
        self.fps = fps

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.frames
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0

    def set(self, _prop, value):
        self.position = int(value)
        return True

    def read(self):
        # Constant BGR pixels make color conversion and normalization testable.
        return True, np.full((12, 16, 3), (10, 20, 30), dtype=np.uint8)

    def release(self):
        self.released = True


class FixedOutput(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def forward(self, _video):
        return self.values


class VideoClassificationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = BACKEND_DIR / "uploads" / f"test_classifier_{uuid.uuid4().hex}"
        cls.fixture_dir.mkdir(parents=True)
        cls.checkpoint_path = cls.fixture_dir / "fixture.pth"
        cls.class_names_path = cls.fixture_dir / "class_names.json"
        fixture_model = CNNLSTM(len(VIDEO_CLASS_NAMES))
        torch.save({
            "model_state_dict": fixture_model.state_dict(),
            "classes": list(VIDEO_CLASS_NAMES),
            "num_frames": 16,
            "img_size": 112,
        }, cls.checkpoint_path)
        cls.class_names_path.write_text(json.dumps(list(VIDEO_CLASS_NAMES)), encoding="utf-8")
        del fixture_model

    @classmethod
    def tearDownClass(cls):
        VideoClassificationModel._cache.clear()
        shutil.rmtree(cls.fixture_dir)

    def setUp(self):
        VideoClassificationModel._cache.clear()

    def make_model(self, checkpoint=None, class_names=None):
        return VideoClassificationModel(
            checkpoint or self.checkpoint_path,
            class_names or self.class_names_path,
        )

    def test_checkpoint_strictly_loads_and_class_order_matches(self):
        model = self.make_model()
        self.assertEqual(model.class_names, ("Assault", "Robbery", "Shooting", "Normal"))
        self.assertEqual((model.sequence_length, model.image_size), (16, 112))
        self.assertFalse(model.model.training)

    def test_missing_checkpoint_and_class_mapping_are_reported(self):
        with self.assertRaisesRegex(VideoModelUnavailableError, "checkpoint is missing"):
            self.make_model(checkpoint=self.fixture_dir / "missing.pth")
        with self.assertRaisesRegex(VideoModelUnavailableError, "class mapping is missing"):
            self.make_model(class_names=self.fixture_dir / "missing.json")

    def test_class_mapping_mismatch_is_rejected(self):
        wrong_names = self.fixture_dir / "wrong_names.json"
        wrong_names.write_text(json.dumps(list(reversed(VIDEO_CLASS_NAMES))), encoding="utf-8")
        with self.assertRaisesRegex(VideoModelUnavailableError, "class order"):
            self.make_model(class_names=wrong_names)

    def test_training_preprocessing_samples_16_rgb_normalized_frames_at_112(self):
        capture = FakeCapture("unused")
        with patch("app.services.video_classification_model.cv2.VideoCapture", return_value=capture):
            sequence = decode_video_sequence("unused", sequence_length=16, image_size=112)
        self.assertEqual(sequence.shape, (16, 3, 112, 112))
        self.assertEqual(sequence.dtype, np.float32)
        self.assertTrue(np.isfinite(sequence).all())
        expected_rgb = (np.asarray((30, 20, 10), dtype=np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
        np.testing.assert_allclose(sequence[0, :, 0, 0], expected_rgb, rtol=1e-6, atol=1e-6)
        self.assertTrue(capture.released)

    def test_classifier_sampling_is_duration_bounded_and_reports_coverage(self):
        capture = FakeCapture("unused", frames=21, fps=2)
        with patch("app.services.video_classification_model.cv2.VideoCapture", return_value=capture):
            sequence, coverage = decode_video_sequence(
                "unused", sequence_length=16, image_size=112,
                max_duration_sec=5, return_metadata=True,
            )
        self.assertEqual(sequence.shape, (16, 3, 112, 112))
        self.assertEqual(coverage["first_sample_frame"], 0)
        self.assertEqual(coverage["last_sample_frame"], 10)
        self.assertEqual(coverage["last_sample_timestamp_sec"], 5.0)
        self.assertFalse(coverage["covered_entire_video"])

    def test_classifier_invalid_fps_keeps_timestamps_unavailable(self):
        capture = FakeCapture("unused", frames=4, fps=0)
        with patch("app.services.video_classification_model.cv2.VideoCapture", return_value=capture):
            _, coverage = decode_video_sequence(
                "unused", sequence_length=4, image_size=112,
                max_duration_sec=5, return_metadata=True,
            )
        self.assertFalse(coverage["timestamps_available"])
        self.assertIsNone(coverage["duration_sec"])
        self.assertIsNone(coverage["first_sample_timestamp_sec"])

    def test_inference_maps_output_to_checkpoint_class_order(self):
        classifier = self.make_model()
        classifier.model = FixedOutput(torch.tensor([[0.0, 5.0, 1.0, -1.0]]))
        with patch("app.services.video_classification_model.decode_video_sequence",
                   return_value=(np.zeros((16, 3, 112, 112), dtype=np.float32), {"sampled_frame_count": 16})):
            result = classifier.predict_path("unused")
        self.assertEqual(result["label"], "Robbery")
        self.assertEqual(list(result["probabilities"]), list(VIDEO_CLASS_NAMES))
        self.assertIn("accuracy has not been independently validated", result["classifier_note"])
        self.assertEqual(result["sampling_coverage"]["sampled_frame_count"], 16)

    def test_temporal_candidates_follow_window_class_even_when_primary_disagrees(self):
        class SequentialCapture(FakeCapture):
            def read(self):
                if self.position >= self.frames:
                    return False, None
                self.position += 1
                return True, np.full((12, 16, 3), (10, 20, 30), dtype=np.uint8)

        classifier = self.make_model()
        classifier.model = FixedOutput(torch.tensor([[5.0, 0.0, 0.0, -1.0]]))
        with patch("app.services.video_classification_model.cv2.VideoCapture",
                   side_effect=lambda _path: SequentialCapture("unused", frames=40, fps=1)):
            assault = classifier.predict_windows_path("unused", confidence_threshold=.1, primary_category="Assault")
            robbery = classifier.predict_windows_path("unused", confidence_threshold=.1, primary_category="Robbery")
        self.assertTrue(assault["activity_events"])
        self.assertTrue(all(event["category"] == "Assault" for event in assault["activity_events"]))
        self.assertTrue(robbery["activity_events"])
        self.assertTrue(all(event["category"] == "Assault" for event in robbery["activity_events"]))

    def test_shooting_temporal_event_preserves_each_window_as_unverified_candidate(self):
        class SequentialCapture(FakeCapture):
            def read(self):
                if self.position >= self.frames:
                    return False, None
                self.position += 1
                return True, np.full((12, 16, 3), (10, 20, 30), dtype=np.uint8)

        classifier = self.make_model()
        classifier.model = FixedOutput(torch.tensor([[0.0, 0.0, 5.0, -1.0]]))
        with patch("app.services.video_classification_model.cv2.VideoCapture",
                   side_effect=lambda _path: SequentialCapture("unused", frames=40, fps=1)):
            result = classifier.predict_windows_path("unused", confidence_threshold=.1,
                                                     primary_category="Shooting")
        self.assertTrue(result["activity_events"])
        for event in result["activity_events"]:
            self.assertEqual(event["category"], "Shooting")
            self.assertEqual(event["verification_status"], "unverified_candidate")
            self.assertLess(event["start_timestamp_sec"], event["end_timestamp_sec"])
            self.assertTrue(event["window_predictions"])
            for window in event["window_predictions"]:
                self.assertEqual(window["label"], "Shooting")
                self.assertGreater(window["confidence"], .1)
                self.assertIn("Shooting", window["probabilities"])

    def test_final_window_covers_40_2_and_82_867_second_videos(self):
        class SequentialCapture(FakeCapture):
            def read(self):
                if self.position >= self.frames:
                    return False, None
                self.position += 1
                return True, np.zeros((4, 4, 3), dtype=np.uint8)

        classifier = self.make_model()
        classifier.model = FixedOutput(torch.tensor([[5.0, 0.0, 0.0, -1.0]]))
        for frame_count, expected_duration, expected_start in (
                (1206, 40.2, 32.0), (2486, 82.8666666667, 72.0)):
            with self.subTest(duration=expected_duration), patch(
                    "app.services.video_classification_model.cv2.VideoCapture",
                    side_effect=lambda _path, count=frame_count: SequentialCapture("unused", count, 30)):
                result = classifier.predict_windows_path("unused", confidence_threshold=.1)
            coverage = result["sampling_coverage"]
            final_window = result["temporal_windows"][-1]
            self.assertAlmostEqual(final_window["start_timestamp_sec"], expected_start, places=3)
            self.assertAlmostEqual(final_window["end_timestamp_sec"], expected_duration, places=3)
            self.assertAlmostEqual(coverage["processed_duration_sec"], expected_duration, places=3)
            self.assertTrue(coverage["covered_entire_video"])

    def test_event_merging_overlaps_short_long_gaps_and_intervening_classes(self):
        def row(label, start, end):
            return {"label": label, "start_timestamp_sec": start, "end_timestamp_sec": end,
                    "confidence": .9, "probabilities": {label: .9}}

        merged = merge_temporal_window_events(
            [row("Assault", 0, 10), row("Assault", 8, 18), row("Assault", 19, 20)],
            VIDEO_CLASS_NAMES, .5, 1.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["end_timestamp_sec"], 20)
        split = merge_temporal_window_events(
            [row("Robbery", 0, 2), row("Robbery", 2.5, 3), row("Robbery", 8, 9)],
            VIDEO_CLASS_NAMES, .5, 1.0)
        self.assertEqual([(item["start_timestamp_sec"], item["end_timestamp_sec"]) for item in split],
                         [(0, 3), (8, 9)])
        intervening = merge_temporal_window_events(
            [row("Shooting", 0, 4), row("Robbery", 3, 5), row("Shooting", 4, 7)],
            VIDEO_CLASS_NAMES, .5, 10.0)
        self.assertEqual([item["category"] for item in intervening], ["Shooting", "Robbery", "Shooting"])

    def test_non_finite_or_wrong_shape_inference_is_rejected(self):
        classifier = self.make_model()
        with patch("app.services.video_classification_model.decode_video_sequence",
                   return_value=(np.zeros((16, 3, 112, 112), dtype=np.float32), {"sampled_frame_count": 16})):
            classifier.model = FixedOutput(torch.full((1, 4), float("nan")))
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                classifier.predict_path("unused")
            classifier.model = FixedOutput(torch.zeros((1, 3)))
            with self.assertRaisesRegex(RuntimeError, "unexpected class-score shape"):
                classifier.predict_path("unused")


if __name__ == "__main__":
    unittest.main()
