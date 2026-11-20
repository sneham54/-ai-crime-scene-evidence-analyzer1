import os
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import BACKEND_DIR, PROJECT_ROOT, _resolve_video_model_path
from app.services.video_classification_model import VIDEO_CLASS_NAMES, VideoClassificationModel


class VideoModelSelectionTests(unittest.TestCase):
    def tearDown(self):
        VideoClassificationModel._cache.clear()

    def test_default_is_original_and_relative_paths_select_both_checkpoints(self):
        original = BACKEND_DIR / "models" / "video_classifier.pth"
        candidate = BACKEND_DIR / "models" / "video_classifier_v2.pth"
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Path(_resolve_video_model_path()), original.resolve())
        with patch.dict(os.environ, {"VIDEO_MODEL_PATH": "models/video_classifier.pth"}, clear=True):
            self.assertEqual(Path(_resolve_video_model_path()), original.resolve())
        with patch.dict(os.environ, {"VIDEO_MODEL_PATH": "backend/models/video_classifier_v2.pth"}, clear=True):
            self.assertEqual(Path(_resolve_video_model_path()), candidate.resolve())
        with patch.dict(os.environ, {"VIDEO_MODEL_PATH": "models/video_classifier_v2.pth"}, clear=True):
            self.assertEqual(Path(_resolve_video_model_path()), candidate.resolve())

    def test_missing_explicit_candidate_does_not_silently_fall_back(self):
        missing = PROJECT_ROOT / "backend" / "models" / "missing-video-model.pth"
        with patch.dict(os.environ, {"VIDEO_MODEL_PATH": "backend/models/missing-video-model.pth"}, clear=True):
            self.assertEqual(Path(_resolve_video_model_path()), missing.resolve())

    def test_original_and_candidate_checkpoints_strictly_load_in_cnn_lstm(self):
        class_names_path = BACKEND_DIR / "models" / "class_names.json"
        for filename in ("video_classifier.pth", "video_classifier_v2.pth"):
            with self.subTest(checkpoint=filename):
                model = VideoClassificationModel(BACKEND_DIR / "models" / filename, class_names_path)
                self.assertEqual(model.class_names, VIDEO_CLASS_NAMES)
                self.assertEqual(model.sequence_length, 16)
                self.assertEqual(model.image_size, 112)
                self.assertFalse(model.model.training)
                VideoClassificationModel._cache.clear()


if __name__ == "__main__":
    unittest.main()
