import json
import sys
import shutil
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FIXTURE_ROOT = PROJECT_ROOT / "backend" / "tests" / "fixture_work"

import torch
from flask import Flask
from flask_jwt_extended import JWTManager
from unittest.mock import patch

from app.services.video_classification_model import (
    CNNLSTM,
    VIDEO_CLASS_NAMES,
    VideoClassificationModel,
    VideoModelUnavailableError,
)
from train_video_classifier import (
    classification_metrics,
    collect_grouped_videos,
    make_group_splits,
)
from tools.prepare_yolo_source_split import collect_splits, make_assignment, source_id, summarize_source_overlap
from app.routes.video import bp as video_blueprint


class VideoTrainingToolsTests(unittest.TestCase):
    def fixture_dir(self, name):
        path = FIXTURE_ROOT / name
        path.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, path)
        return path

    def test_group_split_keeps_x264_variants_together(self):
        root = self.fixture_dir("video_groups")
        for class_name, prefix in (("Assault", "Assault"), ("Robbery", "Robbery"),
                                   ("Shooting", "Shooting"), ("Normal", "Normal_Videos")):
            folder = root / class_name
            folder.mkdir()
            for i in range(1, 5):
                (folder / f"{prefix}{i:03d}_x264.mp4").write_text(f"{class_name}-{i}-A", encoding="utf-8")
                if class_name in {"Assault", "Shooting"}:
                    (folder / f"{prefix}{i:03d}_x264A.mp4").write_text(f"{class_name}-{i}-B", encoding="utf-8")
        rows = collect_grouped_videos(root)
        splits = make_group_splits(rows, seed=17)
        source_splits = {}
        for split, values in splits.items():
            for row in values:
                old = source_splits.setdefault(row["source_group"], split)
                self.assertEqual(old, split)

    def test_missing_video_dataset_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            collect_grouped_videos(FIXTURE_ROOT / "no_such_video_dataset_for_test")

    def test_identical_video_bytes_with_conflicting_classes_are_rejected(self):
        root = self.fixture_dir("conflicting_video_labels")
        for class_name in ("Assault", "Robbery"):
            folder = root / class_name
            folder.mkdir()
            for suffix in ("001", "002", "003"):
                (folder / f"{class_name}{suffix}_x264.mp4").write_bytes(b"same-content")
        # Add enough per-class groups so collection reaches exact duplicate checking.
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            collect_grouped_videos(root)

    def test_metrics_include_classwise_scores_and_confusion(self):
        result = classification_metrics([0, 0, 1, 2], [0, 1, 1, 2])
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0, 0], [0, 1, 0, 0],
                                                       [0, 0, 1, 0], [0, 0, 0, 0]])
        self.assertIn("precision", result["per_class"]["Assault"])
        self.assertIn("recall", result["per_class"]["Shooting"])
        self.assertIn("f1", result["per_class"]["Normal"])

    def test_checkpoint_loader_accepts_inference_contract(self):
        VideoClassificationModel._cache.clear()
        root = self.fixture_dir("checkpoint_load")
        model = CNNLSTM(len(VIDEO_CLASS_NAMES))
        (root / "candidate.pth").touch()
        (root / "class_names.json").write_text(json.dumps(list(VIDEO_CLASS_NAMES)), encoding="utf-8")
        checkpoint = {"model_state_dict": model.state_dict(), "classes": list(VIDEO_CLASS_NAMES),
                      "num_frames": 16, "img_size": 112}
        with patch("app.services.video_classification_model.torch.load", return_value=checkpoint):
            loaded = VideoClassificationModel(root / "candidate.pth", root / "class_names.json")
        self.assertEqual(loaded.sequence_length, 16)
        self.assertEqual(loaded.image_size, 112)
        self.assertEqual(loaded.class_names, VIDEO_CLASS_NAMES)
        VideoClassificationModel._cache.clear()

    def test_missing_checkpoint_raises_specific_error(self):
        VideoClassificationModel._cache.clear()
        with self.assertRaises(VideoModelUnavailableError):
            VideoClassificationModel(FIXTURE_ROOT / "missing.pth", FIXTURE_ROOT / "classes.json")
        VideoClassificationModel._cache.clear()

    def test_api_rejects_video_paths_outside_upload_directory(self):
        app = Flask(__name__)
        app.config.update(TESTING=True, JWT_SECRET_KEY="unit-test", UPLOAD_FOLDER=str(FIXTURE_ROOT / "uploads"), VIDEO_MAX_SAMPLED_FRAMES=12)
        app.db = None
        JWTManager(app)
        app.register_blueprint(video_blueprint, url_prefix="/api/video")
        FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
        video_path = FIXTURE_ROOT / "missing_classifier_api_input.mp4"
        video_path.touch()
        self.addCleanup(video_path.unlink, missing_ok=True)
        response = app.test_client().post("/api/video/analyze", json={"filepath": str(video_path)})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["code"], "invalid_video_path")


class YoloSourceAuditTests(unittest.TestCase):
    def test_video_frames_collapse_to_one_source_id(self):
        self.assertEqual(source_id(Path("68_mp4-0_jpg.rf.hash.jpg")), "68_video")
        self.assertEqual(source_id(Path("68_mp4-9_jpg.rf.other.jpg")), "68_video")

    def test_audit_reports_existing_source_overlap(self):
        rows = {
            "src_a": [{"split": "train", "counts": {1: 2}}, {"split": "test", "counts": {7: 1}}],
            "src_b": [{"split": "validation", "counts": {1: 1}}],
        }
        result = summarize_source_overlap(rows)
        self.assertEqual(result["overlap"]["train/test"]["count"], 1)
        self.assertEqual(result["overlap"]["train/test"]["examples"], ["src_a"])
        self.assertEqual(result["class_instances"]["train"][1], 2)

    def test_group_assignment_is_deterministic_and_70_15_15(self):
        rows = {f"source_{i}": [{"counts": {1: int(i % 3 == 0), 7: int(i % 5 == 0)}}]
                for i in range(20)}
        first = make_assignment(rows, seed=29)
        second = make_assignment(rows, seed=29)
        self.assertEqual(first, second)
        self.assertEqual({split: list(first.values()).count(split) for split in ("train", "validation", "test")},
                         {"train": 14, "validation": 3, "test": 3})

    def test_missing_yolo_label_is_reported(self):
        root = FIXTURE_ROOT / "missing_yolo_label"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root)
        images = root / "train" / "images"
        images.mkdir(parents=True)
        (images / "sample.jpg").write_bytes(b"fixture")
        config = {"path": str(root), "train": "train/images", "names": {0: "hammer"}}
        with self.assertRaises(FileNotFoundError):
            collect_splits(config, root)


if __name__ == "__main__":
    unittest.main()
