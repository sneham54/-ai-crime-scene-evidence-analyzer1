import unittest
import csv
import shutil
import uuid
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from flask import Flask
from flask_jwt_extended import JWTManager

from app.services import video_service
from app.services.video_service import (VideoService, group_evidence_events, _classify_video,
                                        _probe_source_timing, _source_frame_timing)
from app.services.video_classification_model import VideoModelUnavailableError
from app.routes.video import bp as video_blueprint
from app.routes import video as video_route


class FakeCapture:
    def __init__(self, frames=8, fps=4, seek_supported=True):
        self.frames, self.fps = frames, fps
        self.seek_supported = seek_supported
        self.position = 0
        self.released = False

    def isOpened(self): return True
    def get(self, prop):
        if prop == video_service.cv2.CAP_PROP_FRAME_COUNT: return self.frames
        if prop == video_service.cv2.CAP_PROP_FPS: return self.fps
        if prop == video_service.cv2.CAP_PROP_FRAME_WIDTH: return 30
        if prop == video_service.cv2.CAP_PROP_FRAME_HEIGHT: return 20
        return 0
    def set(self, _prop, value):
        if not self.seek_supported: return False
        self.position = int(value)
        return True
    def read(self):
        if self.position >= self.frames: return False, None
        self.position += 1
        return True, np.zeros((20, 30, 3), dtype=np.uint8)
    def release(self): self.released = True


class FakeClassifier:
    def __init__(self, *args): pass
    def predict_path(self, _path, max_duration_sec=None):
        return {"label": "Normal", "confidence": .9, "probabilities": {"Normal": .9},
                "sampling_coverage": {"sampled_frame_count": 16, "first_sample_timestamp_sec": 0.0,
                                      "last_sample_timestamp_sec": 1.75, "covered_entire_video": True,
                                      "max_duration_sec": max_duration_sec}}


class MissingClassifier:
    def __init__(self, *_args):
        raise VideoModelUnavailableError("BiGRU+MIL checkpoint is not configured")


class BrokenClassifier(FakeClassifier):
    def predict_path(self, _path, max_duration_sec=None):
        raise RuntimeError("classifier inference failed")


class FakeYolo:
    def detect_frame(self, _frame):
        return ([{"class": "knife", "confidence": .8, "bbox": [1, 1, 8, 8]}], None)


class TrackingYolo(FakeYolo):
    def __init__(self):
        self.session_count = 0
        self.track_calls = []

    from contextlib import contextmanager
    @contextmanager
    def video_tracking_session(self):
        self.session_count += 1
        yield

    def track_video_frame(self, frame, timestamp, frame_number):
        self.track_calls.append((timestamp, frame_number))
        return ([{"class_id": 3, "class": "knife", "class_name": "knife", "confidence": .8,
                  "bbox": [1, 1, 8, 8], "track_id": 42, "tracking_status": "tracked"}], frame)


class VideoServiceTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config["JWT_SECRET_KEY"] = "video-service-test"
        JWTManager(self.app)
        self.app.register_blueprint(video_blueprint, url_prefix="/api/video")
        self.temp_uploads = Path(__file__).resolve().parents[1] / "uploads" / f"test_artifacts_{uuid.uuid4().hex}"
        self.temp_uploads.mkdir(parents=True)
        self.app.config.update({
            "VIDEO_MODEL_PATH": "classifier.pth", "CLASS_NAMES_PATH": "classes.json",
            "VIDEO_MIN_CONFIDENCE": 0, "VIDEO_MAX_DURATION_SEC": 600,
            "VIDEO_MAX_PROCESSED_FRAMES": 100, "VIDEO_SAMPLE_INTERVAL": 2,
            "VIDEO_SAMPLE_FPS": 1, "VIDEO_EVENT_MAX_GAP_SEC": 3,
            "VIDEO_EVENT_MIN_IOU": .1, "VIDEO_EVENT_MIN_OBSERVATIONS": 1,
            "UPLOAD_FOLDER": str(self.temp_uploads),
        })
        self.app.db = None

    def tearDown(self):
        shutil.rmtree(self.temp_uploads)

    def test_source_pts_are_converted_with_time_base_when_frame_aligned(self):
        source = self.temp_uploads / "pts-source.mp4"
        source.write_bytes(b"fixture")
        payload = {"streams": [{"time_base": "1/1000", "avg_frame_rate": "25/1",
                                "duration": "0.120", "nb_frames": "3"}],
                   "frames": [{"best_effort_timestamp": "0"},
                              {"best_effort_timestamp": "40"},
                              {"best_effort_timestamp": "80"}]}
        with patch("app.services.video_timing.subprocess.run",
                          return_value=SimpleNamespace(stdout=json.dumps(payload))) as probe:
            timing = _probe_source_timing(source, 3, 25)
        probe.assert_called_once()
        self.assertTrue(timing["presentation_timestamps_available"])
        self.assertTrue(timing["constant_frame_rate_verified"])
        self.assertEqual(timing["timestamp_mapping_status"], "verified_cfr_frame_index_fps")
        self.assertEqual(_source_frame_timing(2, 25, timing), {
            "timestamp_sec": .08, "presentation_timestamp": 80, "time_base": "1/1000",
            "timestamp_mapping_status": "verified_cfr_frame_index_fps",
            "timestamp_basis": "zero_based_frame_index_divided_by_source_fps",
            "timestamps_verified": True,
        })

    def test_source_pts_count_mismatch_is_explicitly_unverified(self):
        source = self.temp_uploads / "misaligned-source.mp4"
        source.write_bytes(b"fixture")
        payload = {"streams": [{"time_base": "1/1000", "avg_frame_rate": "25/1",
                                "duration": "0.080", "nb_frames": "2"}],
                   "frames": [{"best_effort_timestamp": "0"},
                              {"best_effort_timestamp": "40"}]}
        with patch("app.services.video_timing.subprocess.run",
                          return_value=SimpleNamespace(stdout=json.dumps(payload))):
            timing = _probe_source_timing(source, 3, 25)
        self.assertFalse(timing["presentation_timestamps_available"])
        self.assertEqual(timing["timestamp_mapping_status"], "unverified_frame_index_fps_fallback")
        fallback = _source_frame_timing(2, 25, timing)
        self.assertEqual(fallback["timestamp_sec"], .08)
        self.assertIsNone(fallback["presentation_timestamp"])
        self.assertEqual(fallback["timestamp_mapping_status"], "unverified_frame_index_fps_fallback")

    def test_aligned_variable_rate_pts_remain_authoritative(self):
        source = self.temp_uploads / "vfr-source.mp4"
        source.write_bytes(b"fixture")
        payload = {"streams": [{"time_base": "1/1000", "avg_frame_rate": "25/1",
                                "duration": "0.120", "nb_frames": "3"}],
                   "frames": [{"best_effort_timestamp": "0"},
                              {"best_effort_timestamp": "40"},
                              {"best_effort_timestamp": "100"}]}
        with patch("app.services.video_timing.subprocess.run",
                          return_value=SimpleNamespace(stdout=json.dumps(payload))):
            timing = _probe_source_timing(source, 3, 25)
        self.assertTrue(timing["presentation_timestamps_available"])
        self.assertFalse(timing["constant_frame_rate_verified"])
        self.assertEqual(_source_frame_timing(2, 25, timing)["timestamp_sec"], .1)
        self.assertEqual(_source_frame_timing(2, 25, timing)["timestamp_mapping_status"],
                         "verified_vfr_presentation_timestamps")

    def analyze(self, cap, detector=None, full_video=False):
        with self.app.app_context(), \
             patch.object(video_service.cv2, "VideoCapture", return_value=cap), \
             patch.object(video_service, "VideoClassificationModel", FakeClassifier), \
             patch.object(video_service, "YOLOService", return_value=detector or FakeYolo()):
            return VideoService().analyze("clip.mp4", full_video=full_video)

    def test_full_video_request_does_not_shortcut_to_saved_tracking_csv(self):
        cap = FakeCapture(frames=8, fps=4)
        catalog = SimpleNamespace(match_tracking_artifact=lambda *_args, **_kwargs: {
            "status": "EXACT_MATCH", "match_status": "EXACT_MATCH",
            "record_count": 1, "records": [], "metadata_verification": {},
        })
        with self.app.app_context(), \
             patch.object(video_service.cv2, "VideoCapture", return_value=cap), \
             patch.object(video_service, "VideoClassificationModel", FakeClassifier), \
             patch.object(video_service, "YOLOService", return_value=FakeYolo()), \
             patch.object(video_service, "EvidenceCatalogService", return_value=catalog), \
             patch.object(VideoService, "_analyze_saved_tracking", side_effect=AssertionError("saved CSV shortcut used")):
            result = VideoService().analyze(
                "clip.mp4", full_video=True, category="assault",
                original_filename="clip.mp4", create_annotated_video=False,
            )
        self.assertEqual(result["video_info"]["frames_analyzed"], 8)
        self.assertEqual(result["video_info"]["inference_coverage_percent"], 100.0)
        self.assertEqual(result["video_info"]["analysis_mode"], "full_video")
        self.assertEqual(result["tracking_data_match"]["status"], "EXACT_MATCH")
        self.assertFalse(result["tracking_data_match"]["artifact_reused"])
        self.assertEqual(result["tracking_data_match"]["reuse_skipped_reason"],
                         "full_video_requested_requires_sequential_inference")

    def test_whole_video_prediction_is_primary_and_window_disagreement_is_provisional(self):
        class DualClassifier:
            def predict_path(self, _path, max_duration_sec=None):
                return {"label": "Assault", "top_class": "Assault", "confidence": .72,
                        "probabilities": {"Assault": .72, "Robbery": .18, "Shooting": .06, "Normal": .04},
                        "sampling_coverage": {"sampled_frame_count": 16}}

            def predict_windows_path(self, _path, max_duration_sec=None, confidence_threshold=.55,
                                     primary_category=None, merge_gap_sec=1.0):
                self.primary_category = primary_category
                return {"label": "Robbery", "top_class": "Robbery", "confidence": .8,
                        "probabilities": {"Assault": .1, "Robbery": .8, "Shooting": .05, "Normal": .05},
                        "classification_mode": "window", "sampling_coverage": {},
                        "temporal_windows": [], "activity_events": []}

        classifier = DualClassifier()
        output = _classify_video(classifier, "clip.mp4", 600, .55)
        self.assertEqual(output["label"], "Unresolved")
        self.assertIsNone(output["confidence"])
        self.assertEqual(output["whole_video_prediction"]["label"], "Assault")
        self.assertEqual(classifier.primary_category, "Assault")
        self.assertEqual(output["classification_status"], "uncertain_prediction_disagreement")
        self.assertEqual(output["alternative_predictions"][0]["label"], "Robbery")

    def test_sampling_metadata_timestamps_and_events(self):
        cap = FakeCapture()
        result = self.analyze(cap)
        self.assertEqual([f["frame_number"] for f in result["frame_results"]], [0, 2, 4, 6])
        self.assertEqual([f["timestamp_sec"] for f in result["frame_results"]], [0, .5, 1, 1.5])
        self.assertEqual(result["video_info"]["sampling_interval_frames"], 2)
        self.assertEqual(result["video_info"]["analysis_mode"], "sampled")
        self.assertFalse(result["video_info"]["analysis_covered_entire_video"])
        self.assertEqual(result["video_info"]["analysis_coverage"], "portion")
        self.assertEqual(result["video_info"]["inference_coverage_percent"], 50.0)
        self.assertEqual(result["video_info"]["processing_status"], "completed_partial_coverage")
        self.assertEqual(result["video_info"]["first_analyzed_timestamp_sec"], 0)
        self.assertEqual(result["video_info"]["last_analyzed_timestamp_sec"], 1.5)
        self.assertTrue(result["video_info"]["classifier_sampling_coverage"]["covered_entire_video"])
        self.assertEqual(len(result["evidence_events"]), 1)
        self.assertEqual(result["evidence_events"][0]["observation_count"], 4)
        self.assertEqual(result["evidence_events"][0]["supporting_frame_numbers"], [0, 2, 4, 6])
        self.assertEqual(result["evidence_events"][0]["review_status"], "candidate")
        self.assertEqual(result["candidate_intervals"], [])
        self.assertEqual(result["candidate_interval_status"], "not_evaluated")
        self.assertEqual(result["verification_status"], "unverified")
        self.assertTrue((self.temp_uploads / "evidence" / result["analysis_id"] / "evidence_timeline.csv").is_file())
        self.assertTrue((self.temp_uploads / "evidence" / result["analysis_id"] / "evidence_events.csv").is_file())
        self.assertTrue((self.temp_uploads / "evidence" / result["analysis_id"] / "tracking_summary.csv").is_file())
        self.assertTrue(result["frame_results"][0]["evidence_image"].endswith("frame_000000000.jpg"))
        self.assertIn("analysis_json", result)
        self.assertIn("video_info", result)
        self.assertIn("classification", result)
        self.assertIn("frame_results", result)
        self.assertIn("detections", result)
        self.assertIn("evidence_events", result)
        analysis_dir = self.temp_uploads / "evidence" / result["analysis_id"]
        record_path = analysis_dir / "analysis.json"
        self.assertTrue(record_path.is_file())
        serialized = record_path.read_text(encoding="utf-8")
        record = json.loads(serialized, parse_constant=lambda value: self.fail(f"invalid JSON constant: {value}"))
        self.assertNotIn(str(self.temp_uploads), serialized)
        self.assertNotIn("C:\\", serialized)
        self.assertNotIn("classifier_note", record["classification"])
        self.assertEqual(record["classification"]["probabilities"]["Normal"], .9)
        self.assertEqual(record["frames"][0]["detections"][0]["bbox"], [1, 1, 8, 8])
        self.assertEqual(record["frames"][0]["detections"][0]["review_status"], "Needs review")
        self.assertTrue((analysis_dir / record["frames"][0]["original_evidence_frame"]).is_file())
        self.assertEqual(record["grouped_events"][0]["review_status"], "candidate")
        self.assertTrue(record["grouped_events"][0]["representative_original_evidence_frame"].startswith("original_frame_"))
        self.assertTrue(Path(self.temp_uploads / "evidence" / result["analysis_id"] / record["grouped_events"][0]["representative_original_evidence_frame"]).is_file())
        self.assertTrue(cap.released)

    def test_explicit_full_video_mode_analyzes_every_source_frame(self):
        result = self.analyze(FakeCapture(frames=8, fps=4), full_video=True)
        self.assertEqual([frame["frame_number"] for frame in result["frame_results"]], list(range(8)))
        self.assertEqual(result["video_info"]["processed_frames"], 8)
        self.assertEqual(result["video_info"]["inference_coverage_percent"], 100.0)
        self.assertTrue(result["video_info"]["analysis_covered_entire_video"])
        self.assertEqual(result["video_info"]["analysis_mode"], "full_video")
        self.assertEqual(result["video_info"]["sampling_interval_frames"], 1)
        self.assertEqual(result["candidate_interval_status"], "not_evaluated")
        self.assertIn("whole-video", result["candidate_interval_limitation"])

    def test_full_video_mode_reports_partial_coverage_at_duration_limit(self):
        with self.app.app_context():
            self.app.config["VIDEO_MAX_DURATION_SEC"] = 1
        result = self.analyze(FakeCapture(frames=12, fps=4), full_video=True)
        self.assertEqual(result["video_info"]["processed_frames"], 4)
        self.assertEqual(result["video_info"]["inference_coverage_percent"], 33.33)
        self.assertTrue(result["video_info"]["duration_limit_reached"])
        self.assertFalse(result["video_info"]["analysis_covered_entire_video"])
        self.assertEqual(result["video_info"]["processing_status"], "completed_partial_coverage")

    def test_byte_track_continuity_preserves_class_id_and_video_timestamps(self):
        detector = TrackingYolo()
        result = self.analyze(FakeCapture(), detector)
        self.assertEqual(detector.session_count, 1)
        self.assertEqual(detector.track_calls, [(0.0, 0), (.5, 2), (1.0, 4), (1.5, 6)])
        self.assertEqual([item["track_id"] for item in result["detections"]], [42] * 4)
        self.assertEqual([item["class_id"] for item in result["detections"]], [3] * 4)
        self.assertEqual(result["timeline"][0]["timestamp_sec"], 0)
        self.assertEqual(result["timeline"][0]["track_id"], 42)
        self.assertEqual(result["tracking"]["method"], "ByteTrack")
        self.assertEqual(result["tracking"]["tracked_detection_count"], 4)
        self.assertEqual(result["tracking"]["untracked_detection_count"], 0)
        self.assertEqual(result["source_video_path"], str(Path("clip.mp4").resolve()))
        self.assertTrue(result["artifact_paths"]["detection_csv"].endswith("evidence_timeline.csv"))
        analysis_dir = self.temp_uploads / "evidence" / result["analysis_id"]
        record = json.loads((analysis_dir / "analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(record["frames"][0]["detections"][0]["class_id"], 3)
        self.assertEqual(record["frames"][0]["detections"][0]["track_id"], 42)
        self.assertIn("timestamp_basis", record["source_metadata"])
        self.assertFalse(record["source_metadata"]["presentation_timestamps_available"])
        self.assertIn("source_integrity", record)
        self.assertEqual(record["tracking"]["tracked_detection_count"], 4)
        with (analysis_dir / "evidence_timeline.csv").open(encoding="utf-8-sig") as csv_file:
            csv_rows = list(csv.DictReader(csv_file))
        self.assertEqual(csv_rows[0]["track_id"], "42")
        self.assertEqual(csv_rows[0]["bbox_xyxy"], "[1, 1, 8, 8]")

    def test_video_analysis_does_not_produce_scene_change_results(self):
        class CutCapture(FakeCapture):
            def read(self):
                if self.position >= self.frames:
                    return False, None
                frame_number = self.position
                self.position += 1
                pattern = np.indices((20, 30)).sum(axis=0) % 2
                if frame_number >= 4:
                    pattern = 1 - pattern
                frame = np.repeat((pattern * 255).astype(np.uint8)[:, :, None], 3, axis=2)
                return True, frame

        result = self.analyze(CutCapture())
        self.assertNotIn("scene_changes", result)
        record = json.loads((self.temp_uploads / "evidence" / result["analysis_id"] / "analysis.json").read_text(encoding="utf-8"))
        self.assertNotIn("scene_changes", record)

        class LightingChangeCapture(FakeCapture):
            def read(self):
                if self.position >= self.frames:
                    return False, None
                frame_number = self.position
                self.position += 1
                level = 0 if frame_number < 4 else 40
                return True, np.full((20, 30, 3), level, dtype=np.uint8)

        lighting_result = self.analyze(LightingChangeCapture())
        self.assertNotIn("scene_changes", lighting_result)

    def test_frame_sampling_falls_back_to_sequential_decode_when_seek_fails(self):
        result = self.analyze(FakeCapture(seek_supported=False))
        self.assertEqual([frame["frame_number"] for frame in result["frame_results"]], [0, 2, 4, 6])
        self.assertEqual([frame["timestamp_sec"] for frame in result["frame_results"]], [0, .5, 1, 1.5])
        self.assertEqual(result["video_info"]["width"], 30)
        self.assertEqual(result["video_info"]["height"], 20)

    def test_video_analysis_uses_fixed_sampling_without_scene_change_processing(self):
        class MotionCapture(FakeCapture):
            def read(self):
                if self.position >= self.frames:
                    return False, None
                frame_number = self.position
                self.position += 1
                pattern = np.indices((20, 30)).sum(axis=0) % 2
                if frame_number == 1:
                    pattern = 1 - pattern
                frame = np.repeat((pattern * 255).astype(np.uint8)[:, :, None], 3, axis=2)
                return True, frame

        result = self.analyze(MotionCapture())
        sampled = [frame["frame_number"] for frame in result["frame_results"]]
        self.assertEqual(sampled, [0, 2, 4, 6])
        self.assertEqual(result["video_info"]["sampling_strategy"], "fixed_interval")
        self.assertNotIn("scene_changes", result)

    def test_timestamps_use_original_frame_index_and_source_fps(self):
        class PositionTimestampCapture(FakeCapture):
            def get(self, prop):
                if prop == video_service.cv2.CAP_PROP_POS_MSEC:
                    frame_index = max(0, self.position - 1)
                    return {0: 0, 2: 580, 4: 1260, 6: 1990}.get(frame_index, 0)
                return super().get(prop)

        result = self.analyze(PositionTimestampCapture())
        self.assertEqual([frame["timestamp_sec"] for frame in result["frame_results"]], [0, .5, 1, 1.5])
        self.assertEqual(result["video_info"]["timestamp_source"], "unverified_frame_index_divided_by_reported_fps")
        self.assertEqual(result["frame_results"][0]["timestamp_mapping_status"], "unverified_frame_index_fps_fallback")
        self.assertEqual(result["source_metadata"]["timestamp_mapping_status"], "unverified_frame_index_fps_fallback")
        self.assertIsNone(result["frame_results"][0]["presentation_timestamp"])
        self.assertTrue(result["video_info"]["timestamps_available"])

    def test_scene_change_comparison_is_not_part_of_video_analysis_service(self):
        self.assertFalse(hasattr(video_service, "_scene_comparison"))

    def test_tracker_failure_falls_back_to_normal_detection(self):
        class BrokenTrackingYolo(TrackingYolo):
            def track_video_frame(self, _frame, _timestamp, _frame_number):
                raise RuntimeError("ByteTrack unavailable")

        cap = FakeCapture()
        result = self.analyze(cap, BrokenTrackingYolo())
        self.assertEqual(result["total_detections"], 4)
        self.assertFalse(result["tracking"]["used"])
        self.assertEqual(result["tracking"]["fallback_reason"], "RuntimeError")
        self.assertTrue(cap.released)

    def test_json_serializer_normalizes_numpy_and_non_finite_values(self):
        record = video_service._json_safe({
            "confidence": np.float32(.75), "bbox": np.asarray([1, 2, 3, 4]),
            "invalid": float("nan"),
        })
        serialized = json.dumps(record, allow_nan=False)
        self.assertEqual(json.loads(serialized), {"confidence": .75, "bbox": [1, 2, 3, 4], "invalid": None})

    def test_missing_evidence_frames_are_serialized_as_unavailable(self):
        with patch.object(video_service.cv2, "imwrite", return_value=False):
            result = self.analyze(FakeCapture())
        record_path = self.temp_uploads / "evidence" / result["analysis_id"] / "analysis.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertIsNone(record["frames"][0]["annotated_evidence_frame"])
        self.assertIsNone(record["frames"][0]["original_evidence_frame"])
        self.assertIsNone(record["grouped_events"][0]["representative_evidence_frame"])
        self.assertIsNone(record["grouped_events"][0]["representative_original_evidence_frame"])
        self.assertIsNone(result["frame_results"][0]["original_evidence_image"])

    def test_evidence_routes_keep_csv_and_annotated_access_and_add_json_original(self):
        analysis_id = uuid.uuid4().hex
        evidence_dir = self.temp_uploads / "evidence" / analysis_id
        evidence_dir.mkdir(parents=True)
        (evidence_dir / "analysis.json").write_text('{"schema_version": 1}', encoding="utf-8")
        (evidence_dir / "evidence_timeline.csv").write_text("frame_number\n0\n", encoding="utf-8")
        (evidence_dir / "evidence_events.csv").write_text("event_id\n", encoding="utf-8")
        (evidence_dir / "frame_000000000.jpg").write_bytes(b"annotated")
        (evidence_dir / "original_frame_000000000.jpg").write_bytes(b"original")
        client = self.app.test_client()
        (evidence_dir / "tracking_summary.csv").write_text("track_id,detection_count\n42,1\n", encoding="utf-8")
        for filename in ("analysis.json", "evidence_timeline.csv", "evidence_events.csv", "tracking_summary.csv", "frame_000000000.jpg", "original_frame_000000000.jpg"):
            response = client.get(f"/api/video/evidence/{analysis_id}/{filename}")
            self.assertEqual(response.status_code, 200, filename)
            response.close()
        missing = client.get(f"/api/video/evidence/{analysis_id}/original_frame_000000001.jpg")
        self.assertEqual(missing.status_code, 404)
        missing.close()

    def test_analyze_api_keeps_existing_fields_and_adds_json_reference(self):
        video_dir = self.temp_uploads / "videos"
        video_dir.mkdir()
        video_path = video_dir / "clip.mp4"
        video_path.write_bytes(b"uploaded-video-placeholder")
        legacy_result = {
            "analysis_id": "a" * 32,
            "video_info": {"duration_sec": 1.0, "total_frames": 4, "fps": 4.0,
                           "frames_analyzed": 1, "processed_frames": 1},
            "classification": {
                "label": "Shooting", "confidence": .9,
                "probabilities": {"Assault": .01, "Robbery": .01, "Shooting": .9, "Normal": .08},
                "temporal_windows": [{"label": "Shooting", "confidence": .88,
                                      "start_timestamp_sec": 0.0, "end_timestamp_sec": 2.0}],
                "activity_events": [{"category": "Shooting", "confidence": .88,
                                      "start_timestamp_sec": 0.0, "end_timestamp_sec": 2.0,
                                      "verification_status": "unverified_candidate",
                                      "window_predictions": [{"label": "Shooting", "confidence": .88,
                                                              "start_timestamp_sec": 0.0,
                                                              "end_timestamp_sec": 2.0,
                                                              "probabilities": {"Shooting": .88}}]}],
            },
            "candidate_intervals": [{"category": "Shooting", "confidence": .88,
                                     "start_timestamp_sec": 0.0, "end_timestamp_sec": 2.0,
                                     "verification_status": "unverified_candidate"}],
            "candidate_interval_status": "unverified_classifier_candidates",
            "frame_results": [{"frame_number": 0, "timestamp_sec": 0.0, "detections": []}],
            "detections": [], "total_detections": 0, "evidence_events": [],
            "evidence_timeline_csv": "/api/video/evidence/" + "a" * 32 + "/evidence_timeline.csv",
            "evidence_events_csv": "/api/video/evidence/" + "a" * 32 + "/evidence_events.csv",
            "analysis_json": "/api/video/evidence/" + "a" * 32 + "/analysis.json",
        }

        analysis_options = []

        class FakeVideoAnalysis:
            def analyze(self, _path, max_frames=None, full_video=False):
                analysis_options.append((max_frames, full_video))
                return legacy_result

        with patch.object(video_route, "VideoService", return_value=FakeVideoAnalysis()):
            response = self.app.test_client().post("/api/video/analyze", json={
                "filepath": str(video_path), "max_frames": 1, "full_video": True,
            })
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        for field in ("video_info", "classification", "frame_results", "detections", "total_detections", "evidence_events", "evidence_timeline_csv", "evidence_events_csv"):
            self.assertIn(field, payload)
        self.assertEqual(payload["analysis_json"], legacy_result["analysis_json"])
        self.assertEqual(payload["candidate_intervals"], legacy_result["candidate_intervals"])
        self.assertEqual(payload["classification"]["activity_events"][0]["verification_status"], "unverified_candidate")
        self.assertEqual(payload["classification"]["activity_events"][0]["window_predictions"][0]["confidence"], .88)
        self.assertEqual(analysis_options, [(None, True)])

    def test_full_video_category_request_enables_annotated_review_video(self):
        video_dir = self.temp_uploads / "videos"
        video_dir.mkdir()
        video_path = video_dir / "clip.mp4"
        video_path.write_bytes(b"uploaded-video-placeholder")
        analysis_dir = self.temp_uploads / "evidence" / ("c" * 32)
        analysis_dir.mkdir(parents=True)
        artifact_path = analysis_dir / "analysis.json"
        artifact_path.write_text('{"schema_version":1}', encoding="utf-8")
        calls = []

        class FakeVideoAnalysis:
            def analyze(self, _path, **kwargs):
                calls.append(kwargs)
                return {"analysis_id": "c" * 32, "video_info": {}, "classification": {},
                        "frame_results": [], "detections": [],
                        "artifact_paths": {"analysis_json": str(artifact_path)}}

        with patch.object(video_route, "VideoService", return_value=FakeVideoAnalysis()):
            response = self.app.test_client().post("/api/video/analyze", json={
                "filepath": str(video_path), "category": "shooting", "full_video": True,
            })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(calls[0]["create_annotated_video"])

    def test_successful_analysis_persists_risk_and_upload_metadata_to_existing_artifact(self):
        video_dir = self.temp_uploads / "videos"
        video_dir.mkdir()
        video_path = video_dir / "stored-upload.mp4"
        video_path.write_bytes(b"uploaded-video-placeholder")
        analysis_id = "b" * 32
        artifact_dir = self.temp_uploads / "evidence" / analysis_id
        artifact_dir.mkdir(parents=True)
        artifact_path = artifact_dir / "analysis.json"
        artifact_path.write_text(json.dumps({
            "schema_version": 1, "analysis_id": analysis_id,
            "classification": {"label": "Assault", "confidence": .9},
            "frames": [{"detections": []}], "tracking": {"used": True},
        }), encoding="utf-8")
        result = {
            "analysis_id": analysis_id,
            "video_info": {"duration_sec": 1.0, "total_frames": 4, "fps": 4.0,
                           "frames_analyzed": 1, "processed_frames": 1},
            "classification": {"label": "Assault", "confidence": .9},
            "frame_results": [{"frame_number": 0, "timestamp_sec": 0.0,
                               "detections": [{"class": "person", "track_id": 8}]}],
            "detections": [{"class": "person", "track_id": 8}],
            "total_detections": 1, "tracking": {"used": True, "method": "ByteTrack"},
            "artifact_paths": {"analysis_json": str(artifact_path)},
        }

        class FakeVideoAnalysis:
            def analyze(self, _path, **_kwargs):
                return result

        with patch.object(video_route, "VideoService", return_value=FakeVideoAnalysis()):
            response = self.app.test_client().post("/api/video/analyze", json={
                "filepath": str(video_path), "category": "assault",
                "original_filename": "Assault001_x264.mp4",
            })

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["persistence"], {"artifact": "saved", "database": "unavailable"})
        saved = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["original_filename"], "Assault001_x264.mp4")
        self.assertEqual(saved["total_detections"], 1)
        self.assertEqual(saved["valid_tracking_ids"], 1)
        self.assertIsNotNone(saved["uploaded_at"])
        self.assertIsNotNone(saved["risk"]["overall_risk"])

    def test_failed_video_analysis_is_persisted_without_classifying_it_as_complete(self):
        video_dir = self.temp_uploads / "videos"
        video_dir.mkdir()
        video_path = video_dir / "failed-upload.mp4"
        video_path.write_bytes(b"invalid-but-uploaded")

        class FailedVideoAnalysis:
            def analyze(self, _path, **_kwargs):
                raise VideoModelUnavailableError("test classifier unavailable")

        with patch.object(video_route, "VideoService", return_value=FailedVideoAnalysis()):
            response = self.app.test_client().post("/api/video/analyze", json={
                "filepath": str(video_path), "original_filename": "Assault001_x264.mp4",
            })

        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertTrue(payload["analysis_id"])
        artifact_path = self.temp_uploads / "evidence" / payload["analysis_id"] / "analysis.json"
        saved = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["failure_code"], "video_classifier_unavailable")
        self.assertEqual(saved["original_filename"], "Assault001_x264.mp4")

    def test_missing_video_checkpoint_still_returns_yolo_evidence(self):
        cap = FakeCapture()
        with self.app.app_context(), \
             patch.object(video_service.cv2, "VideoCapture", return_value=cap), \
             patch.object(video_service, "VideoClassificationModel", MissingClassifier), \
             patch.object(video_service, "YOLOService", return_value=FakeYolo()):
            result = VideoService().analyze("clip.mp4")
        self.assertEqual(result["classification"]["label"], "Unavailable")
        self.assertEqual(result["total_detections"], 4)
        self.assertEqual(result["candidate_interval_status"], "not_evaluated")

    def test_video_classifier_inference_error_does_not_discard_yolo_results(self):
        cap = FakeCapture()
        with self.app.app_context(), \
             patch.object(video_service.cv2, "VideoCapture", return_value=cap), \
             patch.object(video_service, "VideoClassificationModel", BrokenClassifier), \
             patch.object(video_service, "YOLOService", return_value=FakeYolo()):
            result = VideoService().analyze("clip.mp4")
        self.assertEqual(result["classification"]["label"], "Unavailable")
        self.assertIn("classification failed", result["classification"]["classifier_note"])
        self.assertEqual(result["total_detections"], 4)
        self.assertEqual(result["frame_results"][0]["detections"][0]["class"], "knife")
        self.assertTrue(cap.released)

    def test_invalid_fps_keeps_frame_timestamps_unavailable_and_releases(self):
        cap = FakeCapture(fps=0)
        result = self.analyze(cap)
        self.assertFalse(result["video_info"]["timestamps_available"])
        self.assertIsNone(result["frame_results"][0]["timestamp_sec"])
        self.assertIsNone(result["video_info"]["duration_sec"])
        self.assertIsNone(result["video_info"]["first_analyzed_timestamp_sec"])
        self.assertIsNone(result["video_info"]["fps"])
        self.assertTrue(cap.released)

    def test_short_video_sampled_at_stride_two_is_reported_as_partial(self):
        result = self.analyze(FakeCapture(frames=4, fps=2))
        self.assertFalse(result["video_info"]["analysis_covered_entire_video"])
        self.assertEqual(result["video_info"]["processed_frames"], 2)
        self.assertEqual(result["video_info"]["inference_coverage_percent"], 50.0)
        self.assertEqual(result["video_info"]["last_analyzed_timestamp_sec"], 1.0)

    def test_every_frame_inference_is_full_coverage(self):
        self.app.config["VIDEO_SAMPLE_INTERVAL"] = 1
        result = self.analyze(FakeCapture(frames=4, fps=2))
        self.assertTrue(result["video_info"]["analysis_covered_entire_video"])
        self.assertEqual(result["video_info"]["inference_coverage_percent"], 100.0)
        self.assertEqual(result["video_info"]["processing_status"], "completed_full_coverage")

    def test_early_decode_stop_is_interrupted_and_partial(self):
        class TruncatedCapture(FakeCapture):
            def read(self):
                if self.position >= 4:
                    return False, None
                return super().read()

        self.app.config["VIDEO_SAMPLE_INTERVAL"] = 1
        result = self.analyze(TruncatedCapture(frames=8, fps=2))
        info = result["video_info"]
        self.assertEqual(info["processed_frames"], 4)
        self.assertEqual(info["processing_status"], "interrupted")
        self.assertEqual(info["inference_coverage_percent"], 50.0)
        self.assertFalse(info["analysis_covered_entire_video"])

    def test_long_video_frame_cap_reports_partial_coverage(self):
        self.app.config["VIDEO_MAX_PROCESSED_FRAMES"] = 3
        result = self.analyze(FakeCapture(frames=40, fps=2))
        info = result["video_info"]
        self.assertEqual(info["processed_frames"], 3)
        self.assertTrue(info["frame_limit_reached"])
        self.assertFalse(info["analysis_covered_entire_video"])
        self.assertEqual(info["analysis_coverage"], "portion")
        self.assertEqual(info["first_analyzed_timestamp_sec"], 0)
        self.assertEqual(info["last_analyzed_timestamp_sec"], 2.0)

    def test_invalid_frame_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no readable frames"):
            self.analyze(FakeCapture(frames=0, fps=2))

    def test_frame_limit_is_reported_without_claiming_exhaustive_processing(self):
        cap = FakeCapture(frames=20)
        self.app.config["VIDEO_MAX_PROCESSED_FRAMES"] = 2
        result = self.analyze(cap)
        self.assertEqual(result["video_info"]["processed_frames"], 2)
        self.assertTrue(result["video_info"]["frame_limit_reached"])
        self.assertEqual(result["video_info"]["analysis_mode"], "sampled")

    def test_temporal_association_does_not_merge_distant_boxes(self):
        frames = [
            {"frame_number": 0, "timestamp_sec": 0, "detections": [{"class": "knife", "bbox": [0, 0, 10, 10], "confidence": .8}]},
            {"frame_number": 1, "timestamp_sec": 1, "detections": [{"class": "knife", "bbox": [80, 80, 90, 90], "confidence": .7}]},
        ]
        events = group_evidence_events(frames, max_gap_sec=3, min_iou=.1)
        self.assertEqual(len(events), 2)

    def test_repeated_detections_produce_one_event_with_summary_and_evidence(self):
        frames = [
            {"frame_number": 0, "timestamp_sec": 0.0, "frame_width": 100, "frame_height": 100,
             "evidence_image": "/api/video/evidence/a/frame_0.jpg",
             "detections": [{"class": "pistol", "confidence": .7, "bbox": [10, 10, 30, 30]}]},
            {"frame_number": 1, "timestamp_sec": 1.0, "frame_width": 100, "frame_height": 100,
             "evidence_image": "/api/video/evidence/a/frame_1.jpg",
             "detections": [{"class": "pistol", "confidence": .9, "bbox": [11, 10, 31, 30]}]},
            {"frame_number": 2, "timestamp_sec": 2.0, "frame_width": 100, "frame_height": 100,
             "evidence_image": "/api/video/evidence/a/frame_2.jpg",
             "detections": [{"class": "pistol", "confidence": .8, "bbox": [12, 10, 32, 30]}]},
        ]
        events = group_evidence_events(frames)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["observation_count"], 3)
        self.assertEqual(event["start_timestamp_sec"], 0.0)
        self.assertEqual(event["end_timestamp_sec"], 2.0)
        self.assertEqual(event["duration_sec"], 2.0)
        self.assertEqual(event["confidence_summary"], {"min": .7, "mean": .8, "max": .9})
        self.assertEqual(event["representative_frame_number"], 1)
        self.assertTrue(event["representative_evidence_image"].endswith("frame_1.jpg"))
        self.assertEqual(event["incident_status"], "not_assessed")

    def test_gap_limit_joins_nearby_observations_but_splits_long_gap(self):
        frames = [
            {"frame_number": 0, "timestamp_sec": 0.0, "detections": [{"class": "knife", "confidence": .8, "bbox": [0, 0, 10, 10]}]},
            {"frame_number": 1, "timestamp_sec": 2.5, "detections": [{"class": "knife", "confidence": .7, "bbox": [0, 0, 10, 10]}]},
            {"frame_number": 2, "timestamp_sec": 6.0, "detections": [{"class": "knife", "confidence": .9, "bbox": [0, 0, 10, 10]}]},
        ]
        events = group_evidence_events(frames, max_gap_sec=3.0)
        self.assertEqual([event["observation_count"] for event in events], [2, 1])
        self.assertEqual(events[0]["duration_sec"], 2.5)
        self.assertEqual(events[1]["duration_sec"], 0.0)

    def test_two_same_class_objects_are_associated_one_to_one(self):
        frames = [
            {"frame_number": 0, "timestamp_sec": 0.0, "detections": [
                {"class": "person", "confidence": .9, "bbox": [0, 0, 10, 10]},
                {"class": "person", "confidence": .8, "bbox": [30, 30, 40, 40]},
            ]},
            {"frame_number": 1, "timestamp_sec": 1.0, "detections": [
                {"class": "person", "confidence": .85, "bbox": [1, 1, 11, 11]},
                {"class": "person", "confidence": .75, "bbox": [31, 31, 41, 41]},
            ]},
        ]
        events = group_evidence_events(frames)
        self.assertEqual(len(events), 2)
        self.assertEqual([event["observation_count"] for event in events], [2, 2])
        self.assertNotEqual(events[0]["event_id"], events[1]["event_id"])

    def test_single_frame_detection_is_kept_as_unverified_event(self):
        frames = [{
            "frame_number": 7, "timestamp_sec": 3.5,
            "evidence_image": "/api/video/evidence/a/frame_7.jpg",
            "detections": [{"class": "rifle", "confidence": .61, "bbox": [1, 2, 8, 9]}],
        }]
        events = group_evidence_events(frames)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["observation_count"], 1)
        self.assertEqual(events[0]["duration_sec"], 0.0)
        self.assertEqual(events[0]["verification_status"], "unverified")
        self.assertIn("does not confirm", events[0]["description"])

    def test_empty_detections_returns_no_events(self):
        frames = [{"frame_number": 0, "timestamp_sec": 0, "detections": []}]
        self.assertEqual(group_evidence_events(frames), [])

    def test_temporal_confirmation_requires_three_nearby_observations(self):
        frames = [{"timestamp_sec": ts, "detections": [{"class_id": 3, "class": "knife", "track_id": 42,
                  "confidence": confidence, "bbox": [1, 1, 8, 8]}]}
                  for ts, confidence in ((0.0, .5), (.4, .7), (.8, .9))]
        video_service._temporal_confirmation(frames)
        values = [frame["detections"][0] for frame in frames]
        self.assertTrue(all(item["temporal_status"] == "confirmed_repeated_observation" for item in values))
        self.assertAlmostEqual(values[1]["smoothed_confidence"], .7)
        self.assertAlmostEqual(values[1]["track_duration_sec"], .8)

        sparse = [{"timestamp_sec": 2.0, "detections": [{"class_id": 6, "class": "pistol", "track_id": 9,
                  "confidence": .8, "bbox": [1, 1, 8, 8]}]}]
        video_service._temporal_confirmation(sparse)
        self.assertEqual(sparse[0]["detections"][0]["temporal_status"], "unconfirmed_single_or_sparse_observation")

    def test_failure_during_detection_releases_capture(self):
        class BrokenYolo:
            def detect_frame(self, _frame): raise RuntimeError("inference failure")
        cap = FakeCapture()
        with self.assertRaises(RuntimeError): self.analyze(cap, BrokenYolo())
        self.assertTrue(cap.released)


if __name__ == "__main__":
    unittest.main()
