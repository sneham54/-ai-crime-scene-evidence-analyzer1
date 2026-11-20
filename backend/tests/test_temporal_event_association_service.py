import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.services.temporal_event_association_service import TemporalEventAssociationService
from scripts.evaluate_temporal_intervals import evaluate


def _video(path, frames=30, fps=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (48, 32))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((32, 48, 3), index % 255, dtype=np.uint8))
    writer.release()


def _csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["video", "frame", "timestamp_seconds", "class_id", "class_name", "confidence",
              "track_id", "x1", "y1", "x2", "y2"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({"video": "scene.mp4", "class_id": 0, "class_name": "person",
                "confidence": .8, "track_id": 7, "x1": 1, "y1": 1, "x2": 20, "y2": 25, **row})


def _service(tmp_path, records, config=None):
    video = tmp_path / "scene.mp4"
    _video(video)
    csv_path = tmp_path / "selected.csv"
    _csv(csv_path, records)
    options = {"TEMPORAL_EVENT_WEIGHTS": {
        "shooting": {"classifier": .7, "yolo": .15, "tracking": .15},
        "assault": {"classifier": .7, "yolo": .15, "tracking": .15},
        "robbery": {"classifier": .7, "yolo": .15, "tracking": .15},
    }, "TEMPORAL_EVENT_BIN_SEC": 1, "TEMPORAL_EVENT_MIN_SCORE": .5,
        "TEMPORAL_EVENT_MIN_CLASSIFIER_PROBABILITY": .5,
        "TEMPORAL_EVENT_MIN_SUPPORTED_BINS": 1, **(config or {})}
    service = TemporalEventAssociationService(options)
    return service, video, csv_path


def _window(start, end, category="Shooting", probability=.9):
    probabilities = {"Assault": .02, "Robbery": .02, "Shooting": .02, "Normal": .94}
    probabilities[category] = probability
    probabilities["Normal"] = 1 - probability
    return {"start_timestamp_sec": start, "end_timestamp_sec": end,
            "label": category, "confidence": probability, "probabilities": probabilities}


def test_vfr_presentation_timestamps_preferred_and_frame_offset_preserved(tmp_path):
    service, video, csv_path = _service(tmp_path, [
        {"frame": 1, "timestamp_seconds": 0.0},
        {"frame": 2, "timestamp_seconds": .15},
    ])
    pts = {0: 0.0, 1: .15, 2: .31, 3: .48}
    result = service.associate(original_path=video, category="Shooting", start_sec=0, end_sec=1,
        temporal_predictions=[_window(0, 1)], classifier_probabilities={"Shooting": .9},
        csv_path=csv_path, presentation_timestamps=pts)
    assert result["refinement"]["csv_timing_validation"]["method"] == "presentation_timestamp"
    assert result["detections"][0]["frame_number"] == 0
    assert result["detections"][0]["timestamp_seconds"] == 0.0
    assert result["detections"][1]["frame_number"] == 1
    assert result["detections"][1]["timestamp_seconds"] == .15
    assert result["detections"][1]["timestamp_source"] == "presentation_timestamp"


def test_timestamp_mismatch_is_rejected(tmp_path):
    service, video, csv_path = _service(tmp_path, [{"frame": 5, "timestamp_seconds": 8.0}])
    with pytest.raises(ValueError, match="timestamps do not align"):
        service.associate(original_path=video, category="Assault", start_sec=0, end_sec=3,
            temporal_predictions=[_window(0, 2, "Assault")], csv_path=csv_path)


@pytest.mark.parametrize("category,class_name", [("Shooting", "pistol"), ("Assault", "knife"), ("Robbery", "bag")])
def test_category_specific_detection_association_and_track_aggregation(tmp_path, category, class_name):
    service, video, csv_path = _service(tmp_path, [
        {"frame": 10, "timestamp_seconds": 1.0, "class_name": class_name, "track_id": 11, "confidence": .6},
        {"frame": 11, "timestamp_seconds": 1.1, "class_name": class_name, "track_id": 11, "confidence": .9},
        {"frame": 20, "timestamp_seconds": 2.0, "class_name": "person", "track_id": 12, "confidence": .7},
        {"frame": 22, "timestamp_seconds": 2.2, "class_name": class_name, "track_id": 11, "confidence": .75},
        {"frame": 1, "timestamp_seconds": .1, "video": "other.mp4", "track_id": 999},
    ], {"TEMPORAL_EVENT_MAX_TRACK_GAP_SEC": .5})
    result = service.associate(original_path=video, category=category, start_sec=1, end_sec=2.5,
        temporal_predictions=[_window(1, 2.5, category)], csv_path=csv_path)
    assert {item["track_id"] for item in result["detections"]} == {"11", "12"}
    group = next(item for item in result["refinement"]["track_groups"] if item["track_id"] == "11")
    assert group["detection_count"] == 3
    assert group["mean_confidence"] == pytest.approx(.75)
    assert group["max_confidence"] == .9
    assert group["tracking_duration_sec"] == pytest.approx(1.2)
    assert group["detection_gaps_sec"] == pytest.approx([.1, 1.1])
    assert group["continuity_score"] == pytest.approx(.5)
    assert group["fragmented"] is True


def test_classifier_led_refinement_stays_inside_candidate_and_reports_scores(tmp_path):
    service, video, csv_path = _service(tmp_path, [
        {"frame": 10, "timestamp_seconds": 1.0, "class_name": "gun", "confidence": .9},
        {"frame": 11, "timestamp_seconds": 1.1, "class_name": "gun", "confidence": .9},
        {"frame": 20, "timestamp_seconds": 2.0, "class_name": "person", "confidence": .8},
        {"frame": 21, "timestamp_seconds": 2.1, "class_name": "person", "confidence": .8},
    ])
    result = service.associate(original_path=video, category="Shooting", start_sec=0, end_sec=4,
        temporal_predictions=[_window(1, 3)], classifier_probabilities={"Shooting": .9}, csv_path=csv_path)
    refined = result["refinement"]
    assert refined["original_candidate_start_sec"] == 0
    assert refined["original_candidate_end_sec"] == 4
    assert refined["refined_start_sec"] == 1
    assert refined["refined_end_sec"] == 3
    assert refined["refinement_status"] == "refined"
    assert refined["verification_status"] == "unverified_candidate"
    assert refined["supporting_detection_count"] == 4
    assert refined["evidence_score"] is not None
    assert refined["boundary_confidence"]["start"] is not None


def test_missing_classifier_or_insufficient_evidence_preserves_original(tmp_path):
    service, video, csv_path = _service(tmp_path, [
        {"frame": 10, "timestamp_seconds": 1, "class_name": "pistol", "confidence": .99},
        {"frame": 11, "timestamp_seconds": 1.1, "class_name": "person", "confidence": .99},
    ])
    result = service.associate(original_path=video, category="Shooting", start_sec=0, end_sec=3,
        temporal_predictions=[], csv_path=csv_path)
    refined = result["refinement"]
    assert refined["refined_start_sec"] == 0
    assert refined["refined_end_sec"] == 3
    assert refined["refinement_status"] == "insufficient_evidence_preserved"
    assert refined["verification_status"] == "unverified_candidate"


def test_optional_action_signal_requires_validated_result_and_available_weights(tmp_path):
    service, video, csv_path = _service(tmp_path, [
        {"frame": 10, "timestamp_seconds": 1, "class_name": "person", "track_id": 7, "confidence": .8},
    ], {"TEMPORAL_EVENT_WEIGHTS": {"assault": {
        "classifier": .6, "yolo": .1, "tracking": .1, "action": .2,
    }}})
    windows = [_window(1, 2, "Assault", .8)]
    unvalidated = service.associate(original_path=video, category="Assault", start_sec=1, end_sec=2,
        temporal_predictions=windows, csv_path=csv_path,
        additional_evidence={"action": {"validated": False, "score": 1.0}})
    assert "action" not in unvalidated["refinement"]["available_modalities"]
    validated = service.associate(original_path=video, category="Assault", start_sec=1, end_sec=2,
        temporal_predictions=windows, csv_path=csv_path,
        additional_evidence={"action": {"validated": True, "score": .9}})
    assert "action" in validated["refinement"]["available_modalities"]


def test_temporal_service_extracts_clip_and_persists_candidate_and_refined_times(tmp_path):
    service, video, csv_path = _service(tmp_path, [
        {"frame": 10, "timestamp_seconds": 1, "class_name": "pistol", "confidence": .9},
        {"frame": 11, "timestamp_seconds": 1.1, "class_name": "person", "confidence": .8},
    ])
    results = tmp_path / "tracking"
    results.mkdir()
    selected_csv = results / "detections.csv"
    shutil.copyfile(csv_path, selected_csv)
    # A manifest-selected saved CSV is required by event extraction.
    tracked_dir = results / "videos"
    tracked_dir.mkdir()
    tracked = tracked_dir / "annotated.mp4"
    _video(tracked)
    status = results / "status.csv"
    with status.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video", "status", "output_csv", "output_video"])
        writer.writeheader()
        writer.writerow({"video": video.name, "status": "completed", "output_csv": str(selected_csv),
                         "output_video": str(tracked)})
    service = TemporalEventAssociationService({"SHOOTING_RESULTS_DIR": results, "SHOOTING_TRACKING_STATUS_PATH": status,
        "TEMPORAL_EVENT_WEIGHTS": {"shooting": {"classifier": .7, "yolo": .15, "tracking": .15}},
        "TEMPORAL_EVENT_BIN_SEC": 1, "TEMPORAL_EVENT_MIN_SCORE": .5,
        "TEMPORAL_EVENT_MIN_CLASSIFIER_PROBABILITY": .5})
    event = service.create_event(original_path=video, output_root=tmp_path / "evidence", category="Shooting",
        start_sec=0, end_sec=3, temporal_predictions=[_window(1, 2)], classifier_probabilities={"Shooting": .9},
        source_video_name=video.name, context_before_sec=.1, context_after_sec=.1)
    assert event["original_candidate_start_sec"] == 0
    assert event["refined_start_sec"] == 1
    assert event["verification_status"] == "unverified_candidate"
    assert event["evidence_clip_interval"]["start_sec"] <= event["refined_start_sec"]
    assert Path(event["artifacts"]["original_clip"]).is_file()
    assert Path(event["artifacts"]["json"]).is_file()
    event_json = Path(event["artifacts"]["json"]).read_bytes()
    saved = json.loads(event_json)
    assert saved["candidate_start_timestamp_sec"] == 0
    assert saved["refined_start_timestamp_sec"] == 1
    integrity_path = Path(saved["artifacts"]["integrity_manifest"])
    integrity = json.loads(integrity_path.read_text())
    assert integrity["event_json_sha256"] == __import__("hashlib").sha256(event_json).hexdigest()


@pytest.mark.parametrize("category,class_name", [("Assault", "knife"), ("Robbery", "bag")])
def test_temporal_service_creates_category_specific_real_evidence_package(tmp_path, category, class_name):
    service, video, csv_path = _service(tmp_path, [
        {"frame": 10, "timestamp_seconds": 1.0, "class_name": class_name, "confidence": .85},
        {"frame": 11, "timestamp_seconds": 1.1, "class_name": "person", "confidence": .75},
    ])
    category_root = tmp_path / category.casefold()
    if category == "Assault":
        artifact_dir = category_root / video.stem
        artifact_dir.mkdir(parents=True)
        shutil.copyfile(csv_path, artifact_dir / "detections.csv")
        _video(artifact_dir / f"{video.stem}_tracked.mp4")
        config = {
            "ASSAULT_RESULTS_DIR": category_root,
            "TEMPORAL_EVENT_WEIGHTS": {"assault": {"classifier": .7, "yolo": .15, "tracking": .15}},
        }
    else:
        artifact_dir = category_root / video.stem
        artifact_dir.mkdir(parents=True)
        selected_csv = artifact_dir / "detections.csv"
        shutil.copyfile(csv_path, selected_csv)
        tracked = artifact_dir / "annotated.mp4"
        _video(tracked)
        status = category_root / "tracking_status.csv"
        with status.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_name", "status", "detections_csv", "annotated_video"])
            writer.writeheader()
            writer.writerow({"video_name": video.name, "status": "completed",
                             "detections_csv": str(selected_csv), "annotated_video": str(tracked)})
        config = {
            "ROBBERY_RESULTS_DIR": category_root,
            "ROBBERY_TRACKING_STATUS_PATH": status,
            "TEMPORAL_EVENT_WEIGHTS": {"robbery": {"classifier": .7, "yolo": .15, "tracking": .15}},
        }

    config.update({"TEMPORAL_EVENT_BIN_SEC": 1, "TEMPORAL_EVENT_MIN_SCORE": .5,
                   "TEMPORAL_EVENT_MIN_CLASSIFIER_PROBABILITY": .5})
    service = TemporalEventAssociationService(config)
    event = service.create_event(
        original_path=video, output_root=tmp_path / f"{category.casefold()}_evidence",
        category=category, start_sec=0, end_sec=3,
        temporal_predictions=[_window(1, 2, category)], classifier_probabilities={category: .9},
        source_video_name=video.name, context_before_sec=.1, context_after_sec=.1,
        verification_status="unverified_candidate",
    )
    assert event["category"] == category
    assert event["verification_status"] == "unverified_candidate"
    assert event["original_candidate_start_sec"] == 0
    assert event["refined_start_sec"] == 1
    assert len(event["detections"]) == 2
    assert Path(event["artifacts"]["original_clip"]).is_file()
    assert Path(event["artifacts"]["annotated_clip"]).is_file()
    assert Path(event["artifacts"]["csv"]).is_file()
    assert Path(event["artifacts"]["json"]).is_file()


def test_interval_evaluation_reports_no_accuracy_without_manual_references():
    report = evaluate([{"category": "shooting", "video": "scene.mp4",
        "predicted_start_sec": "1", "predicted_end_sec": "2"}])
    assert report["status"] == "not_estimable_no_verified_reference_intervals"
    assert report["refined_intervals"]["start_time_mae_sec"] is None
