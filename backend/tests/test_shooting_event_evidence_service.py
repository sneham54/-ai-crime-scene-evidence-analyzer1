import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import json
from flask import Flask
from flask_jwt_extended import JWTManager, create_access_token
from types import SimpleNamespace
from unittest.mock import patch

from app.services.shooting_event_evidence_service import ShootingEventEvidenceService
from app.services.shooting_event_evidence_service import EventEvidenceService
from app.routes import video as video_route
from app.routes.video import bp as video_blueprint
from app.routes.evidence import bp as evidence_blueprint
from app.models.evidence import serialize_evidence


def _video(path, frames=12, fps=6):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.putText(frame, str(index), (4, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


def _setup(root, *, include_csv=True, status="completed", aligned=True):
    source = root / "Shooting001.mp4"
    _video(source)
    artifact = root / "artifacts"
    artifact.mkdir()
    csv_path = artifact / "Shooting001_detections.csv"
    if include_csv:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video", "frame", "timestamp_seconds", "class_id", "class_name", "confidence", "track_id", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            writer.writerow({"video": source.name, "frame": 6, "timestamp_seconds": 1, "class_id": 8,
                             "class_name": "weapon", "confidence": .8, "track_id": 4,
                             "x1": 4, "y1": 5, "x2": 20, "y2": 25})
            writer.writerow({"video": source.name, "frame": 10, "timestamp_seconds": 1.6667, "class_id": 5,
                             "class_name": "person", "confidence": .9, "track_id": 7,
                             "x1": 12, "y1": 8, "x2": 30, "y2": 40})
            writer.writerow({"video": "Other.mp4", "frame": 7, "timestamp_seconds": 1.16, "class_id": 5,
                             "class_name": "person", "confidence": .9, "track_id": 99,
                             "x1": 1, "y1": 1, "x2": 10, "y2": 10})
    videos = root / "videos"
    videos.mkdir()
    tracked = videos / "Shooting001_tracked.mp4"
    if aligned:
        _video(tracked)
    status_path = root / "status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video", "status", "output_csv", "output_video"])
        writer.writeheader()
        writer.writerow({"video": source.name, "status": status, "output_csv": str(csv_path),
                         "output_video": str(tracked)})
    return source, {"SHOOTING_TRACKING_STATUS_PATH": status_path,
                    "SHOOTING_RESULTS_DIR": artifact}


def test_event_association_extracts_source_clip_and_aligned_tracking(tmp_path):
    source, config = _setup(tmp_path)
    service = ShootingEventEvidenceService(config)
    record = service.create_event(original_path=source, output_root=tmp_path / "out",
                                  start_sec=0.8, end_sec=1.2, confidence=.72,
                                  context_before_sec=.2, context_after_sec=.2)
    assert [item["frame_number"] for item in record["detections"]] == [6]
    assert record["detections"][0]["track_id"] == "4"
    assert record["detections"][0]["bbox"] == [4.0, 5.0, 20.0, 25.0]
    assert record["tracked_video_alignment"]["verified"] is True
    assert Path(record["artifacts"]["original_clip"]).is_file()
    assert Path(record["artifacts"]["annotated_clip"]).is_file()
    assert Path(record["artifacts"]["keyframes"][0]).is_file()
    assert Path(record["artifacts"]["annotated_keyframes"][0]).is_file()
    # Stable source and interval identity makes retry output idempotent.
    assert service.create_event(original_path=source, output_root=tmp_path / "out",
                                start_sec=.8, end_sec=1.2, context_before_sec=.2,
                                context_after_sec=.2)["evidence_id"] == record["evidence_id"]


def test_misaligned_annotated_video_is_not_used(tmp_path):
    source, config = _setup(tmp_path, aligned=False)
    record = ShootingEventEvidenceService(config).create_event(
        original_path=source, output_root=tmp_path / "out", start_sec=.8, end_sec=1.2)
    assert record["tracked_video_alignment"]["verified"] is False
    assert record["artifacts"]["annotated_clip"] is None


def test_empty_detection_csv_still_extracts_original_context_keyframes(tmp_path):
    source, config = _setup(tmp_path)
    with (tmp_path / "artifacts" / "Shooting001_detections.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(["video", "frame", "timestamp_seconds", "class_id", "class_name", "confidence", "track_id", "x1", "y1", "x2", "y2"])
    record = ShootingEventEvidenceService(config).create_event(
        original_path=source, output_root=tmp_path / "out", start_sec=.8, end_sec=1.2)
    assert record["detections"] == []
    assert Path(record["artifacts"]["original_clip"]).is_file()
    assert record["artifacts"]["keyframes"]
    assert record["artifacts"]["annotated_keyframes"] == []


@pytest.mark.parametrize("start,end", [(-1, 1), (1, 1), (2, 1), (0, 5)])
def test_invalid_event_intervals_rejected(tmp_path, start, end):
    source, config = _setup(tmp_path)
    with pytest.raises(ValueError):
        ShootingEventEvidenceService(config).create_event(
            original_path=source, output_root=tmp_path / "out", start_sec=start, end_sec=end)


def test_missing_manifest_csv_and_failed_job_rejected(tmp_path):
    source, config = _setup(tmp_path, include_csv=False)
    with pytest.raises(FileNotFoundError):
        ShootingEventEvidenceService(config).create_event(
            original_path=source, output_root=tmp_path / "out", start_sec=.5, end_sec=1)


def test_duplicate_completed_manifest_records_are_rejected(tmp_path):
    source, config = _setup(tmp_path)
    status_path = config["SHOOTING_TRACKING_STATUS_PATH"]
    with status_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([source.name, "completed", "duplicate.csv", "duplicate.mp4"])
    with pytest.raises(ValueError, match="no unique record"):
        ShootingEventEvidenceService(config).resolve_artifacts(source, "Shooting", source.name)


def test_assault_artifacts_are_selected_from_exact_source_folder(tmp_path):
    source = tmp_path / "source" / "Assault001_x264.mp4"
    source.parent.mkdir()
    source.write_bytes(b"assault video")
    folder = tmp_path / "assault-results" / source.stem
    folder.mkdir(parents=True)
    csv_path = folder / "detections.csv"
    csv_path.write_text("frame,timestamp_seconds,class_id,confidence,x1,y1,x2,y2\n0,0,0,.8,1,2,3,4\n")
    tracked_path = folder / f"{source.stem}_tracked.mp4"
    tracked_path.write_bytes(b"annotated video")
    service = EventEvidenceService({"ASSAULT_RESULTS_DIR": folder.parent})
    manifest, selected_csv, selected_video = service.resolve_artifacts(source, "Assault", source.name)
    assert manifest["status"] == "folder_csv_pair"
    assert selected_csv == csv_path.resolve()
    assert selected_video == tracked_path.resolve()


def test_robbery_artifacts_follow_completed_status_manifest_paths(tmp_path):
    source = tmp_path / "source" / "Robbery001_x264.mp4"
    source.parent.mkdir()
    source.write_bytes(b"robbery video")
    results = tmp_path / "robbery-results"
    run = results / source.stem
    run.mkdir(parents=True)
    csv_path = run / "detections.csv"
    csv_path.write_text("frame,timestamp_seconds,class_id,confidence,x1,y1,x2,y2\n")
    tracked_path = run / "annotated.mp4"
    tracked_path.write_bytes(b"annotated video")
    status_path = results / "tracking_status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_name", "status", "annotated_video", "detections_csv"])
        writer.writeheader()
        writer.writerow({"video_name": source.name, "status": "completed",
                         "annotated_video": str(tracked_path), "detections_csv": str(csv_path)})
    service = EventEvidenceService({"ROBBERY_RESULTS_DIR": results, "ROBBERY_TRACKING_STATUS_PATH": status_path})
    manifest, selected_csv, selected_video = service.resolve_artifacts(source, "Robbery", source.name)
    assert manifest["resolved_status"] == "completed"
    assert selected_csv == csv_path.resolve()
    assert selected_video == tracked_path.resolve()


@pytest.mark.parametrize("category", ["Assault", "Robbery"])
def test_category_event_extraction_preserves_detection_time_track_and_clips(tmp_path, category):
    source = tmp_path / "source" / f"{category}001_x264.mp4"
    _video(source)
    results = tmp_path / f"{category.lower()}-results"
    run = results / source.stem
    run.mkdir(parents=True)
    csv_path = run / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video", "frame", "timestamp_seconds", "class_id",
            "class", "confidence", "track_id", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerow({"video": source.name, "frame": 6, "timestamp_seconds": 1.0,
            "class_id": 0, "class": "person", "confidence": .83, "track_id": 29,
            "x1": 2, "y1": 3, "x2": 24, "y2": 39})
    tracked = run / ("annotated.mp4" if category == "Robbery" else f"{source.stem}_tracked.mp4")
    _video(tracked)
    config = {f"{category.upper()}_RESULTS_DIR": results}
    if category == "Robbery":
        status_path = results / "tracking_status.csv"
        with status_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_name", "status", "annotated_video", "detections_csv"])
            writer.writeheader()
            writer.writerow({"video_name": source.name, "status": "completed",
                "annotated_video": str(tracked), "detections_csv": str(csv_path)})
        config["ROBBERY_TRACKING_STATUS_PATH"] = status_path
    record = EventEvidenceService(config).create_event(original_path=source,
        output_root=tmp_path / "evidence" / category.lower(), start_sec=.9, end_sec=1.1,
        category=category, source_video_name=source.name, context_before_sec=.1, context_after_sec=.1)
    assert record["category"] == category
    assert record["verification_status"] == "unverified"
    assert record["detections"][0]["timestamp_seconds"] == 1.0
    assert record["detections"][0]["frame_number"] == 6
    assert record["detections"][0]["track_id"] == "29"
    assert record["tracked_video_alignment"]["verified"] is True
    assert Path(record["artifacts"]["original_clip"]).is_file()
    assert Path(record["artifacts"]["annotated_clip"]).is_file()
    assert Path(record["artifacts"]["keyframes"][0]).is_file()
    assert record["tracking_groups"][0]["track_id"] == "29"


def test_generic_assault_event_api_persists_category_package(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    source = video_dir / "Assault001_x264.mp4"
    source.write_bytes(b"uploaded source")
    analysis_id = "b" * 32
    analysis_dir = tmp_path / "evidence" / analysis_id
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "analysis.json").write_text(json.dumps({
        "category": "Assault", "original_filename": source.name,
        "source_video_path": str(source), "video_info": {"fps": 6, "total_frames": 12},
        "classification": {"label": "Assault", "confidence": .7,
            "activity_events": [{"category": "Assault", "start_timestamp_sec": 1,
                                 "end_timestamp_sec": 2, "confidence": .7}]},
    }), encoding="utf-8")
    record = {"evidence_id": "f" * 24, "category": "Assault",
              "verification_status": "unverified_candidate", "detections": [], "artifacts": {}}

    class Collection:
        def __init__(self): self.calls = []
        def update_one(self, query, update, upsert=False): self.calls.append((query, update, upsert))

    app = Flask(__name__)
    app.config.update(JWT_SECRET_KEY="assault-test", UPLOAD_FOLDER=str(tmp_path))
    JWTManager(app)
    app.register_blueprint(video_blueprint, url_prefix="/api/video")
    evidence = Collection()
    app.db = SimpleNamespace(evidence=evidence)

    class EventService:
        def __init__(self, _config): pass
        def create_event(self, **kwargs):
            assert kwargs["category"] == "assault"
            assert kwargs["verification_status"] == "unverified_candidate"
            return record

    with patch.object(video_route, "TemporalEventAssociationService", EventService):
        response = app.test_client().post(f"/api/video/event-evidence/{analysis_id}", json={
            "start_timestamp_sec": 1, "end_timestamp_sec": 2,
        })
    assert response.status_code == 200
    assert response.get_json()["event_evidence"] is not None
    assert "shooting_event" not in response.get_json()
    assert evidence.calls[0][1]["$setOnInsert"]["event_evidence"] is record
    assert evidence.calls[0][1]["$setOnInsert"]["file_type"] == "assault_event"


def test_evidence_serialization_keeps_generic_and_legacy_shooting_packages():
    generic = {"evidence_id": "a" * 24, "category": "Robbery"}
    legacy = {"evidence_id": "b" * 24, "category": "Shooting"}
    document = serialize_evidence({"_id": "mongo-id", "event_evidence": generic, "shooting_event": legacy})
    assert document["event_evidence"] is generic
    assert document["shooting_event"] is legacy


def test_category_event_listing_reads_durable_assault_package(tmp_path):
    analysis_id = "c" * 32
    event_id = "d" * 24
    event_root = tmp_path / "evidence" / analysis_id / "category-events" / "assault" / event_id
    event_root.mkdir(parents=True)
    (event_root / "event.json").write_text(json.dumps({
        "evidence_id": event_id, "category": "Assault", "verification_status": "unverified_candidate",
    }), encoding="utf-8")
    app = Flask(__name__)
    app.config.update(TESTING=True, JWT_SECRET_KEY="event-list-test-secret-long-enough-for-sha256",
                      UPLOAD_FOLDER=str(tmp_path))
    JWTManager(app)
    app.register_blueprint(evidence_blueprint, url_prefix="/api/evidence")
    with app.app_context():
        token = create_access_token(identity="reviewer")
    response = app.test_client().get("/api/evidence/event-evidence/assault",
        headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.get_json()["events"] == [{
        "evidence_id": event_id, "category": "Assault", "verification_status": "unverified_candidate",
        "analysis_id": analysis_id,
    }]


def test_shooting_event_api_persists_idempotently_and_preserves_candidate_status(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    source = video_dir / "Shooting001.mp4"
    source.write_bytes(b"uploaded source")
    analysis_id = "a" * 32
    analysis_dir = tmp_path / "evidence" / analysis_id
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "analysis.json").write_text(json.dumps({
        "category": "Shooting", "original_filename": source.name,
        "source_video_path": str(source), "video_info": {"fps": 6, "total_frames": 12},
        "classification": {"label": "Shooting", "confidence": .8},
        "verification_status": "unverified",
    }), encoding="utf-8")
    canonical_csv = tmp_path / "canonical.csv"
    canonical_csv.write_text("frame\n", encoding="utf-8")
    event = {"evidence_id": "e" * 24, "verification_status": "unverified_candidate",
             "detections": [], "artifacts": {}}

    class Collection:
        def __init__(self):
            self.calls = []

        def update_one(self, query, update, upsert=False):
            self.calls.append((query, update, upsert))

    evidence = Collection()
    app = Flask(__name__)
    app.config.update(JWT_SECRET_KEY="shooting-test", UPLOAD_FOLDER=str(tmp_path))
    JWTManager(app)
    app.register_blueprint(video_blueprint, url_prefix="/api/video")
    app.db = SimpleNamespace(evidence=evidence)

    class EventService:
        def __init__(self, _config):
            pass

        def create_event(self, **_kwargs):
            return event

    with patch.object(video_route, "TemporalEventAssociationService", EventService):
        response = app.test_client().post(f"/api/video/shooting-events/{analysis_id}", json={
            "start_timestamp_sec": 1, "end_timestamp_sec": 2,
            "verification_status": "unverified_candidate",
        })
    assert response.status_code == 200
    assert response.get_json()["shooting_event"]["verification_status"] == "unverified_candidate"
    assert len(evidence.calls) == 1
    query, update, upsert = evidence.calls[0]
    assert query == {"evidence_id": event["evidence_id"]}
    assert update["$setOnInsert"]["shooting_event"] is event
    assert upsert is True
    source, config = _setup(tmp_path / "failed", status="failed")
    with pytest.raises(ValueError, match="status"):
        ShootingEventEvidenceService(config).create_event(
            original_path=source, output_root=tmp_path / "out", start_sec=.5, end_sec=1)
