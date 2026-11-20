from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from flask import Flask
from types import SimpleNamespace

import sys
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.evidence_catalog_service import EvidenceCatalogService  # noqa: E402
from app.services.video_service import VideoService, _temporal_confirmation  # noqa: E402
from app.services.yolo_service import YOLOService  # noqa: E402


CLASS_NAMES = ("gun", "hammer", "helmet", "knife", "mask", "person", "pistol", "rifle", "weapon")


def make_catalog(tmp_path, contents_by_folder):
    source_root = tmp_path / "sources"
    assault = source_root / "Assault"
    assault.mkdir(parents=True)
    source = assault / "Sample_x264.mp4"
    source.write_bytes(b"same source bytes")
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(source.read_bytes())
    labels_root = tmp_path / "tracking"
    all_txt = []
    for folder_name, named_files in contents_by_folder:
        labels = labels_root / folder_name / "labels"
        labels.mkdir(parents=True)
        for filename, text in named_files.items():
            path = labels / filename
            path.write_text(text, encoding="utf-8")
            all_txt.append(path)
    catalog = EvidenceCatalogService({
        "EVIDENCE_SOURCE_DIR": str(source_root),
        "ASSAULT_RESULTS_DIR": str(tmp_path / "csv"),
        "IMPORTED_TRACKING_RESULTS_DIR": str(labels_root),
    })
    metadata = {"total_frames": 20, "fps": 25.0, "duration_sec": .8,
                "width": 200, "height": 100}
    return catalog, upload, metadata, all_txt, hashlib.sha256(source.read_bytes()).hexdigest()


def match(catalog, upload, metadata):
    return catalog.match_tracking_artifact("assault", "Sample_x264.mp4", str(upload), metadata)


def test_txt_five_field_rows_map_all_nine_classes_and_keep_confidence_missing(tmp_path):
    rows = "\n".join(f"{i} 0.5 0.5 0.4 0.2" for i in range(9))
    catalog, upload, metadata, _, _ = make_catalog(
        tmp_path, [("Sample_x264", {"Sample_x264_10.txt": rows})])

    result = match(catalog, upload, metadata)

    assert result["artifact_type"] == "txt_labels"
    assert [row["class"] for row in result["records"]] == list(CLASS_NAMES)
    assert [row["class_name"] for row in result["records"]] == list(CLASS_NAMES)
    assert all(row["confidence"] is None for row in result["records"])
    assert all(row["track_id"] is None for row in result["records"])
    assert all(row["category"] == ("Persons" if name == "person" else "Weapons" if name in {
        "gun", "hammer", "knife", "pistol", "rifle", "weapon"} else "Other")
               for row, name in zip(result["records"], CLASS_NAMES))


def test_loaded_detector_class_names_take_precedence_over_fallback(monkeypatch):
    monkeypatch.setattr(YOLOService, "_model", SimpleNamespace(names={0: "detector-gun"}))
    assert YOLOService.class_name_for_id(0) == "detector-gun"
    assert YOLOService.class_name_for_id(8) == "weapon"


def test_six_field_row_preserves_valid_id_and_uses_pixel_box_and_source_fps(tmp_path):
    catalog, upload, metadata, _, _ = make_catalog(
        tmp_path, [("Sample_x264", {"Sample_x264_10.txt": "8 0.5 0.5 0.4 0.2 37"})])

    result = match(catalog, upload, metadata)
    row = result["records"][0]

    assert row["track_id"] == 37
    assert row["confidence"] is None
    assert row["class"] == row["class_name"] == "weapon"
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (60, 40, 140, 60)
    assert row["timestamp_seconds"] == .4


@pytest.mark.parametrize("invalid_id", ["-1", "1.5", "nan", "inf", "x"])
def test_six_field_rows_reject_invalid_track_ids(tmp_path, invalid_id):
    catalog, upload, metadata, _, _ = make_catalog(
        tmp_path, [("Sample_x264", {"Sample_x264_10.txt": f"0 .5 .5 .2 .2 {invalid_id}"})])
    assert match(catalog, upload, metadata)["status"] == "INVALID_TRACKING_DATA"


@pytest.mark.parametrize("row", [
    "0 nan .5 .2 .2", "0 inf .5 .2 .2", "0 -.1 .5 .2 .2",
    "0 .5 .5 1.1 .2", "0 .95 .5 .2 .2", "0 .5 .5 0 .2",
])
def test_invalid_normalized_coordinates_are_rejected(tmp_path, row):
    catalog, upload, metadata, _, _ = make_catalog(
        tmp_path, [("Sample_x264", {"Sample_x264_10.txt": row})])
    assert match(catalog, upload, metadata)["status"] == "INVALID_TRACKING_DATA"


@pytest.mark.parametrize("class_id", ["-1", "9", "1.5", "nan", "inf"])
def test_invalid_class_ids_are_rejected(tmp_path, class_id):
    catalog, upload, metadata, _, _ = make_catalog(
        tmp_path, [("Sample_x264", {"Sample_x264_10.txt": f"{class_id} .5 .5 .2 .2"})])
    assert match(catalog, upload, metadata)["status"] == "INVALID_TRACKING_DATA"


def test_out_of_range_frame_is_rejected_and_in_range_does_not_verify_mapping(tmp_path):
    catalog, upload, metadata, _, _ = make_catalog(
        tmp_path, [("Sample_x264", {"Sample_x264_19.txt": "5 .5 .5 .2 .2"})])
    result = match(catalog, upload, metadata)
    assert result["records"][0]["frame"] == 19
    assert result["records"][0]["timestamp_seconds"] == .76
    assert result["status"] == "UNVERIFIED_MATCH"
    assert result["metadata_verification"]["frame_index_mapping"] is False

    bad_catalog, bad_upload, bad_metadata, _, _ = make_catalog(
        tmp_path / "bad", [("Sample_x264", {"Sample_x264_20.txt": "5 .5 .5 .2 .2"})])
    assert match(bad_catalog, bad_upload, bad_metadata)["status"] == "INVALID_TRACKING_DATA"


def test_duplicate_frame_files_and_source_hash_mismatch_are_not_reused(tmp_path):
    catalog, upload, metadata, _, _ = make_catalog(tmp_path, [
        ("Sample_x264", {"Sample_x264_4.txt": "5 .5 .5 .2 .2"}),
        ("sample-x264", {"sample-x264_4.txt": "6 .5 .5 .2 .2"}),
    ])
    assert match(catalog, upload, metadata)["status"] == "UNVERIFIED_MATCH"

    upload.write_bytes(b"different source bytes")
    result = match(catalog, upload, metadata)
    assert result["status"] == "UNVERIFIED_MATCH"
    assert result["detection_count"] == 0


def test_saved_txt_tracking_metadata_preserves_type_path_ids_and_null_confidence(tmp_path):
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"source")
    labels = tmp_path / "Sample" / "labels"
    labels.mkdir(parents=True)
    app = Flask(__name__)
    app.config.update(UPLOAD_FOLDER=str(tmp_path / "uploads"),
                      VIDEO_EVENT_MAX_GAP_SEC=3.0, VIDEO_EVENT_MIN_IOU=.1,
                      VIDEO_MAX_DURATION_SEC=600)
    service = VideoService.__new__(VideoService)
    service.classifier = None
    service.classifier_error = RuntimeError("not configured for test")
    match_data = {
        "status": "UNVERIFIED_MATCH", "artifact_type": "txt_labels",
        "artifact_path": str(labels), "artifact_filename": "Sample/labels",
        "source_path": str(source), "records": [{
            "frame": 10, "timestamp_seconds": .4, "class": "person",
            "class_name": "person", "class_id": 5, "confidence": None,
            "track_id": 37, "x1": 60, "y1": 40, "x2": 140, "y2": 60,
            "category": "Persons",
        }],
    }

    with app.app_context():
        result = service._analyze_saved_tracking(
            str(source), "sample.mp4", "assault", match_data,
            total_frames=20, fps=25.0, duration=.8, source_width=200, source_height=100)

    detection = result["detections"][0]
    tracking = result["tracking"]
    assert detection["confidence"] is None
    assert detection["track_id"] == 37
    assert detection["tracking_status"] == "unverified_txt_track_id"
    assert tracking["source_csv"] is None
    assert tracking["artifact_type"] == "txt_labels"
    assert tracking["source_artifact"] == str(labels)
    assert tracking["source_artifact_filename"] == "Sample/labels"
    assert tracking["tracking_algorithm"].startswith("Unverified")
    assert tracking["tracked_detection_count"] == 1
    assert result["video_info"]["timestamp_source"] == "unverified_frame_index_divided_by_reported_fps"
    assert result["video_info"]["frame_index_mapping_status"] == "unverified"
    assert result["evidence_events"][0]["confidence_summary"] is None


def test_temporal_confidence_summary_keeps_missing_values_unavailable():
    frames = [{"timestamp_sec": float(index), "detections": [{
        "class_id": 5, "class": "person", "track_id": 7, "confidence": None,
    }]} for index in range(3)]
    _temporal_confirmation(frames)
    assert all(frame["detections"][0]["smoothed_confidence"] is None for frame in frames)


