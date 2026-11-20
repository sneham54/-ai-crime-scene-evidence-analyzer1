import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.recover_assault_videos_from_csv import _detection_rows, _render_one


@pytest.fixture
def recovery_workspace():
    root = PROJECT_ROOT / ".integration-fixtures" / "assault-recovery-tests"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_csv(path, rows):
    fields = ["frame", "timestamp_seconds", "class", "confidence", "track_id", "x1", "y1", "x2", "y2"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_saved_frame_and_timestamp_fields_map_to_source_frames(recovery_workspace):
    csv_path = recovery_workspace / "detections.csv"
    _write_csv(csv_path, [{"frame": 1, "timestamp_seconds": 0, "class": "knife", "confidence": .8,
                          "track_id": 42, "x1": 1, "y1": 2, "x2": 9, "y2": 10}])
    result, count = _detection_rows(csv_path, 4, 10, 64, 48)
    assert count == 1
    assert result[0][0]["track_id"] == "42"
    assert result[0][0]["bbox"] == (1, 2, 9, 10)


def test_missing_bbox_information_is_not_rendered_as_valid(recovery_workspace):
    csv_path = recovery_workspace / "detections.csv"
    csv_path.write_text("frame,timestamp_seconds,class,confidence\n0,0,knife,0.8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required CSV columns"):
        _detection_rows(csv_path, 4, 10, 64, 48)


def test_render_recovery_preserves_csv_and_validates_new_mp4(recovery_workspace):
    source = recovery_workspace / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(4):
        writer.write(np.full((48, 64, 3), index * 25, dtype=np.uint8))
    writer.release()

    csv_path = recovery_workspace / "detections.csv"
    _write_csv(csv_path, [
        {"frame": 1, "timestamp_seconds": 0, "class": "knife", "confidence": .8,
         "track_id": 42, "x1": 5, "y1": 5, "x2": 25, "y2": 25},
        {"frame": 4, "timestamp_seconds": .3, "class": "knife", "confidence": .7,
         "track_id": 42, "x1": 7, "y1": 5, "x2": 27, "y2": 25},
    ])
    original_csv_bytes = csv_path.read_bytes()
    output = recovery_workspace / "Assault042_x264_recovered_from_csv.mp4"
    result = _render_one(source, csv_path, output)
    assert result["status"] == "recovered_from_saved_csv_no_inference"
    assert result["frames"] == 4
    assert (result["width"], result["height"]) == (64, 48)
    assert csv_path.read_bytes() == original_csv_bytes
    assert output.is_file() and output.stat().st_size > 0
    assert _render_one(source, csv_path, output)["status"] == "skipped_output_exists"
