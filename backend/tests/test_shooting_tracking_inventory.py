import csv
import tempfile
from pathlib import Path

from app.services.evidence_catalog_service import EvidenceCatalogService


def _write_detections(path: Path, frame=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "frame", "timestamp_seconds", "class", "confidence", "x1", "y1", "x2", "y2", "track_id",
        ])
        writer.writeheader()
        writer.writerow({"frame": frame, "timestamp_seconds": frame / 30, "class": "person",
                         "confidence": .9, "x1": 1, "y1": 2, "x2": 10, "y2": 12, "track_id": 17})


def _manifest(path: Path, video, status, output_csv, frames=4, processed=2, rows=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "status", "frames_total", "frames_processed",
                                                    "detection_rows", "output_csv"])
        writer.writeheader()
        writer.writerow({"video": video, "status": status, "frames_total": frames,
                         "frames_processed": processed, "detection_rows": rows,
                         "output_csv": f"/remote/detections/{output_csv}"})


def test_shooting_match_uses_manifest_canonical_version_and_excludes_failed_job():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source_root = root / "sources"
        source_dir = source_root / "Shooting"
        source_dir.mkdir(parents=True)
        original = source_dir / "ShootingDemo.mp4"
        original.write_bytes(b"original-unannotated-video")
        upload = root / "upload.mp4"
        upload.write_bytes(original.read_bytes())
        detections = root / "detections"
        old = detections / "ShootingDemo_detections_20261026.csv"
        canonical = detections / "ShootingDemo_detections_20261027.csv"
        _write_detections(old, 1)
        _write_detections(canonical, 2)
        failed_csv = detections / "ShootingFailed_detections_partial.csv"
        _write_detections(failed_csv)
        manifest = root / "shooting_tracking_status.csv"
        _manifest(manifest, "ShootingDemo.mp4", "completed", canonical.name)
        with manifest.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["video", "status", "frames_total", "frames_processed",
                                                        "detection_rows", "output_csv"])
            writer.writerow({"video": "ShootingFailed.mp4", "status": "failed", "frames_total": 4,
                             "frames_processed": 1, "detection_rows": 1,
                             "output_csv": f"/remote/detections/{failed_csv.name}"})
        failed_source = source_dir / "ShootingFailed.mp4"
        failed_source.write_bytes(b"failed-original")

        service = EvidenceCatalogService({"EVIDENCE_SOURCE_DIR": str(source_root),
                                          "SHOOTING_RESULTS_DIR": str(detections),
                                          "SHOOTING_TRACKING_STATUS_PATH": str(manifest)})
        metadata = {"total_frames": 4, "fps": 30.0, "duration_sec": 4 / 30,
                    "width": 20, "height": 20}
        matched = service.match_tracking_artifact("Shooting", original.name, str(upload), metadata)
        assert matched["status"] == "EXACT_MATCH"
        assert Path(matched["csv_path"]) == canonical
        assert matched["artifact_version_count"] == 2
        assert matched["tracking_id_count"] == 1
        assert matched["manifest_video"] == "ShootingDemo.mp4"

        failed_upload = root / "failed-upload.mp4"
        failed_upload.write_bytes(failed_source.read_bytes())
        rejected = service.match_tracking_artifact("Shooting", failed_source.name, str(failed_upload), metadata)
        assert rejected["status"] == "INVALID_TRACKING_DATA"
        assert rejected["job_status"] == "failed"


def test_shooting_inventory_collapses_duplicate_versions_and_reports_eval_overlap():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        detections = root / "main_archive" / "Shooting" / "detections"
        _write_detections(detections / "ShootingDemo_detections_1.csv")
        canonical = detections / "ShootingDemo_detections_2.csv"
        _write_detections(canonical, 1)
        manifest = root / "main_archive" / "Shooting" / "shooting_tracking_status.csv"
        _manifest(manifest, "ShootingDemo.mp4", "completed", canonical.name)
        (root / "main_archive" / "Shooting" / "shooting_all_timestamped_detections.csv").write_text("frame,class\n0,person\n", encoding="utf-8")
        for archive in ("evaluation_1", "evaluation_2"):
            _write_detections(root / archive / "shooting_evaluation" / "ShootingDemo_detections.csv")
        service = EvidenceCatalogService({"EVIDENCE_SOURCE_DIR": str(root / "sources"),
                                          "SHOOTING_RESULTS_DIR": str(detections),
                                          "SHOOTING_TRACKING_STATUS_PATH": str(manifest),
                                          "ASSAULT_RESULTS_DIR": str(root / "none-a"),
                                          "ROBBERY_RESULTS_DIR": str(root / "none-r"),
                                          "IMPORTED_TRACKING_RESULTS_DIR": str(root / "none-txt")})
        inventory = service.tracking_artifact_inventory()
        shooting = [item for item in inventory["artifacts"] if item["category"] == "shooting"]
        assert len(shooting) == 1
        assert shooting[0]["filename"] == canonical.name
        assert shooting[0]["duplicate_version_count"] == 1
        assert inventory["shooting_inventory"]["total_detection_csv_file_count"] == 3
        assert inventory["shooting_inventory"]["shared_evaluation_csv_filename_count"] == 1
        assert inventory["shooting_inventory"]["evaluation_archives_used_for_automatic_reuse"] is False
