"""Run the existing upload and video-analysis Flask routes on four pilot videos.

This harness does not create an alternate analysis service. It registers the
existing upload/video blueprints in an isolated Flask test app with MongoDB
disabled and writes all uploads/evidence under a new run-specific directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT / "backend"
sys.path.insert(0, str(BACKEND))

from flask import Flask  # noqa: E402
from flask_jwt_extended import JWTManager  # noqa: E402

from app.config import Config  # noqa: E402
from app.routes.upload import bp as upload_bp  # noqa: E402
from app.routes.video import bp as video_bp  # noqa: E402


PILOTS = (
    ("Shooting", "Shooting001_x264.mp4"),
    ("Shooting", "Shooting002_x264.mp4"),
    ("Assault", "Assault001_x264.mp4"),
    ("Robbery", "Robbery001_x264.mp4"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def source_manifest(project: Path) -> dict[tuple[str, str], dict]:
    manifest_path = project / "reports" / "source_video_manifest.csv"
    rows = csv.DictReader(manifest_path.open("r", newline="", encoding="utf-8-sig"))
    return {(str(row.get("category", "")).casefold(), Path(row.get("source_path", "")).name.casefold()): row
            for row in rows}


def artifact_hashes(paths: dict) -> dict:
    result = {}
    for name, raw in paths.items():
        if not raw:
            result[name] = None
            continue
        path = Path(raw)
        result[name] = {"path": str(path), "exists": path.is_file(),
                        "size_bytes": path.stat().st_size if path.is_file() else None,
                        "sha256": sha256(path) if path.is_file() else None}
    return result


def detection_summary(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("class_name") or row.get("class") or "unknown")].append(row)
    summary = {}
    for name, values in sorted(grouped.items()):
        confidences = [float(row["confidence"]) for row in values if row.get("confidence") is not None]
        summary[name] = {
            "count": len(values),
            "frames": sorted({int(row["frame_number"]) for row in values if row.get("frame_number") is not None}),
            "timestamps_sec": sorted({float(row["timestamp_sec"]) for row in values if row.get("timestamp_sec") is not None}),
            "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
            "min_confidence": min(confidences) if confidences else None,
            "max_confidence": max(confidences) if confidences else None,
            "track_ids": sorted({str(row["track_id"]) for row in values if row.get("track_id") is not None}),
        }
    return summary


def read_tracks(path: str | None) -> list[dict]:
    if not path or not Path(path).is_file():
        return []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    source_root = Path(Config.EVIDENCE_SOURCE_DIR)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = PROJECT / "reports" / f"four_pilot_acceptance_{run_id}"
    if output_root.exists():
        raise SystemExit(f"Refusing to overwrite existing run directory: {output_root}")
    output_root.mkdir(parents=True)
    manifest = source_manifest(PROJECT)
    verified_sources = []
    for category, filename in PILOTS:
        source = source_root / category / filename
        if not source.is_file():
            raise FileNotFoundError(f"Pilot source is missing: {source}")
        digest = sha256(source)
        row = manifest.get((category.casefold(), filename.casefold()))
        if not row or str(row.get("source_sha256", "")).casefold() != digest.casefold():
            raise RuntimeError(f"Source manifest SHA-256 mismatch or missing row: {source}")
        verified_sources.append({"category": category, "filename": filename,
                                 "path": str(source.resolve()), "sha256": digest,
                                 "manifest_status": row.get("status"), "manifest_reason": row.get("reason")})

    app = Flask("app", root_path=str(BACKEND / "app"))
    app.config.from_object(Config)
    app.config.update(TESTING=True, UPLOAD_FOLDER=str(output_root / "uploads"))
    app.db = None
    app.mongo_client = None
    for folder in ("images", "videos", "evidence", "annotated_videos"):
        (Path(app.config["UPLOAD_FOLDER"]) / folder).mkdir(parents=True, exist_ok=True)
    JWTManager(app)
    app.register_blueprint(upload_bp, url_prefix="/api/upload")
    app.register_blueprint(video_bp, url_prefix="/api/video")

    runs = []
    with app.test_client() as client:
        for source_record in verified_sources:
            category = source_record["category"]
            filename = source_record["filename"]
            started = time.monotonic()
            with Path(source_record["path"]).open("rb") as stream:
                uploaded = client.post("/api/upload/video", data={"file": (stream, filename)},
                                       content_type="multipart/form-data")
            if uploaded.status_code != 200 or not uploaded.json or not uploaded.json.get("success"):
                runs.append({**source_record, "status": "upload_failed", "http_status": uploaded.status_code,
                             "error": uploaded.get_data(as_text=True)[:1000]})
                print(json.dumps(runs[-1]), flush=True)
                continue
            upload_data = uploaded.json
            analyzed = client.post("/api/video/analyze", json={
                "filepath": upload_data["filepath"], "evidence_id": upload_data.get("evidence_id"),
                "category": category.casefold(), "original_filename": filename, "full_video": True,
            })
            elapsed = round(time.monotonic() - started, 3)
            result = analyzed.json or {}
            artifact_paths = result.get("artifact_paths") or {}
            row = {
                **source_record,
                "status": "completed" if analyzed.status_code == 200 and result.get("success") else "analysis_failed",
                "http_status": analyzed.status_code, "elapsed_sec": elapsed,
                "analysis_id": result.get("analysis_id"),
                "upload_persistence": upload_data.get("persistence"),
                "analysis_persistence": result.get("persistence"),
                "source_integrity": result.get("source_integrity"),
                "video_info": result.get("video_info"),
                "classification": result.get("classification"),
                "tracking": result.get("tracking"),
                "tracking_data_match": result.get("tracking_data_match"),
                "total_detections": result.get("total_detections"),
                "class_detection_counts": detection_summary(result.get("detections") or []),
                "tracks": read_tracks(artifact_paths.get("tracking_csv")),
                "candidate_intervals": result.get("candidate_intervals"),
                "activity_events": result.get("activity_events"),
                "evidence_events": result.get("evidence_events"),
                "artifact_paths": artifact_paths,
                "artifact_integrity": artifact_hashes(artifact_paths),
                "error": result.get("error"),
            }
            row_path = output_root / f"{Path(filename).stem}_result.json"
            row_path.write_text(json.dumps(json_safe(row), indent=2, allow_nan=False), encoding="utf-8")
            runs.append({"result_json": str(row_path), **row})
            print(json.dumps({"pilot": filename, "status": row["status"], "analysis_id": row.get("analysis_id"),
                              "frames": (row.get("video_info") or {}).get("frames_analyzed"),
                              "detections": row.get("total_detections"),
                              "tracker_match": (row.get("tracking_data_match") or {}).get("status"),
                              "seconds": elapsed}), flush=True)

    summary = {"run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "pipeline": "existing Flask upload/video blueprints; full_video=true",
               "source_manifest": str(PROJECT / "reports" / "source_video_manifest.csv"),
               "output_directory": str(output_root), "results": runs}
    (output_root / "run_summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(f"RUN_SUMMARY={output_root / 'run_summary.json'}", flush=True)
    return 0 if all(row.get("status") == "completed" for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
