"""Read-only catalog for previously generated category evidence artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from app.services.yolo_service import YOLOService


CATEGORIES = {
    "assault": {
        "label": "Assault",
        "kind": "object_detection",
        "results_config": "ASSAULT_RESULTS_DIR",
        "method": "Saved YOLO detections; track IDs are shown where present, but per-run tracker provenance is not recorded",
    },
    "robbery": {
        "label": "Robbery",
        "kind": "activity_prediction",
        "results_config": "ROBBERY_RESULTS_DIR",
        "method": "Activity classification; CNN-LSTM and VideoMAE are separate runs",
    },
    "shooting": {
        "label": "Shooting",
        "kind": "object_detection",
        "results_config": "SHOOTING_RESULTS_DIR",
        "method": "Saved YOLO detections; tracker provenance depends on each run's recorded metadata",
    },
    "normal": {
        "label": "Normal",
        "kind": "video_classification",
        "results_config": "NORMAL_RESULTS_DIR",
        "method": "Saved normal-video artifacts when configured; application analyses are counted separately from source inventory",
    },
}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
STATUS_WORDS = {"complete", "completed", "success", "successful"}
_CATALOG_CACHE: dict[tuple, dict] = {}


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_. -]", "", value).strip()


def _number(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        result = float(value)
        return result if result == result and abs(result) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _pick(row: dict, *names):
    lowered = {str(key).strip().lower(): value for key, value in row.items() if key is not None}
    for name in names:
        if name in lowered and lowered[name] not in (None, ""):
            return lowered[name]
    return None


def _normalize_record(row: dict, category: str) -> dict:
    timestamp = _number(_pick(row, "timestamp_seconds", "timestamp_sec", "time_seconds", "timestamp", "start_timestamp_seconds"))
    confidence = _number(_pick(row, "confidence", "score", "probability", "prediction_confidence"))
    class_name = _pick(row, "class_name", "class", "predicted_class", "prediction", "predicted_label", "label")
    frame = _number(_pick(row, "frame", "frame_number", "frame_id"))
    class_id = _number(_pick(row, "class_id", "class_index"))
    track_id = _pick(row, "track_id", "tracker_id", "object_id")
    bbox_values = [
        _number(_pick(row, "x1", "xmin", "left")),
        _number(_pick(row, "y1", "ymin", "top")),
        _number(_pick(row, "x2", "xmax", "right")),
        _number(_pick(row, "y2", "ymax", "bottom")),
    ]
    bbox = dict(zip(("x1", "y1", "x2", "y2"), bbox_values)) if all(v is not None for v in bbox_values) else None
    probabilities = _pick(row, "probabilities", "class_probabilities", "scores")
    if isinstance(probabilities, str):
        try:
            probabilities = json.loads(probabilities)
        except json.JSONDecodeError:
            pass
    return {
        "record_type": "activity_prediction" if category == "robbery" else "object_detection",
        "timestamp_seconds": timestamp,
        "frame": int(frame) if frame is not None else None,
        "class_name": str(class_name) if class_name is not None else None,
        "class_id": int(class_id) if class_id is not None else None,
        "confidence": confidence,
        "bbox": bbox,
        "track_id": str(track_id) if track_id not in (None, "") else None,
        "probabilities": probabilities if isinstance(probabilities, (dict, list)) else None,
        "raw": {str(k): v for k, v in row.items() if k is not None},
    }


def _dedupe_records(records: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for record in records:
        key = json.dumps(record.get("raw", record), sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


class EvidenceCatalogService:
    """Resolve configured evidence directories and expose verified artifacts."""

    def __init__(self, config):
        self.config = config
        self.source_root = Path(config.get("EVIDENCE_SOURCE_DIR") or "").expanduser()

    def _results_root(self, category: str) -> Path:
        configured = self.config.get(CATEGORIES[category]["results_config"])
        return (Path(configured).expanduser() if configured else
                Path(__file__).resolve().parent / ".unconfigured-results" / category)

    def _source_dir(self, category: str) -> Path:
        return self.source_root / CATEGORIES[category]["label"]

    def _txt_root(self) -> Path:
        configured = self.config.get("IMPORTED_TRACKING_RESULTS_DIR")
        return Path(configured).expanduser() if configured else Path(__file__).resolve().parents[2] / "tracked_results_drive_import" / "crime_tracking_results" / "bytetrack"

    def _shooting_status_records(self) -> dict[str, dict]:
        configured = self.config.get("SHOOTING_TRACKING_STATUS_PATH")
        default = Path(__file__).resolve().parents[2] / "tracked_results_shooting_import" / "main_archive" / "Shooting" / "shooting_tracking_status.csv"
        manifest = Path(configured).expanduser() if configured else default
        statuses = {}
        if not manifest.is_file():
            return statuses
        try:
            with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    video_name = _pick(row, "video", "video_name", "filename", "source_video")
                    if video_name:
                        statuses[self._normalized_stem(str(video_name))] = {
                            **row, "manifest_path": str(manifest),
                        }
        except (OSError, UnicodeError, csv.Error):
            return {}
        return statuses

    @staticmethod
    def _detection_csv_stem(path: Path) -> str | None:
        match = re.fullmatch(r"(.+)_detections(?:_.*)?", path.stem, flags=re.IGNORECASE)
        return match.group(1) if match else None

    def _shooting_csv_versions(self, root: Path, stem_key: str) -> list[Path]:
        if not root.is_dir():
            return []
        versions = []
        for path in root.glob("*.csv"):
            candidate_stem = self._detection_csv_stem(path)
            if candidate_stem and self._normalized_stem(candidate_stem) == stem_key:
                versions.append(path)
        return sorted(versions, key=lambda item: item.name.casefold())

    def tracking_artifact_inventory(self) -> dict:
        """Read-only inventory of tracking CSVs and imported label artifacts."""
        artifacts = []
        roots = [("csv", self._results_root(category), category)
                 for category in CATEGORIES if category != "shooting"]
        roots.append(("txt", self._txt_root(), "assault"))
        for kind, root, category in roots:
            if not root.is_dir():
                continue
            paths = root.rglob("*.csv") if kind == "csv" else root.rglob("*.txt")
            for path in paths:
                if kind == "csv" and path.name.casefold() != "detections.csv" and not self._detection_csv_stem(path):
                    continue
                stem = (path.parent.parent.name if kind == "txt" else
                        path.parent.name if path.name.casefold() == "detections.csv" else
                        (self._detection_csv_stem(path) or path.stem))
                record = {"category": category, "video_stem": stem, "artifact_type": "csv_tracking" if kind == "csv" else "txt_labels",
                          "root": root.name,
                          "path": str(path.relative_to(root)), "filename": path.name, "detection_count": 0,
                          "tracking_id_count": 0, "frame_min": None, "frame_max": None}
                try:
                    if kind == "csv":
                        with path.open("r", newline="", encoding="utf-8-sig") as stream:
                            reader = csv.DictReader(stream)
                            frame_values, ids = [], set()
                            for row in reader:
                                record["detection_count"] += 1
                                frame = _number(_pick(row, "frame", "frame_number", "frame_id"))
                                track_id = _pick(row, "track_id", "tracker_id", "object_id")
                                if frame is not None: frame_values.append(int(frame))
                                if self._valid_track_id(track_id): ids.add(str(int(float(track_id))))
                            record["tracking_id_count"] = len(ids)
                            record["frame_min"] = min(frame_values) if frame_values else None
                            record["frame_max"] = max(frame_values) if frame_values else None
                    else:
                        frame_values, ids = [], set()
                        match = re.fullmatch(re.escape(path.parent.parent.name) + r"_(\d+)", path.stem, flags=re.IGNORECASE)
                        if match: frame_values.append(int(match.group(1)))
                        for line in path.read_text(encoding="utf-8-sig").splitlines():
                            fields = line.split()
                            if len(fields) in (5, 6):
                                record["detection_count"] += 1
                                if len(fields) == 6 and self._valid_track_id(fields[5]): ids.add(str(int(float(fields[5]))))
                        record["tracking_id_count"] = len(ids)
                        record["frame_min"] = record["frame_max"] = frame_values[0] if frame_values else None
                except (OSError, UnicodeError, csv.Error):
                    record["inventory_error"] = "artifact_unreadable"
                artifacts.append(record)

        shooting_root = self._results_root("shooting")
        shooting_status = self._shooting_status_records()
        shooting_files = [path for path in shooting_root.glob("*.csv")
                          if self._detection_csv_stem(path)] if shooting_root.is_dir() else []
        grouped = {}
        for path in shooting_files:
            stem = self._detection_csv_stem(path)
            grouped.setdefault(self._normalized_stem(stem), []).append(path)
        for stem_key, versions in sorted(grouped.items()):
            status = shooting_status.get(stem_key, {})
            expected_name = Path(str(_pick(status, "output_csv", "csv_path") or "")).name
            selected = next((path for path in versions if expected_name and
                             path.name.casefold() == expected_name.casefold()), None)
            job_status = str(status.get("status") or "unknown").casefold()
            reusable = job_status in STATUS_WORDS and selected is not None
            selected = selected or versions[0]
            record = {"category": "shooting", "video_stem": self._detection_csv_stem(selected),
                      "artifact_type": "csv_tracking", "root": shooting_root.name,
                      "path": str(selected.relative_to(shooting_root)), "filename": selected.name,
                      "status": job_status, "automatic_reuse": reusable,
                      "manifest_video": status.get("video"), "expected_csv_filename": expected_name or None,
                      "detection_count": 0, "tracking_id_count": 0,
                      "frame_min": None, "frame_max": None,
                      "duplicate_version_count": max(0, len(versions) - 1),
                      "artifact_versions": [str(path.relative_to(shooting_root)) for path in versions]}
            try:
                with selected.open("r", newline="", encoding="utf-8-sig") as stream:
                    reader = csv.DictReader(stream)
                    frames, ids = [], set()
                    for row in reader:
                        record["detection_count"] += 1
                        frame = _number(_pick(row, "frame", "frame_number", "frame_id"))
                        track_id = _pick(row, "track_id", "tracker_id", "object_id")
                        if frame is not None:
                            frames.append(int(frame))
                        if self._valid_track_id(track_id):
                            ids.add(str(int(float(track_id))))
                    record["tracking_id_count"] = len(ids)
                    record["frame_min"] = min(frames) if frames else None
                    record["frame_max"] = max(frames) if frames else None
            except (OSError, UnicodeError, csv.Error):
                record["inventory_error"] = "artifact_unreadable"
                record["automatic_reuse"] = False
            artifacts.append(record)

        archive_root = shooting_root.parents[2] if len(shooting_root.parents) > 2 else shooting_root.parent
        evaluation_dirs = sorted(archive_root.glob("evaluation_*/shooting_evaluation"))
        evaluation_names = [{path.name.casefold() for path in directory.glob("*_detections.csv")}
                            for directory in evaluation_dirs]
        shared_evaluation_names = len(set.intersection(*evaluation_names)) if len(evaluation_names) > 1 else 0
        completed_jobs = sum(str(item.get("status", "")).casefold() in STATUS_WORDS
                             for item in shooting_status.values())
        incomplete_jobs = sum(bool(item) and str(item.get("status", "")).casefold() not in STATUS_WORDS
                              for item in shooting_status.values())
        aggregate_csv_count = sum(path.is_file() for path in shooting_root.parent.glob("shooting_all_timestamped_detections.csv")) if shooting_root.parent.is_dir() else 0
        shooting_summary = {"detection_csv_file_count": len(shooting_files),
                            "aggregate_detection_csv_file_count": aggregate_csv_count,
                            "total_detection_csv_file_count": len(shooting_files) + aggregate_csv_count,
                            "logical_video_count": len(grouped),
                            "duplicate_version_file_count": sum(max(0, len(items) - 1) for items in grouped.values()),
                            "completed_job_count": completed_jobs,
                            "incomplete_job_count": incomplete_jobs,
                            "evaluation_archive_count": len(evaluation_dirs),
                            "shared_evaluation_csv_filename_count": shared_evaluation_names,
                            "evaluation_archives_used_for_automatic_reuse": False}
        return {"read_only": True,
                "shooting_covered": any(item["category"] == "shooting" and item.get("automatic_reuse") and item.get("detection_count", 0) > 0 for item in artifacts),
                "artifact_count": len(artifacts), "artifacts": artifacts,
                "shooting_inventory": shooting_summary}

    def match_tracking_artifact(self, category: str, original_filename: str, upload_path: str, video_metadata: dict) -> dict:
        """Match exact category/stem candidates, then require source hash and timeline checks."""
        category = str(category or "").casefold()
        if category not in CATEGORIES:
            return self._match_response("NO_MATCH", "Choose a supported category.")
        stem_key = self._normalized_stem(original_filename)
        if not stem_key:
            return self._match_response("NO_MATCH", "Original filename is unavailable.")
        shooting_status = None
        shooting_csv_path = None
        if category == "shooting":
            shooting_status = self._shooting_status_records().get(stem_key)
            if shooting_status is None:
                return self._match_response("UNVERIFIED_MATCH", "No Shooting tracking status-manifest row matches this original filename; saved tracking data was not reused.")
            job_status = str(shooting_status.get("status", "")).strip().casefold()
            if job_status not in STATUS_WORDS:
                return {**self._match_response("INVALID_TRACKING_DATA", f"Shooting tracking job status is '{job_status or 'missing'}'; incomplete results are excluded from automatic reuse."),
                        "job_status": job_status or None, "manifest_video": shooting_status.get("video")}
            expected_csv = Path(str(_pick(shooting_status, "output_csv", "csv_path") or "")).name
            versions = self._shooting_csv_versions(self._results_root("shooting"), stem_key)
            canonical = [path for path in versions if path.name.casefold() == expected_csv.casefold()]
            if len(canonical) != 1:
                return {**self._match_response("INVALID_TRACKING_DATA", "The completed Shooting status manifest's canonical output CSV is missing or ambiguous; no duplicate version was selected."),
                        "job_status": job_status, "expected_csv_filename": expected_csv or None,
                        "artifact_version_count": len(versions)}
            shooting_csv_path = canonical[0]
        source_dir = self._source_dir(category)
        sources = [p for p in source_dir.iterdir() if p.is_file() and p.suffix.casefold() in VIDEO_EXTENSIONS and self._normalized_stem(p.name) == stem_key] if source_dir.is_dir() else []
        upload = Path(upload_path)
        if not upload.is_file():
            return self._match_response("UNVERIFIED_MATCH", "Uploaded source is unavailable for verification.")
        try:
            digest = self._sha256(upload)
            identical = [p for p in sources if self._sha256(p) == digest]
        except OSError:
            return self._match_response("UNVERIFIED_MATCH", "Could not hash the uploaded source and local original.")
        if not identical:
            return self._match_response("UNVERIFIED_MATCH", "No byte-identical local source with this category and normalized filename was found; artifacts were not reused.")
        if len(identical) > 1:
            return {**self._match_response("UNVERIFIED_MATCH", "Multiple byte-identical local source candidates share this normalized filename; no artifact was selected."), "ambiguity": True, "candidate_count": len(identical)}
        source = identical[0]
        csv_matches = []
        root = self._results_root(category)
        if category == "shooting":
            csv_matches = [shooting_csv_path] if shooting_csv_path else []
        elif root.is_dir():
            for p in root.rglob("detections.csv"):
                if self._normalized_stem(p.parent.name) == stem_key:
                    csv_matches.append(p)
        txt_matches = []
        if category == "assault" and self._txt_root().is_dir():
            txt_matches = [p for p in self._txt_root().glob("*/labels/*.txt") if self._normalized_stem(p.parent.parent.name) == stem_key]
        candidate_groups = ([ ("csv_tracking", csv_matches) ] if csv_matches else []) + ([ ("txt_labels", [p.parent for p in txt_matches]) ] if txt_matches else [])
        candidates = [(kind, path) for kind, paths in candidate_groups for path in paths]
        if not candidates:
            return self._match_response("NO_MATCH", "Verified original has no matching local tracking artifact.")
        # Each label directory is one artifact; duplicate result sets are intentionally ambiguous.
        unique = {(kind, str(path.resolve())): (kind, path) for kind, path in candidates}
        if len(unique) != 1:
            return {**self._match_response("UNVERIFIED_MATCH", f"Multiple matching artifacts ({len(unique)}) were found; none was selected."), "ambiguity": True, "candidate_count": len(unique)}
        artifact_type, artifact_path = next(iter(unique.values()))
        if artifact_type == "csv_tracking":
            checked = self.match_tracking_csv(category, original_filename, upload_path, video_metadata,
                                               csv_path_override=artifact_path)
            if checked.get("status") != "verified_match":
                status = "INVALID_TRACKING_DATA" if "invalid" in checked.get("status", "") or "mismatch" in checked.get("status", "") else "UNVERIFIED_MATCH"
                return {**self._match_response(status, checked.get("reason", "CSV verification failed.")), "artifact_type": artifact_type}
            if category == "shooting":
                manifest_frames = _number(shooting_status.get("frames_total"))
                manifest_rows = _number(_pick(shooting_status, "detection_rows", "detections"))
                processed_frames = _number(shooting_status.get("frames_processed"))
                if manifest_frames is not None and int(manifest_frames) != int(video_metadata.get("total_frames") or 0):
                    return {**self._match_response("INVALID_TRACKING_DATA", "Shooting status-manifest frame count does not match the uploaded original."),
                            "artifact_type": artifact_type, "job_status": "completed"}
                if processed_frames is not None and processed_frames <= 0:
                    return {**self._match_response("INVALID_TRACKING_DATA", "Shooting status manifest reports no processed source frames."),
                            "artifact_type": artifact_type, "job_status": "completed"}
                if manifest_rows is not None and int(manifest_rows) != int(checked.get("record_count", 0)):
                    return {**self._match_response("INVALID_TRACKING_DATA", "Shooting status-manifest detection count does not match its canonical CSV."),
                            "artifact_type": artifact_type, "job_status": "completed"}
                if not checked.get("record_count"):
                    return {**self._match_response("INVALID_TRACKING_DATA", "The completed Shooting job has no detections to reuse."),
                            "artifact_type": artifact_type, "job_status": "completed"}
            return {**checked, "status": "EXACT_MATCH", "artifact_type": artifact_type, "match_status": "EXACT_MATCH",
                    "detection_count": checked["record_count"],
                    "manifest_video": shooting_status.get("video") if shooting_status else None,
                    "job_status": shooting_status.get("status") if shooting_status else None,
                    "artifact_filename": artifact_path.name,
                    "artifact_version_count": len(self._shooting_csv_versions(root, stem_key)) if category == "shooting" else 1,
                    "metadata_verification": {"source_sha256": True, "fps": True, "frame_count": True, "duration": True, "frame_index_mapping": True}}
        rows, frame_numbers, track_ids = [], [], set()
        width, height = int(video_metadata.get("width") or 0), int(video_metadata.get("height") or 0)
        total_frames = int(video_metadata.get("total_frames") or 0)
        fps = _number(video_metadata.get("fps"))
        if width <= 0 or height <= 0 or total_frames <= 0 or fps is None or fps <= 0:
            return self._match_response("INVALID_TRACKING_DATA", "TXT labels require valid source dimensions, frame count, and FPS.")
        files_by_frame = {}
        try:
            for path in txt_matches:
                m = re.fullmatch(re.escape(path.parent.parent.name) + r"_(\d+)", path.stem, flags=re.IGNORECASE)
                if not m:
                    return self._match_response("INVALID_TRACKING_DATA", "TXT frame filename does not follow the verified video-stem_frame-index convention.")
                frame = int(m.group(1))
                if frame < 0 or frame >= total_frames:
                    return self._match_response("INVALID_TRACKING_DATA", "TXT frame indices do not fit the exact source video's frame count.")
                if frame in files_by_frame:
                    return self._match_response("UNVERIFIED_MATCH", "Duplicate TXT files map to the same frame; no labels were selected.")
                files_by_frame[frame] = path
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    fields = line.split()
                    if len(fields) not in (5, 6):
                        return self._match_response("INVALID_TRACKING_DATA", "TXT row is not a 5-field detection or 6-field detection with an ID.")
                    cls, cx, cy, bw, bh = map(float, fields[:5])
                    tolerance = 1e-5
                    if not all(math.isfinite(v) for v in (cls, cx, cy, bw, bh)) or cls < 0 or not cls.is_integer() or cls >= len(YOLOService.DETECTOR_CLASS_NAMES) or min(cx, cy, bw, bh) < 0 or max(cx, cy, bw, bh) > 1 or bw == 0 or bh == 0 or cx-bw/2 < -tolerance or cy-bh/2 < -tolerance or cx+bw/2 > 1+tolerance or cy+bh/2 > 1+tolerance:
                        return self._match_response("INVALID_TRACKING_DATA", "TXT normalized box values are invalid.")
                    track_id = None
                    if len(fields) == 6:
                        track_value = float(fields[5])
                        if not math.isfinite(track_value) or track_value < 0 or not track_value.is_integer():
                            return self._match_response("INVALID_TRACKING_DATA", "Six-column TXT row has an invalid tracking ID.")
                        track_id = int(track_value); track_ids.add(track_id)
                    x1, y1, x2, y2 = ((cx-bw/2)*width, (cy-bh/2)*height,
                                      (cx+bw/2)*width, (cy+bh/2)*height)
                    class_name = YOLOService.class_name_for_id(int(cls))
                    if class_name is None:
                        return self._match_response("INVALID_TRACKING_DATA", "TXT row class ID is not in the trained detector mapping.")
                    forensic_category = YOLOService._map_to_forensic_category(class_name)
                    if forensic_category == "Others":
                        forensic_category = "Other"
                    rows.append({"frame": frame, "timestamp_seconds": frame / fps,
                                 "class": class_name, "class_name": class_name,
                                 "class_id": int(cls), "confidence": None, "track_id": track_id,
                                 "category": forensic_category,
                                 "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        except (OSError, UnicodeError, ValueError, OverflowError, KeyError):
            return self._match_response("INVALID_TRACKING_DATA", "TXT artifact could not be parsed or mapped to source metadata.")
        if not files_by_frame or not rows:
            return self._match_response("INVALID_TRACKING_DATA", "TXT artifact contains no usable detections or source frame mapping.")
        # Filename shape and in-range indices are insufficient to prove that
        # IDs refer to decoded source frames; producer provenance is absent.
        return {"status": "UNVERIFIED_MATCH", "match_status": "UNVERIFIED_MATCH",
                "reason": "Source bytes and label format match, but TXT frame-index provenance is unavailable; saved labels are not automatically reused.",
                "artifact_type": "txt_labels", "artifact_path": str(artifact_path), "artifact_filename": f"{artifact_path.parent.parent.name}/labels", "artifact_file_count": len(txt_matches), "source_path": str(source), "source_sha256": digest,
                "record_count": len(rows), "detection_count": len(rows), "tracked_record_count": sum(r["track_id"] is not None for r in rows), "tracking_id_count": len(track_ids),
                "frame_min": min(files_by_frame), "frame_max": max(files_by_frame), "frame_coverage_count": len(files_by_frame),
                "records": rows, "metadata_verification": {"source_sha256": True, "fps": True, "frame_count": True, "duration": bool(video_metadata.get("duration_sec")), "frame_index_mapping": False},
                "frame_index_mapping_status": "unverified",
                "frame_index_convention": "<normalized-video-stem>_<assumed-frame-index>.txt"}

    @staticmethod
    def _match_response(status, reason):
        return {"status": status, "match_status": status, "reason": reason, "artifact_type": None, "record_count": 0, "detection_count": 0, "tracked_record_count": 0, "tracking_id_count": 0, "frame_coverage_count": 0, "metadata_verification": {"source_sha256": False, "fps": False, "frame_count": False, "duration": False, "frame_index_mapping": False}}

    @staticmethod
    def _normalized_stem(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", Path(str(value)).stem.casefold())

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def match_tracking_csv(self, category: str, original_filename: str, upload_path: str,
                           video_metadata: dict, csv_path_override: Path | None = None) -> dict:
        """Return a tracking CSV only after exact source and timeline validation.

        A filename is used to find candidates, never as proof of identity. The
        uploaded original must hash-identically to a local source under the
        selected category, and the CSV frame/timestamp data must fit its media
        metadata. Annotated render files are deliberately not consulted.
        """
        if category not in CATEGORIES:
            return {"status": "category_required", "reason": "Choose Assault, Robbery, or Shooting to find matching tracking results."}
        upload = Path(upload_path)
        stem_key = self._normalized_stem(original_filename)
        if not stem_key or not upload.is_file():
            return {"status": "source_unavailable", "reason": "The uploaded original or its filename is unavailable."}

        source_dir = self._source_dir(category)
        try:
            source_candidates = [path for path in source_dir.iterdir()
                                 if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
                                 and self._normalized_stem(path.name) == stem_key] if source_dir.is_dir() else []
        except OSError as error:
            return {"status": "source_verification_failed", "reason": f"The local source category directory is unavailable ({type(error).__name__}); no tracking data was reused."}
        try:
            upload_hash = self._sha256(upload)
            identical_sources = [path for path in source_candidates if self._sha256(path) == upload_hash]
        except OSError as error:
            return {"status": "source_verification_failed", "reason": f"Could not verify the uploaded original against local category sources ({type(error).__name__})."}
        if not identical_sources:
            return {"status": "source_not_verified", "reason": "No byte-identical local source video with this category and filename was found; tracking data was not reused."}

        roots = [self._results_root(category)]
        roots = [root for root in roots if root.is_dir()]
        csv_candidates = []
        try:
            if csv_path_override is not None:
                path = Path(csv_path_override)
                if roots and path.is_file() and path.resolve().is_relative_to(roots[0].resolve()):
                    csv_candidates = [path]
            else:
                for root in roots:
                    for path in root.rglob("*.csv"):
                        if path.name.casefold() == "detections.csv":
                            candidate_stem = path.parent.name
                        else:
                            candidate_stem = self._detection_csv_stem(path)
                        if candidate_stem and self._normalized_stem(candidate_stem) == stem_key:
                            csv_candidates.append(path)
        except OSError as error:
            return {"status": "tracking_directory_unavailable", "reason": f"The tracking-results directory could not be read ({type(error).__name__}); no tracking data was reused."}
        if not csv_candidates:
            return {"status": "tracking_data_missing", "reason": "The original was verified, but no matching detections CSV exists in this category's configured tracking-results directory."}
        if len(csv_candidates) != 1:
            return {"status": "tracking_match_ambiguous", "reason": f"Found {len(csv_candidates)} matching tracking CSV files; no result was selected."}

        csv_path = csv_candidates[0]
        try:
            with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                headers = {str(name).strip().casefold() for name in (reader.fieldnames or [])}
        except (OSError, csv.Error, UnicodeError) as error:
            return {"status": "tracking_csv_invalid", "reason": f"Could not read the matching tracking CSV ({type(error).__name__})."}
        frame_column = next((name for name in ("frame", "frame_number", "frame_id") if name in headers), None)
        class_column = next((name for name in ("class", "class_name", "label") if name in headers), None)
        confidence_column = next((name for name in ("confidence", "score") if name in headers), None)
        box_columns = all(name in headers for name in ("x1", "y1", "x2", "y2"))
        if not rows or not frame_column or not class_column or not confidence_column or not box_columns:
            return {"status": "tracking_csv_invalid", "reason": "The CSV is empty or lacks frame, class, confidence, and x1/y1/x2/y2 fields."}

        frames, timestamp_pairs, valid_rows = [], [], 0
        fps = float(video_metadata.get("fps") or 0)
        frame_count = int(video_metadata.get("total_frames") or 0)
        timestamp_column = next((name for name in ("timestamp_seconds", "timestamp_sec", "time_seconds", "timestamp") if name in headers), None)
        presentation_times = video_metadata.get("presentation_timestamps")
        if not isinstance(presentation_times, dict):
            presentation_times = {}
        for row in rows:
            normalized_row = {str(key).strip().casefold(): value for key, value in row.items() if key is not None}
            try:
                frame_index = int(float(normalized_row.get(frame_column, "")))
                confidence = float(normalized_row.get(confidence_column, ""))
                box = [float(normalized_row.get(name, "")) for name in ("x1", "y1", "x2", "y2")]
                if frame_index < 0 or frame_index > frame_count or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("out-of-range frame or confidence")
                if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError("invalid bounding box")
                width, height = int(video_metadata.get("width") or 0), int(video_metadata.get("height") or 0)
                if width and height and (box[0] < 0 or box[1] < 0 or box[2] > width + 2 or box[3] > height + 2):
                    raise ValueError("bounding box is outside source frame dimensions")
                frames.append(frame_index)
                valid_rows += 1
                if timestamp_column and (fps > 0 or presentation_times) and str(normalized_row.get(timestamp_column, "")).strip():
                    csv_time = float(normalized_row[timestamp_column])
                    if not math.isfinite(csv_time) or csv_time < 0:
                        raise ValueError("invalid timestamp")
                    timestamp_pairs.append((frame_index, csv_time))
            except (TypeError, ValueError, OverflowError):
                return {"status": "tracking_csv_invalid", "reason": "The CSV contains an invalid frame index, confidence, timestamp, or bounding box."}
        if not frames or max(frames) > frame_count:
            return {"status": "tracking_metadata_mismatch", "reason": "CSV frame indices do not fit the uploaded video's frame count."}
        offsets = [0]
        pts_complete = bool(presentation_times) and any(
            all(presentation_times.get(str(frame - offset), presentation_times.get(frame - offset)) is not None
                for frame, _ in timestamp_pairs)
            for offset in (0, 1)
        )
        if timestamp_pairs and pts_complete:
            scored_offsets = []
            for offset in (0, 1):
                errors = []
                for frame, timestamp in timestamp_pairs:
                    pts = presentation_times.get(str(frame - offset), presentation_times.get(frame - offset))
                    if pts is None:
                        errors = []
                        break
                    errors.append(abs(timestamp - float(pts)))
                if errors:
                    errors.sort()
                    p95 = errors[int((len(errors) - 1) * 0.95)]
                    if p95 <= max(0.08, 2.5 / fps if fps > 0 else 0.08):
                        scored_offsets.append((p95, offset))
            if not scored_offsets:
                return {"status": "tracking_metadata_mismatch", "reason": "CSV timestamps do not align with original frame presentation timestamps."}
            offsets = [min(scored_offsets)[1]]
        elif timestamp_pairs and fps > 0:
            scored_offsets = []
            for offset in (0, 1):
                if not all(frame - offset >= 0 for frame, _ in timestamp_pairs):
                    continue
                errors = sorted(abs(timestamp - (frame - offset) / fps) for frame, timestamp in timestamp_pairs)
                p95 = errors[int((len(errors) - 1) * 0.95)]
                if p95 <= max(0.08, 2.5 / fps): scored_offsets.append((p95, offset))
            if not scored_offsets:
                return {"status": "tracking_metadata_mismatch", "reason": "CSV timestamps do not align with the uploaded video's FPS and frame indices."}
            offsets = [min(scored_offsets)[1]]
        frame_offset = offsets[0] if timestamp_pairs else (1 if max(frames) == frame_count else 0)

        return {
            "status": "verified_match", "reason": "Uploaded video hash, selected category, normalized filename, frame indices, and available timestamps match.",
            "csv_path": str(csv_path), "source_path": str(identical_sources[0]),
            "source_sha256": upload_hash, "record_count": valid_rows,
            "frame_min": min(frames) - frame_offset, "frame_max": max(frames) - frame_offset,
            "frame_offset": frame_offset, "frame_coverage_count": len(set(frames)),
            "detection_count": valid_rows,
            "tracked_record_count": sum(1 for row in rows if self._valid_track_id(_pick(row, "track_id", "tracker_id", "object_id"))),
            "tracking_id_count": len({str(int(float(_pick(row, "track_id", "tracker_id", "object_id")))) for row in rows if self._valid_track_id(_pick(row, "track_id", "tracker_id", "object_id"))}),
        }

    @staticmethod
    def _valid_track_id(value):
        try:
            numeric = float(value)
            return math.isfinite(numeric) and numeric >= 0 and numeric.is_integer()
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _key(name: str) -> str:
        return Path(name).stem.casefold()

    @staticmethod
    def _read_csv(path: Path) -> list[dict]:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    def _status_records(self, root: Path) -> dict[str, dict]:
        statuses = {}
        if not root.is_dir():
            return statuses
        for path in sorted(root.glob("*status*.csv")):
            try:
                for row in self._read_csv(path):
                    name = _pick(row, "video_name", "video", "filename", "source_video")
                    if name:
                        statuses[self._key(str(name))] = {"source_file": path.name, **row}
            except (OSError, csv.Error, UnicodeError):
                continue
        return statuses

    def _artifact_sets(self, category: str) -> dict[str, list[dict]]:
        root = self._results_root(category)
        sets: dict[str, list[dict]] = {}
        if not root.is_dir():
            return sets

        # Existing Assault/ByteTrack output layout: one folder per source video.
        for path in root.rglob("detections.csv"):
            try:
                rows = self._read_csv(path)
            except (OSError, csv.Error, UnicodeError):
                continue
            video_name = str(_pick(rows[0], "video_name", "video", "filename") or f"{path.parent.name}.mp4") if rows else f"{path.parent.name}.mp4"
            if not Path(video_name).suffix:
                video_name += ".mp4"
            annotated_candidates = list(path.parent.glob("*_tracked.mp4")) + list(path.parent.glob("*_recovered_from_csv.mp4"))
            annotated = next((candidate for candidate in annotated_candidates if candidate.is_file() and candidate.stat().st_size > 0), None)
            events_path = path.parent / "potential_events.csv"
            event_rows = []
            if events_path.is_file():
                try:
                    event_rows = self._read_csv(events_path)
                except (OSError, csv.Error, UnicodeError):
                    pass
            artifact = {
                "artifact_id": f"csv:{path.relative_to(root).as_posix()}",
                "name": "Saved tracked detections",
                "source_file": path.name,
                "result_type": CATEGORIES[category]["kind"],
                "status": "results_available_status_unverified",
                "records": _dedupe_records([_normalize_record(row, category) for row in rows]),
                "events": event_rows,
                "annotated_available": annotated is not None,
                "_csv_path": path,
                "_annotated_path": annotated,
            }
            sets.setdefault(self._key(video_name), []).append(artifact)

        # The Shooting runner stores per-video CSVs in per_video/ and videos in annotated/.
        per_video = root / "per_video"
        if per_video.is_dir():
            for path in per_video.glob("*_detections.csv"):
                try:
                    rows = self._read_csv(path)
                except (OSError, csv.Error, UnicodeError):
                    continue
                video_name = str(_pick(rows[0], "video_name", "video", "filename") or path.name.removesuffix("_detections.csv") + ".mp4") if rows else path.name.removesuffix("_detections.csv") + ".mp4"
                annotated = root / "annotated" / f"{Path(video_name).stem}_tracked.mp4"
                annotated = annotated if annotated.is_file() and annotated.stat().st_size > 0 else None
                sets.setdefault(self._key(video_name), []).append({
                    "artifact_id": f"csv:{path.relative_to(root).as_posix()}",
                    "name": "Saved detections and tracks",
                    "source_file": path.name,
                    "result_type": CATEGORIES[category]["kind"],
                    "status": "results_available_status_unverified",
                    "records": _dedupe_records([_normalize_record(row, category) for row in rows]),
                    "events": [],
                    "annotated_available": annotated is not None,
                    "_csv_path": path,
                    "_annotated_path": annotated,
                })

        # Robbery prediction files vary between experiments. Only read explicitly
        # named prediction/classification files, never evaluation summaries.
        if category == "robbery":
            for path in root.rglob("*.csv"):
                if not re.search(r"predict|classif", path.name, re.I) or "status" in path.name.lower():
                    continue
                try:
                    rows = self._read_csv(path)
                except (OSError, csv.Error, UnicodeError):
                    continue
                grouped: dict[str, list[dict]] = {}
                names = {}
                for row in rows:
                    name = _pick(row, "video_name", "video", "filename", "source_video")
                    if not name:
                        continue
                    name = str(name)
                    grouped.setdefault(self._key(name), []).append(_normalize_record(row, category))
                    names[self._key(name)] = name if Path(name).suffix else name + ".mp4"
                for key, records in grouped.items():
                    sets.setdefault(key, []).append({
                        "artifact_id": f"csv:{path.relative_to(root).as_posix()}",
                        "name": "Saved activity predictions",
                        "source_file": path.name,
                        "result_type": "activity_prediction",
                        "status": "results_available_status_unverified",
                        "records": _dedupe_records(records),
                        "events": [],
                        "annotated_available": False,
                        "_csv_path": path,
                        "_annotated_path": None,
                    })
        return sets

    def _legacy_assault_sets(self) -> dict[str, list[dict]]:
        path = Path(self.config.get("ASSAULT_LEGACY_JSON_PATH") or "")
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        result = {}
        if not isinstance(payload, list):
            return result
        for item in payload:
            name = item.get("video_name") if isinstance(item, dict) else None
            records = item.get("detections") if isinstance(item, dict) else None
            if not name or not isinstance(records, list):
                continue
            result.setdefault(self._key(str(name)), []).append({
                "artifact_id": "legacy:assault_77_video_detections.json",
                "name": "Legacy Assault detection export",
                "source_file": path.name,
                "result_type": "object_detection",
                "status": "results_available_status_unverified",
                "records": _dedupe_records([_normalize_record(row, "assault") for row in records if isinstance(row, dict)]),
                "events": [],
                "annotated_available": False,
                "_csv_path": None,
                "_annotated_path": None,
            })
        return result

    def _category_signature(self, category: str) -> tuple:
        roots = [self._source_dir(category), self._results_root(category)]
        files = []
        for root in roots:
            if root.is_file():
                candidates = (root,)
            elif root.is_dir():
                candidates = root.rglob("*")
            else:
                continue
            for path in candidates:
                try:
                    if path.is_file():
                        stat = path.stat()
                        files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
                except OSError:
                    continue
        return tuple(sorted(files))

    def _category_data(self, category: str) -> dict:
        signature = self._category_signature(category)
        key = (
            category,
            str(self._source_dir(category).resolve()),
            str(self._results_root(category).resolve()),
            signature,
        )
        cached = _CATALOG_CACHE.get(key)
        if cached is not None:
            return cached
        data = self._build_category_data(category)
        # Drop stale snapshots of this category while keeping other categories warm.
        for old_key in [item for item in _CATALOG_CACHE if item[0] == category]:
            _CATALOG_CACHE.pop(old_key, None)
        _CATALOG_CACHE[key] = data
        return data

    def _build_category_data(self, category: str) -> dict:
        root = self._results_root(category)
        source_dir = self._source_dir(category)
        artifacts = self._artifact_sets(category)
        if category == "assault":
            for key, legacy in self._legacy_assault_sets().items():
                artifacts.setdefault(key, []).extend(legacy)
        statuses = self._status_records(root)
        sources = {}
        if source_dir.is_dir():
            for path in sorted(source_dir.iterdir(), key=lambda item: item.name.casefold()):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                    sources[self._key(path.name)] = path

        records = {}
        for key in set(sources) | set(artifacts) | set(statuses):
            source = sources.get(key)
            result_sets = artifacts.get(key, [])
            status = statuses.get(key)
            if status:
                raw_status = str(_pick(status, "status", "processing_status", "result") or "unknown").strip().lower()
                if raw_status in STATUS_WORDS:
                    record_status = "completed_status_record" if result_sets else "complete_status_missing_or_unreadable_result"
                elif any(word in raw_status for word in ("fail", "error", "incomplete", "partial")):
                    record_status = raw_status
                else:
                    record_status = raw_status or "unknown"
            elif result_sets:
                record_status = "results_available_status_unverified"
            else:
                record_status = "no_saved_results_found"
            video_name = source.name if source else next((Path(s.get("_csv_path").parent.name).name + ".mp4" for s in result_sets if s.get("_csv_path") and s["_csv_path"].name == "detections.csv"), None)
            if not video_name and result_sets:
                csv_path = next((s.get("_csv_path") for s in result_sets if s.get("_csv_path")), None)
                video_name = csv_path.name.removesuffix("_detections.csv") + ".mp4" if csv_path and csv_path.name.endswith("_detections.csv") else csv_path.parent.name + ".mp4" if csv_path else key + ".mp4"
            if not video_name and status:
                video_name = str(_pick(status, "video_name", "video", "filename", "source_video") or key)
            if not video_name:
                video_name = key
            records[key] = {
                "video_id": _safe_id(video_name),
                "video_name": video_name,
                "source_available": source is not None,
                "result_status": record_status,
                "status_record": status,
                "artifact_set_count": len(result_sets),
                "record_count": sum(len(s.get("records", [])) for s in result_sets),
                "annotated_available": any(s.get("annotated_available") for s in result_sets),
            }
        completed_status_count = sum(
            str(item.get("status", "")).strip().lower() in STATUS_WORDS
            and bool(artifacts.get(key))
            for key, item in statuses.items()
        )
        if not root.is_dir():
            processing_status = "unavailable"
        elif completed_status_count and completed_status_count == len(statuses):
            processing_status = "completed"
        elif completed_status_count:
            processing_status = "partially_completed"
        elif statuses:
            processing_status = "failed_or_incomplete"
        elif artifacts:
            processing_status = "artifacts_without_status"
        else:
            processing_status = "not_started"
        artifact_video_keys = {key for key, values in artifacts.items() if values}
        completed_status_keys = {
            key for key, item in statuses.items()
            if str(_pick(item, "status", "processing_status", "result") or "").strip().casefold() in STATUS_WORDS
        }
        partial_status_keys = {
            key for key, item in statuses.items()
            if any(word in str(_pick(item, "status", "processing_status", "result") or "").strip().casefold()
                   for word in ("partial", "incomplete"))
        }
        failed_status_keys = {
            key for key, item in statuses.items()
            if any(word in str(_pick(item, "status", "processing_status", "result") or "").strip().casefold()
                   for word in ("fail", "error", "invalid"))
        }
        failed_status_keys.update(completed_status_keys - artifact_video_keys)
        source_keys = set(sources)
        unresolved_source_status_keys = source_keys - artifact_video_keys - set(statuses)
        # A source directory can be empty; a missing one is unknown rather than zero.
        source_inventory_available = source_dir.is_dir()
        artifact_video_count = len(artifact_video_keys) if root.is_dir() else None
        artifact_set_count = sum(int(item.get("artifact_set_count", 0) or 0) for item in records.values()) if root.is_dir() else None
        return {
            "category": category,
            "label": CATEGORIES[category]["label"],
            "result_type": CATEGORIES[category]["kind"],
            "method": CATEGORIES[category]["method"],
            "source_available": source_inventory_available,
            "source_inventory_status": "available" if source_inventory_available else "unavailable",
            "source_video_count": len(sources) if source_inventory_available else None,
            "results_directory_available": root.is_dir(),
            "videos_with_results": artifact_video_count,
            "videos_with_valid_catalog_artifacts": artifact_video_count,
            "videos_with_completed_status": len(completed_status_keys & artifact_video_keys) if root.is_dir() else None,
            "videos_with_partial_status": len(partial_status_keys) if root.is_dir() else None,
            "videos_with_failed_or_invalid_status": len(failed_status_keys) if root.is_dir() else None,
            "videos_with_completed_status_missing_artifacts": len(completed_status_keys - artifact_video_keys) if root.is_dir() else None,
            "videos_not_yet_processed": len(unresolved_source_status_keys) if source_inventory_available else None,
            "videos_with_unverified_processing_status": len(artifact_video_keys - set(statuses)) if root.is_dir() else None,
            "result_record_count": sum(sum(len(s.get("records", [])) for s in artifacts.get(key, [])) for key in records) if root.is_dir() else None,
            "artifact_set_count": artifact_set_count,
            "duplicate_artifact_version_count": max(0, artifact_set_count - artifact_video_count) if artifact_set_count is not None and artifact_video_count is not None else None,
            "status_record_count": len(statuses) if root.is_dir() else None,
            "processing_status": processing_status,
            "videos": sorted(records.values(), key=lambda item: item["video_name"].casefold()),
            "_artifacts": artifacts,
            "_statuses": statuses,
            "_source_paths": sources,
        }

    def list_categories(self) -> list[dict]:
        summaries = []
        for category in CATEGORIES:
            data = self._category_data(category)
            summaries.append({key: value for key, value in data.items() if not key.startswith("_") and key != "videos"})
        return summaries

    def list_videos(self, category: str) -> dict:
        data = self._category_data(category)
        return {key: value for key, value in data.items() if not key.startswith("_")}

    def video_detail(self, category: str, video_id: str) -> dict | None:
        if not _safe_id(video_id) or _safe_id(video_id) != video_id:
            return None
        data = self._category_data(category)
        key = self._key(video_id)
        summary = next((item for item in data["videos"] if self._key(item["video_name"]) == key), None)
        if not summary:
            return None
        result_sets = data["_artifacts"].get(key, [])
        public_sets = []
        for artifact in result_sets:
            public_sets.append({k: v for k, v in artifact.items() if not k.startswith("_")})
        return {
            **summary,
            "result_type": CATEGORIES[category]["kind"],
            "model_provenance": "Per-run model metadata was not present in the saved artifact.",
            "result_sets": public_sets,
        }

    def media_path(self, category: str, video_id: str, kind: str, artifact_id: str | None = None) -> Path | None:
        detail = self.video_detail(category, video_id)
        if not detail:
            return None
        data = self._category_data(category)
        key = self._key(video_id)
        if kind == "source":
            return data["_source_paths"].get(key)
        if kind != "annotated":
            return None
        for artifact in data["_artifacts"].get(key, []):
            if artifact_id and artifact["artifact_id"] != artifact_id:
                continue
            path = artifact.get("_annotated_path")
            if path and path.is_file() and path.stat().st_size > 0:
                return path
        return None

    def status(self, category: str) -> dict:
        data = self._category_data(category)
        return {"category": category, "status_record_count": len(data["_statuses"]), "videos": data["_statuses"]}

    def reports(self, category: str) -> dict:
        data = self._category_data(category)
        root = self._results_root(category)
        files = []
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md", ".txt"}:
                    continue
                if re.search(r"report|evaluation|summary|coverage|audit", path.name, re.I):
                    report_id = path.relative_to(root).as_posix()
                    files.append({
                        "report_id": report_id,
                        "name": path.name,
                        "size_bytes": path.stat().st_size,
                        "download_url": f"/api/evidence/{category}/reports/{report_id}",
                    })
        return {"category": category, "files": files}

    def report_file_path(self, category: str, report_id: str) -> Path | None:
        if not report_id or "\\" in report_id:
            return None
        root = self._results_root(category)
        candidate = (root / report_id).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return None
        if candidate.suffix.lower() not in {".csv", ".json", ".md", ".txt"} or not candidate.is_file():
            return None
        return candidate

    def report_payload(self, category: str, video_id: str, artifact_id: str) -> dict | None:
        detail = self.video_detail(category, video_id)
        if not detail:
            return None
        artifact = next((item for item in detail["result_sets"] if item["artifact_id"] == artifact_id), None)
        if not artifact:
            return None
        return {
            "case": {"case_id": "", "description": f"Saved {CATEGORIES[category]['label']} evidence for review"},
            "source_category": CATEGORIES[category]["label"],
            "evidence_kind": artifact["result_type"],
            "verification_status": "unverified_model_output",
            "human_review_required": True,
            "file_name": detail["video_name"],
            "artifact_name": artifact["name"],
            "processing_status": detail["result_status"],
            "results": artifact["records"],
            "event_candidates": artifact.get("events", []),
            "limitations": "Saved model outputs are investigative leads. Object detections and activity predictions do not establish criminal conduct.",
        }
