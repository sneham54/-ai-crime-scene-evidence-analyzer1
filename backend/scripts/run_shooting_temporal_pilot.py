"""Run one real-data temporal association pilot for Shooting, Assault or Robbery.

Reuses the production-default classifier and saved tracking outputs. Never runs
YOLO or ByteTrack.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.config import Config  # noqa: E402
from app.services.evidence_catalog_service import EvidenceCatalogService  # noqa: E402
from app.services.shooting_event_evidence_service import EventEvidenceService  # noqa: E402
from app.services.temporal_event_association_service import TemporalEventAssociationService  # noqa: E402
from app.services.video_classification_model import VideoClassificationModel  # noqa: E402


CATEGORIES = {"Shooting": "shooting", "Assault": "assault", "Robbery": "robbery"}


def config_dict():
    return {name: getattr(Config, name) for name in dir(Config) if name.isupper()}


def _local_path(value):
    return Path(value) if value else Path()


def manifest_records(category):
    """Normalize each category's recorded artifact format without guessing runs."""
    key = CATEGORIES[category]
    cfg = config_dict()
    source_root = Path(cfg["EVIDENCE_SOURCE_DIR"]) / category
    if key == "shooting":
        root = BACKEND / "tracked_results_shooting_import" / "main_archive" / "Shooting"
        with (root / "shooting_artifact_manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
            records = list(csv.DictReader(stream))
        with (root / "shooting_tracking_status.csv").open(newline="", encoding="utf-8-sig") as stream:
            statuses = {row.get("video"): row for row in csv.DictReader(stream)}
        return root, records, statuses

    results = Path(cfg["ASSAULT_RESULTS_DIR"] if key == "assault" else cfg["ROBBERY_RESULTS_DIR"])
    status_file = Path(cfg["ASSAULT_TRACKING_STATUS_PATH"] if key == "assault" else cfg["ROBBERY_TRACKING_STATUS_PATH"])
    if status_file.is_file():
        with status_file.open(newline="", encoding="utf-8-sig") as stream:
            statuses = list(csv.DictReader(stream))
    else:
        statuses = []

    if key == "assault":
        records = []
        # Assault has no status CSV. The evidence service only selects a unique
        # exact source-stem folder and its saved CSV/render pair.
        for original in sorted(source_root.glob("*.mp4")):
            stem = original.stem
            artifact_dir = results / stem
            csv_path = artifact_dir / "detections.csv"
            render_candidates = [p for p in artifact_dir.glob("*.mp4") if p.stem.casefold() in
                                 {f"{stem}_tracked".casefold(), "annotated", "annotated_video"}]
            records.append({"original_video": original.name, "original_video_local": str(original),
                "tracking_status": "completed" if csv_path.is_file() and len(render_candidates) == 1 else "missing_or_ambiguous",
                "detection_csv_expected": str(csv_path), "detection_csv_local": str(csv_path),
                "tracked_video_expected": str(render_candidates[0]) if len(render_candidates) == 1 else "",
                "tracked_video_local": str(render_candidates[0]) if len(render_candidates) == 1 else "",
                "artifact_selection": "exact_source_stem_folder"})
        return results, records, {}

    normalized, status_map = [], {}
    for status in statuses:
        lowered = {str(k).strip().casefold(): v for k, v in status.items() if k is not None}
        name = Path(lowered.get("video") or lowered.get("video_name") or lowered.get("filename") or "").name
        if not name:
            continue
        csv_ref = lowered.get("output_csv") or lowered.get("detections_csv") or ""
        video_ref = lowered.get("output_video") or lowered.get("annotated_video") or ""
        status_map[name] = {**status, "video": name, "status": lowered.get("status", ""),
                            "output_csv": csv_ref, "output_video": video_ref}
        normalized.append({"original_video": name, "original_video_local": str(source_root / name),
            "tracking_status": lowered.get("status", ""), "detection_csv_expected": csv_ref,
            "detection_csv_local": csv_ref, "tracked_video_expected": video_ref,
            "tracked_video_local": video_ref, "artifact_selection": "status_record_exact_paths"})
    return results, normalized, status_map


def validate_record(record, status, category, catalog, evidence):
    name = record["original_video"]
    original = _local_path(record.get("original_video_local"))
    tracking_csv = _local_path(record.get("detection_csv_local"))
    tracked_video = _local_path(record.get("tracked_video_local"))
    if record.get("tracking_status", "").casefold() not in {"completed", "complete", "success", "successful"}:
        raise ValueError(f"tracking artifact status is {record.get('tracking_status') or 'missing'}")

    if category != "Assault":
        if not status or status.get("status", "").casefold() not in {"completed", "complete", "success", "successful"}:
            raise ValueError(f"tracking status is {status.get('status') if status else 'missing'}")
        for key, selected_path, label in (("output_csv", tracking_csv, "CSV"),
                                          ("output_video", tracked_video, "annotated video")):
            reference = Path(status.get(key, ""))
            if category == "Robbery":
                matches = reference.resolve() == selected_path.resolve()
            else:
                # The Shooting status CSV contains the original Drive path;
                # its companion artifact manifest records the exact local copy.
                matches = reference.name.casefold() == selected_path.name.casefold()
            if not matches:
                raise ValueError(f"status {label} reference does not match selected artifact")

    if not original.is_file():
        raise FileNotFoundError("original video is missing")
    if not tracking_csv.is_file() or not tracked_video.is_file():
        raise FileNotFoundError("manifest/status referenced CSV or annotated video is missing")

    info = evidence._metadata(original)
    metadata = {**info, "total_frames": info["frames"], "duration_sec": info["frames"] / info["fps"]}
    key = CATEGORIES[category]
    check = catalog.match_tracking_csv(key, name, str(original), metadata, csv_path_override=tracking_csv)
    if check.get("status") != "verified_match":
        raise ValueError(check.get("reason") or check.get("status") or "source/timeline validation failed")
    # Second exact-selection guard: status record for Shooting/Robbery; exact
    # source-stem folder for Assault.
    selected_status, selected_csv, selected_video = evidence.resolve_artifacts(original, category, name)
    if selected_csv.resolve() != tracking_csv.resolve():
        raise ValueError("evidence service selected a different tracking CSV")
    if not selected_video or selected_video.resolve() != tracked_video.resolve():
        raise ValueError("evidence service selected a different or missing annotated video")
    return {"video": name, "original": original, "csv": tracking_csv, "tracked": tracked_video,
            "status": selected_status, "check": check}


def inspect_excluded(records, statuses, selected_name, category, catalog, evidence):
    excluded = []
    for record in records:
        name = record.get("original_video", "")
        if name == selected_name:
            continue
        try:
            validate_record(record, statuses.get(name), category, catalog, evidence)
        except Exception as error:
            excluded.append({"video": name, "reason": str(error)})
    return excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=tuple(CATEGORIES), required=True)
    parser.add_argument("--video", required=True, help="Exact source video filename")
    args = parser.parse_args()
    category = args.category

    cfg = config_dict()
    root, records, statuses = manifest_records(category)
    row = next((item for item in records if item.get("original_video") == args.video), None)
    if row is None:
        raise SystemExit(f"No {category} artifact record for {args.video}")
    evidence = EventEvidenceService(cfg)
    catalog = EvidenceCatalogService(cfg)
    selected = validate_record(row, statuses.get(args.video), category, catalog, evidence)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = BACKEND / "real_data_pilots" / category.casefold() / f"{Path(args.video).stem}_{stamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    # Pin production model. Never consult video_classifier_v2.pth.
    model_path = BACKEND / "models" / "video_classifier.pth"
    classifier = VideoClassificationModel(model_path, Config.CLASS_NAMES_PATH)
    prediction = classifier.predict_windows_path(selected["original"], primary_category=category)
    prediction_record = {"source_video": args.video, "checkpoint": str(model_path),
                         "primary_category": category, **prediction}
    (output_root / "classifier_predictions.json").write_text(
        json.dumps(prediction_record, allow_nan=False, indent=2), encoding="utf-8")

    service = TemporalEventAssociationService(cfg, evidence_service=evidence)
    event_records = []
    for candidate in prediction["activity_events"]:
        event_records.append(service.create_event(
            original_path=selected["original"], output_root=output_root / "events",
            start_sec=candidate["start_timestamp_sec"], end_sec=candidate["end_timestamp_sec"],
            category=category, temporal_predictions=prediction["temporal_windows"],
            classifier_probabilities=prediction["probabilities"], classifier_confidence=candidate["confidence"],
            video_info={"classifier_window_seconds": prediction["sampling_coverage"]["window_seconds"]},
            source_video_name=args.video, verification_status="unverified_candidate"))

    excluded = inspect_excluded(records, statuses, args.video, category, catalog, evidence)
    excluded_path = output_root / "excluded_tracking_artifacts.csv"
    with excluded_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video", "reason"])
        writer.writeheader()
        writer.writerows(excluded)

    summaries = []
    for event in event_records:
        detections = event.get("detections", [])
        confidences = [float(item["confidence"]) for item in detections if item.get("confidence") is not None]
        summaries.append({
            "evidence_id": event["evidence_id"],
            "original_candidate_start_sec": event["original_candidate_start_sec"],
            "original_candidate_end_sec": event["original_candidate_end_sec"],
            "candidate_confidence": event.get("classifier_confidence"),
            "refined_start_sec": event["refined_start_sec"], "refined_end_sec": event["refined_end_sec"],
            "refinement_status": event["refinement_status"], "verification_status": event["verification_status"],
            "detection_count": len(detections),
            "object_classes": dict(Counter(item.get("class_name", "unknown") for item in detections)),
            "tracking_ids": sorted({str(item["track_id"]) for item in detections if item.get("track_id") not in (None, "")}),
            "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
            "maximum_confidence": max(confidences) if confidences else None,
            "tracking_continuity": [{key: group.get(key) for key in
                ("class_name", "track_id", "detection_count", "continuity_score", "tracking_duration_sec", "max_gap_sec", "fragmented")}
                for group in event.get("temporal_evidence", {}).get("track_groups", [])],
            "csv_timing_validation": event["temporal_evidence"]["csv_timing_validation"],
            "tracked_video_alignment": event["tracked_video_alignment"], "artifacts": event["artifacts"],
        })

    report = {
        "pilot_status": "completed" if summaries else "no_classifier_candidate",
        "source_video": str(selected["original"]), "source_video_name": args.video,
        "source_sha256": selected["check"].get("source_sha256"),
        "tracking_csv": str(selected["csv"]), "tracking_csv_validation": selected["check"],
        "tracking_status_record": selected["status"], "classifier_checkpoint": str(model_path),
        "classifier_candidate_count": len(prediction["activity_events"]),
        "classifier_probabilities": prediction["probabilities"], "sampling_metadata": prediction["sampling_coverage"],
        "temporal_window_count": len(prediction["temporal_windows"]), "events": summaries,
        "excluded_artifact_count": len(excluded), "excluded_artifacts_csv": str(excluded_path),
        "output_directory": str(output_root),
    }
    (output_root / "pilot_report.json").write_text(json.dumps(report, allow_nan=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
