#!/usr/bin/env python3
"""Validate local Assault pairs against an existing combined reconciliation CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return rows, reader.fieldnames or []


def validate_video(path: Path, *, full_decode: bool = False) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        return {"valid": False, "reason": "missing_or_empty_video"}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"valid": False, "reason": "video_cannot_be_opened"}
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count_property = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = 0
    if full_decode:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            if frame.shape[1] != width or frame.shape[0] != height:
                capture.release()
                return {"valid": False, "reason": "variable_or_inconsistent_dimensions"}
    else:
        ok, frame = capture.read()
        decoded = 1 if ok else 0
    capture.release()
    if decoded <= 0 or width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        return {"valid": False, "reason": "invalid_video_metadata_or_no_decodable_frames"}
    if full_decode and frame_count_property > 0 and decoded != frame_count_property:
        return {"valid": False, "reason": f"decoded_frame_count_mismatch:{decoded}/{frame_count_property}"}
    return {"valid": True, "fps": fps, "frames": frame_count_property or decoded,
            "decoded_frames": decoded, "width": width, "height": height}


def validate_detection_csv(path: Path, source_video: dict) -> dict:
    try:
        rows, fields = read_csv(path)
    except (OSError, csv.Error, UnicodeError) as error:
        return {"valid": False, "reason": f"csv_read_error:{type(error).__name__}"}
    required = {"class", "confidence", "x1", "y1", "x2", "y2"}
    time_fields = {"frame", "frame_number", "timestamp_seconds", "timestamp_sec", "timestamp"}
    missing = sorted(required - set(fields))
    if missing:
        return {"valid": False, "reason": "missing_columns:" + ",".join(missing), "rows": len(rows)}
    if not (set(fields) & time_fields):
        return {"valid": False, "reason": "missing_frame_or_timestamp_column", "rows": len(rows)}
    if not rows:
        return {"valid": False, "reason": "empty_detection_csv", "rows": 0}

    invalid = 0
    tracked = 0
    max_timestamp = 0.0
    max_frame = 0
    for row in rows:
        try:
            label = str(row.get("class") or row.get("class_name") or "").strip()
            confidence = float(row["confidence"])
            box = [float(row[name]) for name in ("x1", "y1", "x2", "y2")]
            timestamp_value = row.get("timestamp_seconds") or row.get("timestamp_sec") or row.get("timestamp")
            frame_value = row.get("frame") or row.get("frame_number")
            timestamp = float(timestamp_value) if timestamp_value not in (None, "") else None
            frame_number = int(float(frame_value)) if frame_value not in (None, "") else None
            if not label or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError
            if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError
            if timestamp is None and frame_number is None:
                raise ValueError
            if timestamp is not None:
                if timestamp < 0 or not math.isfinite(timestamp):
                    raise ValueError
                max_timestamp = max(max_timestamp, timestamp)
            if frame_number is not None:
                if frame_number < 0:
                    raise ValueError
                max_frame = max(max_frame, frame_number)
            tracked += bool(str(row.get("track_id") or "").strip())
        except (ValueError, TypeError, KeyError, OverflowError):
            invalid += 1
    if invalid:
        return {"valid": False, "reason": f"invalid_detection_rows:{invalid}", "rows": len(rows)}
    if source_video.get("valid"):
        if max_frame >= source_video["frames"] + 2:
            return {"valid": False, "reason": "detection_frame_outside_source_video", "rows": len(rows)}
        if max_timestamp > source_video["frames"] / source_video["fps"] + 1 / source_video["fps"]:
            return {"valid": False, "reason": "detection_timestamp_outside_source_video", "rows": len(rows)}
    return {"valid": True, "rows": len(rows), "tracked_rows": tracked,
            "max_frame": max_frame, "max_timestamp_seconds": max_timestamp}


def audit(manifest: Path, source_dir: Path, results_dir: Path) -> list[dict]:
    rows, _ = read_csv(manifest)
    results = []
    for item in rows:
        status = item.get("reconciliation_status", "")
        local_mp4s = [Path(value) for value in (item.get("local_final_mp4s") or "").split(";") if value]
        # Manifests may include potential_events.csv in the same CSV list;
        # only detections.csv (or *_detections.csv) is the per-frame record.
        local_csvs = [
            Path(value.strip()) for value in (item.get("local_detection_csvs") or "").split(";")
            if value.strip() and Path(value.strip()).name.lower() == "detections.csv"
        ]
        local_partials = [Path(value) for value in (item.get("local_partial_mp4s") or "").split(";") if value]
        video_path = source_dir / (item.get("source_video_local") or "")
        source_needed = bool(local_mp4s or local_partials or status == "CSV_ONLY_NO_FINAL_MP4_ANYWHERE")
        source_info = validate_video(video_path) if source_needed else {
            "valid": None, "reason": "not_decoded_no_local_result_to_validate",
            "exists": video_path.is_file(),
        }
        final_info = validate_video(local_mp4s[0], full_decode=True) if len(local_mp4s) == 1 else {"valid": False, "reason": "final_mp4_count_not_one"}
        csv_info = validate_detection_csv(local_csvs[0], source_info) if len(local_csvs) == 1 else {"valid": False, "reason": "detection_csv_count_not_one"}
        partial_info = validate_video(local_partials[0]) if len(local_partials) == 1 else {"valid": None, "reason": "no_single_partial_file"}
        alternate_csvs = []
        if not local_csvs and results_dir.is_dir():
            source_key = item.get("source_key", "").lower()
            alternate_csvs = [path for path in results_dir.rglob("*.csv")
                              if ((path.name.lower() == "detections.csv" and path.parent.name.lower() == source_key)
                                  or path.stem.lower() == source_key + "_detections")]
        if final_info.get("valid") and source_info.get("valid"):
            if (final_info["width"], final_info["height"]) != (source_info["width"], source_info["height"]):
                final_info = {**final_info, "valid": False, "reason": "output_dimensions_differ_from_source"}
            elif abs(final_info["fps"] - source_info["fps"]) > 0.05:
                final_info = {**final_info, "valid": False, "reason": "output_fps_differs_from_source"}
            elif abs(final_info["frames"] - source_info["frames"]) > 1:
                final_info = {**final_info, "valid": False, "reason": "output_frame_count_differs_from_source"}
        if status == "COMPLETE_PAIR_IN_ONE_LOCATION" and final_info.get("valid") and csv_info.get("valid"):
            audit_status = "COMPLETE_EXISTING_LOCAL_PAIR_VALIDATED"
        elif status == "CSV_ONLY_NO_FINAL_MP4_ANYWHERE":
            audit_status = "DRIVE_CSV_REPORTED_COPY_REQUIRED_FOR_LOCAL_RECONSTRUCTION"
        elif status == "PARTIAL_ONLY_NEEDS_REVIEW":
            audit_status = "PARTIAL_OUTPUT_PRESERVED_REQUIRES_REVIEW"
        elif status == "FINAL_MP4_NO_CSV_ANYWHERE":
            audit_status = "FINAL_MP4_WITHOUT_CSV_DRIVE_SEARCH_REQUIRED"
        else:
            audit_status = "NO_MATCH_IN_AUDITED_LOCATIONS_WIDER_DRIVE_SEARCH_PENDING"
        results.append({
            "source_key": item.get("source_key"),
            "source_video": item.get("source_video_local"),
            "reconciliation_status": status,
            "audit_status": audit_status,
            "source_path": str(video_path),
            "source_video_exists": source_info.get("exists", video_path.is_file()),
            "source_video_valid": source_info.get("valid"),
            "source_video_validation": source_info.get("reason", "valid"),
            "local_final_mp4": str(local_mp4s[0]) if len(local_mp4s) == 1 else "",
            "final_mp4_valid": final_info.get("valid"),
            "final_mp4_validation": final_info.get("reason", "valid"),
            "final_mp4_frames": final_info.get("frames"),
            "final_mp4_fps": final_info.get("fps"),
            "detection_csv": str(local_csvs[0]) if len(local_csvs) == 1 else "",
            "detection_csv_valid": csv_info.get("valid"),
            "detection_csv_validation": csv_info.get("reason", "valid"),
            "alternate_local_detection_csvs": ";".join(str(path) for path in alternate_csvs),
            "detection_rows": csv_info.get("rows"),
            "rows_with_track_id": csv_info.get("tracked_rows"),
            "drive_mp4_paths_from_manifest": item.get("drive_mp4_paths"),
            "drive_csv_paths_from_manifest": item.get("drive_csv_paths"),
            "partial_paths_preserved": item.get("local_partial_mp4s"),
            "partial_mp4_valid_first_frame": partial_info.get("valid"),
            "partial_mp4_validation": partial_info.get("reason", "valid"),
            "wider_drive_search_complete": False,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path.home() / "Downloads" / "assault_combined_reconciliation.csv")
    parser.add_argument("--source-dir", type=Path,
                        default=Path.home() / "OneDrive" / "Documents" / "crime_video_dataset" / "Assault")
    parser.add_argument("--results-dir", type=Path, default=Path("backend/tracked_results_400/Assault"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports") / "assault_local_validation.csv")
    parser.add_argument("--force", action="store_true", help="Replace the report at --output if it exists")
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"reconciliation manifest not found: {args.manifest}")
    if not args.source_dir.is_dir():
        parser.error(f"Assault source folder not found: {args.source_dir}")
    if args.output.exists() and not args.force:
        parser.error(f"refusing to overwrite existing report: {args.output}; use --force intentionally")
    results = audit(args.manifest, args.source_dir, args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0]) if results else []
    with args.output.open("x" if not args.force else "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    counts = {}
    for row in results:
        counts[row["audit_status"]] = counts.get(row["audit_status"], 0) + 1
    print(f"videos={len(results)} statuses={counts} report={args.output}")


if __name__ == "__main__":
    main()
