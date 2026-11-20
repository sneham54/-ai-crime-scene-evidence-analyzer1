#!/usr/bin/env python3
"""Render missing Assault annotated MP4s from saved detections without inference.

The script never changes an existing detection CSV, final MP4, or partial MP4.
It accepts byte-for-byte copied CSVs from the Colab export under --drive-export.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def _rows(csv_path: Path) -> tuple[list[dict], set[str]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = {str(value).strip().lower() for value in (reader.fieldnames or []) if value}
    return rows, fields


def _value(row: dict, *names: str):
    lowered = {str(key).strip().lower(): value for key, value in row.items() if key is not None}
    for name in names:
        value = lowered.get(name)
        if value not in (None, ""):
            return value
    return None


def _detection_rows(csv_path: Path, frame_count: int, fps: float, width: int, height: int):
    rows, fields = _rows(csv_path)
    required = {"class", "confidence", "x1", "y1", "x2", "y2"}
    if not required <= fields:
        raise ValueError("missing required CSV columns: " + ", ".join(sorted(required - fields)))
    if not ({"frame", "frame_number", "timestamp_seconds", "timestamp_sec", "timestamp"} & fields):
        raise ValueError("CSV has neither frame numbers nor timestamps")
    if not rows:
        raise ValueError("CSV has no detection rows; an empty export cannot establish a usable annotated result")

    raw_frames = []
    parsed = []
    for row_number, row in enumerate(rows, start=2):
        try:
            class_name = str(_value(row, "class", "class_name", "label") or "").strip()
            confidence = float(_value(row, "confidence", "score"))
            coords = [float(_value(row, name)) for name in ("x1", "y1", "x2", "y2")]
            frame_text = _value(row, "frame", "frame_number")
            timestamp_text = _value(row, "timestamp_seconds", "timestamp_sec", "timestamp", "time_seconds")
            frame = int(float(frame_text)) if frame_text is not None else None
            timestamp = float(timestamp_text) if timestamp_text is not None else None
            if not class_name or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("invalid class or confidence")
            if not all(math.isfinite(value) for value in coords) or coords[2] <= coords[0] or coords[3] <= coords[1]:
                raise ValueError("invalid bounding box")
            if frame is None and timestamp is None:
                raise ValueError("missing frame and timestamp")
            if frame is not None and frame < 0:
                raise ValueError("negative frame number")
            if timestamp is not None and (not math.isfinite(timestamp) or timestamp < 0):
                raise ValueError("invalid timestamp")
            if timestamp is not None and timestamp * fps > frame_count + 1:
                raise ValueError("timestamp is outside source video")
            if frame is not None and frame > frame_count + 1:
                raise ValueError("frame is outside source video")
            if frame is not None:
                raw_frames.append(frame)
            parsed.append((row_number, frame, timestamp, class_name, confidence, coords,
                           str(_value(row, "track_id", "tracker_id", "object_id") or "").strip()))
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"invalid detection CSV row {row_number}: {error}") from error

    one_based = bool(raw_frames) and 0 not in raw_frames and min(raw_frames) >= 1
    by_frame = defaultdict(list)
    for _, frame, timestamp, class_name, confidence, coords, track_id in parsed:
        if timestamp is not None:
            frame_index = int(round(timestamp * fps))
            if frame is not None:
                expected = frame - 1 if one_based else frame
                if abs(frame_index - expected) > 2:
                    raise ValueError("CSV frame and timestamp columns disagree by more than two frames")
        else:
            frame_index = frame - 1 if one_based else frame
        if frame_index is None or not 0 <= frame_index < frame_count:
            raise ValueError("detection does not map to a frame in the source video")
        x1, y1, x2, y2 = coords
        x1, y1 = max(0, min(width - 1, round(x1))), max(0, min(height - 1, round(y1)))
        x2, y2 = max(0, min(width - 1, round(x2))), max(0, min(height - 1, round(y2)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bounding box falls outside source image dimensions")
        by_frame[frame_index].append({"class": class_name, "confidence": confidence,
                                      "bbox": (x1, y1, x2, y2), "track_id": track_id})
    return by_frame, len(parsed)


def _render_one(source_video: Path, detections_csv: Path, output_video: Path) -> dict:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        return {"status": "blocked_source_video_unreadable", "reason": "source video could not be opened"}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0 or width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        cap.release()
        return {"status": "blocked_source_video_metadata_invalid", "reason": "invalid source FPS, dimensions, or frame count"}
    try:
        by_frame, detection_count = _detection_rows(detections_csv, frame_count, fps, width, height)
    except (OSError, csv.Error, UnicodeError, ValueError) as error:
        cap.release()
        return {"status": "blocked_csv_not_renderable", "reason": str(error)}

    if output_video.exists():
        cap.release()
        return {"status": "skipped_output_exists", "reason": "refusing to overwrite existing recovered video"}
    output_video.parent.mkdir(parents=True, exist_ok=True)
    temp_video = output_video.with_name(output_video.stem + ".partial.mp4")
    if temp_video.exists():
        cap.release()
        return {"status": "skipped_temp_exists", "reason": "existing partial recovery file preserved"}
    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return {"status": "blocked_video_writer_unavailable", "reason": "OpenCV mp4v writer did not open"}

    frame_index = 0
    read_error = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (height, width):
            read_error = True
            break
        for detection in by_frame.get(frame_index, []):
            x1, y1, x2, y2 = detection["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 80), 2)
            track_label = f" ID:{detection['track_id']}" if detection["track_id"] else ""
            label = f"{detection['class']} {detection['confidence']:.2f}{track_label}"
            cv2.putText(frame, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (40, 220, 80), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_index += 1
    writer.release()
    cap.release()
    if read_error or frame_index != frame_count:
        return {"status": "blocked_source_decode_incomplete",
                "reason": f"decoded {frame_index} of {frame_count} source frames; partial file preserved at {temp_video}"}

    check = cv2.VideoCapture(str(temp_video))
    valid = check.isOpened()
    actual_count = int(check.get(cv2.CAP_PROP_FRAME_COUNT)) if valid else 0
    actual_fps = float(check.get(cv2.CAP_PROP_FPS)) if valid else 0.0
    actual_width = int(check.get(cv2.CAP_PROP_FRAME_WIDTH)) if valid else 0
    actual_height = int(check.get(cv2.CAP_PROP_FRAME_HEIGHT)) if valid else 0
    decoded_output = 0
    while valid:
        ok, _frame = check.read()
        if not ok:
            break
        decoded_output += 1
    check.release()
    if (not valid or decoded_output != frame_count or abs(actual_fps - fps) > 0.05
            or (actual_width, actual_height) != (width, height) or temp_video.stat().st_size == 0):
        return {"status": "blocked_render_validation_failed",
                "reason": f"output validation mismatch (frames={decoded_output}/{frame_count}, fps={actual_fps}/{fps}, size={actual_width}x{actual_height}/{width}x{height}); preserved at {temp_video}"}
    temp_video.rename(output_video)
    return {"status": "recovered_from_saved_csv_no_inference", "reason": "validated full decode, dimensions, FPS, and frame count",
            "detections": detection_count, "frames": frame_count, "fps": fps,
            "width": width, "height": height, "output": str(output_video)}


def _find_csvs(search_roots: list[Path], source_key: str) -> list[Path]:
    found = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.csv"):
            # Avoid scanning package caches and source datasets when callers
            # choose a broad local search root.
            if any(part.casefold() in {"node_modules", ".git", "venv", "__pycache__"}
                   for part in path.parts):
                continue
            name = path.name.lower()
            if path.stem.lower() == source_key.lower() + "_detections" or (
                path.name.lower() == "detections.csv" and path.parent.name.lower() == source_key.lower()
            ):
                found.append(path)
    unique = {}
    for path in found:
        try:
            unique[(path.stat().st_size, path.read_bytes())] = path
        except OSError:
            continue
    return list(unique.values())


def _local_output_roots(project_root: Path, results_dir: Path, drive_export: Path,
                        extra_roots: list[Path]) -> list[Path]:
    """Search configured outputs and known local result/report locations first."""
    candidates = [
        results_dir, drive_export,
        project_root / "backend" / "robbery_videomae",
        project_root / "backend" / "shooting_evaluation",
        project_root / "backend" / "uploads",
        project_root / "runs", project_root / "reports", project_root / "scratch",
        *extra_roots,
    ]
    roots = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in seen and resolved.is_dir():
            roots.append(resolved)
            seen.add(resolved)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path.home() / "Downloads" / "assault_combined_reconciliation.csv")
    parser.add_argument("--source-dir", type=Path,
                        default=Path.home() / "OneDrive" / "Documents" / "crime_video_dataset" / "Assault")
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results-dir", type=Path,
                        default=project_root / "backend" / "tracked_results_400" / "Assault")
    parser.add_argument("--drive-export", type=Path,
                        default=project_root / "backend" / "assault_drive_recovery_export" / "artifacts")
    parser.add_argument("--search-root", type=Path, action="append", default=[],
                        help="Additional local output folder to search before marking a Drive CSV unavailable; repeatable")
    parser.add_argument("--status-report", type=Path,
                        default=Path("reports/assault_csv_video_recovery_status.csv"))
    parser.add_argument("--force-report", action="store_true")
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"reconciliation manifest not found: {args.manifest}")
    if not args.source_dir.is_dir():
        parser.error(f"Assault source directory not found: {args.source_dir}")
    if args.status_report.exists() and not args.force_report:
        parser.error(f"refusing to overwrite status report: {args.status_report}")

    with args.manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reconciled = list(csv.DictReader(stream))
    search_roots = _local_output_roots(project_root, args.results_dir, args.drive_export,
                                       args.search_root)
    status_rows = []
    for item in reconciled:
        if item.get("reconciliation_status") != "CSV_ONLY_NO_FINAL_MP4_ANYWHERE":
            continue
        source_key = item["source_key"]
        candidates = _find_csvs(search_roots, source_key)
        row = {"source_key": source_key, "source_video": item.get("source_video_local"),
               "status": "blocked_drive_csv_not_copied", "reason": "CSV listed in Drive manifest is unavailable locally"}
        if len(candidates) > 1:
            row.update(status="blocked_multiple_csv_versions", reason="multiple non-identical CSV files found; review their provenance")
        elif len(candidates) == 1:
            source_video = next((args.source_dir / f"{source_key}{suffix}" for suffix in VIDEO_EXTENSIONS
                                 if (args.source_dir / f"{source_key}{suffix}").is_file()), None)
            if source_video is None:
                row.update(status="blocked_source_missing", reason="matching source video is unavailable locally")
            else:
                target_dir = args.results_dir / source_key
                target_csv = target_dir / "detections.csv"
                target_video = target_dir / f"{source_key}_recovered_from_csv.mp4"
                existing_finals = [path for path in args.results_dir.rglob("*.mp4")
                                   if source_key.lower() in path.name.lower()
                                   and not path.name.lower().endswith(".partial.mp4")]
                if existing_finals:
                    row.update(status="existing_alternate_mp4_requires_validation",
                               reason="an alternate final MP4 exists; inspect it before rendering another")
                elif target_csv.exists():
                    try:
                        same = target_csv.read_bytes() == candidates[0].read_bytes()
                    except OSError:
                        same = False
                    if not same:
                        row.update(status="blocked_csv_destination_conflict", reason="detections.csv already exists with different bytes")
                    else:
                        row.update(_render_one(source_video, target_csv, target_video))
                else:
                    try:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        with target_csv.open("xb") as destination:
                            destination.write(candidates[0].read_bytes())
                        row.update(_render_one(source_video, target_csv, target_video))
                    except FileExistsError:
                        row.update(status="blocked_csv_destination_conflict", reason="detections.csv appeared during recovery; preserved")
                    except OSError as error:
                        row.update(status="blocked_csv_copy_failed", reason=type(error).__name__)
        status_rows.append(row)

    args.status_report.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in status_rows for key in row})
    with args.status_report.open("x" if not args.force_report else "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(status_rows)
    print(f"csv_only_videos={len(status_rows)} report={args.status_report}")
    for status in sorted({row["status"] for row in status_rows}):
        print(f"{status}={sum(row['status'] == status for row in status_rows)}")


if __name__ == "__main__":
    main()
