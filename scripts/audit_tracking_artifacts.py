"""Read-only per-video inventory for locally accessible tracking artifacts."""
from __future__ import annotations

import csv
import os
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
TRACK_ID_COLUMNS = {"track_id", "tracker_id", "object_id"}
SOURCE_ROOT = Path(os.getenv("EVIDENCE_SOURCE_DIR", Path.home() / "OneDrive" / "Documents" / "crime_video_dataset"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOTS = {
    "Assault": [Path(os.getenv("ASSAULT_RESULTS_DIR", PROJECT_ROOT / "backend" / "tracked_results_400" / "Assault"))],
    "Robbery": [
        Path(os.getenv("ROBBERY_RESULTS_DIR", PROJECT_ROOT / "backend" / "robbery_drive_recovery_export")),
        PROJECT_ROOT / "backend" / "robbery_drive_recovery_export_20261026_172720",
        PROJECT_ROOT / "backend" / "tracked_results_400" / "Robbery",
    ],
    "Shooting": [Path(os.getenv("SHOOTING_RESULTS_DIR", PROJECT_ROOT / "backend" / "shooting_evaluation"))],
}


def readable_video(path: Path) -> tuple[bool, int | None, str | None]:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            opened = bool(cap.isOpened())
            if not opened:
                return False, None, "VideoCapture could not open file"
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            ok, _ = cap.read()
            return bool(ok), frame_count, None if ok else "First frame could not be decoded"
        finally:
            cap.release()
    except Exception as error:  # report a probe failure as unavailable, never valid
        return False, None, f"Probe failed: {type(error).__name__}: {error}"


def csv_details(path: Path) -> tuple[str, int | None, bool | None, int | None, str | None, str | None, str | None, str | None]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            if not columns:
                return "", None, None, None, None, None, None, "CSV has no header"
            rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            return "|".join(columns), None, None, None, None, None, None, "CSV unreadable: row width does not match its header"
        normalized = {column.strip().lower() for column in columns if column}
        has_track_id = bool(normalized & TRACK_ID_COLUMNS)
        populated_track_id = has_track_id and any(
            str(row.get(column, "")).strip()
            for row in rows for column in columns
            if column and column.strip().lower() in TRACK_ID_COLUMNS
        )
        frame_column = next((column for column in columns if column.strip().lower() in {"frame", "frame_number", "frame_id"}), None)
        time_column = next((column for column in columns if column.strip().lower() in {"timestamp_seconds", "timestamp_sec", "time_seconds", "timestamp"}), None)
        distinct_frames = len({str(row.get(frame_column, "")).strip() for row in rows if frame_column and str(row.get(frame_column, "")).strip()}) if frame_column else None
        times = []
        if time_column:
            for row in rows:
                try:
                    times.append(float(row.get(time_column, "")))
                except (TypeError, ValueError):
                    pass
        return "|".join(columns), len(rows), bool(populated_track_id), distinct_frames, str(min(times)) if times else None, str(max(times)) if times else None, "", None
    except (OSError, csv.Error, UnicodeError) as error:
        return "", None, None, None, None, None, None, f"CSV unreadable: {type(error).__name__}: {error}"


def _candidate_outputs(category: str, source: Path) -> tuple[list[Path], list[Path], list[Path]]:
    stem = source.stem.casefold()
    mp4s, csvs, statuses = [], [], []
    for root in RESULT_ROOTS[category]:
        if not root.is_dir():
            continue
        statuses.extend(root.glob("*status*.csv"))
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lower = path.name.casefold()
            parent_matches = path.parent.name.casefold() == stem
            stem_matches = path.stem.casefold() in {stem, f"{stem}_tracked", f"{stem}_annotated", f"{stem}_annotated_working", f"{stem}_tracked.partial"}
            if path.suffix.lower() == ".mp4" and (parent_matches or stem_matches):
                mp4s.append(path)
            if path.suffix.lower() == ".csv" and (
                (parent_matches and lower in {"detections.csv", "tracking_status.csv", "status.csv"})
                or path.stem.casefold() in {f"{stem}_detections", f"{stem}_tracking_status"}
            ):
                (statuses if "status" in lower else csvs).append(path)
    return sorted(set(mp4s)), sorted(set(csvs)), sorted(set(statuses))


def audit_row(category: str, source: Path) -> dict:
    source_ok, source_frames, source_error = readable_video(source)
    mp4s, csvs, statuses = _candidate_outputs(category, source)
    final_mp4s = [path for path in mp4s if "working" not in path.stem.casefold() and ".partial" not in path.stem.casefold()]
    mp4_results = [(path, *readable_video(path)) for path in final_mp4s]
    valid_mp4s = [item for item in mp4_results if item[1]]
    schemas, row_counts, track_flags, distinct_frames, first_times, last_times, csv_errors = [], [], [], [], [], [], []
    for path in csvs:
        schema, count, track_ids, frame_count, first_time, last_time, _, error = csv_details(path)
        schemas.append(f"{path} [{schema}]")
        row_counts.append(count)
        track_flags.append(track_ids)
        distinct_frames.append(frame_count)
        first_times.append(first_time)
        last_times.append(last_time)
        if error:
            csv_errors.append(f"{path}: {error}")
    status_value, saved_error = "not_started", ""
    if statuses:
        for path in statuses:
            try:
                with path.open("r", newline="", encoding="utf-8-sig") as stream:
                    for item in csv.DictReader(stream):
                        name = item.get("video_name", item.get("video", item.get("filename", item.get("source_video", ""))))
                        if Path(name).stem.casefold() == source.stem.casefold():
                            status_value = item.get("status", item.get("processing_status", "unknown")) or "unknown"
                            saved_error = item.get("error", item.get("error_message", "")) or ""
                            break
            except (OSError, csv.Error, UnicodeError):
                continue
    if status_value == "not_started" and (mp4s or csvs):
        status_value = "results_present_status_unrecorded"
    incomplete = [path for path in mp4s if path not in final_mp4s]
    if csv_errors or any(not item[1] for item in mp4_results):
        disposition = "invalid"
    elif len(mp4s) > len(final_mp4s) or (valid_mp4s and not csvs):
        disposition = "partial"
    elif valid_mp4s and csvs:
        output_counts = [frames for _, ok, frames, _ in mp4_results if ok and frames]
        mismatch = bool(source_frames and output_counts and any(frames != source_frames for frames in output_counts))
        recorded_status = status_value.lower()
        status_allows_completion = recorded_status in {"completed", "complete", "success", "successful", "results_present_status_unrecorded"}
        disposition = "complete" if source_ok and source_frames is not None and output_counts and not mismatch and status_allows_completion else "needs_human_review"
    elif mp4s or csvs:
        disposition = "partial"
    else:
        disposition = "missing"
    if status_value.lower() in {"processing", "running"}:
        disposition = "partial"
    elif any(word in status_value.lower() for word in ("fail", "error", "interrupt")):
        disposition = "partial" if mp4s or csvs else "missing"
    if len(mp4s) > 1 or len(csvs) > 1:
        disposition = "needs_human_review"
    return {
        "category": category, "video": source.name, "source_path": str(source),
        "source_readable_first_frame": source_ok, "source_frame_count_metadata": source_frames,
        "source_read_error": source_error or "", "tracking_status": status_value,
        "saved_error": saved_error, "annotated_mp4_paths": ";".join(map(str, mp4s)),
        "final_mp4_paths": ";".join(map(str, final_mp4s)),
        "valid_final_mp4_count": len(valid_mp4s),
        "output_frame_counts_metadata": ";".join(f"{path}={frames}" for path, ok, frames, error in mp4_results),
        "csv_paths": ";".join(map(str, csvs)), "csv_schemas": ";".join(schemas),
        "csv_row_counts": ";".join("unknown" if value is None else str(value) for value in row_counts),
        "csv_has_populated_track_ids": ";".join("unknown" if value is None else str(value) for value in track_flags),
        "csv_distinct_detection_frames": ";".join("unknown" if value is None else str(value) for value in distinct_frames),
        "csv_first_detection_timestamp_sec": ";".join(value or "unknown" for value in first_times),
        "csv_last_detection_timestamp_sec": ";".join(value or "unknown" for value in last_times),
        "csv_detection_frame_coverage_percent": ";".join(
            "unknown" if value is None or not source_frames else f"{value / source_frames * 100:.2f}" for value in distinct_frames
        ),
        "csv_errors": ";".join(csv_errors), "ambiguous_match": len(mp4s) > 1 or len(csvs) > 1,
        "disposition": disposition,
        "coverage": "detection-record frame coverage only; not inference coverage" if csvs else "unavailable",
        "notes": ";".join([*(f"incomplete candidate: {p}" for p in incomplete), *(f"invalid MP4: {p}: {err}" for p, ok, _, err in mp4_results if not ok)]),
    }


def main() -> int:
    output = PROJECT_ROOT / "reports" / "tracking_artifact_inventory.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["category", "video", "source_path", "source_readable_first_frame", "source_frame_count_metadata", "source_read_error", "tracking_status", "saved_error", "annotated_mp4_paths", "final_mp4_paths", "valid_final_mp4_count", "output_frame_counts_metadata", "csv_paths", "csv_schemas", "csv_row_counts", "csv_has_populated_track_ids", "csv_distinct_detection_frames", "csv_first_detection_timestamp_sec", "csv_last_detection_timestamp_sec", "csv_detection_frame_coverage_percent", "csv_errors", "ambiguous_match", "disposition", "coverage", "notes"]
    rows = []
    for category in ("Assault", "Robbery", "Shooting"):
        source_dir = SOURCE_ROOT / category
        if source_dir.is_dir():
            sources = sorted((path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS), key=lambda path: path.name.casefold())
            rows.extend(audit_row(category, source) for source in sources)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Inventory: {output}")
    print(f"Rows: {len(rows)}")
    for category in ("Assault", "Robbery", "Shooting"):
        subset = [row for row in rows if row["category"] == category]
        print(f"{category}: " + ", ".join(f"{state}={sum(row['disposition'] == state for row in subset)}" for state in ("complete", "partial", "missing", "invalid", "needs_human_review")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
