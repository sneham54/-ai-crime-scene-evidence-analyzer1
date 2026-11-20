"""Read-only audit of labeled crime video folders and their video files.

The audit hashes files but never changes anything under the dataset root. It
decodes videos sequentially with OpenCV and refreshes the reports as it goes so
the latest completed records survive an interrupted run.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import os
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(r"C:\Users\admin\OneDrive\Documents\crime_video_dataset")
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
CATEGORIES = ("Assault", "Robbery", "Shooting", "Normal")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
HASH_CHUNK_BYTES = 1024 * 1024


def sha256_file(path):
    """Return a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _error(record, code, message):
    record["errors"].append({"code": code, "message": str(message)})


def inspect_video(path, dataset_root, category, cv2_module):
    """Hash and decode one file; always release an allocated VideoCapture."""
    path = Path(path)
    record = {
        "path": str(path.relative_to(dataset_root)),
        "category": category,
        "extension": path.suffix.lower(),
        "sha256": None,
        "status": "pending",
        "fps": None,
        "reported_frame_count": None,
        "decoded_frame_count": 0,
        "duration_seconds": None,
        "resolution": None,
        "errors": [],
    }

    try:
        record["sha256"] = sha256_file(path)
    except Exception as error:
        _error(record, "sha256_read_failed", f"{type(error).__name__}: {error}")

    capture = None
    try:
        capture = cv2_module.VideoCapture(str(path))
        if capture is None or not capture.isOpened():
            _error(record, "video_open_failed", "OpenCV could not open this video.")
            record["status"] = "unreadable"
            return record

        try:
            fps = _finite_number(capture.get(cv2_module.CAP_PROP_FPS))
        except Exception as error:
            fps = None
            _error(record, "fps_metadata_read_failed", f"{type(error).__name__}: {error}")
        if fps is None or fps <= 0:
            record["fps"] = None
            _error(record, "invalid_fps", "FPS metadata is missing, non-finite, or not positive.")
        else:
            record["fps"] = round(fps, 4)

        try:
            reported_frames_value = _finite_number(capture.get(cv2_module.CAP_PROP_FRAME_COUNT))
        except Exception as error:
            reported_frames_value = None
            _error(record, "frame_count_metadata_read_failed", f"{type(error).__name__}: {error}")
        reported_frames = (
            max(0, int(round(reported_frames_value)))
            if reported_frames_value is not None else None
        )
        record["reported_frame_count"] = reported_frames

        first_frame = None
        while True:
            try:
                ok, frame = capture.read()
            except Exception as error:
                _error(record, "frame_read_exception", f"{type(error).__name__}: {error}")
                break
            if not ok or frame is None:
                break
            record["decoded_frame_count"] += 1
            if first_frame is None:
                first_frame = frame

        if first_frame is not None:
            height, width = first_frame.shape[:2]
            if width > 0 and height > 0:
                record["resolution"] = f"{width}x{height}"

        decoded_frames = record["decoded_frame_count"]
        if decoded_frames == 0:
            _error(record, "zero_decodable_frames", "No video frames could be decoded.")
            record["status"] = "unreadable"
        else:
            record["status"] = "readable"
            if reported_frames is not None and reported_frames <= 0:
                _error(record, "zero_reported_frames", "Container reports zero frames although decoding returned frames.")
            elif reported_frames is not None and decoded_frames < reported_frames:
                _error(
                    record,
                    "decode_ended_before_reported_frame_count",
                    f"Decoded {decoded_frames} of {reported_frames} reported frames; file may be truncated or damaged.",
                )
                record["status"] = "corrupt_or_incomplete"
            elif any(error["code"] == "frame_read_exception" for error in record["errors"]):
                record["status"] = "corrupt_or_incomplete"

        if record["fps"] is not None and decoded_frames > 0:
            record["duration_seconds"] = round(decoded_frames / record["fps"], 3)
    except Exception as error:
        _error(record, "video_inspection_failed", f"{type(error).__name__}: {error}")
        if record["status"] == "pending":
            record["status"] = "inspection_error"
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception as error:
                _error(record, "capture_release_failed", f"{type(error).__name__}: {error}")

    return record


def _duplicate_groups(records):
    paths_by_digest = defaultdict(list)
    for record in records:
        if record.get("sha256"):
            paths_by_digest[record["sha256"]].append(record["path"])
    groups = [
        {"sha256": digest, "paths": sorted(paths)}
        for digest, paths in sorted(paths_by_digest.items()) if len(paths) > 1
    ]
    duplicate_paths = sorted(path for group in groups for path in group["paths"])
    return groups, duplicate_paths


def _category_summary(category, records, folder_exists):
    fps_values = [item["fps"] for item in records if item.get("fps") is not None]
    resolutions = Counter(item["resolution"] for item in records if item.get("resolution"))
    durations = [item["duration_seconds"] for item in records if item.get("duration_seconds") is not None]
    return {
        "folder_exists": folder_exists,
        "video_count": len(records),
        "readable_count": sum(item["status"] == "readable" for item in records),
        "unreadable_count": sum(item["status"] == "unreadable" for item in records),
        "corrupt_or_incomplete_count": sum(item["status"] == "corrupt_or_incomplete" for item in records),
        "inspection_error_count": sum(item["status"] == "inspection_error" for item in records),
        "total_duration_seconds": round(sum(durations), 3),
        "average_fps": round(sum(fps_values) / len(fps_values), 4) if fps_values else None,
        "fps_sample_count": len(fps_values),
        "resolution_breakdown": dict(sorted(resolutions.items())),
    }


def build_report(dataset_root, records, missing_categories, scan_state, fatal_error=None):
    category_records = {
        category: [item for item in records if item["category"] == category]
        for category in CATEGORIES
    }
    category_summaries = {
        category: _category_summary(
            category,
            category_records[category],
            category not in missing_categories,
        )
        for category in CATEGORIES
    }
    duplicate_groups, duplicate_paths = _duplicate_groups(records)
    all_fps = [item["fps"] for item in records if item.get("fps") is not None]
    all_durations = [item["duration_seconds"] for item in records if item.get("duration_seconds") is not None]
    all_resolutions = Counter(item["resolution"] for item in records if item.get("resolution"))
    errors = []
    for category in missing_categories:
        errors.append({"path": category, "code": "missing_category_folder", "message": f"Required category folder is missing: {category}"})
    for item in records:
        for issue in item["errors"]:
            errors.append({"path": item["path"], **issue})
    if fatal_error:
        errors.append({"path": None, "code": "audit_interrupted", "message": fatal_error})

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(Path(dataset_root).resolve()),
        "status": scan_state,
        "supported_extensions": sorted(VIDEO_EXTENSIONS),
        "expected_categories": list(CATEGORIES),
        "missing_categories": list(missing_categories),
        "category_summary": category_summaries,
        "summary": {
            "video_count": len(records),
            "readable_count": sum(item["status"] == "readable" for item in records),
            "unreadable_count": sum(item["status"] == "unreadable" for item in records),
            "corrupt_or_incomplete_count": sum(item["status"] == "corrupt_or_incomplete" for item in records),
            "inspection_error_count": sum(item["status"] == "inspection_error" for item in records),
            "total_duration_seconds": round(sum(all_durations), 3),
            "average_fps": round(sum(all_fps) / len(all_fps), 4) if all_fps else None,
            "fps_sample_count": len(all_fps),
            "resolution_breakdown": dict(sorted(all_resolutions.items())),
            "exact_duplicate_group_count": len(duplicate_groups),
            "duplicate_paths": duplicate_paths,
            "error_count": len(errors),
        },
        "duplicate_groups": duplicate_groups,
        "errors": errors,
        "files": records,
    }


def _markdown_report(report):
    summary = report["summary"]
    lines = [
        "# Video Dataset Accuracy Evaluation Audit",
        "",
        f"- Dataset root: `{report['dataset_root']}`",
        f"- Audit status: **{report['status']}**",
        f"- Supported extensions: {', '.join(report['supported_extensions'])}",
        f"- Video files inspected: {summary['video_count']}",
        f"- Total decoded duration: {summary['total_duration_seconds']:.3f} seconds",
        f"- Average valid FPS: {summary['average_fps'] if summary['average_fps'] is not None else 'unavailable'}",
        f"- Exact duplicate groups: {summary['exact_duplicate_group_count']}",
        f"- Errors and warnings: {summary['error_count']}",
        "",
        "## Required category folders",
        "",
    ]
    if report["missing_categories"]:
        lines.extend(f"- **MISSING:** `{name}`" for name in report["missing_categories"])
    else:
        lines.append("All four required category folders are present.")
    lines.extend([
        "",
        "## Per-category summary",
        "",
        "| Category | Folder | Files | Readable | Unreadable | Corrupt/incomplete | Duration (s) | Average FPS | Resolution breakdown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for category, values in report["category_summary"].items():
        resolution_text = ", ".join(f"{size}: {count}" for size, count in values["resolution_breakdown"].items()) or "none"
        lines.append(
            f"| {category} | {'yes' if values['folder_exists'] else 'MISSING'} | {values['video_count']} "
            f"| {values['readable_count']} | {values['unreadable_count']} | {values['corrupt_or_incomplete_count']} "
            f"| {values['total_duration_seconds']:.3f} | {values['average_fps'] if values['average_fps'] is not None else 'n/a'} "
            f"| {resolution_text} |"
        )
    lines.extend(["", "## Exact SHA-256 duplicates", ""])
    if report["duplicate_groups"]:
        for group in report["duplicate_groups"]:
            lines.append(f"- SHA-256 `{group['sha256']}`")
            lines.extend(f"  - `{path}`" for path in group["paths"])
    else:
        lines.append("No exact duplicates found among files successfully hashed.")
    lines.extend(["", "## File errors and warnings", ""])
    if report["errors"]:
        for issue in report["errors"]:
            location = f"`{issue['path']}`: " if issue.get("path") else ""
            lines.append(f"- {location}**{issue['code']}** — {issue['message']}")
    else:
        lines.append("No audit errors or warnings.")
    lines.extend([
        "",
        "## Interpretation note",
        "",
        "This report checks file integrity and metadata only. It does not evaluate model accuracy or establish that category labels are correct.",
        "",
    ])
    return "\n".join(lines)


def _atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", delete=False,
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        ) as stream:
            temp_name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def save_reports(output_dir, report):
    """Atomically refresh the Markdown and JSON checkpoint reports."""
    output_dir = Path(output_dir)
    json_path = output_dir / "accuracy_evaluation_report.json"
    markdown_path = output_dir / "accuracy_evaluation_report.md"
    _atomic_write(json_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _atomic_write(markdown_path, _markdown_report(report))
    return markdown_path, json_path


def run_audit(dataset_root, output_dir, cv2_module=None):
    """Audit all category videos; return report and both saved output paths."""
    if cv2_module is None:
        import cv2 as cv2_module

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    if dataset_root.exists() and output_dir.resolve().is_relative_to(dataset_root.resolve()):
        raise ValueError("Report output must be outside the dataset directory; dataset contents are read-only.")
    records = []
    fatal_error = None
    scan_state = "complete"
    if not dataset_root.is_dir():
        fatal_error = f"Dataset root does not exist or is not a directory: {dataset_root}"
        report = build_report(dataset_root, records, list(CATEGORIES), "failed", fatal_error)
        md_path, json_path = save_reports(output_dir, report)
        return report, md_path, json_path

    try:
        root_directories = {item.name.casefold(): item for item in dataset_root.iterdir() if item.is_dir()}
    except Exception as error:
        fatal_error = f"Could not inspect dataset root: {type(error).__name__}: {error}"
        report = build_report(dataset_root, records, list(CATEGORIES), "failed", fatal_error)
        md_path, json_path = save_reports(output_dir, report)
        return report, md_path, json_path
    category_directories = {}
    missing_categories = []
    for category in CATEGORIES:
        found = root_directories.get(category.casefold())
        if found is None:
            missing_categories.append(category)
        else:
            category_directories[category] = found
    for category in missing_categories:
        print(f"[MISSING CATEGORY] {category}", flush=True)

    try:
        files_by_category = {
            category: sorted(
                (
                    path for path in folder.rglob("*")
                    if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
                ),
                key=lambda path: str(path).casefold(),
            )
            for category, folder in category_directories.items()
        }
    except Exception as error:
        fatal_error = f"Could not enumerate category files: {type(error).__name__}: {error}"
        scan_state = "partial"
        files_by_category = {category: [] for category in category_directories}

    total_files = sum(len(paths) for paths in files_by_category.values())
    current = 0
    report = build_report(dataset_root, records, missing_categories, scan_state)
    save_reports(output_dir, report)

    try:
        for category in CATEGORIES:
            for video_path in files_by_category.get(category, []):
                current += 1
                print(f"[{current}/{total_files}] Auditing {category}: {video_path}", flush=True)
                try:
                    record = inspect_video(video_path, dataset_root, category, cv2_module)
                except Exception as error:
                    record = {
                        "path": str(video_path.relative_to(dataset_root)),
                        "category": category,
                        "extension": video_path.suffix.lower(),
                        "sha256": None,
                        "status": "inspection_error",
                        "fps": None,
                        "reported_frame_count": None,
                        "decoded_frame_count": 0,
                        "duration_seconds": None,
                        "resolution": None,
                        "errors": [{
                            "code": "unhandled_video_error",
                            "message": f"{type(error).__name__}: {error}",
                        }],
                    }
                records.append(record)
                report = build_report(dataset_root, records, missing_categories, scan_state)
                save_reports(output_dir, report)
    except KeyboardInterrupt:
        fatal_error = "Audit interrupted by user; reports contain completed files up to interruption."
        scan_state = "partial"
    except Exception as error:
        fatal_error = f"{type(error).__name__}: {error}"
        scan_state = "partial"

    if fatal_error:
        report = build_report(dataset_root, records, missing_categories, scan_state, fatal_error)
    else:
        report = build_report(dataset_root, records, missing_categories, "complete")
    md_path, json_path = save_reports(output_dir, report)
    return report, md_path, json_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args(argv)

    try:
        report, markdown_path, json_path = run_audit(args.dataset_root, args.output_dir)
    except Exception as error:
        print(f"[FATAL] Could not save or complete audit reports: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    for category in report["missing_categories"]:
        print(f"[MISSING CATEGORY] {category}", flush=True)
    summary = report["summary"]
    print(
        f"Audit {report['status']}: {summary['video_count']} videos, "
        f"{summary['unreadable_count']} unreadable, "
        f"{summary['corrupt_or_incomplete_count']} corrupt/incomplete, "
        f"{summary['exact_duplicate_group_count']} exact duplicate groups.",
        flush=True,
    )
    print(f"Markdown report: {markdown_path}")
    print(f"JSON report: {json_path}")
    return 130 if report["status"] == "partial" and report["errors"] and any(
        item["code"] == "audit_interrupted" for item in report["errors"]
    ) else (1 if report["status"] == "failed" else 0)


if __name__ == "__main__":
    raise SystemExit(main())
