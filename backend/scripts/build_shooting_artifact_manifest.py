"""Build a read-only-source manifest for existing Shooting tracking outputs."""
import argparse
import csv
import os
from pathlib import Path


def build(status_path: Path, results_dir: Path, videos_dir: Path, source_dir: Path, output: Path):
    records = []
    with status_path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            video_name = Path(row.get("video") or "").name
            csv_name = Path(row.get("output_csv") or "").name
            tracked_name = Path(row.get("output_video") or "").name
            original_candidates = [path for path in source_dir.rglob(video_name) if path.is_file()] if video_name and source_dir.is_dir() else []
            detection = results_dir / csv_name if csv_name else None
            tracked = videos_dir / tracked_name if tracked_name else None
            records.append({
                "original_video": video_name,
                "original_video_local": str(original_candidates[0]) if len(original_candidates) == 1 else "",
                "original_match_count": len(original_candidates),
                "tracking_status": row.get("status", ""),
                "detection_csv_expected": row.get("output_csv", ""),
                "detection_csv_local": str(detection.resolve()) if detection and detection.is_file() else "",
                "detection_csv_exists": bool(detection and detection.is_file()),
                "tracked_video_expected": row.get("output_video", ""),
                "tracked_video_local": str(tracked.resolve()) if tracked and tracked.is_file() else "",
                "tracked_video_exists": bool(tracked and tracked.is_file()),
                "error": row.get("error", ""),
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["original_video", "original_video_local", "original_match_count", "tracking_status",
              "detection_csv_expected", "detection_csv_local", "detection_csv_exists",
              "tracked_video_expected", "tracked_video_local", "tracked_video_exists", "error"]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[1] / "tracked_results_shooting_import" / "main_archive" / "Shooting"
    parser.add_argument("--status", type=Path, default=default_root / "shooting_tracking_status.csv")
    parser.add_argument("--results", type=Path, default=default_root / "detections")
    parser.add_argument("--tracked", type=Path, default=default_root / "videos")
    parser.add_argument("--sources", type=Path, default=Path(os.getenv("EVIDENCE_SOURCE_DIR", Path.home() / "OneDrive" / "Documents" / "crime_video_dataset")) / "Shooting")
    parser.add_argument("--output", type=Path, default=default_root / "shooting_artifact_manifest.csv")
    args = parser.parse_args()
    rows = build(args.status, args.results, args.tracked, args.sources, args.output)
    print(f"Wrote {len(rows)} status-linked artifact rows to {args.output}")
    print(f"Missing canonical CSVs: {sum(not row['detection_csv_exists'] for row in rows)}")
    print(f"Missing tracked MP4s: {sum(not row['tracked_video_exists'] for row in rows)}")
    print(f"Missing/ambiguous originals: {sum(row['original_match_count'] != 1 for row in rows)}")
