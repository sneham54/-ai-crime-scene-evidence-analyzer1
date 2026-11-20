"""Run the normal video detector and summarize per-class observations."""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
CLASS_NAMES = {
    0: "gun", 1: "hammer", 2: "helmet", 3: "knife", 4: "mask",
    5: "person", 6: "pistol", 7: "rifle", 8: "weapon",
}
CSV_FIELDS = [
    "video", "class_id", "class_name", "detections", "avg_conf", "min_conf",
    "max_conf", "roi_detections", "time_sec", "frames_analyzed",
    "false_positives", "missed_detections",
]


def _load_ground_truth(video_path):
    """Load optional ``<video name>.gt.json`` class-count annotations."""
    candidates = (
        video_path.with_name(video_path.name + ".gt.json"),
        video_path.with_suffix(".gt.json"),
    )
    ground_truth_path = next((path for path in candidates if path.is_file()), None)
    if ground_truth_path is None:
        return None, None
    with ground_truth_path.open(encoding="utf-8") as ground_truth_file:
        records = json.load(ground_truth_file)
    if not isinstance(records, list):
        raise ValueError(f"Ground truth must be a list: {ground_truth_path}")
    counts = Counter()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "class_id" not in record or "bbox" not in record:
            raise ValueError(f"Ground-truth record {index} must include class_id and bbox: {ground_truth_path}")
        try:
            class_id = int(record["class_id"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid class_id in {ground_truth_path} record {index}") from error
        if not isinstance(record["bbox"], (list, tuple)) or len(record["bbox"]) != 4:
            raise ValueError(f"Ground-truth bbox in {ground_truth_path} record {index} must have four values")
        counts[class_id] += 1
    return ground_truth_path, counts


def _analyze_videos(video_paths):
    """Analyze each video once using the same VideoService used by the API."""
    sys.path.insert(0, str(BACKEND_DIR))
    from dotenv import load_dotenv
    from flask import Flask

    load_dotenv(PROJECT_ROOT / ".env")
    from app.config import Config
    from app.services.video_service import VideoService

    app = Flask("benchmark_backend", root_path=str(BACKEND_DIR / "app"))
    app.config.from_object(Config)
    reports = []
    with app.app_context():
        service = VideoService()
        for video_path in video_paths:
            started_at = time.perf_counter()
            result = service.analyze(str(video_path))
            elapsed = time.perf_counter() - started_at
            detections = result.get("detections", [])
            class_groups = defaultdict(list)
            roi_counts = Counter()
            for detection in detections:
                class_id = detection.get("class_id")
                if class_id is None:
                    continue
                class_id = int(class_id)
                class_groups[class_id].append(detection)
                if detection.get("source") == "roi-crop":
                    roi_counts[class_id] += 1

            ground_truth_path, ground_truth_counts = _load_ground_truth(video_path)
            class_ids = set(class_groups)
            if ground_truth_counts is not None:
                class_ids.update(ground_truth_counts)
            per_class = {}
            for class_id in sorted(class_ids):
                items = class_groups.get(class_id, [])
                confidences = [float(item.get("confidence", 0.0)) for item in items]
                gt_count = ground_truth_counts.get(class_id, 0) if ground_truth_counts is not None else None
                detection_count = len(items)
                per_class[class_id] = {
                    "class_name": (
                        items[0].get("class", CLASS_NAMES.get(class_id, f"class_{class_id}"))
                        if items else CLASS_NAMES.get(class_id, f"class_{class_id}")
                    ),
                    "detections": detection_count,
                    "avg_conf": round(sum(confidences) / len(confidences), 3) if confidences else None,
                    "min_conf": round(min(confidences), 3) if confidences else None,
                    "max_conf": round(max(confidences), 3) if confidences else None,
                    "roi_detections": roi_counts[class_id],
                    "false_positives": max(0, detection_count - gt_count) if gt_count is not None else None,
                    "missed_detections": max(0, gt_count - detection_count) if gt_count is not None else None,
                    "ground_truth": gt_count,
                }
            report = {
                "video": str(video_path),
                "ground_truth_file": str(ground_truth_path) if ground_truth_path else None,
                "ground_truth_available": ground_truth_counts is not None,
                "time_sec": round(elapsed, 3),
                "frames_analyzed": len(result.get("frame_results", [])),
                "false_positives": sum(item["false_positives"] or 0 for item in per_class.values()),
                "missed_detections": sum(item["missed_detections"] or 0 for item in per_class.values()),
                "per_class": per_class,
            }
            reports.append(report)
    return reports


def _write_reports(output_dir, reports):
    """Write CSV and JSON benchmark reports to the requested directory."""
    csv_path = output_dir / "benchmark_report.csv"
    json_path = output_dir / "benchmark_report.json"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for report in reports:
            for class_id, stats in report["per_class"].items():
                writer.writerow({
                    "video": report["video"],
                    "class_id": class_id,
                    "class_name": stats["class_name"],
                    "detections": stats["detections"],
                    "avg_conf": stats["avg_conf"],
                    "min_conf": stats["min_conf"],
                    "max_conf": stats["max_conf"],
                    "roi_detections": stats["roi_detections"],
                    "time_sec": report["time_sec"],
                    "frames_analyzed": report["frames_analyzed"],
                    "false_positives": stats["false_positives"],
                    "missed_detections": stats["missed_detections"],
                })
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(reports, json_file, indent=2)
    return csv_path, json_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos_folder", type=Path, help="Folder to scan recursively for videos")
    args = parser.parse_args(argv)
    if not args.videos_folder.is_dir():
        print(f"[ERROR] {args.videos_folder} is not a directory")
        return 1

    videos = sorted(
        path for path in args.videos_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    try:
        reports = _analyze_videos(videos) if videos else []
        csv_path, json_path = _write_reports(Path.cwd(), reports)
    except Exception as error:
        print(f"[ERROR] Benchmark failed: {type(error).__name__}: {error}")
        return 1

    if not videos:
        print(f"Warning: no supported video files found in {args.videos_folder}")
    print(f"Benchmark completed. CSV saved to {csv_path}, JSON saved to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
