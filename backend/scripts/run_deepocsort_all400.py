"""Preflight and resumable 5-video-batch Deep OC-SORT evaluation."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2


BACKEND = Path(__file__).resolve().parents[1]
PROJECT = BACKEND.parent
SOURCE_MANIFEST = PROJECT / "reports" / "source_video_manifest.csv"
OUTPUT_ROOT = BACKEND / "tracking_outputs" / "deepocsort_all400_run_20261119"
REPORTS = PROJECT / "reports"
MASTER_CSV = REPORTS / "deepocsort_all400_processing_manifest.csv"
SUMMARY_JSON = REPORTS / "deepocsort_all400_summary.json"
OVERALL_MD = REPORTS / "deepocsort_all400_overall_report.md"
CATEGORY_MD = REPORTS / "deepocsort_all400_category_report.md"
FAILURE_MD = REPORTS / "deepocsort_all400_failure_analysis.md"
REVIEW_CSV = REPORTS / "deepocsort_all400_visual_review_index.csv"
EXPECTED = {"Assault": 100, "Shooting": 100, "Robbery": 150, "Normal": 50}
FIELDS = ["manifest_index", "batch_number", "video_path", "category", "source_sha256",
          "status", "started_at_utc", "ended_at_utc", "output_folder", "error_details",
          "frames_processed", "source_frame_count", "frames_with_detections", "raw_yolo_detections",
          "tracked_detections", "unique_track_ids", "frames_with_no_detections",
          "detection_to_track_coverage", "average_track_duration_seconds",
          "median_track_duration_seconds", "processing_seconds", "tracker_error_count"]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_sources():
    if not SOURCE_MANIFEST.is_file():
        raise RuntimeError(f"source_video_manifest.csv missing: {SOURCE_MANIFEST}")
    rows = []
    with SOURCE_MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        for index, row in enumerate(csv.DictReader(f), 1):
            category = row["category"].strip().title()
            rows.append({"manifest_index": index, "batch_number": (index - 1) // 5 + 1,
                         "video_path": str(Path(row["source_path"]).resolve()), "category": category,
                         "source_sha256": row.get("source_sha256", "").strip(),
                         "status": "pending", "started_at_utc": "", "ended_at_utc": "",
                         "output_folder": "", "error_details": ""})
    counts = Counter(r["category"] for r in rows)
    paths = [r["video_path"].casefold() for r in rows]
    if len(rows) != 400 or dict(counts) != EXPECTED:
        raise RuntimeError(f"Manifest counts mismatch: rows={len(rows)}, categories={dict(counts)}")
    if len(set(paths)) != 400:
        raise RuntimeError(f"Manifest has duplicate source paths: unique={len(set(paths))}")
    return rows


def preflight(rows, verify_hashes=False):
    issues, hash_to_paths, total_bytes = [], defaultdict(list), 0
    for row in rows:
        path = Path(row["video_path"])
        if not path.is_file() or path.stat().st_size <= 0:
            issues.append({"path": str(path), "category": row["category"], "problem": "missing_or_empty"})
            continue
        total_bytes += path.stat().st_size
        expected_hash = row["source_sha256"]
        if expected_hash:
            hash_to_paths[expected_hash.casefold()].append(str(path))
            if verify_hashes:
                actual = digest(path)
                if actual.casefold() != expected_hash.casefold():
                    issues.append({"path": str(path), "category": row["category"],
                                   "problem": f"sha256_mismatch:{actual}"})
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            issues.append({"path": str(path), "category": row["category"], "problem": "container_open_failed"})
            continue
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        ok, frame = cap.read()
        cap.release()
        if frames <= 0 or fps <= 0 or not ok or frame is None:
            issues.append({"path": str(path), "category": row["category"], "problem": "invalid_metadata_or_first_frame"})
    duplicate_content = {key: values for key, values in hash_to_paths.items() if len(values) > 1}
    disk = shutil.disk_usage(BACKEND)
    return {"source_count": len(rows), "category_counts": dict(Counter(r["category"] for r in rows)),
            "unique_source_paths": len({r["video_path"].casefold() for r in rows}),
            "unique_manifest_hashes": len(hash_to_paths), "duplicate_content_groups": duplicate_content,
            "missing_empty_corrupt_quickcheck": issues, "source_bytes_total": total_bytes,
            "disk_free_bytes": disk.free, "disk_total_bytes": disk.total,
            "hashes_verified": bool(verify_hashes)}


def save_manifest(rows):
    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    temp = MASTER_CSV.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(MASTER_CSV)


def load_resume(rows):
    if not MASTER_CSV.exists():
        return rows
    with MASTER_CSV.open(encoding="utf-8-sig", newline="") as f:
        old = list(csv.DictReader(f))
    old_by_path = {r["video_path"].casefold(): r for r in old}
    for row in rows:
        prior = old_by_path.get(row["video_path"].casefold())
        if prior:
            if prior.get("source_sha256") != row["source_sha256"]:
                raise RuntimeError(f"Source manifest changed after evaluation began: {row['video_path']}")
            row.update(prior)
            row["manifest_index"] = int(row["manifest_index"])
            row["batch_number"] = int(row["batch_number"])
            if row["status"] == "processing":
                row["status"] = "pending"
                row["error_details"] = "Previous process interrupted; resuming this video from frame zero."
    return rows


def collect_metadata(row):
    meta_path = Path(row["output_folder"]) / "metadata.json"
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mapping = {"frames_processed": "frames_processed", "source_frame_count": "source_frame_count",
               "frames_with_detections": "frames_with_detections", "raw_yolo_detections": "raw_yolo_detections",
               "tracked_detections": "tracked_detections", "unique_track_ids": "unique_track_ids",
               "frames_with_no_detections": "frames_with_no_detections",
               "detection_to_track_coverage": "detection_to_track_coverage",
               "average_track_duration_seconds": ("track_duration_seconds", "average"),
               "median_track_duration_seconds": ("track_duration_seconds", "median"),
               "processing_seconds": "processing_seconds", "tracker_error_count": "tracker_error_count"}
    for out_key, key in mapping.items():
        value = meta
        for part in key if isinstance(key, tuple) else (key,):
            value = value.get(part) if isinstance(value, dict) else None
        row[out_key] = value
    row["status"] = meta.get("processing_status", "failed")
    if row["status"] == "completed":
        row["error_details"] = ""
    elif row["status"] == "partial":
        row["error_details"] = f"Partial frame decode: {meta.get('frames_processed')}/{meta.get('source_frame_count')} frames"


def derive(rows, preflight_result, annotate, run_state):
    category = {}
    for name, expected in EXPECTED.items():
        group = [r for r in rows if r["category"] == name]
        category[name] = summarize(group, expected)
    all_summary = summarize(rows, 400)
    valid_rows = [r for r in rows if r.get("status") in ("completed", "partial")]
    coverage_values = [float(r["detection_to_track_coverage"]) for r in valid_rows
                       if r.get("detection_to_track_coverage") not in (None, "")]
    best_category = None
    candidates = [(v["mean_detection_to_track_coverage"], k) for k, v in category.items()
                  if v["mean_detection_to_track_coverage"] is not None]
    if candidates:
        best_category = max(candidates)[1]
    failures = []
    for r in rows:
        if r.get("status") in ("failed", "partial") or not r.get("unique_track_ids") or \
                (r.get("detection_to_track_coverage") not in (None, "") and
                 float(r["detection_to_track_coverage"]) < .5):
            failures.append({"video_path": r["video_path"], "category": r["category"],
                             "status": r.get("status"), "raw_detections": safe_num(r.get("raw_yolo_detections")),
                             "unique_track_ids": safe_num(r.get("unique_track_ids")),
                             "track_coverage": safe_float(r.get("detection_to_track_coverage")),
                             "output_folder": r.get("output_folder", ""),
                             "reason": r.get("error_details") or failure_reason(r)})
    payload = {"expected_videos": 400, "accounted_videos": len(rows),
               "overall": all_summary, "categories": category,
               "best_diagnostic_category_by_mean_detection_to_track_coverage": best_category,
               "mean_detection_to_track_coverage_across_videos": sum(coverage_values)/len(coverage_values) if coverage_values else None,
               "ground_truth_available": False,
               "accuracy_statement": "IDF1, HOTA and identity-switch accuracy are unavailable without valid identity ground truth.",
               "annotated_outputs_enabled": annotate,
               "annotation_skip_reason": None if annotate else "Insufficient free disk space for duplicate video outputs; tracking artifacts still generated.",
               "preflight": preflight_result, "run_state": run_state,
               "evaluation_output_folder": str(OUTPUT_ROOT), "failure_review_count": len(failures),
               "failure_review": failures}
    return payload


def safe_num(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def safe_float(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def failure_reason(row):
    detections, ids = safe_num(row.get("raw_yolo_detections")), safe_num(row.get("unique_track_ids"))
    if row.get("status") == "pending":
        return "pending"
    if row.get("status") == "failed":
        return row.get("error_details") or "processing_failure"
    if detections == 0:
        return "no_yolo_detections"
    if ids == 0:
        return "yolo_detections_but_no_track_ids"
    if safe_float(row.get("detection_to_track_coverage")) is not None and safe_float(row.get("detection_to_track_coverage")) < .5:
        return "low_detection_to_track_coverage"
    return "review_short_or_fragmented_tracks"


def summarize(rows, expected):
    completed = [r for r in rows if r.get("status") == "completed"]
    partial = [r for r in rows if r.get("status") == "partial"]
    failed = [r for r in rows if r.get("status") == "failed"]
    processed = completed + partial
    coverages = [safe_float(r.get("detection_to_track_coverage")) for r in processed]
    coverages = [x for x in coverages if x is not None]
    return {"expected": expected, "accounted": len(rows), "completed": len(completed),
            "partial": len(partial), "failed": len(failed),
            "pending": sum(r.get("status") in ("pending", "processing", "") for r in rows),
            "videos_with_no_detections": sum(safe_num(r.get("raw_yolo_detections")) == 0 for r in processed),
            "videos_with_zero_track_ids": sum(safe_num(r.get("unique_track_ids")) == 0 for r in processed),
            "videos_with_track_ids": sum(safe_num(r.get("unique_track_ids")) > 0 for r in processed),
            "total_frames_processed": sum(safe_num(r.get("frames_processed")) for r in processed),
            "total_yolo_detections": sum(safe_num(r.get("raw_yolo_detections")) for r in processed),
            "total_tracked_detections": sum(safe_num(r.get("tracked_detections")) for r in processed),
            "total_unique_track_ids_video_local_sum": sum(safe_num(r.get("unique_track_ids")) for r in processed),
            "mean_detection_to_track_coverage": sum(coverages)/len(coverages) if coverages else None,
            "common_failure_reasons": dict(Counter(failure_reason(r) for r in rows
                                                   if r.get("status") not in ("completed", "partial") or
                                                   failure_reason(r) != "review_short_or_fragmented_tracks"))}


def write_reports(rows, preflight_result, annotate, run_state):
    payload = derive(rows, preflight_result, annotate, run_state)
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    o = payload["overall"]
    OVERALL_MD.write_text(
        "# Deep OC-SORT all-source-video evaluation\n\n"
        f"Run state: **{run_state}**. Accounted: **{len(rows)}/400**. Completed: **{o['completed']}**; "
        f"partial: **{o['partial']}**; failed: **{o['failed']}**; pending: **{o['pending']}**.\n\n"
        f"Videos with IDs: **{o['videos_with_track_ids']}**; zero IDs: **{o['videos_with_zero_track_ids']}**; "
        f"no YOLO detections: **{o['videos_with_no_detections']}**.\n\n"
        f"Raw detections: **{o['total_yolo_detections']}**; tracked detections: **{o['total_tracked_detections']}**; "
        f"video-local ID sum: **{o['total_unique_track_ids_video_local_sum']}**.\n\n"
        f"Diagnostic mean detection-to-track coverage: **{o['mean_detection_to_track_coverage']}**; "
        "this is not tracking accuracy. No IDF1, HOTA, or identity-switch accuracy is reported because valid "
        "identity ground truth is unavailable.\n\n"
        f"Tracker: Ultralytics Deep OC-SORT 8.4.174, CPU, `best.pt`, official project YAML, ReID off, GMC none, "
        f"Byte association off. Annotated video enabled: `{annotate}`.\n\n"
        f"Artifacts: `{OUTPUT_ROOT}`. Processing manifest: `{MASTER_CSV}`.\n", encoding="utf-8")
    table = ["# Deep OC-SORT category report", "", "| Category | Expected | Completed | Partial | Failed | Pending | Zero detections | Zero IDs | IDs | Detections | Tracked | Mean coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, c in payload["categories"].items():
        table.append(f"| {name} | {c['expected']} | {c['completed']} | {c['partial']} | {c['failed']} | {c['pending']} | {c['videos_with_no_detections']} | {c['videos_with_zero_track_ids']} | {c['videos_with_track_ids']} | {c['total_yolo_detections']} | {c['total_tracked_detections']} | {c['mean_detection_to_track_coverage']} |")
    table += ["", f"Best category by mean diagnostic detection-to-track coverage: **{payload['best_diagnostic_category_by_mean_detection_to_track_coverage']}**. This comparison is not an accuracy measure.", ""]
    CATEGORY_MD.write_text("\n".join(table), encoding="utf-8")
    failure_lines = ["# Deep OC-SORT failure and visual review", "",
                     f"Review rows: {len(payload['failure_review'])}. See `{REVIEW_CSV}`.", "",
                     f"Annotated outputs enabled: `{annotate}`. " + ("All output videos are present in the review index." if annotate else payload["annotation_skip_reason"]), "",
                     "Every listed row distinguishes no detections, detections without IDs, partial reads, and processing failures. Non-empty IDs do not prove identity correctness.", ""]
    FAILURE_MD.write_text("\n".join(failure_lines), encoding="utf-8")
    with REVIEW_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["video_path", "category", "status", "annotated_video_path", "raw_detections", "unique_track_ids", "track_coverage", "review_reason"])
        w.writeheader()
        for r in payload["failure_review"]:
            annotated_path = str(Path(r["output_folder"]) / "annotated.mp4") if annotate and r.get("output_folder") else ""
            w.writerow({"video_path": r["video_path"], "category": r["category"], "status": r["status"],
                        "annotated_video_path": annotated_path, "raw_detections": r["raw_detections"],
                        "unique_track_ids": r["unique_track_ids"], "track_coverage": r["track_coverage"],
                        "review_reason": r["reason"]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify-hashes", action="store_true", help="Hash all 8 GB of source files in preflight")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--save-annotated", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    args = p.parse_args()
    rows = read_sources()
    pf = preflight(rows, args.verify_hashes)
    if pf["missing_empty_corrupt_quickcheck"]:
        raise RuntimeError(f"Preflight found source issues: {pf['missing_empty_corrupt_quickcheck'][:10]}")
    if pf["duplicate_content_groups"]:
        print(f"WARNING duplicate content hashes: {len(pf['duplicate_content_groups'])}")
    estimate = pf["source_bytes_total"]
    annotation_enabled = args.save_annotated and pf["disk_free_bytes"] > estimate + 2 * 1024**3
    print(json.dumps({"preflight": pf, "annotated_video_enabled": annotation_enabled,
                      "annotation_requested": args.save_annotated,
                      "manifest": str(SOURCE_MANIFEST), "tracker": "DeepOCSORT", "model": str(BACKEND / "models" / "best.pt")}, indent=2))
    if args.preflight_only or not args.run:
        return
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if MASTER_CSV.exists():
        rows = load_resume(rows)
    else:
        # Refuse to replace any pre-existing evaluation report set.
        existing = [str(x) for x in (SUMMARY_JSON, OVERALL_MD, CATEGORY_MD, FAILURE_MD) if x.exists()]
        if existing:
            raise RuntimeError(f"Refusing to overwrite existing report(s): {existing}")
    for row in rows:
        if row.get("status") == "processing":
            row["status"] = "pending"
    save_manifest(rows)
    write_reports(rows, pf, annotation_enabled, "running")
    runner = BACKEND / "scripts" / "run_deepocsort.py"
    python = BACKEND / "deepocsort_venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError(f"Isolated tracker Python is missing: {python}")
    for batch_start in range(0, len(rows), 5):
        batch = rows[batch_start:batch_start+5]
        todo = [r for r in batch if r.get("status") not in ("completed", "partial") and
                (r.get("status") != "failed" or args.retry_failed)]
        if not todo:
            continue
        print(f"BATCH {batch[0]['batch_number']:03d}/080 start ({len(todo)} outstanding in 5-video group)", flush=True)
        for row in todo:
            attempt = 1
            base = f"{row['manifest_index']:03d}_{Path(row['video_path']).stem}"
            while True:
                dest = OUTPUT_ROOT / (base if attempt == 1 else f"{base}_attempt{attempt:02d}")
                if not dest.exists():
                    break
                attempt += 1
            row.update({"status": "processing", "started_at_utc": now(), "ended_at_utc": "",
                        "output_folder": str(dest), "error_details": ""})
            save_manifest(rows)
            command = [str(python), str(runner), row["video_path"], "--output", str(dest),
                       "--model", str(BACKEND / "models" / "best.pt"),
                       "--tracker", str(BACKEND / "config" / "deepocsort.yaml")]
            if annotation_enabled:
                command.append("--save-annotated")
            print(f"START {row['manifest_index']:03d}/400 {row['category']} {Path(row['video_path']).name}", flush=True)
            try:
                result = subprocess.run(command, cwd=BACKEND, text=True, capture_output=True)
                (dest / "runner.stdout.log").write_text(result.stdout or "", encoding="utf-8")
                (dest / "runner.stderr.log").write_text(result.stderr or "", encoding="utf-8")
                if result.returncode != 0:
                    row["status"] = "failed"
                    row["error_details"] = (result.stderr or result.stdout or f"runner exit {result.returncode}")[-4000:]
                else:
                    collect_metadata(row)
                    if row["status"] not in ("completed", "partial"):
                        row["status"] = "failed"
                        row["error_details"] = "Runner metadata had an invalid processing_status"
            except Exception as exc:
                row["status"] = "failed"
                row["error_details"] = f"{type(exc).__name__}: {exc}"
            row["ended_at_utc"] = now()
            save_manifest(rows)
            print(f"DONE {row['manifest_index']:03d}/400 status={row['status']} frames={row.get('frames_processed')} detections={row.get('raw_yolo_detections')} ids={row.get('unique_track_ids')} error={row.get('error_details','')[:160]}", flush=True)
            write_reports(rows, pf, annotation_enabled, "running")
        print(f"BATCH {batch[0]['batch_number']:03d}/080 complete", flush=True)
    final_state = "complete" if all(r.get("status") == "completed" for r in rows) else "accounted_with_failures_or_partial_videos"
    write_reports(rows, pf, annotation_enabled, final_state)
    print(f"FINISHED completed={sum(r['status']=='completed' for r in rows)} partial={sum(r['status']=='partial' for r in rows)} failed={sum(r['status']=='failed' for r in rows)} pending={sum(r['status']=='pending' for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
