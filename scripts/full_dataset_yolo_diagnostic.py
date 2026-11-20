#!/usr/bin/env python3
"""Resumable 1-fps YOLO inventory for the crime-video dataset.

Placeholder labels are written for workflow compatibility only. A full-frame
"gun" box is not ground truth; this tool therefore never presents placeholder
comparisons as accuracy or computes mAP. best.pt is loaded read-only.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASSES = ("Assault", "Robbery", "Shooting", "Normal")
DEFAULT_DATASET = Path(r"C:\Users\admin\OneDrive\Documents\crime_video_dataset")
DEFAULT_PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = Path(r"C:\Users\admin\.gemini\antigravity\brain\9bf660b7-ad56-4623-b73c-0719aa8d43c2")
CLASS_NAMES = {
    0: "gun", 1: "hammer", 2: "helmet", 3: "knife", 4: "mask",
    5: "person", 6: "pistol", 7: "rifle", 8: "weapon",
}


def markdown_table(headers, rows):
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    return ("| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" +
            "".join("| " + " | ".join(clean(v) for v in row) + " |\n" for row in rows))


def atomic_json(path: Path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def write_report(path, dataset, split, records, class_names, interval, confidence):
    totals = Counter()
    videos_ok = videos_error = sampled_frames = 0
    for record in records.values():
        if record.get("error"):
            videos_error += 1
            continue
        videos_ok += 1
        sampled_frames += int(record.get("sampled_frames", 0))
        totals.update({int(k): int(v) for k, v in record.get("detections", {}).items()})
    all_count = sum(totals.values())
    names = {int(k): v for k, v in class_names.items()}
    count_rows = [[names.get(cid, str(cid)), totals[cid], f"{100*totals[cid]/all_count:.2f}%" if all_count else "0.00%"]
                  for cid in sorted(names)]
    labeled_names = {Path(p).name for p in split["labeled"]}
    labeled_records = [r for r in records.values() if Path(r["path"]).name in labeled_names and not r.get("error")]
    synthetic_rows = []
    for cid in sorted(names):
        tp = sum(int(r.get("placeholder_diagnostic", {}).get(str(cid), {}).get("tp", 0)) for r in labeled_records)
        fp = sum(int(r.get("placeholder_diagnostic", {}).get(str(cid), {}).get("fp", 0)) for r in labeled_records)
        fn = sum(int(r.get("placeholder_diagnostic", {}).get(str(cid), {}).get("fn", 0)) for r in labeled_records)
        synthetic_rows.append([names[cid], tp, fp, fn, "not reported", "not reported", "not reported"])
    body = [
        "# Full-dataset YOLO diagnostic report\n\n",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n",
        "## Test split and method\n\n",
        f"- Dataset root: `{dataset}`\n- Videos discovered: {len(split['labeled']) + len(split['unlabeled'])}\n",
        f"- Placeholder subset: {len(split['labeled'])} videos ({split['ratio']:.0%}); seed {split['seed']}.\n",
        f"- Unlabeled subset: {len(split['unlabeled'])} videos.\n",
        f"- Inference cadence: one sampled frame every {interval:g} second(s); confidence threshold: {confidence:.2f}.\n",
        f"- Videos completed: {videos_ok}; errors: {videos_error}; sampled frames processed: {sampled_frames}.\n",
        "- YOLO checkpoint is loaded read-only.\n\n",
        "## Placeholder-label status and diagnostic\n\n",
        "Each selected video's scratch label file contains a synthetic class-0 full-frame box. This is not a manual annotation and does not describe the objects in the frames. The table below is only a placeholder-overlap diagnostic on the first sampled frame per selected video; it is not model accuracy. Precision, recall, F1, and mAP are intentionally not reported because placeholder boxes are not ground truth.\n\n",
        markdown_table(["Class", "Synthetic TP", "Synthetic FP", "Synthetic FN", "Precision", "Recall", "F1"], synthetic_rows),
        "\n## Detection frequencies across the dataset\n\n",
        f"Counts include detections at confidence ≥ {confidence:.2f} from all completed videos and sampled frames. These are model detections, not verified object counts.\n\n",
        markdown_table(["Class", "Detection instances", "% of detections"], count_rows),
        f"\nTotal detections: **{all_count}**.\n\n",
        "## Replacing placeholders with real annotations\n\n",
        "Manually inspect sampled frames in LabelImg (or another annotation tool), choose YOLO format, and save one label text file for each annotated frame with rows `class_id x_center y_center width height` normalized to [0,1]. Store the real frame images and labels in a separate dataset tree with matching image/label basenames; do not overwrite the scratch placeholders. Then evaluate predictions against those verified frame-level labels using one-to-one class-aware IoU matching. Only compute mAP when a valid labeled evaluation set and confidence-ranked detections are available.\n\n",
        "## Run status\n\n",
        "The run is resumable. Re-run the same command to skip completed videos and continue from the per-video checkpoint JSON.\n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(body), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=.20)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=.30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--limit-videos", type=int, default=0, help="Optional smoke run; 0 processes all videos")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Write the seeded split and synthetic placeholder files, but do not run inference")
    args = parser.parse_args()
    if not 0 < args.ratio < 1 or args.interval <= 0 or args.batch <= 0:
        parser.error("ratio, interval, and batch must be positive (ratio < 1)")
    if not args.dataset.is_dir():
        parser.error(f"Dataset directory does not exist: {args.dataset}")

    videos = []
    for category in CLASSES:
        folder = args.dataset / category
        if folder.is_dir():
            videos.extend(sorted(folder.rglob("*.mp4")))
    if not videos:
        parser.error("No .mp4 files found in expected class subfolders")
    rng = random.Random(args.seed)
    labeled = sorted(rng.sample(videos, max(1, round(len(videos) * args.ratio))))
    labeled_set = {str(p.resolve()) for p in labeled}
    report_root = args.report_root
    report_root.mkdir(parents=True, exist_ok=True)
    scratch = args.project / "scratch" / "full_dataset_yolo_diagnostic"
    label_dir = scratch / "placeholder_labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    for video in labeled:
        # Full-frame class-0 pseudo-label: generated only under scratch and never
        # used as a claim of real ground truth.
        (label_dir / f"{video.stem}.txt").write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
    split = {
        "seed": args.seed, "ratio": args.ratio,
        "labeled": [str(p.resolve()) for p in labeled],
        "unlabeled": [str(p.resolve()) for p in videos if str(p.resolve()) not in labeled_set],
        "warning": "Generated placeholder labels are synthetic and are not ground truth.",
    }
    atomic_json(report_root / "test_split.json", split)
    if args.prepare_only:
        print(f"Prepared {len(labeled)} placeholder labels under {label_dir}")
        print(f"Split: {report_root / 'test_split.json'}")
        print("No inference or accuracy evaluation was performed.")
        return
    state_tag = f"i{args.interval:g}_c{args.confidence:g}_s{args.imgsz}_b{args.batch}"
    state_path = scratch / f"progress_{state_tag}.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"records": {}}
    except (OSError, json.JSONDecodeError):
        state = {"records": {}}
    records = state.setdefault("records", {})
    model_path = args.project / "backend" / "models" / "best.pt"
    model = YOLO(str(model_path))
    names = {int(k): str(v) for k, v in model.names.items()} if isinstance(model.names, dict) else {i: str(v) for i, v in enumerate(model.names)}
    work = videos[:args.limit_videos] if args.limit_videos else videos
    started = time.monotonic()
    for video_number, video in enumerate(work, 1):
        key = str(video.resolve())
        if key in records and records[key].get("completed"):
            continue
        record = {"path": key, "detections": {}, "sampled_frames": 0}
        cap = cv2.VideoCapture(key)
        if not cap.isOpened():
            cap.release()
            record.update(completed=True, error="OpenCV could not open video")
            records[key] = record
            atomic_json(state_path, state)
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            cap.release()
            record.update(completed=True, error="Invalid FPS or frame count")
            records[key] = record
            atomic_json(state_path, state)
            continue
        stride = max(1, round(fps * args.interval))
        target_indices = range(0, frame_count, stride)
        first_frame = None
        batch_frames = []
        def infer_batch(frames, is_first_batch):
            if not frames:
                return None
            results = model.predict(frames, verbose=False, conf=args.confidence,
                                    imgsz=args.imgsz, batch=len(frames))
            first_diag = None
            for batch_i, result in enumerate(results):
                boxes = result.boxes
                detections = []
                if boxes is not None:
                    for b in boxes:
                        cls_id = int(b.cls.item())
                        record["detections"][str(cls_id)] = record["detections"].get(str(cls_id), 0) + 1
                        detections.append((cls_id, float(b.conf.item()), b.xyxy[0].tolist()))
                record["sampled_frames"] += 1
                if is_first_batch and batch_i == 0 and key in labeled_set:
                    # Synthetic full-frame class-0 match on one frame only.
                    h, w = frames[0].shape[:2]
                    full_box = [0, 0, w, h]
                    best = max((box for box in detections if box[0] == 0),
                               key=lambda box: _iou(box[2], full_box), default=None)
                    diag = {str(cid): {"tp": 0, "fp": 0, "fn": 0} for cid in names}
                    if best is not None and _iou(best[2], full_box) >= .5:
                        diag["0"]["tp"] = 1
                    else:
                        diag["0"]["fn"] = 1
                    for cls_id, _conf, _box in detections:
                        diag.setdefault(str(cls_id), {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
                    # If class-0 prediction matched the pseudo-label, remove its FP.
                    if best is not None and _iou(best[2], full_box) >= .5:
                        diag["0"]["fp"] = max(0, diag["0"]["fp"] - 1)
                    first_diag = diag
            return first_diag
        try:
            first_batch = True
            for frame_index in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    continue
                batch_frames.append(frame)
                if len(batch_frames) >= args.batch:
                    diag = infer_batch(batch_frames, first_batch)
                    if diag is not None:
                        record["placeholder_diagnostic"] = diag
                    first_batch = False
                    batch_frames = []
            if batch_frames:
                diag = infer_batch(batch_frames, first_batch)
                if diag is not None:
                    record["placeholder_diagnostic"] = diag
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            cap.release()
        record["completed"] = True
        records[key] = record
        atomic_json(state_path, state)
        write_report(report_root / "full_dataset_accuracy_report.md", args.dataset,
                     split, records, names, args.interval, args.confidence)
        elapsed = time.monotonic() - started
        print(f"[{video_number}/{len(work)}] {video.name}: {record.get('sampled_frames',0)} frames; "
              f"{len(records)}/{len(videos)} records checkpointed; elapsed {elapsed/60:.1f} min", flush=True)
    write_report(report_root / "full_dataset_accuracy_report.md", args.dataset,
                 split, records, names, args.interval, args.confidence)
    print(f"Report: {report_root / 'full_dataset_accuracy_report.md'}")
    print(f"Split: {report_root / 'test_split.json'}")
    print(f"Progress/resume state: {state_path}")


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


if __name__ == "__main__":
    main()
