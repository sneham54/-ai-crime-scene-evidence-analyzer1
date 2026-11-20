#!/usr/bin/env python3
"""Audit the crime video dataset and evaluate the repository's existing models.

Uses only packages already installed in the backend virtual environment. The
report defaults to the requested Antigravity brain location; pass --report to
choose a writable alternate location. best.pt is opened read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2


CLASSES = ("Assault", "Robbery", "Shooting", "Normal")
ENCODING_SUFFIX = re.compile(r"(?:[_ -](?:x264[a-z]?|copy(?:\s*\(\d+\))?|duplicate\d*))+$", re.I)
PROJECT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT / "backend"
DEFAULT_DATA = Path(r"C:\Users\admin\OneDrive\Documents\crime_video_dataset")
DEFAULT_REPORT = Path(r"C:\Users\admin\.gemini\antigravity\brain\6523bb00-4a5c-4155-9724-cd70bf00eafc\accuracy_evaluation_report.md")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(root: Path):
    rows, errors, duplicates = [], [], defaultdict(list)
    for category in CLASSES:
        folder = root / category
        if not folder.is_dir():
            errors.append({"path": str(folder), "error": "Category directory is missing"})
            continue
        for path in sorted(folder.rglob("*.mp4")):
            size = path.stat().st_size
            digest = sha256(path)
            duplicates[digest].append(str(path))
            cap = cv2.VideoCapture(str(path))
            opened = cap.isOpened()
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
            cap.release()
            good = opened and fps > 0 and count > 0 and width > 0 and height > 0
            duration = count / fps if good else None
            row = {"category": category, "path": str(path), "size": size,
                   "duration": duration, "fps": fps if opened else None,
                   "resolution": f"{width}x{height}" if good else "unavailable",
                   "sha256": digest, "status": "OK" if good else "CORRUPT/UNREADABLE"}
            rows.append(row)
            if not good:
                errors.append({"path": str(path), "error": "OpenCV could not open video or returned invalid metadata"})
    dupes = [(digest, paths) for digest, paths in duplicates.items() if len(paths) > 1]
    return rows, errors, dupes


def md_table(headers, rows):
    esc = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "".join(
        "| " + " | ".join(esc(x) for x in row) + " |\n" for row in rows)


def audit_section(rows, errors, dupes):
    out = ["## Dataset audit\n\n### Per-video inventory\n\n",
           md_table(["Category", "Full path", "Size (bytes)", "Duration (s)", "FPS", "Resolution", "SHA-256", "Status"],
                    [[r["category"], r["path"], r["size"], f'{r["duration"]:.3f}' if r["duration"] is not None else "—",
                      f'{r["fps"]:.3f}' if r["fps"] is not None else "—", r["resolution"], r["sha256"], r["status"]] for r in rows]),
           "\n### Category summary\n\n"]
    summary = []
    for c in CLASSES:
        items = [r for r in rows if r["category"] == c]
        durations = [r["duration"] for r in items if r["duration"] is not None]
        fps = [r["fps"] for r in items if r["fps"] and r["fps"] > 0]
        resolutions = Counter(r["resolution"] for r in items)
        summary.append([c, len(items), f"{sum(durations)/3600:.3f}", f"{sum(fps)/len(fps):.3f}" if fps else "—",
                        ", ".join(f"{k}: {v}" for k, v in sorted(resolutions.items())) or "—"])
    out.append(md_table(["Category", "Videos", "Total duration (h)", "Average FPS", "Resolution breakdown"], summary))
    out.append("\n### Corrupt or unreadable videos\n\n")
    out.append(md_table(["Path", "Error"], [[e["path"], e["error"]] for e in errors]) if errors else "None.\n")
    out.append("\n### Duplicate files (SHA-256)\n\n")
    out.append(md_table(["SHA-256", "Matching paths"], [[h, "<br>".join(p)] for h, p in dupes]) if dupes else "None found.\n")
    return "".join(out)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def yolo_section(root, rows, scratch):
    by_stem = defaultdict(list)
    for txt in root.rglob("*.txt"):
        by_stem[txt.stem].append(txt)
    annotated = [(r, by_stem.get(Path(r["path"]).stem, [])) for r in rows]
    annotated = [(r, p[0]) for r, p in annotated if p]
    if not annotated:
        return ("## YOLO annotation status\n\nNo ground-truth YOLO bounding-box labels were found whose base names match dataset videos. ") + (
            "A manually annotated subset can be created in LabelImg: open sampled video frames, draw boxes, choose YOLO format, "
            "and save one `.txt` per image/frame using `class_id x_center y_center width height` normalized to [0,1]. "
            "For this evaluator, name each label file to match its source video basename and document the labeled frame.\n")
    from ultralytics import YOLO
    model = YOLO(str(BACKEND / "models" / "best.pt"))
    counts = {c: Counter() for c in CLASSES}
    records = []
    for row, label in annotated:
        cap = cv2.VideoCapture(row["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            records.append([Path(row["path"]).name, "frame unavailable", 0, 0, 0])
            continue
        height, width = frame.shape[:2]
        gt = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, xc, yc, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
            gt.append((cls, [(xc-bw/2)*width, (yc-bh/2)*height, (xc+bw/2)*width, (yc+bh/2)*height]))
        results = model.predict(frame, verbose=False)
        preds = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                preds.append((int(box.cls.item()), float(box.conf.item()), box.xyxy[0].tolist()))
        used = set()
        for cls in sorted(set([g[0] for g in gt] + [p[0] for p in preds])):
            cname = model.names.get(cls, str(cls)) if isinstance(model.names, dict) else str(cls)
            counts.setdefault(cname, Counter())
            gs = [b for k, b in gt if k == cls]
            ps = [(conf, b) for k, conf, b in preds if k == cls]
            candidates = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gs) for pi, (_, p) in enumerate(ps)), reverse=True)
            matched_g, matched_p = set(), set()
            for score, gi, pi in candidates:
                if score >= .5 and gi not in matched_g and pi not in matched_p:
                    matched_g.add(gi); matched_p.add(pi)
            counts[cname]["FP"] += len(ps)-len(matched_p)
            counts[cname]["FN"] += len(gs)-len(matched_g)
            counts[cname]["TP"] += len(matched_g)
        records.append([Path(row["path"]).name, str(label), len(preds), len(gt), f'{len(preds)-len(gt):+d}'])
    return "## YOLO annotation status\n\nMatching annotation files were found; comparison uses the middle decoded frame and greedy class-aware IoU matching at 0.50.\n\n" + \
        "## YOLO detection summary\n\n" + md_table(["Class", "TP", "FP", "FN"], [[c, x["TP"], x["FP"], x["FN"]] for c, x in sorted(counts.items())]) + "\n" + \
        md_table(["Video", "Label file", "Predictions", "Ground truth", "Count difference"], records)


def classifier_section(rows, root, scratch, seed):
    rng = random.Random(seed)
    split = {"seed": seed, "classes": list(CLASSES), "test": [], "ignored": []}
    for c in CLASSES:
        items = [r for r in rows if r["category"] == c and r["status"] == "OK"]
        # Treat re-encodes (e.g. x264A) and exact duplicates as one source group.
        groups = defaultdict(list)
        digest_source = {}
        for row in items:
            source = ENCODING_SUFFIX.sub("", Path(row["path"]).stem).casefold()
            source = digest_source.get(row["sha256"], source)
            digest_source[row["sha256"]] = source
            groups[source].append(row)
        sources = sorted(groups)
        rng.shuffle(sources)
        n = max(1, round(len(sources)*.2)) if sources else 0
        chosen = set(sources[:n])
        for source, members in groups.items():
            target = "test" if source in chosen else "ignored"
            for row in members:
                split[target].append({"path": row["path"], "label": c, "source_group": f"{c}:{source}"})
    split_path = scratch / "test_split.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    model_path = BACKEND / "models" / "video_classifier.pth"
    # This repo's loader requires class_names.json. Supply equivalent temporary metadata if absent.
    names_path = BACKEND / "models" / "class_names.json"
    temp_names = scratch / "class_names.json"
    if not names_path.is_file():
        temp_names.write_text(json.dumps(list(CLASSES)), encoding="utf-8")
        names_path = temp_names
    try:
        sys.path.insert(0, str(BACKEND))
        from app.services.video_classification_model import VideoClassificationModel
        model = VideoClassificationModel(model_path, names_path)
    except Exception as e:
        return split_path, "## Video-classifier evaluation\n\nEvaluation unavailable: " + str(e).replace("\n", " ") + ". The checkpoint or its metadata could not be loaded.\n"
    cm = {a: Counter() for a in CLASSES}
    predictions = []
    for item in split["test"]:
        try:
            pred = model.predict_path(item["path"])["label"]
        except Exception as e:
            pred = "ERROR"
            item["error"] = str(e)
        cm[item["label"]][pred] += 1
        predictions.append({**item, "prediction": pred})
    split["predictions"] = predictions
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    labels = list(CLASSES)
    table = [[truth] + [cm[truth][pred] for pred in labels] for truth in labels]
    metrics = []
    total = len(predictions)
    correct = sum(cm[c][c] for c in labels)
    for c in labels:
        tp = cm[c][c]; fp = sum(cm[t][c] for t in labels if t != c); fn = sum(cm[c][p] for p in labels if p != c)
        precision = tp/(tp+fp) if tp+fp else 0
        recall = tp/(tp+fn) if tp+fn else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
        metrics.append([c, f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", sum(cm[c].values())])
    text = f"## Video-classifier evaluation\n\nTest set: {total} videos (20% per category, seed {seed}); remaining valid videos are ignored.\n\n### Confusion matrix (rows=true, columns=predicted)\n\n"
    text += md_table(["True \\ Predicted"]+labels, table) + "\n### Per-class metrics\n\n"
    text += md_table(["Class", "Precision", "Recall", "F1", "Support"], metrics)
    text += f"\nOverall accuracy: **{correct/total:.4f}** ({correct}/{total}).\n\n"
    if total:
        confusions = [(cm[t][p], t, p) for t in labels for p in labels if t != p and cm[t][p]]
        dominant = sorted(confusions, reverse=True)[:3]
        details = "; ".join(f"{t} predicted as {p}: {n}" for n, t, p in dominant) or "no errors in this split"
        text += f"\nInterpretation: the most frequent errors were {details}. The group-separated split keeps normalized re-encodes of the same named source together; labels still come from folder names and may contain annotation noise.\n"
    else:
        text += "No valid videos were available for evaluation.\n"
    return split_path, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scratch", type=Path, default=PROJECT / "scratch" / "accuracy_evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    if not args.dataset.is_dir():
        parser.error(f"Dataset directory not found: {args.dataset}")
    args.scratch.mkdir(parents=True, exist_ok=True)
    rows, errors, dupes = audit(args.dataset)
    sections = ["# Full-dataset accuracy evaluation\n\n", audit_section(rows, errors, dupes), "\n\n"]
    sections.append(yolo_section(args.dataset, rows, args.scratch))
    sections.append("\n\n")
    split_path, classifier = classifier_section(rows, args.dataset, args.scratch, args.seed)
    sections.extend([classifier, "\n\n## Automated tests and regression\n\n"])
    if args.skip_tests:
        sections.append("Skipped by command-line option.\n")
    else:
        py = sys.executable
        test = subprocess.run([py, "-m", "pytest", "backend/tests"], cwd=PROJECT, capture_output=True, text=True)
        sections.append(f"`python -m pytest backend/tests` exit code: {test.returncode}.\n\n```text\n{(test.stdout+test.stderr)[-12000:]}\n```\n")
        sample = next((r["path"] for r in rows if r["status"] == "OK"), None)
        if sample:
            cap = cv2.VideoCapture(sample)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
            ok, frame = cap.read()
            cap.release()
            if ok:
                frame_path = args.scratch / "regression_middle_frame.png"
                cv2.imwrite(str(frame_path), frame)
                regression = subprocess.run([py, "run_regression.py", str(frame_path)], cwd=PROJECT, capture_output=True, text=True)
                sections.append(f"\n`python run_regression.py <middle frame>` exit code: {regression.returncode}; source video: `{sample}`.\n\n```text\n{(regression.stdout+regression.stderr)[-12000:]}\n```\n")
            else:
                sections.append(f"\nRegression script not run: could not extract a frame from `{sample}`.\n")
        else:
            sections.append("\nRegression script not run: no valid sample video was available.\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("".join(sections), encoding="utf-8")
    print(f"Report: {args.report}\nSplit: {split_path}\nVideos audited: {len(rows)}\nReport sections written.")


if __name__ == "__main__":
    main()
