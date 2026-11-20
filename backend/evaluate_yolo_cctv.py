"""Sweep YOLO confidence thresholds on labeled CCTV frames or export a review bundle.

Run from backend/:
  python evaluate_yolo_cctv.py --labels-csv path/to/labels.csv --report-json eval.json
  python evaluate_yolo_cctv.py --labels-yolo-images path/to/valid/images --report-json eval.json
  python evaluate_yolo_cctv.py --review-video path/to/clip.mp4 --review-output-dir review_bundle

CSV columns are image_path,class,x1,y1,x2,y2 with pixel coordinates. Include
one row with empty class/coordinates for every negative frame. Relative image
paths resolve from the CSV directory. For CCTV videos, export labeled frames
at their timestamps and evaluate those images.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from app.config import Config
from app.services.yolo_evaluation import (
    empty_counts,
    filter_predictions,
    finalize_counts,
    map_class_id,
    match_frame_detections,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_THRESHOLDS = (0.10, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75)
VIDEO_IOU = 0.45
EVAL_MATCH_IOU = 0.50


def parse_thresholds(value):
    try:
        thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Thresholds must be comma-separated numbers.") from error
    if not thresholds or any(not math.isfinite(item) or not 0 <= item <= 1 for item in thresholds):
        raise argparse.ArgumentTypeError("Provide one or more finite thresholds between 0 and 1.")
    if not any(math.isclose(item, Config.YOLO_CONFIDENCE, abs_tol=1e-9) for item in thresholds):
        thresholds.append(Config.YOLO_CONFIDENCE)
    return sorted(set(thresholds))


def read_labels_csv(csv_path, known_classes):
    csv_path = Path(csv_path).resolve()
    grouped = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"image_path", "class", "x1", "y1", "x2", "y2"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Label CSV must include image_path,class,x1,y1,x2,y2 columns.")
        for row_number, row in enumerate(reader, start=2):
            image_value = (row.get("image_path") or "").strip()
            if not image_value:
                raise ValueError(f"Missing image_path on CSV row {row_number}.")
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = (csv_path.parent / image_path).resolve()
            label = (row.get("class") or "").strip()
            coordinates = [(row.get(name) or "").strip() for name in ("x1", "y1", "x2", "y2")]
            if not label and not any(coordinates):
                grouped[str(image_path)]
                continue
            if label not in known_classes:
                raise ValueError(f"Unknown ground-truth class {label!r} on CSV row {row_number}.")
            try:
                box = [float(value) for value in coordinates]
            except ValueError as error:
                raise ValueError(f"Invalid pixel box on CSV row {row_number}.") from error
            if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"Invalid pixel box on CSV row {row_number}.")
            grouped[str(image_path)].append({"class": label, "bbox": box})
    if not grouped:
        raise ValueError("Label CSV contains no image rows.")
    for image_path in grouped:
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Labeled frame does not exist: {image_path}")
    return grouped


def read_yolo_label_directory(images_dir, class_names):
    """Read standard normalized xywh labels beside an images/ directory."""
    images_dir = Path(images_dir).resolve()
    labels_dir = images_dir.parent / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected image directory and sibling labels directory: {images_dir}")
    image_paths = sorted(path for path in images_dir.rglob("*")
                         if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"No supported images found in {images_dir}")
    grouped = {}
    for image_path in image_paths:
        relative = image_path.relative_to(images_dir)
        label_path = labels_dir / relative.with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing YOLO label file for frame: {label_path}")
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not decode labeled image: {image_path}")
        height, width = frame.shape[:2]
        boxes = []
        for row_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            fields = raw_line.split()
            if len(fields) != 5:
                raise ValueError(f"Expected class_id x_center y_center width height in {label_path}:{row_number}")
            try:
                class_id = int(fields[0])
                center_x, center_y, box_width, box_height = map(float, fields[1:])
                label = map_class_id(class_names, class_id)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid YOLO label in {label_path}:{row_number}: {error}") from error
            values = (center_x, center_y, box_width, box_height)
            if not all(math.isfinite(value) for value in values) or box_width <= 0 or box_height <= 0:
                raise ValueError(f"Invalid normalized box in {label_path}:{row_number}")
            boxes.append({"class": label, "bbox": [
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            ]})
        grouped[str(image_path)] = boxes
    return grouped


def predict_frame(model, frame, class_names, confidence, image_size):
    result = model.predict(
        frame, conf=confidence, iou=VIDEO_IOU, imgsz=image_size,
        device="cpu", verbose=False, agnostic_nms=False,
    )[0]
    predictions = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            predictions.append({
                "class_id": class_id,
                "class": map_class_id(class_names, class_id),
                "confidence": float(box.conf[0]),
                "bbox": [float(value) for value in box.xyxy[0].tolist()],
            })
    return predictions, result.plot(line_width=2)


def evaluate(model, labels, class_names, thresholds, image_size):
    class_names_list = list(class_names.values()) if isinstance(class_names, dict) else list(class_names)
    rows = []
    for index, (image_path, ground_truth) in enumerate(labels.items(), start=1):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not decode labeled image: {image_path}")
        predictions, _ = predict_frame(model, frame, class_names, min(thresholds), image_size)
        rows.append((ground_truth, predictions))
        if index % 100 == 0 or index == len(labels):
            print(f"Evaluated {index}/{len(labels)} labeled frames", flush=True)

    report = {
        "evaluation_type": "labeled_cctv_frame_confidence_sweep",
        "checkpoint": str(Path(Config.YOLO_MODEL_PATH).resolve()),
        "class_mapping": {str(key): value for key, value in class_names.items()},
        "labeled_frames": len(rows),
        "ground_truth_objects": sum(len(gt) for gt, _ in rows),
        "inference_confidence_floor": min(thresholds),
        "inference_iou": VIDEO_IOU,
        "matching_iou": EVAL_MATCH_IOU,
        "thresholds": {},
        "limitations": [
            "Metrics describe only the supplied labeled frames and are not general accuracy claims.",
            "Confidence thresholds alter the false-positive/missed-detection trade-off; inspect class-wise results.",
        ],
    }
    for threshold in thresholds:
        counts = {name: empty_counts() for name in class_names_list}
        for ground_truth, raw_predictions in rows:
            predictions = filter_predictions(raw_predictions, threshold)
            matches, false_positives, false_negatives = match_frame_detections(
                ground_truth, predictions, EVAL_MATCH_IOU
            )
            for prediction, _truth, _iou in matches:
                counts[prediction["class"]]["true_positives"] += 1
            for prediction in false_positives:
                counts[prediction["class"]]["false_positives"] += 1
            for truth in false_negatives:
                counts[truth["class"]]["missed_detections"] += 1
        total = empty_counts()
        per_class = {}
        for name, values in counts.items():
            per_class[name] = finalize_counts(values)
            for metric in total:
                total[metric] += values[metric]
        report["thresholds"][str(threshold)] = {
            "overall_micro": finalize_counts(total), "per_class": per_class,
        }
    return report


def export_review_video(model, class_names, video_path, output_dir, image_size):
    video_path, output_dir = Path(video_path).resolve(), Path(output_dir).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Review video does not exist: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not decode review video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Review video must have readable frames and valid FPS metadata.")
    duration = frame_count / fps
    if duration > Config.VIDEO_MAX_DURATION_SEC:
        capture.release()
        raise ValueError(f"Review video exceeds configured duration limit ({Config.VIDEO_MAX_DURATION_SEC} seconds).")
    stride = max(1, int(round(fps / max(0.01, Config.VIDEO_SAMPLE_FPS))))
    frame_limit = max(1, int(Config.VIDEO_MAX_PROCESSED_FRAMES))
    csv_path = output_dir / "review.csv"
    processed = 0
    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=[
                "video", "frame_number", "timestamp_sec", "annotated_frame", "class_id",
                "predicted_class", "confidence", "x1", "y1", "x2", "y2",
                "review_status", "reviewed_class", "review_notes",
            ])
            writer.writeheader()
            for frame_number in range(0, frame_count, stride):
                if processed >= frame_limit or frame_number / fps > Config.VIDEO_MAX_DURATION_SEC:
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    continue
                timestamp = frame_number / fps
                predictions, annotated = predict_frame(
                    model, frame, class_names, Config.YOLO_CONFIDENCE, image_size
                )
                filename = f"frame_{frame_number:09d}_{timestamp:.3f}s.jpg"
                cv2.putText(annotated, f"{timestamp:.3f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                if not cv2.imwrite(str(output_dir / filename), annotated):
                    raise OSError(f"Could not save review frame: {output_dir / filename}")
                for prediction in predictions or [None]:
                    box = prediction["bbox"] if prediction else ["", "", "", ""]
                    writer.writerow({
                        "video": str(video_path), "frame_number": frame_number,
                        "timestamp_sec": round(timestamp, 3), "annotated_frame": filename,
                        "class_id": prediction["class_id"] if prediction else "",
                        "predicted_class": prediction["class"] if prediction else "",
                        "confidence": round(prediction["confidence"], 6) if prediction else "",
                        "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                        "review_status": "pending", "reviewed_class": "", "review_notes": "",
                    })
                processed += 1
    finally:
        capture.release()
    if not processed:
        raise ValueError("No review frames could be decoded.")
    return {"review_csv": str(csv_path), "annotated_frames": processed, "duration_sec": duration,
            "frame_stride": stride, "confidence_threshold": Config.YOLO_CONFIDENCE}


def load_detector():
    checkpoint = Path(Config.YOLO_MODEL_PATH)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Configured YOLO checkpoint not found: {checkpoint}")
    model = YOLO(str(checkpoint))
    names = model.names
    if not isinstance(names, dict) or not names:
        raise ValueError("YOLO checkpoint does not expose a valid class-id mapping.")
    return model, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--labels-csv", help="Ground-truth frame boxes in pixel-coordinate CSV format")
    mode.add_argument("--labels-yolo-images", help="Images directory with sibling labels/ YOLO-format annotations")
    mode.add_argument("--review-video", help="Video to export as annotated, timestamped human-review frames")
    parser.add_argument("--max-frames", type=int,
                        help="Deterministically sample at most this many labeled frames (seed 42)")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Random seed used with --max-frames (default: 42)")
    parser.add_argument("--thresholds", type=parse_thresholds,
                        default=parse_thresholds(",".join(map(str, DEFAULT_THRESHOLDS))))
    parser.add_argument("--report-json", help="Evaluation JSON output (labeled mode only)")
    parser.add_argument("--review-output-dir", help="New output directory (review mode only)")
    args = parser.parse_args()
    try:
        model, names = load_detector()
        if args.labels_csv or args.labels_yolo_images:
            if not args.report_json:
                parser.error("--report-json is required with a labeled-frame input.")
            if args.max_frames is not None and args.max_frames <= 0:
                parser.error("--max-frames must be greater than zero.")
            labels = (read_labels_csv(args.labels_csv, set(names.values())) if args.labels_csv
                      else read_yolo_label_directory(args.labels_yolo_images, names))
            if args.max_frames is not None and args.max_frames < len(labels):
                sampled_paths = set(random.Random(args.sample_seed).sample(
                    list(labels), args.max_frames
                ))
                labels = {path: boxes for path, boxes in labels.items() if path in sampled_paths}
            report = evaluate(model, labels, names, args.thresholds, Config.YOLO_IMAGE_SIZE)
            report["sampling"] = {
                "method": "deterministic uniform sample without replacement" if args.max_frames else "all supplied labeled frames",
                "seed": args.sample_seed if args.max_frames else None,
                "requested_max_frames": args.max_frames,
                "evaluated_frames": len(labels),
            }
            output = Path(args.report_json).resolve()
            if output.exists():
                parser.error(f"Refusing to overwrite existing evaluation report: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))
        else:
            if not args.review_output_dir:
                parser.error("--review-output-dir is required with --review-video.")
            result = export_review_video(model, names, args.review_video,
                                         args.review_output_dir, Config.YOLO_IMAGE_SIZE)
            print(json.dumps(result, indent=2))
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
