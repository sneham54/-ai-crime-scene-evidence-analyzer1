"""Isolated full-video Deep OC-SORT runner. One process and tracker per video."""
import argparse
import csv
import gzip
import json
import logging
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import IterableSimpleNamespace, YAML


FIELDS = ["frame", "timestamp_seconds", "class", "class_id", "confidence",
          "track_id", "x1", "y1", "x2", "y2"]


def run(source, output, model_path, tracker_config, imgsz=640, confidence=0.25,
        save_annotated=False, progress_every=300):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "processing.log"
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    started_wall = time.time()
    started = time.perf_counter()
    detections_csv = output / "tracking.csv"
    tracker_io_path = output / "tracker_io.jsonl.gz"
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"video_read_failure: cannot open {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"video_read_failure: invalid metadata fps={fps}, size={width}x{height}")

    config = YAML.load(str(Path(tracker_config).resolve()))
    if config.get("tracker_type") != "deepocsort" or bool(config.get("use_byte", False)):
        cap.release()
        raise RuntimeError("tracker_configuration_failure: expected DeepOCSORT with use_byte=false")
    tracker = TRACKER_MAP["deepocsort"](args=IterableSimpleNamespace(**config))
    model = YOLO(str(Path(model_path).resolve()))
    if not isinstance(model.names, dict) or sorted(model.names) != list(range(9)):
        cap.release()
        raise RuntimeError("model_configuration_failure: best.pt class IDs must be 0 through 8")

    writer = None
    frame_number = raw_count = tracked_count = frames_with_detections = frames_without_detections = 0
    class_counts, class_track_ids, track_frames = Counter(), defaultdict(set), defaultdict(list)
    tracker_errors = []
    try:
        with detections_csv.open("w", newline="", encoding="utf-8") as csv_handle, \
                gzip.open(tracker_io_path, "wt", encoding="utf-8", compresslevel=6) as io_handle:
            table = csv.DictWriter(csv_handle, fieldnames=FIELDS)
            table.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = frame_number / fps
                prediction = model.predict(frame, conf=confidence, iou=0.45, imgsz=imgsz,
                                           device="cpu", verbose=False)[0]
                boxes = prediction.boxes.cpu().numpy()
                raw = []
                for index, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    coords = [float(v) for v in box.xyxy[0].tolist()]
                    raw.append({"detection_index": index, "class_id": cls_id,
                                "class": model.names.get(cls_id, str(cls_id)),
                                "confidence": float(box.conf[0]), "bbox_xyxy_pixels": coords})

                tracks = tracker.update(boxes, frame)
                by_detection = {}
                tracker_rows = []
                if tracks is not None:
                    for track_row in tracks:
                        # Ultralytics tracker API row: xyxy, track_id, confidence, class_id, detection_index.
                        detection_index = int(track_row[-1])
                        track_id = int(track_row[4])
                        class_id = int(track_row[6])
                        tracker_rows.append({"detection_index": detection_index,
                                             "track_id": track_id, "class_id": class_id,
                                             "confidence": float(track_row[5]),
                                             "bbox_xyxy_pixels": [float(v) for v in track_row[:4]]})
                        by_detection[detection_index] = (track_id, class_id)

                annotated = frame.copy() if save_annotated else None
                frame_tracked = 0
                for item in raw:
                    assignment = by_detection.get(item["detection_index"])
                    track_id = assignment[0] if assignment and assignment[1] == item["class_id"] else None
                    if assignment and assignment[1] != item["class_id"]:
                        tracker_errors.append({"frame": frame_number, "type": "class_mismatch",
                                               "detection_index": item["detection_index"],
                                               "detection_class_id": item["class_id"],
                                               "tracker_class_id": assignment[1]})
                    x1, y1, x2, y2 = item["bbox_xyxy_pixels"]
                    table.writerow({"frame": frame_number, "timestamp_seconds": f"{timestamp:.9f}",
                                    "class": item["class"], "class_id": item["class_id"],
                                    "confidence": f"{item['confidence']:.8f}", "track_id": track_id or "",
                                    "x1": f"{x1:.3f}", "y1": f"{y1:.3f}",
                                    "x2": f"{x2:.3f}", "y2": f"{y2:.3f}"})
                    raw_count += 1
                    class_counts[item["class"]] += 1
                    if track_id is not None:
                        frame_tracked += 1
                        tracked_count += 1
                        class_track_ids[item["class"]].add(track_id)
                        track_frames[(item["class"], track_id)].append(frame_number)
                    if annotated is not None:
                        color = (40, 210, 180) if track_id is not None else (0, 160, 255)
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{item['class']} #{track_id}" if track_id is not None else f"{item['class']} no-id"
                        cv2.putText(annotated, label, (int(x1), max(18, int(y1)-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                raw_len = len(raw)
                frames_with_detections += int(raw_len > 0)
                frames_without_detections += int(raw_len == 0)
                io_handle.write(json.dumps({"frame": frame_number, "timestamp_seconds": timestamp,
                                           "raw_yolo_detections": raw, "tracker_input_count": raw_len,
                                           "tracker_output": tracker_rows,
                                           "tracker_output_count": len(tracker_rows),
                                           "detections_with_ids": frame_tracked}, allow_nan=False) + "\n")

                if save_annotated:
                    if writer is None:
                        writer = cv2.VideoWriter(str(output / "annotated.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError("annotated_video_failure: output writer unavailable")
                    writer.write(annotated)
                frame_number += 1
                if progress_every > 0 and frame_number % progress_every == 0:
                    logging.info("progress frames=%d detections=%d tracked=%d", frame_number, raw_count, tracked_count)
    except Exception:
        logging.exception("video_processing_failure")
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    processing_seconds = time.perf_counter() - started
    unique_tracks = len(track_frames)
    track_lengths = [len(frames) for frames in track_frames.values()]
    durations = [(max(frames) - min(frames)) / fps for frames in track_frames.values()]
    processing_status = "completed" if (source_frames <= 0 or frame_number == source_frames) else "partial"
    metadata = {
        "source_path": str(source), "source_bytes": source.stat().st_size,
        "model_path": str(Path(model_path).resolve()), "model_class_names": model.names,
        "tracker_config": str(Path(tracker_config).resolve()), "tracker_type": "deepocsort",
        "tracker_use_byte": False, "reid_enabled": bool(config.get("with_reid", False)),
        "gmc_method": config.get("gmc_method", "none"),
        "fps": fps, "source_frame_count": source_frames, "frames_processed": frame_number,
        "width": width, "height": height, "timestamp_method": "zero_based_frame_index_divided_by_source_fps",
        "coordinate_format": "pixel_xyxy", "processing_started_unix": started_wall,
        "processing_seconds": processing_seconds, "processing_status": processing_status,
        "frames_with_detections": frames_with_detections,
        "frames_with_no_detections": frames_without_detections,
        "raw_yolo_detections": raw_count, "tracker_input_detections": raw_count,
        "tracked_detections": tracked_count, "unique_track_ids": unique_tracks,
        "detection_to_track_coverage": tracked_count / raw_count if raw_count else None,
        "track_ids_by_class": {k: sorted(v) for k, v in class_track_ids.items()},
        "track_count_by_class": {k: len(v) for k, v in class_track_ids.items()},
        "class_detection_counts": dict(class_counts),
        "track_length_observations": {"average": statistics.mean(track_lengths) if track_lengths else None,
                                      "median": statistics.median(track_lengths) if track_lengths else None,
                                      "min": min(track_lengths) if track_lengths else None,
                                      "max": max(track_lengths) if track_lengths else None},
        "track_duration_seconds": {"average": statistics.mean(durations) if durations else None,
                                   "median": statistics.median(durations) if durations else None,
                                   "min": min(durations) if durations else None,
                                   "max": max(durations) if durations else None},
        "short_tracks_under_3_observations": sum(n < 3 for n in track_lengths),
        "tracker_error_count": len(tracker_errors), "annotated_video_generated": bool(save_annotated),
        "accuracy_metrics": "not calculated; no valid identity ground truth",
    }
    summary = {"processing_status": processing_status, "class_detection_counts": dict(class_counts),
               "class_track_ids": {k: sorted(v) for k, v in class_track_ids.items()},
               "track_lengths": {f"{cls}:{tid}": len(frames) for (cls, tid), frames in track_frames.items()},
               "track_duration_seconds": {f"{cls}:{tid}": (max(frames)-min(frames))/fps
                                           for (cls, tid), frames in track_frames.items()},
               "tracker_errors": tracker_errors}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    logging.info("finished status=%s frames=%d detections=%d tracked=%d", processing_status,
                 frame_number, raw_count, tracked_count)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--tracker", default="config/deepocsort.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--save-annotated", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output, args.model, args.tracker,
                         args.imgsz, args.confidence, args.save_annotated), indent=2))


if __name__ == "__main__":
    main()
