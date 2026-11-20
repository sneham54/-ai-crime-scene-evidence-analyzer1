"""Run isolated, all-frame YOLO tracker comparisons on explicitly selected videos."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_timestamps(path):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "stream=time_base,avg_frame_rate,r_frame_rate,nb_frames,duration:frame=best_effort_timestamp,best_effort_timestamp_time",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    stream = data["streams"][0]
    time_base = Fraction(stream["time_base"])
    rows = data.get("frames", [])
    values = []
    for row in rows:
        raw_pts = row.get("best_effort_timestamp")
        pts = int(raw_pts) if raw_pts not in (None, "N/A") else None
        raw_seconds = row.get("best_effort_timestamp_time")
        seconds = float(pts * time_base) if pts is not None else (float(raw_seconds) if raw_seconds not in (None, "N/A") else None)
        values.append({"pts": pts, "timestamp_sec": seconds})
    nominal_fps = Fraction(stream["avg_frame_rate"]) if stream.get("avg_frame_rate") not in (None, "0/0") else None
    deltas = [b["timestamp_sec"] - a["timestamp_sec"] for a, b in zip(values, values[1:])
              if a["timestamp_sec"] is not None and b["timestamp_sec"] is not None]
    cfr = bool(nominal_fps and len(deltas) == max(0, len(values) - 1) and
               all(abs(delta - float(1 / nominal_fps)) <= 1e-6 for delta in deltas))
    return {
        "time_base": f"{time_base.numerator}/{time_base.denominator}",
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "duration_sec": float(stream["duration"]) if stream.get("duration") else None,
        "declared_frames": int(stream["nb_frames"]) if stream.get("nb_frames", "N/A").isdigit() else None,
        "frame_timestamps": values,
        "pts_frames": sum(item["pts"] is not None for item in values),
        "verified_cfr_from_pts": cfr,
    }


def reset_predictor(model):
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    for tracker in getattr(predictor, "trackers", []):
        tracker.reset()
    if hasattr(predictor, "vid_path"):
        predictor.vid_path = [None] * len(getattr(predictor, "trackers", []))


def process_video(model, video_path, tracker_name, confidence, imgsz, run_dir):
    video_path = Path(video_path)
    timing = probe_timestamps(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_cv_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    reset_predictor(model)  # one fresh tracker state per video; never reset within this loop
    frame_counts = defaultdict(int)
    track_observations = defaultdict(list)
    detections_path = run_dir / "detections.csv"
    started = time.perf_counter()
    frame_index = 0
    with detections_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "source_video", "source_frame_index", "pts", "time_base", "timestamp_sec",
            "class_id", "class_name", "confidence", "track_id", "x1", "y1", "x2", "y2",
        ))
        writer.writeheader()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pts = timing["frame_timestamps"][frame_index] if frame_index < len(timing["frame_timestamps"]) else {"pts": None, "timestamp_sec": None}
            timestamp = pts["timestamp_sec"]
            if timestamp is None and timing["verified_cfr_from_pts"] and source_fps > 0:
                timestamp = frame_index / source_fps
            result = model.track(
                frame, persist=True, tracker=tracker_name, conf=confidence, iou=0.45,
                imgsz=imgsz, device="cpu", verbose=False,
            )[0]
            boxes = result.boxes
            if boxes is not None and len(boxes):
                coords = boxes.xyxy.detach().cpu().tolist()
                classes = boxes.cls.detach().cpu().tolist()
                scores = boxes.conf.detach().cpu().tolist()
                ids = boxes.id.detach().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                names = result.names
                for box, class_id, score, track_id in zip(coords, classes, scores, ids):
                    class_id = int(class_id)
                    label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                    track_id = int(track_id) if track_id is not None else None
                    frame_counts["detections"] += 1
                    if track_id is not None:
                        frame_counts["tracked_detections"] += 1
                        track_observations[track_id].append((frame_index, timestamp, label))
                    if 0.10 <= float(score) < 0.25:
                        frame_counts["detections_0p10_to_0p25"] += 1
                    writer.writerow({
                        "source_video": video_path.name, "source_frame_index": frame_index,
                        "pts": pts["pts"], "time_base": timing["time_base"], "timestamp_sec": timestamp,
                        "class_id": class_id, "class_name": label, "confidence": round(float(score), 7),
                        "track_id": track_id, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                    })
            frame_counts["frames_with_detections"] += bool(boxes is not None and len(boxes))
            frame_index += 1
    capture.release()
    elapsed = time.perf_counter() - started
    track_rows = []
    fragments = 0
    short_tracks = 0
    for track_id, observations in sorted(track_observations.items()):
        observations.sort(key=lambda row: row[0])
        segments = []
        segment = [observations[0]]
        for current in observations[1:]:
            if current[0] != segment[-1][0] + 1:
                segments.append(segment)
                segment = [current]
            else:
                segment.append(current)
        segments.append(segment)
        fragments += max(0, len(segments) - 1)
        for part, segment in enumerate(segments, start=1):
            timestamps = [row[1] for row in segment if row[1] is not None]
            end = timestamps[-1] if timestamps else None
            start = timestamps[0] if timestamps else None
            duration = (end - start + (1.0 / source_fps if source_fps > 0 else 0.0)) if start is not None else None
            is_short = len(segment) < 3 or (duration is not None and duration < 1.0)
            short_tracks += is_short
            track_rows.append({
                "track_id": track_id, "fragment_index": part,
                "observation_count": len(segment), "start_frame": segment[0][0],
                "end_frame": segment[-1][0], "start_timestamp_sec": start,
                "end_timestamp_sec": end, "duration_sec": round(duration, 6) if duration is not None else None,
                "short_track_lt_1s_or_lt3_observations": is_short,
                "class_names": ";".join(sorted({row[2] for row in segment})),
            })
    track_path = run_dir / "tracks.csv"
    with track_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=track_rows[0].keys() if track_rows else (
            "track_id", "fragment_index", "observation_count", "start_frame", "end_frame",
            "start_timestamp_sec", "end_timestamp_sec", "duration_sec",
            "short_track_lt_1s_or_lt3_observations", "class_names",
        ))
        writer.writeheader()
        writer.writerows(track_rows)
    detections = frame_counts["detections"]
    summary = {
        "video": video_path.name, "video_path": str(video_path),
        "tracker": tracker_name, "detector_confidence": confidence, "imgsz": imgsz,
        "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
        "device": "cpu", "source_sha256": sha256(video_path),
        "source_fps_opencv": source_fps, "opencv_declared_frames": declared_cv_frames,
        "decoded_frames": frame_index, "frames_with_detections": frame_counts["frames_with_detections"],
        "detections": detections, "tracked_detections": frame_counts["tracked_detections"],
        "untracked_detections": detections - frame_counts["tracked_detections"],
        "track_id_detection_coverage": frame_counts["tracked_detections"] / detections if detections else None,
        "detections_0p10_to_0p25": frame_counts["detections_0p10_to_0p25"],
        "unique_track_ids": len(track_observations), "track_fragments_proxy": fragments,
        "track_segments": len(track_rows), "short_track_segments_lt_1s_or_lt3_obs": short_tracks,
        "processing_seconds": round(elapsed, 4),
        "throughput_frames_per_second": round(frame_index / elapsed, 3) if elapsed else None,
        "idF1": None, "HOTA": None,
        "identity_metric_status": "not_computed_no_ground_truth_track_annotations",
        "timestamp_metadata": {key: value for key, value in timing.items() if key != "frame_timestamps"},
        "detections_csv": str(detections_path), "tracks_csv": str(track_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--video", action="append", required=True, help="Repeat for each selected source video")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing evaluation output: {output}")
    output.mkdir(parents=True)
    model_hash = sha256(args.weights)
    tracker_root = Path(ultralytics.__file__).parent / "cfg" / "trackers"
    configurations = [
        ("bytetrack", 0.25), ("bytetrack", 0.10),
        ("botsort", 0.25), ("botsort", 0.10),
    ]
    summaries = []
    for tracker, confidence in configurations:
        model = YOLO(args.weights)
        names = model.names
        if not isinstance(names, dict) or sorted(names) != list(range(9)):
            raise RuntimeError("Selected detector does not expose the expected nine class IDs")
        tracker_yaml = tracker_root / f"{tracker}.yaml"
        for video in args.video:
            video_path = Path(video)
            run_dir = output / video_path.stem / f"{tracker}_conf_{confidence:.2f}".replace(".", "p")
            run_dir.mkdir(parents=True)
            summary = process_video(model, video_path, str(tracker_yaml), confidence, args.imgsz, run_dir)
            summary["detector_sha256"] = model_hash
            summary["detector_class_names"] = names
            summary["tracker_yaml"] = str(tracker_yaml)
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            summaries.append(summary)
            print(json.dumps({key: summary[key] for key in (
                "video", "tracker", "detector_confidence", "decoded_frames", "detections",
                "tracked_detections", "unique_track_ids", "track_fragments_proxy",
                "short_track_segments_lt_1s_or_lt3_obs", "processing_seconds",
            )}), flush=True)
        del model
    summary_path = output / "comparison_summary.csv"
    columns = list(summaries[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    print(f"summary_csv={summary_path}", flush=True)


if __name__ == "__main__":
    main()
