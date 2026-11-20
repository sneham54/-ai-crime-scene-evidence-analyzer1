"""Merge clip predictions into review-only timestamped events and keyframes."""
import uuid
from pathlib import Path

import cv2

from .config import CLASSES
from app.services.video_timing import probe_video_timeline


def merge_prediction_events(predictions, video_id, source_video, confidence_threshold=0.55,
                            max_gap_sec=2.0, suspicious_classes=None):
    if not 0 <= confidence_threshold <= 1 or max_gap_sec < 0:
        raise ValueError("Invalid event threshold or merge gap")
    suspicious = set(suspicious_classes or CLASSES[:-1])
    candidates = []
    previous_candidate = None
    for prediction in sorted(predictions, key=lambda item: item.get("start_timestamp_sec", 0)):
        name = prediction.get("predicted_class")
        confidence = float(prediction.get("confidence", 0))
        start, end = prediction.get("start_timestamp_sec"), prediction.get("end_timestamp_sec")
        if name not in suspicious or confidence < confidence_threshold or start is None or end is None:
            # Any intervening class, low-confidence window, or missing timing
            # is a segmentation boundary; a later same-class window starts a
            # separate candidate instead of bridging over it.
            previous_candidate = None
            continue
        if end < start:
            raise ValueError("Prediction end timestamp precedes its start")
        if (previous_candidate and previous_candidate["predicted_class"] == name
                and start <= previous_candidate["end_timestamp_sec"] + max_gap_sec):
            event = previous_candidate
            event["end_timestamp_sec"] = max(event["end_timestamp_sec"], float(end))
            event["_clips"].append(prediction)
        else:
            previous_candidate = {"predicted_class": name, "start_timestamp_sec": float(start),
                                  "end_timestamp_sec": float(end), "_clips": [prediction]}
            candidates.append(previous_candidate)
    events = []
    for candidate in candidates:
        clips = candidate.pop("_clips")
        best = max(clips, key=lambda item: float(item.get("confidence", 0)))
        events.append({"event_id": uuid.uuid4().hex, "video_id": str(video_id),
                       "source_video": str(source_video), "predicted_class": candidate["predicted_class"],
                       "confidence": round(sum(float(item.get("confidence", 0)) for item in clips) / len(clips), 4),
                       "start_timestamp": candidate["start_timestamp_sec"],
                       "end_timestamp": candidate["end_timestamp_sec"],
                       "representative_frame": best.get("representative_frame"),
                       "frame_index": best.get("representative_frame_index"),
                       "keyframes": [], "verification_status": "pending_review",
                       "model_prediction_only": True,
                       "timestamp_mapping_status": best.get("timestamp_mapping_status", "unavailable"),
                       "timestamps_verified": bool(best.get("timestamps_verified", False))})
    return events


def extract_event_keyframes(video_path, events, predictions, output_dir=None, max_keyframes=3):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Video cannot be opened for keyframe extraction")
    prediction_rows = sorted(predictions, key=lambda p: p.get("start_timestamp_sec", 0))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    timeline = probe_video_timeline(video_path, frame_count, fps)
    output = Path(output_dir) if output_dir else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    try:
        for event in events:
            rows = [p for p in prediction_rows
                    if p.get("predicted_class") == event["predicted_class"] and
                    event["start_timestamp"] <= p.get("start_timestamp_sec", -1) <= event["end_timestamp"]]
            rows.sort(key=lambda p: float(p.get("confidence", 0)), reverse=True)
            selected = []
            for row in rows:
                index = row.get("representative_frame_index")
                timestamp = row.get("representative_timestamp_sec", row.get("start_timestamp_sec"))
                if index is None and timestamp is not None:
                    index = timeline.frame_at_or_after(float(timestamp))
                if index is not None and index not in {item["frame_index"] for item in selected}:
                    index = int(index)
                    selected.append({"frame_index": index, "timestamp_sec": timeline.timestamp(index),
                                     "confidence": float(row.get("confidence", 0)),
                                     "timestamp_mapping_status": timeline.mapping_status,
                                     "timestamps_verified": timeline.timestamps_verified})
                if len(selected) >= max_keyframes:
                    break
            for item in selected:
                capture.set(cv2.CAP_PROP_POS_FRAMES, item["frame_index"])
                ok, frame = capture.read()
                if not ok:
                    continue
                if output:
                    path = output / f"original_frame_{item['frame_index']:08d}_{event['event_id']}.jpg"
                    if not cv2.imwrite(str(path), frame):
                        continue
                    item["path"] = str(path)
                event["keyframes"].append(item)
            if event["keyframes"]:
                representative = event["keyframes"][0]
                event["representative_frame"] = representative.get("path")
                event["frame_index"] = representative["frame_index"]
    finally:
        capture.release()
    return events
