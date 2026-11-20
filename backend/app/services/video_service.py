"""Bounded-memory video classification and sampled-frame YOLO evidence analysis."""
import json
import math
import uuid
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from flask import current_app, has_app_context

from app.services.video_classification_model import VideoClassificationModel
from app.services.yolo_service import YOLOService
from app.services.evidence_catalog_service import EvidenceCatalogService
from app.services.video_timing import probe_video_timeline


def _sha256_file(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _probe_source_timing(path, expected_frames, fps=None):
    """Compatibility wrapper around the shared source-frame timeline probe."""
    timeline = probe_video_timeline(path, expected_frames, fps)
    return {
        "presentation_timestamps_available": timeline.presentation_timestamps_available,
        "time_base": timeline.time_base,
        "presentation_timestamps": list(timeline.presentation_timestamps),
        "frame_timestamps_sec": list(timeline.timestamps_sec),
        "probed_frame_count": len(timeline.presentation_timestamps) or None,
        "duration_sec": timeline.duration_sec,
        "average_fps": timeline.fps,
        "constant_frame_rate_verified": timeline.constant_frame_rate_verified,
        "timestamp_mapping_status": timeline.mapping_status,
        "timestamp_basis": timeline.timestamp_basis,
        "timestamps_verified": timeline.timestamps_verified,
    }


def _source_frame_timing(frame_index, fps, timing):
    """Return source time provenance for one zero-based decoded frame."""
    aligned_pts = timing.get("presentation_timestamps") or []
    frame_times = timing.get("frame_timestamps_sec") or []
    status = timing.get("timestamp_mapping_status", "unavailable")
    if frame_index < len(frame_times) and status == "verified_cfr_frame_index_fps":
        return {"timestamp_sec": frame_index / fps, "presentation_timestamp":
                aligned_pts[frame_index] if frame_index < len(aligned_pts) else None,
                "time_base": timing.get("time_base"), "timestamp_mapping_status": status,
                "timestamp_basis": "zero_based_frame_index_divided_by_source_fps",
                "timestamps_verified": True}
    if frame_index < len(frame_times) and status == "verified_vfr_presentation_timestamps":
        return {"timestamp_sec": frame_times[frame_index],
                "presentation_timestamp": aligned_pts[frame_index] if frame_index < len(aligned_pts) else None,
                "time_base": timing.get("time_base"), "timestamp_mapping_status": status,
                "timestamp_basis": "frame_pts_relative_to_first_presentation_timestamp",
                "timestamps_verified": True}
    valid_fps = np.isfinite(fps) and fps > 0
    return {"timestamp_sec": frame_index / fps if valid_fps else None,
            "presentation_timestamp": None, "time_base": timing.get("time_base"),
            "timestamp_mapping_status": ("unverified_frame_index_fps_fallback" if valid_fps else "unavailable"),
            "timestamp_basis": "unverified_frame_index_divided_by_reported_fps" if valid_fps else "unavailable",
            "timestamps_verified": False}


def _box_iou(first, second):
    if not first or not second or len(first) < 4 or len(second) < 4:
        return 0.0
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, first[2]-first[0]) * max(0, first[3]-first[1])
    area_b = max(0, second[2]-second[0]) * max(0, second[3]-second[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def _json_safe(value):
    """Convert NumPy objects and non-finite numbers into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _classify_video(classifier, video_path, max_duration_sec, confidence_threshold):
    """Keep whole-clip classification primary and attach window candidates."""
    primary = classifier.predict_path(video_path, max_duration_sec=max_duration_sec)
    temporal_method = getattr(classifier, "predict_windows_path", None)
    if not callable(temporal_method):
        return primary
    primary_category = primary.get("top_class") or primary.get("label")
    merge_gap_sec = (float(current_app.config.get("VIDEO_ACTIVITY_MERGE_GAP_SEC", 1.0))
                     if has_app_context() else 1.0)
    temporal = temporal_method(
        video_path,
        max_duration_sec=max_duration_sec,
        confidence_threshold=confidence_threshold,
        primary_category=primary_category,
        merge_gap_sec=merge_gap_sec,
    )
    whole_top = primary.get("top_class") or primary.get("label")
    temporal_top = temporal.get("top_class") or temporal.get("label")
    if primary.get("label") == "Uncertain":
        status = "uncertain_low_confidence"
    elif whole_top != temporal_top:
        status = "uncertain_prediction_disagreement"
    elif whole_top == "Normal":
        status = "provisional_normal_prediction"
    else:
        status = "provisional"
    primary.update({
        "classification_status": status,
        "temporal_prediction": {
            "label": temporal.get("label"), "top_class": temporal_top,
            "confidence": temporal.get("confidence"),
            "probabilities": temporal.get("probabilities", {}),
            "classification_mode": temporal.get("classification_mode"),
            "sampling_coverage": temporal.get("sampling_coverage"),
        },
        "temporal_windows": temporal.get("temporal_windows", []),
        "activity_events": temporal.get("activity_events", []),
        "temporal_sampling_coverage": temporal.get("sampling_coverage"),
        "alternative_predictions": ([{
            "label": temporal_top, "confidence": temporal.get("confidence"),
            "source": "temporal_window_aggregate",
        }] if temporal_top and temporal_top != whole_top else []),
        "classifier_note": (
            (primary.get("classifier_note") or "") + " " +
            ("Whole-video and temporal-window predictions disagree; treat both as provisional and review the source video. "
             if whole_top != temporal_top else "") +
            ("A Normal prediction is not proof that no incident occurred. " if whole_top == "Normal" else "") +
            "Temporal intervals are candidate locations from a whole-video-trained classifier; their boundaries are not independently validated."
        ).strip(),
    })
    if whole_top != temporal_top:
        primary["whole_video_prediction"] = {
            "label": primary.get("label"),
            "top_class": whole_top,
            "confidence": primary.get("confidence"),
            "probabilities": primary.get("probabilities", {}),
            "sampling_coverage": primary.get("sampling_coverage"),
        }
        primary["label"] = "Unresolved"
        primary["top_class"] = "Unresolved"
        primary["confidence"] = None
    return primary


def _attach_activity_support(activity_events, detections):
    for event in activity_events or []:
        start, end = event.get("start_timestamp_sec"), event.get("end_timestamp_sec")
        supporting = [item for item in detections
                      if start is not None and end is not None and start <= item.get("timestamp_sec", -1) <= end]
        event["detected_objects"] = sorted({item.get("class", "unknown") for item in supporting})
        event["tracking_ids"] = sorted({item["track_id"] for item in supporting if item.get("track_id") is not None})
        event["supporting_detection_count"] = len(supporting)
        event.setdefault("verification_status", "unverified_candidate")
    return activity_events or []


def group_evidence_events(frames, max_gap_sec=3.0, min_iou=0.1):
    """Group temporally close, spatially overlapping same-class detections.

    Tracking IDs take precedence when present. Otherwise, each frame is matched
    one-to-one against the most recent compatible event. Events are evidence
    summaries only and carry no incident confirmation.
    """
    events, active = [], []
    for frame in frames or []:
        detections = frame.get("detections") or []
        timestamp = frame.get("timestamp_sec")
        frame_number = frame.get("frame_number")
        used = set()
        for detection in detections:
            class_key = detection.get("class_id", detection.get("class", detection.get("class_name", "unknown")))
            track_id = detection.get("track_id")
            candidates = []
            for event in active:
                if event["class_key"] != class_key:
                    continue
                if timestamp is not None and event["last_timestamp"] is not None:
                    gap = float(timestamp) - event["last_timestamp"]
                    if gap < 0 or gap > float(max_gap_sec):
                        continue
                else:
                    gap = 0
                if track_id is not None and event["track_id"] is not None:
                    overlap = 1.0 if track_id == event["track_id"] else 0.0
                else:
                    overlap = _box_iou(detection.get("bbox"), event["last_bbox"])
                if overlap >= min_iou:
                    candidates.append((overlap, -gap, event))
            event = max(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None
            if event is None or id(event) in used:
                event = {
                    "class_key": class_key,
                    "class_id": detection.get("class_id"),
                    "class": detection.get("class", detection.get("class_name", str(class_key))),
                    "track_id": track_id,
                    "observations": [],
                    "first_timestamp": timestamp,
                    "last_timestamp": timestamp,
                    "last_bbox": detection.get("bbox"),
                }
                events.append(event)
                active.append(event)
            used.add(id(event))
            observation = {**detection, "timestamp_sec": timestamp, "frame_number": frame_number,
                           "frame_width": frame.get("frame_width"),
                           "frame_height": frame.get("frame_height"),
                           "evidence_image": frame.get("evidence_image")}
            event["observations"].append(observation)
            event["last_timestamp"] = timestamp
            event["last_bbox"] = detection.get("bbox")

    results = []
    for index, event in enumerate(events):
        observations = event["observations"]
        times = [item["timestamp_sec"] for item in observations if item.get("timestamp_sec") is not None]
        confidences = [float(item["confidence"]) for item in observations if item.get("confidence") is not None]
        representative = max(observations, key=lambda item: float(item.get("confidence") or 0))
        start = min(times) if times else None
        end = max(times) if times else None
        mean = round(sum(confidences) / len(confidences), 10) if confidences else None
        results.append({
            "event_id": f"event_{index + 1:04d}", "class_id": event["class_id"],
            "class": event["class"], "class_name": event["class"], "track_id": event["track_id"],
            "observation_count": len(observations), "start_timestamp_sec": start,
            "end_timestamp_sec": end, "duration_sec": (end - start) if start is not None and end is not None else None,
            "confidence_summary": {"min": min(confidences), "mean": mean, "max": max(confidences)} if confidences else None,
            "representative_frame_number": representative.get("frame_number"),
            "representative_bbox": representative.get("bbox"),
            "representative_frame_width": representative.get("frame_width"),
            "representative_frame_height": representative.get("frame_height"),
            "representative_evidence_image": representative.get("evidence_image"),
            "incident_status": "not_assessed", "verification_status": "unverified",
            "description": f"Repeated {event['class']} detection evidence; this does not confirm an incident.",
            "observations": observations,
        })
    return results


def _temporal_confirmation(frames):
    """Annotate detections with conservative repeated-track context."""
    groups = {}
    for frame in frames or []:
        timestamp = frame.get("timestamp_sec")
        for item in frame.get("detections") or []:
            key = (item.get("class_id", item.get("class")), item.get("track_id"))
            if key[1] is not None and timestamp is not None:
                groups.setdefault(key, []).append((float(timestamp), item))
    for values in groups.values():
        values.sort(key=lambda pair: pair[0])
        timestamps = [time for time, _ in values]
        for index, (timestamp, item) in enumerate(values):
            nearby = [time for time in timestamps if abs(time - timestamp) <= 2.0]
            confidences = [float(entry["confidence"]) for time_value, entry in values
                           if abs(time_value - timestamp) <= 2.0
                           and entry.get("confidence") is not None
                           and math.isfinite(float(entry["confidence"]))]
            item["temporal_status"] = ("confirmed_repeated_observation" if len(nearby) >= 3
                                       else "unconfirmed_single_or_sparse_observation")
            item["smoothed_confidence"] = sum(confidences) / len(confidences) if confidences else None
            item["track_duration_sec"] = timestamps[-1] - timestamps[0]


class VideoService:
    def __init__(self):
        self.yolo = None
        self.classifier_error = None
        self.classifier_backend = "legacy_resnet18_lstm"
        if current_app.config.get("VIDEO_CLASSIFIER_BACKEND") == "densenet_attention_convlstm":
            try:
                from app.video_models.densenet_attention_convlstm.service_adapter import DenseNetVideoClassifier
                self.classifier = DenseNetVideoClassifier(
                    current_app.config["VIDEO_DENSENET_MODEL_PATH"],
                    current_app.config["VIDEO_MIN_CONFIDENCE"],
                    float(current_app.config.get("VIDEO_ACTIVITY_MIN_CONFIDENCE", 0.55)),
                )
                self.classifier_backend = "experimental_densenet3d_attention_biconvlstm"
                return
            except Exception as error:
                self.classifier_error = error
                current_app.logger.warning(
                    "Configured experimental video classifier unavailable; falling back to existing ResNet18 + LSTM: %s",
                    type(error).__name__,
                )
        try:
            self.classifier = VideoClassificationModel(
                current_app.config["VIDEO_MODEL_PATH"],
                current_app.config["CLASS_NAMES_PATH"],
                current_app.config["VIDEO_MIN_CONFIDENCE"],
            )
            self.classifier_backend = "legacy_resnet18_lstm"
        except Exception as error:
            self.classifier = None
            if self.classifier_error is None:
                self.classifier_error = error

    def analyze(self, video_path, max_frames=None, full_video=False,
                category=None, original_filename=None, create_annotated_video=True):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Uploaded video is invalid or cannot be decoded")
        analysis_id = uuid.uuid4().hex
        evidence_dir = Path(current_app.config["UPLOAD_FOLDER"]) / "evidence" / analysis_id
        evidence_dir.mkdir(parents=True, exist_ok=True)
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            source_timing = _probe_source_timing(video_path, total_frames, fps)
            source_path = Path(video_path).resolve()
            source_hash = _sha256_file(source_path)
            source_size = source_path.stat().st_size if source_path.is_file() else None
            codec_code = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
            codec_fourcc = "".join(chr((codec_code >> (8 * index)) & 0xFF) for index in range(4)).rstrip("\x00")
            if total_frames <= 0:
                raise ValueError("Video has no readable frames")
            valid_fps = np.isfinite(fps) and fps > 0
            duration = source_timing.get("duration_sec")
            if duration is None:
                duration = total_frames / fps if valid_fps else None
            tracking_match = {"status": "NO_MATCH", "match_status": "NO_MATCH",
                              "reason": ("Normal-category videos have no saved tracking inventory; YOLO fallback will be used." if category == "normal" else
                                         "Choose Assault, Robbery, or Shooting to match saved ByteTrack results."),
                              "artifact_type": None, "record_count": 0, "detection_count": 0,
                              "artifact_reused": False,
                              "tracked_record_count": 0, "tracking_id_count": 0,
                              "frame_coverage_count": 0,
                              "metadata_verification": {"source_sha256": False, "fps": False,
                                                        "frame_count": False, "duration": False,
                                                        "frame_index_mapping": False}}
            if category and category != "normal" and original_filename and valid_fps:
                tracking_match = EvidenceCatalogService(current_app.config).match_tracking_artifact(
                    category, original_filename, video_path,
                    {"total_frames": total_frames, "fps": fps, "duration_sec": duration,
                     "width": source_width, "height": source_height},
                )
                # A full-video request promises sequential inference over the
                # source frames. A byte-identical saved CSV can still be a
                # partial/filtered track export, so it cannot satisfy that
                # coverage contract without an independently recorded frame
                # processing manifest.
                if (not full_video and
                        tracking_match.get("status") in {"EXACT_MATCH", "METADATA_MATCH"}):
                    tracking_match["artifact_reused"] = True
                    cap.release()
                    return self._analyze_saved_tracking(video_path, original_filename, category,
                                                        tracking_match, total_frames, fps, duration,
                                                        source_width, source_height, source_timing)
                if full_video and tracking_match.get("status") in {"EXACT_MATCH", "METADATA_MATCH"}:
                    tracking_match["artifact_reused"] = False
                    tracking_match["reuse_skipped_reason"] = "full_video_requested_requires_sequential_inference"
            configured_stride = int(current_app.config.get("VIDEO_SAMPLE_INTERVAL", 0))
            target_sample_fps = max(0.1, float(current_app.config.get("VIDEO_SAMPLE_FPS", 1.0)))
            # A zero frame-interval means use the configured sampling rate. The
            # old fallback coerced zero to a one-frame stride and silently ran
            # YOLO on every frame, despite VIDEO_SAMPLE_FPS=1 in the config.
            stride = (1 if full_video else max(
                1, configured_stride if configured_stride > 0 else
                int(round(fps / target_sample_fps)) if valid_fps else 1
            ))
            configured_limit = max(1, int(current_app.config.get("VIDEO_MAX_PROCESSED_FRAMES", 1000)))
            duration_limit = float(current_app.config.get("VIDEO_MAX_DURATION_SEC", 600))
            if duration_limit > 0 and source_timing.get("frame_timestamps_sec"):
                duration_frame_limit = sum(value < duration_limit
                                            for value in source_timing["frame_timestamps_sec"])
            else:
                duration_frame_limit = (int(math.ceil(duration_limit * fps))
                                        if valid_fps and duration_limit > 0 else total_frames)
            frame_limit = min(total_frames, duration_frame_limit) if full_video else configured_limit
            if max_frames is not None:
                frame_limit = min(frame_limit, max(1, int(max_frames)))
            baseline_indices = set(range(0, total_frames, stride))
            frame_results = []
            if self.yolo is None:
                self.yolo = YOLOService()
            tracker = {"used": hasattr(self.yolo, "track_video_frame"), "fallback_reason": None}
            tracking_session = getattr(self.yolo, "video_tracking_session", None)
            if tracking_session:
                tracking_context = tracking_session()
            else:
                from contextlib import nullcontext
                tracking_context = nullcontext()
            tracking_context.__enter__()
            frame_number = 0
            while frame_number < total_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_timing = _source_frame_timing(frame_number, fps, source_timing)
                timestamp = frame_timing["timestamp_sec"]
                if full_video and duration_limit > 0 and timestamp is not None and timestamp >= duration_limit:
                    break
                analyze_frame = frame_number in baseline_indices
                if not analyze_frame or len(frame_results) >= frame_limit:
                    frame_number += 1
                    continue
                try:
                    if tracker["used"]:
                        detections, annotated = self.yolo.track_video_frame(frame, timestamp, frame_number)
                    else:
                        detections, annotated = self.yolo.detect_frame(frame)
                except Exception as error:
                    if not tracker["used"]:
                        raise
                    tracker.update({"used": False, "fallback_reason": type(error).__name__})
                    try:
                        detections, annotated = self.yolo.detect_frame(frame)
                    except AttributeError:
                        raise error
                annotated = annotated if annotated is not None else frame.copy()
                for detection in detections:
                    detection.setdefault("frame_number", int(frame_number))
                    detection["timestamp_sec"] = timestamp
                    detection["presentation_timestamp"] = frame_timing["presentation_timestamp"]
                    detection["time_base"] = frame_timing["time_base"]
                    detection["timestamp_mapping_status"] = frame_timing["timestamp_mapping_status"]
                    detection["timestamp_basis"] = frame_timing["timestamp_basis"]
                    detection["timestamps_verified"] = frame_timing["timestamps_verified"]
                    detection.setdefault("class_name", detection.get("class", "unknown"))
                    detection.setdefault("review_status", "Needs review")
                base_name = f"frame_{frame_number:09d}.jpg"
                original_name = f"original_{base_name}"
                annotated_ok = bool(cv2.imwrite(str(evidence_dir / base_name), annotated))
                original_ok = bool(cv2.imwrite(str(evidence_dir / original_name), frame))
                frame_results.append({
                    "frame_number": int(frame_number),
                    "timestamp_sec": timestamp,
                    "presentation_timestamp": frame_timing["presentation_timestamp"],
                    "time_base": frame_timing["time_base"],
                    "timestamp_mapping_status": frame_timing["timestamp_mapping_status"],
                    "timestamp_basis": frame_timing["timestamp_basis"],
                    "timestamps_verified": frame_timing["timestamps_verified"],
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "detections": detections,
                    "detection_count": len(detections),
                    "evidence_image": f"/api/video/evidence/{analysis_id}/{base_name}" if annotated_ok else None,
                    "original_evidence_image": f"/api/video/evidence/{analysis_id}/{original_name}" if original_ok else None,
                    "annotated_evidence_frame": base_name if annotated_ok else None,
                    "original_evidence_frame": original_name if original_ok else None,
                })
                frame_number += 1
        finally:
            try:
                tracking_context.__exit__(None, None, None)
            except (UnboundLocalError, AttributeError):
                pass
            cap.release()

        # Produce a review copy at source frame rate. Only frames on which YOLO
        # ran receive overlays; gaps remain untouched source frames and are
        # reflected by the coverage fields below.
        annotated_video = evidence_dir / "annotated_tracking.mp4"
        annotated_video_available = False
        if create_annotated_video:
            review_cap = cv2.VideoCapture(str(video_path))
            writer = None
            if review_cap.isOpened() and valid_fps:
                frame_map = {item["frame_number"]: item["annotated_evidence_frame"] for item in frame_results}
                codec = cv2.VideoWriter_fourcc(*"mp4v")
                width = int(review_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(review_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width > 0 and height > 0:
                    writer = cv2.VideoWriter(str(annotated_video), codec, fps, (width, height))
                    if not writer.isOpened():
                        writer.release()
                        writer = None
                    else:
                        review_frame = 0
                        while True:
                            ok, source_frame = review_cap.read()
                            if not ok:
                                break
                            overlay_name = frame_map.get(review_frame)
                            if overlay_name:
                                overlay = cv2.imread(str(evidence_dir / overlay_name))
                                if overlay is not None:
                                    source_frame = overlay
                            writer.write(source_frame)
                            review_frame += 1
            if writer is not None:
                writer.release()
            review_cap.release()
            annotated_video_available = annotated_video.is_file() and annotated_video.stat().st_size > 0

        _temporal_confirmation(frame_results)
        classifier_coverage = None
        try:
            if self.classifier is None:
                raise self.classifier_error or RuntimeError("Video classifier is unavailable")
            set_context = getattr(self.classifier, "set_evidence_context", None)
            if callable(set_context):
                set_context(analysis_id, evidence_dir)
            classification = _classify_video(
                self.classifier, video_path,
                current_app.config.get("VIDEO_MAX_DURATION_SEC"),
                float(current_app.config.get("VIDEO_ACTIVITY_MIN_CONFIDENCE", 0.55)),
            )
            classification["classifier_backend"] = self.classifier_backend
            checkpoint_path = getattr(self.classifier, "checkpoint_path", None)
            if checkpoint_path is not None:
                classification["model_checkpoint"] = str(checkpoint_path)
            classifier_coverage = classification.get("sampling_coverage") if isinstance(classification, dict) else None
        except Exception as error:
            classification = {"label": "Unavailable", "confidence": None, "probabilities": {},
                              "top_class": "Unavailable", "classification_status": "unavailable",
                              "temporal_windows": [], "activity_events": [],
                              "classifier_note": f"Video classification failed: {type(error).__name__}"}
        all_detections = [d for frame in frame_results for d in frame["detections"]]
        tracked_detections = sum(d.get("track_id") is not None for d in all_detections)
        tracker.update({
            "method": "ByteTrack" if tracker.get("used") else "untracked_detection_fallback",
            "tracker_config": current_app.config.get("VIDEO_TRACKER", "bytetrack.yaml"),
            "model_checkpoint": getattr(self.yolo, "_model_path_used", None) or current_app.config.get("YOLO_MODEL_PATH"),
            "confidence_threshold": float(current_app.config.get("VIDEO_CONFIDENCE_THRESHOLD", current_app.config.get("YOLO_CONFIDENCE", 0.25))),
            "inference_stride_frames": stride,
            "detection_count": len(all_detections),
            "tracked_detection_count": tracked_detections,
            "untracked_detection_count": len(all_detections) - tracked_detections,
            "track_ids_available": tracked_detections > 0,
            "track_id_count": len({d.get("track_id") for d in all_detections if d.get("track_id") is not None}),
            "frames_with_detections": len({d.get("frame_number") for d in all_detections if d.get("frame_number") is not None}),
        })
        timeline = [
            {"frame_number": d["frame_number"], "timestamp_sec": d["timestamp_sec"],
             "presentation_timestamp": d.get("presentation_timestamp"),
             "time_base": d.get("time_base"),
             "timestamp_mapping_status": d.get("timestamp_mapping_status"),
             "class_id": d.get("class_id"), "class": d["class"], "class_name": d["class"],
             "confidence": d["confidence"], "bbox": d["bbox"], "track_id": d.get("track_id")}
            for d in all_detections
        ]
        unique_objects = sorted({d["class"] for d in all_detections})
        whole_prediction = classification.get("whole_video_prediction") or {}
        event_name = whole_prediction.get("label") or classification.get("label", "Unavailable")
        evidence = ", ".join(unique_objects) if unique_objects else "no objects"
        classifier_confidence = whole_prediction.get("confidence", classification.get("confidence"))
        confidence_summary = (f" ({classifier_confidence:.1%} confidence)"
                              if isinstance(classifier_confidence, (int, float)) else "")
        if classification.get("classification_status") == "uncertain_prediction_disagreement":
            temporal_prediction = classification.get("temporal_prediction") or {}
            summary = (
                f"Classification is unresolved: the whole-video model predicted {event_name}"
                f"{confidence_summary}, while temporal windows predicted {temporal_prediction.get('label', 'Unavailable')}. "
            )
        else:
            summary = f"The provisional whole-video classifier prediction is {event_name}{confidence_summary}. "
        coverage_word = "all analyzed frames" if full_video and stride == 1 else "sampled frames"
        summary += (f"YOLO visual evidence in {coverage_word} includes {evidence}. "
                    "Object detections do not establish that a crime occurred.")
        processed = len(frame_results)
        # Processing every planned sample is not full-video inference when the
        # configured stride skips source frames.
        reached_end = frame_number >= total_frames
        intentional_duration_stop = bool(full_video and duration_limit > 0 and duration is not None
                                          and duration > duration_limit and frame_number < total_frames)
        covered = reached_end and len(frame_results) >= total_frames
        first_time = next((item["timestamp_sec"] for item in frame_results if item.get("timestamp_sec") is not None), None)
        last_time = next((item["timestamp_sec"] for item in reversed(frame_results) if item.get("timestamp_sec") is not None), None)
        events = group_evidence_events(frame_results,
            max_gap_sec=float(current_app.config.get("VIDEO_EVENT_MAX_GAP_SEC", 3.0)),
            min_iou=float(current_app.config.get("VIDEO_EVENT_MIN_IOU", .1)))
        source_video_path = str(Path(video_path).resolve())
        for event in events:
            event["review_status"] = "candidate"
            event["event_label"] = event["class"]
            event["source_video_identity"] = Path(video_path).name
            event["supporting_frame_numbers"] = sorted({
                item["frame_number"] for item in event.get("observations", [])
                if item.get("frame_number") is not None
            })
            event["related_track_ids"] = sorted({
                item["track_id"] for item in event.get("observations", [])
                if item.get("track_id") is not None
            })
            representative_number = event.get("representative_frame_number")
            representative = next((item for item in frame_results if item["frame_number"] == representative_number), None)
            event["representative_original_evidence_frame"] = representative.get("original_evidence_frame") if representative else None
            event["representative_original_evidence_image"] = representative.get("original_evidence_image") if representative else None
            event["representative_evidence_frame"] = (Path(event["representative_evidence_image"]).name
                                                       if event.get("representative_evidence_image") else None)
            event["evidence_frame_path"] = event.get("representative_evidence_image")
            event["annotated_video_path"] = f"/api/video/evidence/{analysis_id}/annotated_tracking.mp4" if annotated_video_available else None
        for event in events:
            if event.get("observation_count", 0) < int(current_app.config.get("VIDEO_EVENT_MIN_OBSERVATIONS", 3)):
                event["verification_status"] = "unverified"
        timeline_path = evidence_dir / "evidence_timeline.csv"
        columns = ["frame_number", "timestamp_sec", "presentation_timestamp", "time_base", "timestamp_mapping_status", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "bbox_xyxy", "track_id", "tracking_status"]
        with timeline_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            for detection in all_detections:
                bbox = detection.get("bbox") or [None] * 4
                writer.writerow({"frame_number": detection.get("frame_number"),
                                 "timestamp_sec": detection.get("timestamp_sec"),
                                 "presentation_timestamp": detection.get("presentation_timestamp"),
                                 "time_base": detection.get("time_base"),
                                 "timestamp_mapping_status": detection.get("timestamp_mapping_status"),
                                 "class_id": detection.get("class_id"),
                                 "class_name": detection.get("class_name", detection.get("class")),
                                 "confidence": detection.get("confidence"),
                                 **dict(zip(("x1", "y1", "x2", "y2"), bbox)),
                                 "bbox_xyxy": json.dumps(bbox),
                                 "track_id": detection.get("track_id"),
                                 "tracking_status": detection.get("tracking_status", "untracked")})
        track_observations = defaultdict(list)
        for detection in all_detections:
            track_id = detection.get("track_id")
            if track_id is not None and not isinstance(track_id, bool):
                track_observations[str(track_id)].append(detection)
        tracking_path = evidence_dir / "tracking_summary.csv"
        tracking_columns = ["track_id", "class_names", "detection_count", "start_frame", "end_frame",
                            "start_timestamp_sec", "end_timestamp_sec", "duration_sec", "continuity_fraction"]
        with tracking_path.open("w", newline="", encoding="utf-8-sig") as tracking_file:
            tracking_writer = csv.DictWriter(tracking_file, fieldnames=tracking_columns)
            tracking_writer.writeheader()
            for track_id, observations in sorted(track_observations.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0])):
                observed_frames = sorted({int(item["frame_number"]) for item in observations if item.get("frame_number") is not None})
                observed_times = sorted(float(item["timestamp_sec"]) for item in observations
                                        if isinstance(item.get("timestamp_sec"), (int, float)) and math.isfinite(float(item["timestamp_sec"])))
                tracking_writer.writerow({
                    "track_id": track_id,
                    "class_names": ";".join(sorted({str(item.get("class_name") or item.get("class") or "unknown") for item in observations})),
                    "detection_count": len(observations),
                    "start_frame": min(observed_frames) if observed_frames else None,
                    "end_frame": max(observed_frames) if observed_frames else None,
                    "start_timestamp_sec": min(observed_times) if observed_times else None,
                    "end_timestamp_sec": max(observed_times) if observed_times else None,
                    "duration_sec": round(max(observed_times) - min(observed_times), 6) if observed_times else None,
                    "continuity_fraction": round(len(observed_frames) / (max(observed_frames) - min(observed_frames) + 1), 6) if observed_frames else None,
                })
        events_path = evidence_dir / "evidence_events.csv"
        event_columns = ["analysis_id", "source_video_path", "event_id", "event_label", "start_timestamp_sec",
                         "end_timestamp_sec", "frame_numbers", "related_track_ids", "confidence_mean",
                         "evidence_frame_path", "annotated_video_path", "review_status", "verification_status"]
        with events_path.open("w", newline="", encoding="utf-8-sig") as event_file:
            event_writer = csv.DictWriter(event_file, fieldnames=event_columns)
            event_writer.writeheader()
            for event in events:
                event_writer.writerow({
                    "analysis_id": analysis_id, "source_video_path": source_video_path,
                    "event_id": event.get("event_id"), "event_label": event.get("event_label"),
                    "start_timestamp_sec": event.get("start_timestamp_sec"),
                    "end_timestamp_sec": event.get("end_timestamp_sec"),
                    "frame_numbers": json.dumps(event.get("supporting_frame_numbers", [])),
                    "related_track_ids": json.dumps(event.get("related_track_ids", [])),
                    "confidence_mean": (event.get("confidence_summary") or {}).get("mean"),
                    "evidence_frame_path": event.get("evidence_frame_path"),
                    "annotated_video_path": event.get("annotated_video_path"),
                    "review_status": event.get("review_status"),
                    "verification_status": event.get("verification_status"),
                })
        classifier_checkpoint = classification.get("model_checkpoint")
        detector_checkpoint = tracker.get("model_checkpoint")
        model_metadata = {
            "video_classifier": {"path": classifier_checkpoint,
                                 "sha256": _sha256_file(classifier_checkpoint) if classifier_checkpoint else None},
            "object_detector": {"path": detector_checkpoint,
                                "sha256": _sha256_file(detector_checkpoint) if detector_checkpoint else None},
        }
        source_metadata = {
            "filename": Path(video_path).name, "file_size_bytes": source_size,
            "fps": round(fps, 6) if valid_fps else None,
            "total_frames": total_frames, "width": source_width,
            "height": source_height, "codec_fourcc": codec_fourcc or None,
            "presentation_timestamps_available": source_timing["presentation_timestamps_available"],
            "presentation_time_base": source_timing["time_base"],
            "probed_frame_count": source_timing["probed_frame_count"],
            "source_duration_sec": duration,
            "constant_frame_rate_verified": source_timing["constant_frame_rate_verified"],
            "timestamp_mapping_status": source_timing["timestamp_mapping_status"],
            "timestamp_basis": ("source_presentation_timestamp_times_time_base"
                                if source_timing["timestamp_mapping_status"] == "verified_vfr_presentation_timestamps" else
                                source_timing["timestamp_basis"]),
            "timestamps_verified": source_timing["timestamps_verified"],
        }
        processing_configuration = {
            "full_video_requested": bool(full_video), "frame_stride": stride,
            "frame_limit": frame_limit, "duration_limit_sec": duration_limit,
            "configured_sample_fps": target_sample_fps,
            "configured_sample_interval_frames": configured_stride,
            "detector_confidence_threshold": float(current_app.config.get("YOLO_CONFIDENCE", .25)),
            "tracker_config": tracker.get("tracker_config"),
            "tracking_method": tracker.get("method"),
        }
        record = _json_safe({
            "schema_version": 1, "analysis_id": analysis_id,
            "verification_status": "unverified", "classification": classification,
            "source_video_identity": Path(video_path).name,
            "source_integrity": {"sha256": source_hash, "algorithm": "SHA-256",
                                 "hash_scope": "exact source file bytes"},
            "source_metadata": source_metadata,
            "model_metadata": model_metadata,
            "processing_configuration": processing_configuration,
            "frames": [{**frame, "evidence_image": None, "original_evidence_image": None} for frame in frame_results],
            "grouped_events": events, "tracking": tracker,
        })
        (evidence_dir / "analysis.json").write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
        result = {
            "analysis_id": analysis_id,
            "source_video_path": source_video_path,
            "source_metadata": source_metadata,
            "source_integrity": {"sha256": source_hash, "algorithm": "SHA-256",
                                 "hash_scope": "exact source file bytes"},
            "model_metadata": model_metadata,
            "processing_configuration": processing_configuration,
            "artifact_paths": {
                "annotated_video": str(annotated_video.resolve()) if annotated_video_available else None,
                "detection_csv": str(timeline_path.resolve()),
                "event_csv": str(events_path.resolve()),
                "tracking_csv": str(tracking_path.resolve()),
                "analysis_json": str((evidence_dir / "analysis.json").resolve()),
            },
            "video_info": {
                "total_frames": total_frames,
                "fps": round(fps, 3) if valid_fps else None,
                "duration_sec": round(duration, 3) if duration is not None else None,
                "frames_analyzed": len(frame_results),
                "timestamps_available": bool(valid_fps),
                "sampling_interval_frames": stride, "analysis_mode": "full_video" if covered else "sampled",
                "processed_frames": processed, "frame_limit_reached": processed >= frame_limit and len(baseline_indices) > frame_limit,
                "duration_limit_reached": bool(full_video and duration_limit > 0 and duration is not None and duration > duration_limit),
                "analysis_covered_entire_video": covered,
                "analysis_coverage": "entire_video" if covered else "portion",
                "inference_coverage_fraction": round(processed / total_frames, 6) if total_frames else None,
                "inference_coverage_percent": round(processed / total_frames * 100, 2) if total_frames else None,
                "processing_status": (
                    "completed_full_coverage" if covered else
                    "completed_partial_coverage" if reached_end or intentional_duration_stop else "interrupted"
                ),
                "first_analyzed_timestamp_sec": first_time, "last_analyzed_timestamp_sec": last_time,
                "classifier_sampling_coverage": classifier_coverage,
                "width": frame_results[0]["frame_width"] if frame_results else None,
                "height": frame_results[0]["frame_height"] if frame_results else None,
                "sampling_strategy": "fixed_interval",
                "timestamp_source": source_timing["timestamp_basis"],
                "source_video_path": source_video_path,
                "file_size_bytes": source_size,
                "codec_fourcc": codec_fourcc or None,
                "source_sha256": source_hash,
                "presentation_timestamps_available": source_timing["presentation_timestamps_available"],
                "timestamp_mapping_status": source_timing["timestamp_mapping_status"],
                "timestamp_basis": source_timing["timestamp_basis"],
                "timestamps_verified": source_timing["timestamps_verified"],
                "constant_frame_rate_verified": source_timing["constant_frame_rate_verified"],
                "timestamp_warning": (None if source_timing["timestamps_verified"] else
                    "Timestamp mapping is estimated from frame index and reported FPS; variable-frame-rate timing is unverified."),
                "annotated_video_available": annotated_video_available,
            },
            "classification": classification,
            "event_classification": classification,
            "frame_results": frame_results,
            "detections": all_detections,
            "unique_objects": unique_objects,
            "total_detections": len(all_detections),
            "timeline": timeline,
            "event_timeline": timeline,
            "evidence_events": events,
            "activity_events": _attach_activity_support(classification.get("activity_events", []), all_detections),
            "candidate_intervals": classification.get("activity_events", []),
            "candidate_interval_status": ("unverified_classifier_candidates" if classification.get("temporal_windows") else "not_evaluated"),
            "verification_status": "unverified",
            "candidate_interval_limitation": ("Intervals are unverified candidate locations derived from windows of the whole-video-trained classifier; window categories can disagree with the whole-video prediction, and boundaries are not independently validated." if classification.get("temporal_windows") else "Temporal action interval recognition is unavailable from this whole-video classifier output."),
            "tracking": tracker,
            "tracking_data_match": tracking_match,
            "category": category.title() if category else None,
            "file_name": original_filename or Path(video_path).name,
            "evidence_timeline_csv": f"/api/video/evidence/{analysis_id}/evidence_timeline.csv",
            "evidence_events_csv": f"/api/video/evidence/{analysis_id}/evidence_events.csv",
            "tracking_summary_csv": f"/api/video/evidence/{analysis_id}/tracking_summary.csv",
            "analysis_json": f"/api/video/evidence/{analysis_id}/analysis.json",
            "annotated_video": f"/api/video/evidence/{analysis_id}/annotated_tracking.mp4" if annotated_video_available else None,
            "investigation_summary": summary,
            "disclaimer": "AI-generated results are intended to assist investigation and require verification by authorized personnel.",
            "blood_stain_supported": False,
        }
        return result

    def _analyze_saved_tracking(self, video_path, original_filename, category,
                                match, total_frames, fps, duration,
                                source_width=None, source_height=None, timing=None):
        """Use a verified saved CSV without invoking YOLO or opening its render."""
        timing = timing or _probe_source_timing(video_path, total_frames, fps)
        csv_path = Path(match.get("csv_path") or match.get("artifact_path") or "")
        evidence_dir = Path(current_app.config["UPLOAD_FOLDER"]) / "evidence" / uuid.uuid4().hex
        evidence_dir.mkdir(parents=True, exist_ok=True)
        frame_map = {}
        all_detections = []
        saved_rows = match.get("records")
        if saved_rows is None:
            with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
                saved_rows = list(csv.DictReader(handle))
        for row in saved_rows:
            values = {str(key).strip().casefold(): value for key, value in row.items() if key is not None}
            frame_number = int(float(values.get("frame", values.get("frame_number", values.get("frame_id"))))) - int(match.get("frame_offset", 0))
            frame_timing = _source_frame_timing(frame_number, fps, timing)
            timestamp = round(frame_timing["timestamp_sec"], 3) if frame_timing["timestamp_sec"] is not None else None
            label = str(values.get("class", values.get("class_name", values.get("label", "unknown")))).strip()
            class_value = values.get("class_id", values.get("class_index"))
            class_id = int(float(class_value)) if class_value not in (None, "") else None
            confidence_value = values.get("confidence", values.get("score"))
            confidence = float(confidence_value) if confidence_value not in (None, "") else None
            bbox = [float(values[name]) for name in ("x1", "y1", "x2", "y2")]
            raw_track_id = str(values.get("track_id", values.get("tracker_id", values.get("object_id", ""))) or "").strip()
            try:
                numeric_track_id = float(raw_track_id)
                track_id = int(numeric_track_id) if numeric_track_id >= 0 and numeric_track_id.is_integer() else None
            except (TypeError, ValueError):
                track_id = None
            label_key = label.casefold()
            forensic_category = ("Weapons" if any(word in label_key for word in ("weapon", "gun", "pistol", "rifle", "knife"))
                                 else "Persons" if "person" in label_key
                                 else "Vehicles" if any(word in label_key for word in ("vehicle", "car", "truck", "bus"))
                                 else "Other")
            detection = {
                "frame_number": frame_number, "timestamp_sec": timestamp,
                "timestamp_mapping_status": frame_timing["timestamp_mapping_status"],
                "timestamps_verified": frame_timing["timestamps_verified"],
                "class_id": class_id, "class": label, "class_name": label,
                "confidence": confidence, "bbox": bbox, "track_id": track_id,
                "tracking_status": ("tracked_from_saved_bytetrack" if track_id is not None else "untracked_saved_detection") if match.get("artifact_type") != "txt_labels" else ("unverified_txt_track_id" if track_id is not None else "untracked_txt_label"),
                "category": forensic_category, "review_status": "Needs review",
            }
            all_detections.append(detection)
            frame_map.setdefault(frame_number, []).append(detection)

        frame_results = [{
            "frame_number": number, "timestamp_sec": (
                round(_source_frame_timing(number, fps, timing)["timestamp_sec"], 3)
                if _source_frame_timing(number, fps, timing)["timestamp_sec"] is not None else None),
            "timestamp_mapping_status": _source_frame_timing(number, fps, timing)["timestamp_mapping_status"],
            "timestamp_basis": _source_frame_timing(number, fps, timing)["timestamp_basis"],
            "timestamps_verified": _source_frame_timing(number, fps, timing)["timestamps_verified"],
            "frame_width": source_width, "frame_height": source_height,
            "detections": detections, "detection_count": len(detections),
            "evidence_image": None, "original_evidence_image": None,
        } for number, detections in sorted(frame_map.items())]
        object_events = group_evidence_events(
            frame_results,
            max_gap_sec=float(current_app.config.get("VIDEO_EVENT_MAX_GAP_SEC", 3.0)),
            min_iou=float(current_app.config.get("VIDEO_EVENT_MIN_IOU", 0.1)),
        )
        try:
            if self.classifier is None:
                raise self.classifier_error or RuntimeError("Video classifier unavailable")
            set_context = getattr(self.classifier, "set_evidence_context", None)
            if callable(set_context):
                set_context(evidence_dir.name, evidence_dir)
            classification = _classify_video(
                self.classifier, video_path,
                current_app.config.get("VIDEO_MAX_DURATION_SEC"),
                float(current_app.config.get("VIDEO_ACTIVITY_MIN_CONFIDENCE", 0.55)),
            )
            classification["classifier_backend"] = self.classifier_backend
            checkpoint_path = getattr(self.classifier, "checkpoint_path", None)
            if checkpoint_path is not None:
                classification["model_checkpoint"] = str(checkpoint_path)
        except Exception as error:
            classification = {"label": "Unavailable", "confidence": None,
                              "top_class": "Unavailable", "classification_status": "unavailable",
                              "probabilities": {}, "temporal_windows": [],
                              "activity_events": [],
                              "classifier_note": f"Video classification failed: {type(error).__name__}"}

        activity_events = _attach_activity_support(classification.get("activity_events", []), all_detections)

        unique_objects = sorted({item["class"] for item in all_detections})
        tracked_count = sum(item["track_id"] is not None for item in all_detections)
        tracking = {
            "used": True, "method": "saved_tracking_csv" if match.get("artifact_type") == "csv_tracking" else "saved_txt_labels", "status": match.get("status"),
            "tracking_algorithm": ("Unverified; imported TXT labels contain IDs but tracker provenance is unavailable"
                                   if match.get("artifact_type") == "txt_labels" else
                                   "ByteTrack results directory; IDs are accepted only when valid on the individual CSV row"),
            "source_csv": csv_path.name if match.get("artifact_type") == "csv_tracking" else None,
            "source_artifact": str(match.get("artifact_path") or csv_path),
            "source_artifact_filename": match.get("artifact_filename") or csv_path.name,
            "artifact_type": match.get("artifact_type"), "record_count": len(all_detections),
            "detection_count": len(all_detections), "tracked_detection_count": tracked_count,
            "untracked_detection_count": len(all_detections) - tracked_count,
            "track_ids_available": tracked_count > 0,
            "annotated_video_required": False, "annotated_video_available": False,
        }
        public_match = {key: match.get(key) for key in (
            "status", "match_status", "reason", "artifact_type", "artifact_path", "record_count", "detection_count", "frame_min", "frame_max", "frame_coverage_count", "tracked_record_count", "tracking_id_count", "metadata_verification", "frame_index_mapping_status", "frame_index_convention", "artifact_file_count", "candidate_count", "ambiguity", "job_status", "manifest_video", "artifact_filename", "artifact_version_count", "artifact_reused", "reuse_skipped_reason"
        )}
        public_match["artifact_reused"] = True
        public_match["source_filename"] = Path(match["source_path"]).name
        public_match["artifact_filename"] = match.get("artifact_filename") or csv_path.name
        analysis_id = evidence_dir.name
        timeline_path = evidence_dir / "evidence_timeline.csv"
        columns = ["frame_number", "timestamp_sec", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "track_id", "tracking_status"]
        with timeline_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for item in all_detections:
                writer.writerow({"frame_number": item["frame_number"], "timestamp_sec": item["timestamp_sec"],
                                 "class_id": item["class_id"], "class_name": item["class"],
                                 "confidence": item["confidence"], **dict(zip(("x1", "y1", "x2", "y2"), item["bbox"])),
                                 "track_id": item["track_id"], "tracking_status": item["tracking_status"]})
        result = {
            "analysis_id": analysis_id, "source_video_path": str(Path(video_path).resolve()),
            "file_name": original_filename, "category": category.title(),
            "artifact_paths": {"detection_csv": str(timeline_path.resolve()),
                               "analysis_json": str((evidence_dir / "analysis.json").resolve())},
            "video_info": {"total_frames": total_frames, "fps": round(fps, 3),
                           "duration_sec": round(duration, 3), "frames_analyzed": len(frame_results),
                           "width": source_width, "height": source_height,
                           "processed_frames": len(frame_results), "timestamps_available": True,
                           "analysis_coverage": "saved_tracking_frames", "processing_status": "completed_saved_tracking_reused",
                           "timestamp_source": timing["timestamp_basis"],
                           "timestamp_mapping_status": timing["timestamp_mapping_status"],
                           "timestamps_verified": timing["timestamps_verified"],
                           "timestamp_warning": (None if timing["timestamps_verified"] else
                               "Timestamp mapping is estimated from frame index and reported FPS; variable-frame-rate timing is unverified."),
                           "frame_index_mapping_status": ("unverified" if match.get("artifact_type") == "txt_labels" else "verified")},
            "classification": classification, "event_classification": classification,
            "frame_results": frame_results, "detections": all_detections,
            "unique_objects": unique_objects, "total_detections": len(all_detections),
            "tracking": tracking, "tracking_data_match": public_match,
            "evidence_events": object_events, "activity_events": activity_events,
            "candidate_intervals": activity_events,
            "candidate_interval_status": "unverified_classifier_candidates",
            "verification_status": "unverified",
            "candidate_interval_limitation": "Activity intervals are candidates from overlapping windows of a whole-video-trained classifier; they are not independently validated event boundaries.",
            "annotated_video": None,
            "evidence_timeline_csv": f"/api/video/evidence/{analysis_id}/evidence_timeline.csv",
            "analysis_json": f"/api/video/evidence/{analysis_id}/analysis.json",
            "investigation_summary": (f"The provisional whole-video classifier prediction for the uploaded original is {classification.get('label')}. "
                                      + ("Saved TXT labels were read with unverified frame and tracker provenance; no annotated render was required."
                                         if match.get("artifact_type") == "txt_labels" else
                                         "Verified saved tracking results were reused; no annotated render was required.")),
            "disclaimer": "AI-generated results are unverified investigative leads and require human review.",
            "blood_stain_supported": False,
        }
        (evidence_dir / "analysis.json").write_text(json.dumps(_json_safe(result), allow_nan=False), encoding="utf-8")
        return result
