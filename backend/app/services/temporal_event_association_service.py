"""Temporal classifier and saved-tracking association for candidate intervals.

Scores are review aids, not calibrated probabilities. Defaults are explicitly
uncalibrated and must be tuned against independently verified intervals.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from app.services.shooting_event_evidence_service import EventEvidenceService, _num, sha256_file
from app.services.video_timing import probe_video_timeline


class TemporalEventAssociationService:
    CATEGORIES = {"shooting": "Shooting", "assault": "Assault", "robbery": "Robbery"}
    DEFAULT_WEIGHTS = {
        # Starting configuration only; no measured accuracy claim is attached.
        "shooting": {"classifier": .70, "yolo": .15, "tracking": .15},
        "assault": {"classifier": .70, "yolo": .15, "tracking": .15},
        "robbery": {"classifier": .70, "yolo": .15, "tracking": .15},
    }
    RELEVANT_CLASSES = {
        "shooting": {"person", "gun", "pistol", "rifle", "weapon"},
        "assault": {"person", "weapon", "gun", "pistol", "rifle", "knife"},
        "robbery": {"person", "bag", "backpack", "handbag", "phone", "object", "weapon", "gun", "pistol"},
    }

    def __init__(self, config, evidence_service=None):
        self.config = config
        self.evidence = evidence_service or EventEvidenceService(config)

    @staticmethod
    def presentation_timestamps(video_path):
        """Return normalized PTS only after the shared probe verifies frame alignment."""
        # The shared probe requires OpenCV's decoded frame count and FPS so
        # it can verify that ffprobe's frame-level PTS align with source frames.
        metadata = EventEvidenceService._metadata(video_path)
        timeline = probe_video_timeline(
            video_path, expected_frames=metadata["frames"], fps=metadata["fps"]
        )
        if timeline.mapping_status != "verified_vfr_presentation_timestamps":
            return None
        return {index: timestamp for index, timestamp in enumerate(timeline.timestamps_sec)}

    @staticmethod
    def _normalise_prediction(window, category):
        probs = window.get("probabilities") or {}
        probability = _num(probs.get(category.title()))
        if probability is None and str(window.get("label", "")).casefold() == category:
            probability = _num(window.get("confidence"), 0.0)
        return {
            "start": _num(window.get("start_timestamp_sec", window.get("start_sec"))),
            "end": _num(window.get("end_timestamp_sec", window.get("end_sec"))),
            "probability": probability,
            "label": str(window.get("label") or ""),
            "probabilities": probs,
        }

    @staticmethod
    def _read_tracking_csv(csv_path, video_name, frame_offset, fps, pts_by_frame=None):
        rows = []
        with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            for raw in reader:
                row = {str(key).strip().casefold(): value for key, value in raw.items() if key is not None}
                declared_video = row.get("video") or row.get("video_name") or row.get("filename")
                if declared_video and Path(declared_video).name.casefold() != Path(video_name).name.casefold():
                    continue
                frame = _num(row.get("frame", row.get("frame_number", row.get("frame_id"))))
                if frame is None:
                    continue
                csv_frame = int(frame)
                source_frame = csv_frame - int(frame_offset)
                csv_time = _num(row.get("timestamp_seconds", row.get("timestamp_sec", row.get("timestamp"))))
                pts = (pts_by_frame or {}).get(source_frame)
                timestamp = pts if pts is not None else (csv_time if csv_time is not None else source_frame / fps)
                confidence = _num(row.get("confidence", row.get("score")))
                bbox = [_num(row.get(key)) for key in ("x1", "y1", "x2", "y2")]
                class_name = str(row.get("class_name") or row.get("class") or row.get("label") or "unknown").strip()
                if source_frame < 0 or timestamp is None or timestamp < 0:
                    continue
                rows.append({"video": Path(video_name).name, "frame_number": source_frame,
                    "csv_frame_number": csv_frame, "timestamp_seconds": float(timestamp),
                    "csv_timestamp_seconds": csv_time, "timestamp_source": "presentation_timestamp" if pts is not None else ("csv_timestamp" if csv_time is not None else "frame_fps_fallback"),
                    "class_name": class_name, "class_id": row.get("class_id"),
                    "confidence": confidence, "track_id": row.get("track_id") or row.get("tracker_id") or None,
                    "bbox": bbox if all(value is not None for value in bbox) else None})
        return rows

    @staticmethod
    def _validate_csv_timing(csv_path, fps, frame_count, pts_by_frame=None, preferred_offset=0, video_name=None):
        pairs, frames = [], []
        with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as stream:
            for raw in csv.DictReader(stream):
                row = {str(key).strip().casefold(): value for key, value in raw.items() if key is not None}
                declared_video = row.get("video") or row.get("video_name") or row.get("filename")
                if video_name and declared_video and Path(declared_video).name.casefold() != Path(video_name).name.casefold():
                    continue
                frame = _num(row.get("frame", row.get("frame_number", row.get("frame_id"))))
                if frame is None:
                    continue
                frame = int(frame); frames.append(frame)
                timestamp = _num(row.get("timestamp_seconds", row.get("timestamp_sec", row.get("timestamp"))))
                if timestamp is not None:
                    pairs.append((frame, timestamp))
        if any(frame < 0 or frame > frame_count for frame in frames):
            raise ValueError("Tracking CSV frame numbers exceed the original video's frame range.")
        if not pairs:
            return int(preferred_offset or (1 if frames and max(frames) == frame_count else 0)), {
                "verified": False, "method": "frame_range_only_no_csv_timestamps"}
        candidates = []
        for offset in (0, 1):
            expected = []
            for frame, timestamp in pairs:
                source_frame = frame - offset
                if source_frame < 0:
                    expected = []; break
                pts = (pts_by_frame or {}).get(source_frame, (pts_by_frame or {}).get(str(source_frame)))
                expected_time = _num(pts) if pts is not None else source_frame / fps
                expected.append(abs(timestamp - expected_time))
            if expected:
                expected.sort()
                p95 = expected[int((len(expected) - 1) * .95)]
                candidates.append((p95, offset))
        if not candidates:
            raise ValueError("Tracking CSV timestamps could not be mapped to original frames.")
        error, offset = min(candidates)
        tolerance = max(.08, 2.5 / fps)
        if error > tolerance:
            raise ValueError("Tracking CSV timestamps do not align with original frame presentation timestamps or FPS mapping.")
        return offset, {"verified": True, "method": "presentation_timestamp" if pts_by_frame else "frame_fps_fallback",
                        "p95_absolute_error_sec": error, "tolerance_sec": tolerance}

    @staticmethod
    def _track_groups(rows, max_gap_sec):
        groups = defaultdict(list)
        for row in rows:
            if row.get("track_id") not in (None, ""):
                groups[(row["video"], row["class_name"].casefold(), str(row["track_id"]))].append(row)
        result = []
        for (video, object_class, track_id), observations in sorted(groups.items()):
            observations.sort(key=lambda item: (item["timestamp_seconds"], item["frame_number"]))
            times = [item["timestamp_seconds"] for item in observations]
            gaps = [max(0.0, right - left) for left, right in zip(times, times[1:])]
            result.append({"video": video, "class_name": object_class, "track_id": track_id,
                "detection_count": len(observations), "mean_confidence": _mean(item.get("confidence") for item in observations),
                "max_confidence": max((item["confidence"] for item in observations if item.get("confidence") is not None), default=None),
                "frame_count": len({item["frame_number"] for item in observations}),
                "first_timestamp_sec": times[0], "last_timestamp_sec": times[-1],
                "tracking_duration_sec": max(0.0, times[-1] - times[0]),
                "max_gap_sec": max(gaps, default=0.0), "detection_gaps_sec": gaps,
                "continuity_score": (sum(gap <= max_gap_sec for gap in gaps) / len(gaps)) if gaps else 0.0,
                "fragmented": any(gap > max_gap_sec for gap in gaps), "detections": observations})
        return result

    def _weights(self, category):
        configured = self.config.get("TEMPORAL_EVENT_WEIGHTS") or self.DEFAULT_WEIGHTS
        if isinstance(configured, str):
            configured = json.loads(configured)
        values = configured.get(category, configured.get(category.title().lower(), {}))
        valid = {key: float(value) for key, value in values.items()
                 if key in {"classifier", "yolo", "tracking", "action", "audio"}
                 and _num(value) is not None and float(value) >= 0}
        if not valid or valid.get("classifier", 0) <= max(valid.get("yolo", 0), valid.get("tracking", 0), valid.get("action", 0), valid.get("audio", 0)):
            raise ValueError("Temporal event weights must include classifier as the largest non-zero modality weight.")
        return valid

    def associate(self, *, original_path, category, start_sec, end_sec,
                  temporal_predictions=None, classifier_probabilities=None,
                  yolo_detections=None, video_info=None, additional_evidence=None,
                  csv_path=None, tracking_status=None, frame_offset=0,
                  presentation_timestamps=None, source_video_name=None):
        category_key = str(category or "").casefold()
        if category_key not in self.CATEGORIES:
            raise ValueError("Temporal association supports Shooting, Assault, and Robbery.")
        start, end = _num(start_sec), _num(end_sec)
        if start is None or end is None or start < 0 or end <= start:
            raise ValueError("Candidate interval must have finite non-negative start and later end timestamps.")
        original = Path(original_path).resolve()
        fps = _num((video_info or {}).get("fps"))
        if not fps or fps <= 0:
            fps = _num(self.evidence._metadata(original).get("fps"))
        pts_by_frame = presentation_timestamps
        if pts_by_frame is None:
            pts_by_frame = self.presentation_timestamps(original)

        artifact_manifest = None
        resolved_csv = Path(csv_path).resolve() if csv_path else None
        resolved_video = None
        if resolved_csv is None:
            artifact_manifest, resolved_csv, resolved_video = self.evidence.resolve_artifacts(
                original, self.CATEGORIES[category_key], (video_info or {}).get("original_filename") or original.name)
        if not resolved_csv.is_file():
            raise FileNotFoundError("Selected tracking CSV is unavailable.")
        metadata = self.evidence._metadata(original)
        canonical_video_name = Path(source_video_name or original.name).name
        frame_offset, csv_timing = self._validate_csv_timing(resolved_csv, fps, metadata["frames"], pts_by_frame,
            frame_offset, canonical_video_name)
        detections = self._read_tracking_csv(resolved_csv, canonical_video_name, frame_offset, fps, pts_by_frame)
        for raw in yolo_detections or []:
            if not isinstance(raw, dict):
                continue
            frame = _num(raw.get("frame_number", raw.get("frame")))
            timestamp = _num(raw.get("timestamp_seconds", raw.get("timestamp_sec")))
            if timestamp is None and frame is not None:
                timestamp = (pts_by_frame or {}).get(int(frame), int(frame) / fps)
            bbox = raw.get("bbox")
            if not bbox and all(key in raw for key in ("x1", "y1", "x2", "y2")):
                bbox = [raw.get(key) for key in ("x1", "y1", "x2", "y2")]
            detections.append({"video": canonical_video_name, "frame_number": int(frame) if frame is not None else None,
                "csv_frame_number": None, "timestamp_seconds": timestamp, "csv_timestamp_seconds": None,
                "timestamp_source": "analysis_yolo_timestamp", "class_name": str(raw.get("class_name", raw.get("class", "unknown"))),
                "class_id": raw.get("class_id"), "confidence": _num(raw.get("confidence")),
                "track_id": raw.get("track_id"), "bbox": bbox})
        unique = {}
        for item in detections:
            timestamp = _num(item.get("timestamp_seconds"))
            if timestamp is None:
                continue
            item["timestamp_seconds"] = timestamp
            identity = (item.get("video"), item.get("frame_number"), str(item.get("class_name", "")).casefold(),
                        str(item.get("track_id") or ""), tuple(item.get("bbox") or []))
            unique.setdefault(identity, item)
        detections = list(unique.values())
        detections = [item for item in detections if start <= _num(item.get("timestamp_seconds"), -1) <= end]
        relevant_labels = self.RELEVANT_CLASSES[category_key]
        relevant = [item for item in detections if str(item.get("class_name", item.get("class", "unknown"))).casefold() in relevant_labels]

        classifier_windows = [self._normalise_prediction(item, category_key) for item in (temporal_predictions or [])]
        classifier_windows = [item for item in classifier_windows if item["start"] is not None and item["end"] is not None
                              and item["end"] > item["start"] and item["probability"] is not None
                              and item["start"] < end and item["end"] > start]
        weights = self._weights(category_key)
        bin_sec = max(.1, _num(self.config.get("TEMPORAL_EVENT_BIN_SEC"), 1.0))
        bin_count = max(1, int(math.ceil((end - start) / bin_sec)))
        min_classifier = _num(self.config.get("TEMPORAL_EVENT_MIN_CLASSIFIER_PROBABILITY"), .5)
        min_score = _num(self.config.get("TEMPORAL_EVENT_MIN_SCORE"), .55)
        min_bins = max(1, int(_num(self.config.get("TEMPORAL_EVENT_MIN_SUPPORTED_BINS"), 1)))
        max_gap = max(0.0, _num(self.config.get("TEMPORAL_EVENT_MAX_TRACK_GAP_SEC"), 2.0))
        track_groups = self._track_groups(relevant, max_gap)
        bins = []
        for index in range(bin_count):
            bin_start = start + index * bin_sec
            bin_end = min(end, bin_start + bin_sec)
            windows = [item for item in classifier_windows if item["start"] < bin_end and item["end"] > bin_start]
            probabilities = [item["probability"] for item in windows if item["probability"] is not None]
            classifier_probability = _mean(probabilities)
            neighboring = [item for item in classifier_windows
                           if item["start"] < min(end, bin_end + bin_sec)
                           and item["end"] > max(start, bin_start - bin_sec)]
            consistency = (_mean(1.0 if item["label"].casefold() == category_key else
                                 (1.0 if item["probability"] >= min_classifier else 0.0)
                                 for item in neighboring) if neighboring else None)
            classifier_score = (classifier_probability * (.5 + .5 * consistency)
                                if classifier_probability is not None and consistency is not None
                                else classifier_probability)
            bin_dets = [item for item in relevant if bin_start <= item["timestamp_seconds"] < bin_end or (index == bin_count - 1 and item["timestamp_seconds"] == end)]
            yolo_score = _mean(item.get("confidence") for item in bin_dets)
            if yolo_score is not None:
                persistence_target = max(1, int(_num(self.config.get("TEMPORAL_EVENT_YOLO_PERSISTENCE_OBSERVATIONS"), 2)))
                yolo_score *= min(1.0, len(bin_dets) / persistence_target)
            ids = {str(item["track_id"]) for item in bin_dets if item.get("track_id") not in (None, "")}
            tracks = [item for item in track_groups if item["track_id"] in ids]
            track_score = _mean(item["continuity_score"] if item["detection_count"] > 1 else 0.0 for item in tracks) if tracks else None
            local_track_gaps = []
            local_durations = []
            for track_id in ids:
                track_rows = sorted((item for item in bin_dets if str(item.get("track_id")) == track_id),
                                    key=lambda item: item["timestamp_seconds"])
                local_times = [item["timestamp_seconds"] for item in track_rows]
                local_durations.append(max(0.0, local_times[-1] - local_times[0]) if local_times else 0.0)
                local_track_gaps.extend(right - left for left, right in zip(local_times, local_times[1:]))
            additional = additional_evidence or {}
            modalities = {"classifier": classifier_score, "yolo": yolo_score, "tracking": track_score}
            for modality in ("action", "audio"):
                raw_values = additional.get(modality)
                values = raw_values if isinstance(raw_values, list) else [raw_values]
                valid_scores = []
                for value in values:
                    if not isinstance(value, dict) or value.get("validated") is not True:
                        continue
                    evidence_start = _num(value.get("start_timestamp_sec"))
                    evidence_end = _num(value.get("end_timestamp_sec"))
                    if evidence_start is not None and evidence_end is not None and not (evidence_start < bin_end and evidence_end > bin_start):
                        continue
                    score = _num(value.get("score"))
                    if score is not None and 0 <= score <= 1:
                        valid_scores.append(score)
                if valid_scores:
                    modalities[modality] = max(valid_scores)
            available_weights = {key: weight for key, weight in weights.items() if modalities.get(key) is not None and weight > 0}
            total_weight = sum(available_weights.values())
            score = (sum(float(modalities[key]) * weight for key, weight in available_weights.items()) / total_weight) if total_weight else None
            classifier_supported = classifier_score is not None and classifier_score >= min_classifier
            bins.append({"start_sec": bin_start, "end_sec": bin_end, "score": score,
                "modalities": modalities, "classifier_probability": classifier_probability,
                "classifier_window_consistency": consistency,
                "available_modalities": sorted(available_weights),
                "supporting_detection_count": len(bin_dets), "supporting_track_ids": sorted(ids),
                "mean_detection_confidence": _mean(item.get("confidence") for item in bin_dets),
                "max_detection_confidence": max((item["confidence"] for item in bin_dets if item.get("confidence") is not None), default=None),
                "observed_frame_count": len({item.get("frame_number") for item in bin_dets if item.get("frame_number") is not None}),
                "tracking_duration_sec": sum(local_durations),
                "max_detection_gap_sec": max(local_track_gaps, default=0.0),
                "tracking_continuity_score": track_score,
                "track_fragmentation_count": sum(any(gap > max_gap for gap in group.get("detection_gaps_sec", [])) for group in tracks),
                "classifier_supported": classifier_supported,
                "supported": bool(score is not None and score >= min_score and classifier_supported)})

        supported = [item for item in bins if item["supported"]]
        refine = len(supported) >= min_bins
        boundary_tolerance = max(0.0, _num(self.config.get("TEMPORAL_EVENT_BOUNDARY_TOLERANCE_SEC"), 0.0))
        refined_start = max(start, supported[0]["start_sec"] - boundary_tolerance) if refine else start
        refined_end = min(end, supported[-1]["end_sec"] + boundary_tolerance) if refine else end
        # Avoid pretending the bin edge is more precise than model and evidence resolution.
        resolution = max(bin_sec, _num((video_info or {}).get("classifier_window_seconds"), 0.0) or 0.0)
        refined_start = max(start, math.floor(refined_start / resolution) * resolution) if refine else start
        refined_end = min(end, math.ceil(refined_end / resolution) * resolution) if refine else end
        scoring = [item["score"] for item in (supported if refine else bins) if item["score"] is not None]
        boundary = [supported[0]["score"], supported[-1]["score"]] if refine else []
        probabilities = classifier_probabilities or {}
        aggregate = {name: _mean(item["probabilities"].get(name) for item in classifier_windows) for name in ("Assault", "Robbery", "Shooting", "Normal")}
        aggregate = {name: value for name, value in aggregate.items() if value is not None}
        if probabilities:
            aggregate = {**aggregate, **probabilities}
        groups = self._track_groups(detections, max_gap)
        detection_count = sum(item["supporting_detection_count"] for item in (supported if refine else bins))
        track_ids = sorted({tid for item in (supported if refine else bins) for tid in item["supporting_track_ids"]})
        refinement = {
            "original_candidate_start_sec": start, "original_candidate_end_sec": end,
            "refined_start_sec": refined_start, "refined_end_sec": refined_end,
            "refinement_status": "refined" if refine and (refined_start > start or refined_end < end) else ("supported_unmodified" if refine else "insufficient_evidence_preserved"),
            "refinement_method": "classifier_primary_weighted_temporal_bins" if refine else "original_classifier_candidate_preserved",
            "evidence_score": _mean(scoring), "supporting_detection_count": detection_count,
            "supporting_track_ids": track_ids, "classifier_probabilities": aggregate,
            "boundary_confidence": {"start": boundary[0] if boundary else None, "end": boundary[1] if boundary else None},
            "verification_status": "unverified_candidate", "bin_seconds": bin_sec,
            "boundary_tolerance_sec": boundary_tolerance,
            "score_bins": bins, "track_groups": groups,
            "available_modalities": sorted({modality for item in bins for modality in item["available_modalities"]}),
            "configured_weights": weights, "weights_calibration_status": "uncalibrated_configuration",
            "timestamp_source": "presentation_timestamp" if pts_by_frame else "csv_timestamp_or_frame_fps_fallback",
            "csv_timing_validation": csv_timing,
            "tracking_status_record": tracking_status or artifact_manifest or {},
            "tracking_csv_path": str(resolved_csv), "tracking_csv_sha256": sha256_file(resolved_csv),
            "candidate_interval": {"start_sec": start, "end_sec": end},
            "refined_interval": {"start_sec": refined_start, "end_sec": refined_end},
            "evidence_clip_interval": {"start_sec": refined_start, "end_sec": refined_end,
                "context_before_sec": max(0.0, _num(self.config.get("EVENT_CONTEXT_BEFORE_SEC"), 2.0)),
                "context_after_sec": max(0.0, _num(self.config.get("EVENT_CONTEXT_AFTER_SEC"), 2.0))},
            "limitations": ["YOLO and tracking evidence support review but do not prove a crime.",
                "Weights are uncalibrated; timestamp accuracy is not established without verified reference intervals."]
        }
        return {"refinement": refinement, "detections": detections, "csv_path": resolved_csv,
                "annotated_video_path": resolved_video, "presentation_timestamps": pts_by_frame}

    def create_event(self, *, original_path, output_root, start_sec, end_sec,
                     category="Shooting", temporal_predictions=None,
                     classifier_probabilities=None, classifier_confidence=None,
                     video_info=None, yolo_detections=None, additional_evidence=None,
                     source_video_name=None, context_before_sec=2.0, context_after_sec=2.0,
                     verification_status="unverified_candidate"):
        category_key = str(category).casefold()
        _, csv_path, _ = self.evidence.resolve_artifacts(original_path, category, source_video_name)
        initial_info = dict(video_info or {})
        pts = self.presentation_timestamps(original_path)
        if pts:
            initial_info["presentation_timestamps"] = pts
        manifest_record, selected_csv, _ = self.evidence.resolve_artifacts(original_path, category, source_video_name)
        source_metadata = self.evidence._source_verified(Path(original_path).resolve(), category_key, source_video_name)
        frame_offset = 0
        if source_metadata.get("verified"):
            from app.services.evidence_catalog_service import EvidenceCatalogService
            source_info = self.evidence._metadata(Path(original_path).resolve())
            metadata = {**initial_info, **source_info, "total_frames": source_info["frames"],
                "fps": source_info["fps"], "duration_sec": source_info["frames"] / source_info["fps"]}
            checked = EvidenceCatalogService(self.config).match_tracking_csv(category_key,
                Path(source_video_name or Path(original_path).name).name, str(Path(original_path).resolve()),
                metadata, csv_path_override=selected_csv)
            if checked.get("status") != "verified_match":
                raise ValueError(checked.get("reason") or "Tracking CSV does not align with the original video.")
            frame_offset = int(checked.get("frame_offset", 0))
        association = self.associate(original_path=original_path, category=category,
            start_sec=start_sec, end_sec=end_sec, temporal_predictions=temporal_predictions,
            classifier_probabilities=classifier_probabilities, video_info=initial_info,
            yolo_detections=yolo_detections, additional_evidence=additional_evidence,
            csv_path=selected_csv, source_video_name=source_video_name,
            presentation_timestamps=pts, frame_offset=frame_offset,
            tracking_status=manifest_record)
        refined = association["refinement"]
        event = self.evidence.create_event(original_path=original_path, output_root=output_root,
            start_sec=refined["refined_start_sec"], end_sec=refined["refined_end_sec"],
            category=category, source_video_name=source_video_name, video_info=initial_info,
            confidence=classifier_confidence, verification_status=verification_status,
            classifier_metadata={"probabilities": classifier_probabilities or {},
                                 "temporal_windows": temporal_predictions or []},
            context_before_sec=context_before_sec, context_after_sec=context_after_sec)
        event.update({key: value for key, value in refined.items() if key not in {"track_groups", "score_bins"}})
        event["temporal_evidence"] = refined
        event["source_video_tracking_manifest"] = {
            "category": self.CATEGORIES[category_key],
            "source_video_name": Path(source_video_name or Path(original_path).name).name,
            "source_video_sha256": event.get("original_video_sha256"),
            "status_record": refined.get("tracking_status_record"),
            "resolved_csv_path": refined.get("tracking_csv_path"),
            "resolved_csv_sha256": refined.get("tracking_csv_sha256"),
            "annotated_video_alignment": event.get("tracked_video_alignment"),
        }
        event["candidate_start_timestamp_sec"] = refined["original_candidate_start_sec"]
        event["candidate_end_timestamp_sec"] = refined["original_candidate_end_sec"]
        event["refined_start_timestamp_sec"] = refined["refined_start_sec"]
        event["refined_end_timestamp_sec"] = refined["refined_end_sec"]
        event["evidence_clip_interval"] = {"start_sec": event["context_start_sec"], "end_sec": event["context_end_sec"]}
        json_path = Path(event["artifacts"]["json"])
        event["artifact_hashes"].pop("json", None)
        integrity_path = json_path.with_name("event.integrity.json")
        event["artifacts"]["integrity_manifest"] = str(integrity_path)
        json_path.write_text(json.dumps(event, allow_nan=False, indent=2), encoding="utf-8")
        integrity = {"event_json_path": str(json_path), "event_json_sha256": sha256_file(json_path),
                     "hash_scope": "exact event.json bytes", "source_video_sha256": event.get("original_video_sha256"),
                     "tracking_csv_sha256": refined.get("tracking_csv_sha256")}
        integrity_path.write_text(json.dumps(integrity, allow_nan=False, indent=2), encoding="utf-8")
        return event


def _mean(values):
    numbers = [float(value) for value in values if value is not None and _num(value) is not None]
    return sum(numbers) / len(numbers) if numbers else None
