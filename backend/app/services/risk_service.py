"""Explainable, rule-based investigative triage score (not crime probability).

The point scale is a transparent operational rubric, not a learned or calibrated
model. Confidence values describe individual model outputs; they are not model
accuracy. Missing inputs produce completeness warnings and never add points.
"""
from __future__ import annotations

import math
import os


ACTIVITY_SEVERITY = {"assault": 35.0, "robbery": 40.0, "shooting": 50.0}
OBJECT_SEVERITY = {"weapon": 18.0, "blood": 12.0}
DEFAULT_THRESHOLDS = {"moderate": 25, "high": 50, "critical": 75}
WEAPON_LABELS = {"weapon", "gun", "pistol", "rifle", "knife", "handgun", "shotgun", "revolver", "firearm", "blade", "dagger", "machete"}


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _confidence(value):
    number = _number(value)
    if number is None:
        return None
    # Older clients may send percentages; normalize but keep the raw value out
    # of the explanation so it is never mistaken for accuracy.
    return max(0.0, min(1.0, number / 100 if number > 1 else number))


def _classify(detection):
    label = str(detection.get("class") or detection.get("class_name") or "").strip().casefold()
    category = str(detection.get("category") or "").strip().casefold()
    tokens = set(label.replace("/", " ").replace("_", " ").replace("-", " ").split())
    if category == "weapons" or tokens.intersection(WEAPON_LABELS):
        return "weapon"
    if category in {"blood / biological", "blood", "biological"} or "blood" in label:
        return "blood"
    if category == "persons" or "person" in label:
        return "person"
    if category == "vehicles" or any(word in label for word in ("vehicle", "car", "truck", "bus")):
        return "vehicle"
    return "other"


def _activity_name(value):
    label = str(value or "").strip().casefold()
    return label if label in ACTIVITY_SEVERITY else None


class RiskService:
    """Score distinct evidence signals and return their provenance and rationale."""

    def __init__(self, weights=None, thresholds=None):
        self.severity = dict(ACTIVITY_SEVERITY)
        self.object_severity = dict(OBJECT_SEVERITY)
        self.bonuses = {
            "consistent_tracking": 5.0,
            "independent_corroboration": 10.0,
            "additional_event": 3.0,
            "duration_max": 5.0,
        }
        for name in self.severity:
            self.severity[name] = self._setting(f"RISK_ACTIVITY_{name.upper()}_SEVERITY", self.severity[name])
        for name in self.object_severity:
            self.object_severity[name] = self._setting(f"RISK_OBJECT_{name.upper()}_SEVERITY", self.object_severity[name])
        for name in self.bonuses:
            self.bonuses[name] = self._setting(f"RISK_{name.upper()}_POINTS", self.bonuses[name])
        self.duration_cap_sec = self._setting("RISK_EVENT_DURATION_CAP_SEC", 10.0, minimum=0.1)
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        self.thresholds.update(thresholds or {})
        for name in self.thresholds:
            env_value = self._setting(f"RISK_THRESHOLD_{name.upper()}", self.thresholds[name], minimum=0, maximum=100)
            self.thresholds[name] = int(env_value)
        self.thresholds = {key: max(0, min(100, int(value))) for key, value in self.thresholds.items()}
        if not self.thresholds["moderate"] < self.thresholds["high"] < self.thresholds["critical"]:
            self.thresholds = dict(DEFAULT_THRESHOLDS)
        for group in (self.severity, self.object_severity, self.bonuses):
            for key in group:
                group[key] = max(0.0, min(100.0, float(group[key])))
        for group_name, supplied in (weights or {}).items():
            if group_name in self.severity:
                self.severity[group_name] = max(0.0, min(100.0, float(supplied)))
            elif group_name in self.object_severity:
                self.object_severity[group_name] = max(0.0, min(100.0, float(supplied)))

    @staticmethod
    def _setting(name, fallback, minimum=0.0, maximum=100.0):
        value = _number(os.getenv(name))
        if value is None:
            return fallback
        return max(minimum, min(maximum, value))

    def _level(self, score):
        if score >= self.thresholds["critical"]:
            return "Critical"
        if score >= self.thresholds["high"]:
            return "High"
        if score >= self.thresholds["moderate"]:
            return "Moderate"
        return "Low"

    @staticmethod
    def _event_key(event):
        category = _activity_name(event.get("category") or event.get("label") or event.get("class"))
        start = _number(event.get("start_timestamp_sec", event.get("start_sec")))
        end = _number(event.get("end_timestamp_sec", event.get("end_sec")))
        if category and start is not None and end is not None:
            return category, round(start, 2), round(max(start, end), 2)
        # Without temporal identity, a repeated candidate of the same class is
        # not assumed to be a separate event.
        return (category, "unbounded") if category else None

    def calculate(self, detections, context=None):
        detections = [item for item in (detections or []) if isinstance(item, dict)]
        context = context if isinstance(context, dict) else {}
        classification = context.get("classification") or context.get("event_classification") or {}
        tracker = context.get("tracking") if isinstance(context.get("tracking"), dict) else {}
        match = context.get("tracking_data_match") if isinstance(context.get("tracking_data_match"), dict) else {}
        verified_tracking = (bool(tracker.get("used")) or
                             match.get("status") in {"EXACT_MATCH", "METADATA_MATCH"})
        if not isinstance(classification, dict):
            classification = {}
        temporal_events = context.get("activity_events") or classification.get("activity_events") or []
        if not isinstance(temporal_events, list):
            temporal_events = []
        unique_events = {}
        for event in temporal_events:
            if not isinstance(event, dict):
                continue
            key = self._event_key(event)
            category = _activity_name(event.get("category") or event.get("label") or event.get("class"))
            if key is None or category is None:
                continue
            confidence = _confidence(event.get("confidence"))
            previous = unique_events.get(key)
            if previous is None or (confidence or 0) > (previous["confidence"] or 0):
                unique_events[key] = {**event, "category": category, "confidence": confidence}

        activity_sources = list(unique_events.values())
        overall_activity = _activity_name(classification.get("label"))
        if not activity_sources and overall_activity:
            confidence = _confidence(classification.get("confidence"))
            if confidence is not None:
                activity_sources = [{"category": overall_activity, "confidence": confidence,
                                     "start_timestamp_sec": None, "end_timestamp_sec": None}]
        primary_activity = max(activity_sources,
                               key=lambda event: self.severity[event["category"]] * (event["confidence"] or 0),
                               default=None)
        activity_points = (self.severity[primary_activity["category"]] * (primary_activity["confidence"] or 0)
                           if primary_activity else 0.0)
        distinct_event_points = min(self.bonuses["additional_event"] * max(0, len(unique_events) - 1),
                                    self.bonuses["additional_event"] * 3)
        durations = []
        for event in unique_events.values():
            start = _number(event.get("start_timestamp_sec", event.get("start_sec")))
            end = _number(event.get("end_timestamp_sec", event.get("end_sec")))
            if start is not None and end is not None and end >= start:
                durations.append(end - start)
        duration_points = (self.bonuses["duration_max"] * min(1.0, max(durations, default=0.0) / self.duration_cap_sec)
                          if durations else 0.0)

        best_object_confidence = {key: 0.0 for key in self.object_severity}
        object_labels = {key: set() for key in self.object_severity}
        track_frames = {}
        detection_timestamps = []
        for detection in detections:
            kind = _classify(detection)
            confidence = _confidence(detection.get("confidence")) or 0.0
            if kind in best_object_confidence:
                best_object_confidence[kind] = max(best_object_confidence[kind], confidence)
                object_labels[kind].add(str(detection.get("class") or detection.get("class_name") or kind))
            track_id = detection.get("track_id")
            if track_id is not None:
                frame = detection.get("frame_number", detection.get("frame"))
                timestamp = _number(detection.get("timestamp_sec", detection.get("timestamp_seconds")))
                track_frames.setdefault(str(track_id), set()).add(frame if frame is not None else timestamp)
            timestamp = _number(detection.get("timestamp_sec", detection.get("timestamp_seconds")))
            if timestamp is not None and kind in self.object_severity:
                detection_timestamps.append((timestamp, kind))
        object_points_by_kind = {key: self.object_severity[key] * best_object_confidence[key]
                                 for key in self.object_severity}
        object_points = sum(object_points_by_kind.values())
        consistent_track_ids = sorted(track_id for track_id, observations in track_frames.items()
                                      if len(observations) >= 2)
        tracking_points = self.bonuses["consistent_tracking"] if consistent_track_ids and verified_tracking else 0.0

        corroborated = False
        for event in unique_events.values():
            start = _number(event.get("start_timestamp_sec", event.get("start_sec")))
            end = _number(event.get("end_timestamp_sec", event.get("end_sec")))
            if start is None or end is None:
                continue
            if any(start <= timestamp <= end for timestamp, _kind in detection_timestamps):
                corroborated = True
                break
        corroboration_points = self.bonuses["independent_corroboration"] if corroborated else 0.0

        raw_score = activity_points + object_points + distinct_event_points + duration_points + tracking_points + corroboration_points
        score = int(round(max(0.0, min(100.0, raw_score))))
        level = self._level(score)
        activity_confidence = max((_confidence(item.get("confidence")) or 0.0 for item in activity_sources), default=None)
        detection_confidence = max((_confidence(item.get("confidence")) or 0.0 for item in detections), default=None)
        scene_changes = context.get("scene_changes") if isinstance(context.get("scene_changes"), list) else []
        is_video = bool(context.get("video_info") is not None or "activity_events" in context or "tracking" in context)
        tracking_available = verified_tracking
        sources = {"activity_classifier": bool(_activity_name(classification.get("label")) and _confidence(classification.get("confidence")) is not None) or bool(temporal_events),
                   "visual_detections": bool(detections or context.get("detections") is not None),
                   "tracking": tracking_available,
                   "timestamped_activity_events": bool(temporal_events),
                   "scene_change": "scene_changes" in context}
        applicable = ["visual_detections"]
        if is_video:
            applicable = list(sources)
        completeness = sum(sources[key] for key in applicable) / len(applicable) if applicable else 0.0

        factors = []
        if primary_activity:
            factors.append({"name": f"{primary_activity['category'].title()} classifier signal",
                            "points": round(activity_points, 2), "source": "video_classifier",
                            "model_confidence": primary_activity["confidence"],
                            "note": "Confidence is the model's output strength, not measured accuracy."})
        for kind, points in object_points_by_kind.items():
            if points > 0:
                factors.append({"name": f"{kind.title()} detections",
                                "points": round(points, 2), "source": "visual_detector",
                                "model_confidence": best_object_confidence[kind],
                                "detected_objects": sorted(object_labels[kind]),
                                "note": "Strongest confidence per object class; repeated boxes do not stack."})
        if distinct_event_points:
            factors.append({"name": "Additional distinct activity events", "points": round(distinct_event_points, 2),
                            "source": "temporal_classifier", "event_count": len(unique_events)})
        if duration_points:
            factors.append({"name": "Observed activity duration", "points": round(duration_points, 2),
                            "source": "temporal_classifier", "longest_event_sec": round(max(durations), 2)})
        if tracking_points:
            factors.append({"name": "Consistent tracked observations", "points": round(tracking_points, 2),
                            "source": tracker.get("method", "tracking_ids"), "tracking_ids": consistent_track_ids,
                            "note": "Track persistence supports continuity; it does not establish an offense."})
        if corroboration_points:
            factors.append({"name": "Temporal cross-source corroboration", "points": round(corroboration_points, 2),
                            "source": "classifier_and_detections",
                            "note": "A timestamped activity candidate overlaps a relevant object detection."})

        warnings = []
        if not primary_activity and not object_points:
            warnings.append("No scored activity or relevant-object evidence is available; zero risk points means insufficient evidence, not proof of safety.")
        if not activity_sources:
            warnings.append("No usable Assault, Robbery, or Shooting classifier confidence was supplied.")
        if not detections:
            warnings.append("No visual detections were supplied; missing detections do not add risk points.")
        if is_video and not tracking_available:
            warnings.append("Tracking evidence is unavailable or not included.")
        if is_video and temporal_events and context.get("verification_status") == "unverified":
            warnings.append("Timestamped activity intervals are unverified classifier candidates.")
        classification_status = str(classification.get("classification_status") or "")
        if classification_status:
            warnings.append(f"Video classifier status: {classification_status}; predictions are provisional and have not been independently validated.")
        if match and match.get("status") not in {"EXACT_MATCH", "METADATA_MATCH"}:
            warnings.append(f"Saved tracking match status: {match.get('status')}; tracking provenance is not verified.")
        if scene_changes:
            factors.append({"name": "Scene-change context", "points": 0,
                            "source": "scene_change", "candidate_count": len(scene_changes),
                            "note": "Scene changes are context only and do not independently increase risk."})

        tracked_observations = sum(1 for item in detections if item.get("track_id") is not None)
        return {
            "overall_risk": score,
            "impact_score": score,
            "risk_level": level,
            "risk_label": f"{level} Risk",
            "threat_level": level,
            "severity": level,
            "weapon_count": len(object_labels["weapon"]),
            "person_count": len({str(item.get("class") or item.get("class_name") or "person").casefold()
                                 for item in detections if _classify(item) == "person"}),
            "has_blood": best_object_confidence["blood"] > 0,
            "has_person": any(_classify(item) == "person" for item in detections),
            "contributing_factors": factors,
            "reasons": [f"{factor['name']}: +{factor['points']:.1f} points" for factor in factors if factor["points"] > 0],
            "detected_objects": sorted({str(item.get("class") or item.get("class_name") or "Unknown") for item in detections}),
            "tracking_information": {"method": tracker.get("method"), "match_status": match.get("status"),
                                     "tracked_observation_count": tracked_observations,
                                     "consistent_track_ids": consistent_track_ids,
                                     "unique_track_id_count": len(track_frames)},
            "timestamped_events": temporal_events,
            "confidence_indicators": {"classifier_output_confidence": activity_confidence,
                                       "detector_output_confidence": detection_confidence,
                                       "confidence_is_accuracy": False},
            "evidence_completeness": {"available_sources": sources, "coverage": round(completeness, 2),
                                      "applicable_source_count": len(applicable)},
            "evidence_reliability": {"tracking_source_match": match.get("status", "not_provided"),
                                     "model_outputs": "unverified_model_output" if detections or classification else "unavailable"},
            "warnings": warnings,
            "score_explanation": "Uncalibrated investigative prioritization rubric. Activity severity is scaled by output confidence; relevant object classes contribute their strongest detection only; bounded bonuses represent distinct temporal events, persistent tracks, and timestamp overlap. Missing inputs add no points.",
            "score_disclaimer": "Investigative prioritization indicator only. It is not proof of criminal activity, a crime probability, or model accuracy.",
            "scoring_method": "triage_rubric_v2_unvalidated",
            "thresholds": dict(self.thresholds),
            "threshold_rationale": "Low < moderate: no meaningful combined signal; Moderate: a single modest signal; High: a stronger activity signal or corroborated signals; Critical: multiple strong signals. Thresholds are operational review bands and require validation before deployment decisions.",
            "score_components": {"activity": round(activity_points, 2),
                                 "objects": round(object_points, 2),
                                 "distinct_events": round(distinct_event_points, 2),
                                 "duration": round(duration_points, 2),
                                 "tracking_consistency": round(tracking_points, 2),
                                 "cross_source_corroboration": round(corroboration_points, 2)},
            "scene_change_candidate_count": len(scene_changes),
        }
