"""Adapter for optional use through the existing Flask video service contract."""
from collections import OrderedDict
from pathlib import Path
import threading

import torch

from .classifier import load_checkpoint
from .config import PreprocessConfig
from .inference import VideoInference


class DenseNetVideoClassifier:
    """Implements predict_path/predict_windows_path expected by VideoService.

    Predictions are cached by resolved source path so the existing service's
    primary and temporal calls do not decode or infer on the same video twice.
    """
    _model_cache = {}
    _model_lock = threading.RLock()

    def __init__(self, checkpoint_path, min_confidence=0.0, event_confidence_threshold=0.55,
                 event_gap_sec=2.0):
        self.checkpoint_path = Path(checkpoint_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_key = (str(self.checkpoint_path.resolve()), str(device))
        with self._model_lock:
            cached_model = self._model_cache.get(cache_key)
            if cached_model is None:
                model, checkpoint = load_checkpoint(self.checkpoint_path, device)
                checkpoint = {"preprocess_config": checkpoint.get("preprocess_config", {})}
                self._model_cache[cache_key] = (model, checkpoint)
            else:
                model, checkpoint = cached_model
        raw = checkpoint.get("preprocess_config", {})
        raw["mean"] = tuple(raw.get("mean", (0.485, 0.456, 0.406)))
        raw["std"] = tuple(raw.get("std", (0.229, 0.224, 0.225)))
        self.inference = VideoInference(model, PreprocessConfig(**raw), device)
        self.min_confidence = float(min_confidence)
        self.event_confidence_threshold = float(event_confidence_threshold)
        self.event_gap_sec = float(event_gap_sec)
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._evidence_context = None

    def set_evidence_context(self, analysis_id, evidence_dir):
        self._evidence_context = (str(analysis_id), Path(evidence_dir))

    def _result(self, video_path, max_duration_sec=None, confidence_threshold=None, event_gap_sec=None):
        key = str(Path(video_path).resolve())
        confidence_threshold = self.event_confidence_threshold if confidence_threshold is None else confidence_threshold
        event_gap_sec = self.event_gap_sec if event_gap_sec is None else float(event_gap_sec)
        key = (key, max_duration_sec, float(confidence_threshold), event_gap_sec)
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                if max_duration_sec is not None:
                    from dataclasses import replace
                    preprocess = replace(self.inference.preprocess, max_duration_sec=float(max_duration_sec))
                    runner = VideoInference(self.inference.model, preprocess, self.inference.device)
                else:
                    runner = self.inference
                analysis_id, keyframe_dir = self._evidence_context or (None, None)
                cached = runner.predict(key[0], video_id=analysis_id, confidence_threshold=confidence_threshold,
                                        event_gap_sec=event_gap_sec,
                                        keyframe_dir=keyframe_dir)
                self._cache[key] = cached
                while len(self._cache) > 2:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(key)
            return cached

    @staticmethod
    def _to_contract(result):
        classification = result["classification"]
        predictions = result["clip_predictions"]
        events = result["events"]
        windows = [{"label": row["predicted_class"], "confidence": row["confidence"],
                    "probabilities": row["probabilities"],
                    "start_timestamp_sec": row["start_timestamp_sec"],
                    "end_timestamp_sec": row["end_timestamp_sec"],
                    "representative_frame_index": row["representative_frame_index"],
                    "representative_timestamp_sec": row["representative_timestamp_sec"],
                    "timestamp_mapping_status": row.get("timestamp_mapping_status"),
                    "timestamps_verified": row.get("timestamps_verified", False)}
                   for row in predictions]
        activity_events = [{**event,
                            "keyframes": [dict(keyframe) for keyframe in event.get("keyframes", [])],
                            "source_video": Path(event["source_video"]).name,
                            "category": event["predicted_class"],
                            "start_timestamp_sec": event["start_timestamp"],
                            "end_timestamp_sec": event["end_timestamp"],
                            "window_count": sum(1 for row in predictions
                                if row["predicted_class"] == event["predicted_class"] and
                                event["start_timestamp"] <= row["start_timestamp_sec"] <= event["end_timestamp"]),
                            "verification_status": event.get("verification_status", "pending_review")}
                           for event in events]
        for event in activity_events:
            event_id = event.get("video_id")
            for keyframe in event.get("keyframes", []):
                local_path = keyframe.pop("path", None)
                if event_id and local_path:
                    keyframe["url"] = f"/api/video/evidence/{event_id}/{Path(local_path).name}"
            representative = event.get("representative_frame")
            if event_id and representative:
                event["representative_frame"] = f"/api/video/evidence/{event_id}/{Path(representative).name}"
        metadata = result["video_metadata"]
        return {"label": classification["label"], "top_class": classification["label"],
                "confidence": classification["confidence"], "probabilities": classification["probabilities"],
                "classification_mode": classification["classification_mode"],
                "classification_status": "provisional", "temporal_windows": windows,
                "activity_events": activity_events,
                "sampling_coverage": {"total_frames": metadata["frame_count"], "fps": metadata["fps"],
                    "duration_sec": metadata["duration_sec"],
                    "sampled_frame_count": result["sampling_coverage"]["sampled_frame_count"],
                    "window_count": len(predictions),
                    "covered_entire_video": result["sampling_coverage"]["covered_entire_video"],
                    "timestamps_available": metadata.get("fps") is not None or metadata.get("timestamp_mapping_status") == "verified_vfr_presentation_timestamps",
                    "timestamp_mapping_status": metadata.get("timestamp_mapping_status", "unavailable"),
                    "timestamp_basis": metadata.get("timestamp_basis", "unavailable"),
                    "timestamps_verified": metadata.get("timestamps_verified", False),
                    "timestamp_warning": metadata.get("timestamp_warning")},
                "inference_seconds_per_clip": result["inference_seconds_per_clip"],
                "classifier_note": "Experimental four-class DenseNet-3D, temporal attention and bidirectional ConvLSTM. Predictions and intervals are unverified candidates; model confidence is not calibrated accuracy."}

    def predict_path(self, video_path, max_duration_sec=None):
        result = self._result(video_path, max_duration_sec)
        output = self._to_contract(result)
        if output["confidence"] < self.min_confidence:
            output["label"] = "Uncertain"
        return output

    def predict_windows_path(self, video_path, max_duration_sec=None, confidence_threshold=0.55,
                             overlap_fraction=0.5, primary_category=None, merge_gap_sec=None):
        result = self._result(video_path, max_duration_sec, confidence_threshold, merge_gap_sec)
        return self._to_contract(result)
