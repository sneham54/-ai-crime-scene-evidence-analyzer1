"""Streaming inference and clip-to-event extraction for the experimental model."""
import time

import torch

from .config import CLASSES, PreprocessConfig
from .event_extraction import extract_event_keyframes, merge_prediction_events
from .preprocessing import iter_video_clips, probe_video


class VideoInference:
    def __init__(self, model, preprocess: PreprocessConfig | None = None, device=None):
        self.preprocess = preprocess or PreprocessConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, video_path, video_id=None, confidence_threshold=0.55, event_gap_sec=2.0,
                keyframe_dir=None):
        path = str(video_path)
        metadata = probe_video(path)
        predictions = []
        latencies = []
        for clip in iter_video_clips(path, self.preprocess):
            started = time.perf_counter()
            logits = self.model(clip.tensor.unsqueeze(0).to(self.device))
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
            latencies.append(time.perf_counter() - started)
            index = max(range(len(probs)), key=probs.__getitem__)
            center = len(clip.frame_indices) // 2
            predictions.append({"predicted_class": CLASSES[index], "confidence": float(probs[index]),
                                "probabilities": dict(zip(CLASSES, map(float, probs))),
                                "start_timestamp_sec": clip.start_timestamp_sec,
                                "end_timestamp_sec": clip.end_timestamp_sec,
                                "representative_frame_index": clip.frame_indices[center],
                                "representative_timestamp_sec": clip.timestamps_sec[center],
                "frame_indices": clip.frame_indices,
                "timestamp_mapping_status": clip.timestamp_mapping_status,
                "timestamps_verified": clip.timestamps_verified})
        aggregate = {name: sum(row["probabilities"][name] for row in predictions) / len(predictions)
                     for name in CLASSES}
        label = max(aggregate, key=aggregate.get)
        events = merge_prediction_events(predictions, video_id or path, path, confidence_threshold, event_gap_sec)
        events = extract_event_keyframes(path, events, predictions, keyframe_dir)
        processed_duration = predictions[-1]["end_timestamp_sec"]
        requested_duration = metadata["duration_sec"]
        if self.preprocess.max_duration_sec is not None and requested_duration is not None:
            requested_duration = min(requested_duration, self.preprocess.max_duration_sec)
        return {"video_id": video_id, "video_metadata": metadata, "classification": {
                    "label": label, "confidence": aggregate[label], "probabilities": aggregate,
                    "classification_mode": "experimental_densenet3d_attention_biconvlstm_four_class",
                    "verification_status": "model_prediction_requires_human_review"},
                "clip_predictions": predictions, "events": events,
                "sampling_coverage": {
                    "processed_duration_sec": processed_duration,
                    "window_count": len(predictions),
                    "sampled_frame_count": len({index for row in predictions for index in row["frame_indices"]}),
                    "timestamps_available": metadata["fps"] is not None or metadata["timestamp_mapping_status"] == "verified_vfr_presentation_timestamps",
                    "timestamp_mapping_status": metadata["timestamp_mapping_status"],
                    "timestamp_basis": metadata["timestamp_basis"],
                    "timestamps_verified": metadata["timestamps_verified"],
                    "timestamp_warning": metadata["timestamp_warning"],
                    "covered_entire_video": requested_duration is not None and processed_duration >= requested_duration - 0.001},
                "inference_seconds_per_clip": sum(latencies) / max(1, len(latencies)),
                "processing_status": "completed"}
