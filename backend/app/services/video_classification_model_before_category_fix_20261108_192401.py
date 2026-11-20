"""CPU-compatible loader for the existing ResNet18 + LSTM video checkpoint."""
import json
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models


VIDEO_CLASS_NAMES = ("Assault", "Robbery", "Shooting", "Normal")
IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class VideoModelUnavailableError(FileNotFoundError):
    """Required video model artifact is absent or cannot be configured."""


class CNNLSTM(nn.Module):
    """Architecture used to train backend/models/video_classifier.pth."""

    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=None)
        feature_size = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.cnn = backbone
        self.lstm = nn.LSTM(feature_size, 256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, video):
        batch, frames, channels, height, width = video.shape
        features = self.cnn(video.reshape(batch * frames, channels, height, width))
        features = features.reshape(batch, frames, -1)
        sequence, _ = self.lstm(features)
        return self.classifier(self.dropout(sequence[:, -1, :]))


def decode_video_sequence(video_path, sequence_length=16, image_size=112,
                          max_duration_sec=None, return_metadata=False):
    """Read evenly spaced frames from a duration-bounded clip window."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Uploaded video cannot be decoded")
    frames = []
    sampled_indices = []
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            raise ValueError("Video has no readable frames")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        valid_fps = np.isfinite(fps) and fps > 0
        duration = count / fps if valid_fps else None
        eligible_count = count
        if valid_fps and max_duration_sec is not None:
            max_duration_sec = max(0.0, float(max_duration_sec))
            eligible_count = min(count, int(np.floor(max_duration_sec * fps)) + 1)
        indices = np.linspace(0, max(0, eligible_count - 1), sequence_length).round().astype(int)
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frame = (frame - IMAGE_MEAN) / IMAGE_STD
            frames.append(np.transpose(frame, (2, 0, 1)))
            sampled_indices.append(int(index))
    finally:
        capture.release()
    if not frames:
        raise ValueError("Video contains no decodable frames")
    while len(frames) < sequence_length:
        frames.append(frames[-1].copy())
    sequence = np.stack(frames[:sequence_length]).astype(np.float32)
    if not return_metadata:
        return sequence
    first_index = sampled_indices[0] if sampled_indices else None
    last_index = sampled_indices[-1] if sampled_indices else None
    metadata = {
        "total_frames": count,
        "fps": float(fps) if valid_fps else None,
        "duration_sec": float(duration) if duration is not None else None,
        "sampled_frame_count": len(sampled_indices),
        "first_sample_frame": first_index,
        "last_sample_frame": last_index,
        "first_sample_timestamp_sec": first_index / fps if valid_fps and first_index is not None else None,
        "last_sample_timestamp_sec": last_index / fps if valid_fps and last_index is not None else None,
        "sampling_window_start_sec": 0.0 if valid_fps else None,
        "sampling_window_end_sec": (eligible_count - 1) / fps if valid_fps and eligible_count else None,
        "covered_entire_video": eligible_count == count,
        "timestamps_available": bool(valid_fps),
        "max_duration_sec": max_duration_sec,
    }
    return sequence, metadata


class VideoClassificationModel:
    _cache = {}

    def __new__(cls, checkpoint_path, class_names_path=None, min_confidence=0.0):
        key = (str(Path(checkpoint_path)), str(Path(class_names_path)) if class_names_path else None)
        if key in cls._cache:
            return cls._cache[key]
        instance = super().__new__(cls)
        cls._cache[key] = instance
        return instance

    def __init__(self, checkpoint_path, class_names_path=None, min_confidence=0.0):
        if getattr(self, "_initialized", False):
            return
        self.checkpoint_path = Path(checkpoint_path)
        self.class_names_path = Path(class_names_path) if class_names_path else self.checkpoint_path.with_name("class_names.json")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 16
        self.image_size = 112
        self.class_names = VIDEO_CLASS_NAMES
        self.model = None
        self.min_confidence = float(min_confidence)
        if not self.checkpoint_path.is_file():
            raise VideoModelUnavailableError(
                f"Video classifier checkpoint is missing: {self.checkpoint_path}. "
                "Set VIDEO_MODEL_PATH to a trained compatible checkpoint."
            )
        if not self.class_names_path.is_file():
            raise VideoModelUnavailableError(
                f"Video classifier class mapping is missing: {self.class_names_path}. "
                "Set CLASS_NAMES_PATH to the matching class_names.json."
            )
        try:
            with self.class_names_path.open(encoding="utf-8") as source:
                self.class_names = tuple(json.load(source))
        except (OSError, json.JSONDecodeError, TypeError) as error:
            raise VideoModelUnavailableError(
                f"Video classifier class mapping could not be read: {self.class_names_path}"
            ) from error
        if self.class_names != VIDEO_CLASS_NAMES:
            raise VideoModelUnavailableError(
                "Video classifier must use Assault, Robbery, Shooting, Normal class order."
            )
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise VideoModelUnavailableError(
                f"Video classifier checkpoint could not be safely loaded: {error}"
            ) from error
        required = {"model_state_dict", "classes", "num_frames", "img_size"}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise VideoModelUnavailableError(
                "Video classifier checkpoint is missing model_state_dict, classes, num_frames, or img_size."
            )
        if checkpoint["classes"] != list(self.class_names):
            raise VideoModelUnavailableError("Checkpoint class order does not match class_names.json.")
        try:
            self.sequence_length = int(checkpoint["num_frames"])
            self.image_size = int(checkpoint["img_size"])
        except (TypeError, ValueError) as error:
            raise VideoModelUnavailableError("Checkpoint frame count and image size must be positive integers.") from error
        if self.sequence_length <= 0 or self.image_size <= 0:
            raise VideoModelUnavailableError("Checkpoint frame count and image size must be positive integers.")
        self.model = CNNLSTM(len(self.class_names))
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except (KeyError, RuntimeError, TypeError) as error:
            raise VideoModelUnavailableError(
                f"Video classifier checkpoint does not match the CNNLSTM architecture: {error}"
            ) from error
        self.model.to(self.device).eval()
        self._initialized = True

    def predict_path(self, video_path, max_duration_sec=None):
        sequence, sampling_coverage = decode_video_sequence(
            video_path, self.sequence_length, self.image_size,
            max_duration_sec=max_duration_sec, return_metadata=True,
        )
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        if tuple(logits.shape) != (1, len(self.class_names)):
            raise RuntimeError("Video classifier returned an unexpected class-score shape.")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Video classifier produced non-finite class scores.")
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[index].item())
        top_class = self.class_names[index]
        return {
            "label": top_class if confidence >= self.min_confidence else "Uncertain",
            "top_class": top_class,
            "confidence": round(confidence, 4),
            "probabilities": {name: round(float(probabilities[i].item()), 4) for i, name in enumerate(self.class_names)},
            "classification_mode": "trained_resnet18_lstm",
            "sampling_coverage": sampling_coverage,
            "classifier_note": "Checkpoint loaded and inference completed; accuracy has not been independently validated here. Softmax confidence is not calibrated or a legal conclusion.",
        }

    def predict_windows_path(self, video_path, max_duration_sec=None,
                             confidence_threshold=0.55, overlap_fraction=0.5,
                             primary_category=None):
        """Classify overlapping 16-sampled-frame windows from one sequential decode.

        Frames are sampled at approximately one per second. This temporal mode
        is an inference adaptation of the existing whole-clip checkpoint; its
        event intervals are candidates and have not been independently validated.
        """
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Uploaded video cannot be decoded")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        valid_fps = math.isfinite(fps) and fps > 0
        if total_frames <= 0 or not valid_fps:
            capture.release()
            raise ValueError("Video frame count or FPS is unavailable for temporal classification")
        duration = total_frames / fps
        eligible_count = total_frames
        if max_duration_sec is not None:
            eligible_count = min(total_frames, int(math.floor(max(0.0, float(max_duration_sec)) * fps)) + 1)
        duration = min(duration, eligible_count / fps)
        sample_stride = max(1, int(round(fps)))
        overlap = max(1, min(self.sequence_length - 1, int(round(self.sequence_length * overlap_fraction))))
        sample_count = 0
        last_window_sample_count = 0
        samples = deque(maxlen=self.sequence_length)
        all_predictions = []
        all_probabilities = []

        def infer_window(window_samples):
            stacked = np.stack([entry[1] for entry in window_samples]).astype(np.float32)
            tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logits = self.model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
            values = [float(value) for value in probabilities.detach().cpu().tolist()]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError("Video classifier produced non-finite class scores")
            index = int(np.argmax(values))
            start = float(window_samples[0][0])
            end = min(duration, float(window_samples[-1][0]) + sample_stride / fps)
            result = {"label": self.class_names[index], "confidence": round(values[index], 4),
                      "probabilities": {name: round(values[i], 4) for i, name in enumerate(self.class_names)},
                      "start_timestamp_sec": round(start, 3), "end_timestamp_sec": round(end, 3)}
            all_predictions.append(result)
            all_probabilities.append(values)

        try:
            frame_index = 0
            while frame_index < eligible_count:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % sample_stride == 0:
                    resized = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    normalized = (rgb - IMAGE_MEAN) / IMAGE_STD
                    samples.append((frame_index / fps, np.transpose(normalized, (2, 0, 1))))
                    sample_count += 1
                    if len(samples) == self.sequence_length and (
                        sample_count == self.sequence_length or
                        (sample_count - self.sequence_length) % overlap == 0
                    ):
                        window = list(samples)
                        infer_window(window)
                        last_window_sample_count = sample_count
                frame_index += 1
            if samples and not all_predictions:
                window = list(samples)
                while len(window) < self.sequence_length:
                    window.append(window[-1])
                infer_window(window)
            elif samples and sample_count - last_window_sample_count >= max(1, overlap // 2):
                infer_window(list(samples))
        finally:
            capture.release()
        if not all_predictions:
            raise ValueError("Video contains no decodable frames for temporal classification")

        # Suppress a single-window label fluctuation only when both neighboring
        # windows agree; preserve raw label and confidence for review.
        smoothed = list(all_predictions)
        for index in range(1, len(all_predictions) - 1):
            before, current, after = all_predictions[index - 1:index + 2]
            if before["label"] == after["label"] != current["label"]:
                smoothed_confidence = current["probabilities"][before["label"]]
                smoothed[index] = {**current, "raw_label": current["label"], "label": before["label"],
                                   "confidence": smoothed_confidence, "smoothed": True}
        events = []
        # Score each event class from its own window probability. This allows
        # Assault and Shooting candidates to coexist in one video; a weapon
        # detector is not used as evidence that a shooting occurred.
        event_classes = ("Shooting", "Assault")
        if not math.isfinite(float(confidence_threshold)) or not 0 <= float(confidence_threshold) <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        for category in event_classes:
            for window in smoothed:
                category_confidence = window["probabilities"][category]
                if category_confidence < confidence_threshold:
                    continue
                if events and events[-1]["category"] == category and window["start_timestamp_sec"] <= events[-1]["end_timestamp_sec"] + 1.0:
                    events[-1]["end_timestamp_sec"] = window["end_timestamp_sec"]
                    events[-1]["window_count"] += 1
                    events[-1]["window_confidences"].append(category_confidence)
                else:
                    events.append({"category": category,
                                   "start_timestamp_sec": window["start_timestamp_sec"],
                                   "end_timestamp_sec": window["end_timestamp_sec"],
                                   "window_count": 1, "window_confidences": [category_confidence]})
        events.sort(key=lambda event: (event["start_timestamp_sec"], event["category"]))
        for event in events:
            event["confidence"] = round(sum(event.pop("window_confidences")) / event["window_count"], 4)
            event["verification_status"] = "unverified_candidate"

        mean_probs = np.mean(np.asarray(all_probabilities, dtype=np.float64), axis=0)
        best_index = int(np.argmax(mean_probs))
        best_confidence = float(mean_probs[best_index])
        return {
            "label": self.class_names[best_index] if best_confidence >= self.min_confidence else "Uncertain",
            "top_class": self.class_names[best_index], "confidence": round(best_confidence, 4),
            "probabilities": {name: round(float(mean_probs[i]), 4) for i, name in enumerate(self.class_names)},
            "classification_mode": "trained_resnet18_lstm_overlapping_windows",
            "temporal_windows": smoothed, "activity_events": events,
            "sampling_coverage": {"total_frames": total_frames, "fps": fps,
                                   "duration_sec": round(total_frames / fps, 3),
                                   "sampled_frame_count": math.ceil(eligible_count / sample_stride),
                                   "first_sample_frame": 0,
                                   "last_sample_frame": max(0, (math.ceil(eligible_count / sample_stride) - 1) * sample_stride),
                                   "first_sample_timestamp_sec": 0.0,
                                   "last_sample_timestamp_sec": round(max(0, (math.ceil(eligible_count / sample_stride) - 1) * sample_stride) / fps, 3),
                                   "window_count": len(all_predictions),
                                   "window_seconds": round(self.sequence_length * sample_stride / fps, 3),
                                   "overlap_fraction": overlap / self.sequence_length,
                                   "covered_entire_video": eligible_count == total_frames,
                                   "timestamps_available": True},
            "classifier_note": "Temporal windows use the existing whole-clip checkpoint. Candidate intervals are unverified and softmax confidence is not calibrated accuracy.",
        }
