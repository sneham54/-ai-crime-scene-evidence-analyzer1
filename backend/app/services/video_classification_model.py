"""CPU-compatible loader for the existing ResNet18 + LSTM video checkpoint."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models


VIDEO_CLASS_NAMES = ("Assault", "Robbery", "Shooting", "Normal")
IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


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


def decode_video_sequence(video_path, sequence_length=16, image_size=112):
    """Read evenly spaced frames without retaining the complete video in memory."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Uploaded video cannot be decoded")
    frames = []
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            raise ValueError("Video has no readable frames")
        for index in np.linspace(0, count - 1, sequence_length).round().astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frame = (frame - IMAGE_MEAN) / IMAGE_STD
            frames.append(np.transpose(frame, (2, 0, 1)))
    finally:
        capture.release()
    if not frames:
        raise ValueError("Video contains no decodable frames")
    while len(frames) < sequence_length:
        frames.append(frames[-1].copy())
    return np.stack(frames[:sequence_length]).astype(np.float32)


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
        self.device = torch.device("cpu")
        self.sequence_length = 16
        self.image_size = 112
        self.class_names = VIDEO_CLASS_NAMES
        self.model = None
        self.min_confidence = float(min_confidence)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("Video classifier checkpoint was not found")
        if not self.class_names_path.is_file():
            raise FileNotFoundError("Video classifier class names were not found")
        with self.class_names_path.open(encoding="utf-8") as source:
            self.class_names = tuple(json.load(source))
        if self.class_names != VIDEO_CLASS_NAMES:
            raise ValueError("Video classifier must use Assault, Robbery, Shooting, Normal class order")
        # This is a local trusted trained checkpoint, loaded with CPU mapping.
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("classes") != list(self.class_names):
            raise ValueError("Checkpoint class order does not match class_names.json")
        self.sequence_length = int(checkpoint.get("num_frames", 16))
        self.image_size = int(checkpoint.get("img_size", 112))
        self.model = CNNLSTM(len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        self._initialized = True

    def predict_path(self, video_path):
        sequence = decode_video_sequence(video_path, self.sequence_length, self.image_size)
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[index].item())
        top_class = self.class_names[index]
        return {
            "label": top_class if confidence >= self.min_confidence else "Uncertain",
            "top_class": top_class,
            "confidence": round(confidence, 4),
            "probabilities": {name: round(float(probabilities[i].item()), 4) for i, name in enumerate(self.class_names)},
            "classification_mode": "trained_resnet18_lstm",
            "classifier_note": "Softmax confidence is not a calibrated probability or a legal conclusion.",
        }
