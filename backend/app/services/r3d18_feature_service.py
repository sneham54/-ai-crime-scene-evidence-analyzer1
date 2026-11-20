"""Experimental video-to-R3D-18 feature extraction.

This preprocessing is a documented candidate and is not known to match the
preprocessing used to train the BiGRU/MIL checkpoint.
"""
import math
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18
from torch import nn

from app.services.bigru_mil_model import FEATURE_DIM, SEGMENT_COUNT

FRAMES_PER_SEGMENT = 16
MAX_FEATURE_FRAMES = SEGMENT_COUNT * FRAMES_PER_SEGMENT
MAX_FRAME_WIDTH = 4096
MAX_FRAME_HEIGHT = 4096


class R3D18UnavailableError(RuntimeError):
    """Pretrained R3D-18 weights or the supported model are unavailable."""


class InvalidExperimentalVideoError(ValueError):
    """The video metadata or its decoded frames are invalid."""


class R3D18FeatureService:
    _model = None
    _transform = None
    _load_lock = threading.Lock()

    @classmethod
    def _load_model(cls):
        if cls._model is not None:
            return cls._model, cls._transform
        with cls._load_lock:
            if cls._model is not None:
                return cls._model, cls._transform
            try:
                weights = R3D_18_Weights.KINETICS400_V1
                model = r3d_18(weights=weights)
                model.fc = nn.Identity()
                model.cpu().eval()
                cls._model = model
                cls._transform = weights.transforms()
            except Exception as error:
                # torchvision's weights loader verifies the published hash. Never
                # fall back to weights=None, which would produce random features.
                raise R3D18UnavailableError(
                    "Pretrained R3D-18 Kinetics-400 weights are unavailable. "
                    "Check the backend network connection or torchvision weight cache and retry."
                ) from error
        return cls._model, cls._transform

    @staticmethod
    def inspect_video(video_path, max_duration_sec):
        limit = float(max_duration_sec)
        if not math.isfinite(limit) or limit <= 0:
            raise InvalidExperimentalVideoError("Experimental maximum video duration must be finite and positive.")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise InvalidExperimentalVideoError("Uploaded video is invalid or cannot be decoded.")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
                raise InvalidExperimentalVideoError("Video must have readable frames and valid FPS metadata.")
            duration = frame_count / fps
            if not math.isfinite(duration) or duration <= 0:
                raise InvalidExperimentalVideoError("Video duration could not be determined.")
            if duration > limit:
                raise InvalidExperimentalVideoError(
                    f"Video duration ({duration:.3f} s) exceeds the experimental limit of {limit:.3f} s."
                )
            return frame_count, fps, duration
        finally:
            capture.release()

    @staticmethod
    def map_segment_timestamps(duration_sec):
        duration = float(duration_sec)
        if not math.isfinite(duration) or duration <= 0:
            raise InvalidExperimentalVideoError("Video duration must be finite and greater than zero.")
        return [((index + 0.5) * duration / SEGMENT_COUNT) for index in range(SEGMENT_COUNT)]

    @staticmethod
    def _read_segment(capture, segment_index, frame_count, fps, duration):
        start = segment_index * duration / SEGMENT_COUNT
        end = (segment_index + 1) * duration / SEGMENT_COUNT
        # Uniformly sample 16 temporal positions within the half-open segment.
        times = np.linspace(start, max(start, end - 1.0 / fps), FRAMES_PER_SEGMENT)
        frames = []
        for seconds in times:
            frame_index = min(frame_count - 1, max(0, int(round(seconds * fps))))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                continue
            if frame.ndim != 3 or frame.shape[2] < 3:
                continue
            height, width = frame.shape[:2]
            if width <= 0 or height <= 0 or width > MAX_FRAME_WIDTH or height > MAX_FRAME_HEIGHT:
                raise InvalidExperimentalVideoError(
                    f"Decoded frame dimensions exceed the {MAX_FRAME_WIDTH}x{MAX_FRAME_HEIGHT} experimental limit."
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))
        if not frames:
            raise InvalidExperimentalVideoError(f"No frames could be decoded for temporal segment {segment_index}.")
        while len(frames) < FRAMES_PER_SEGMENT:
            frames.append(frames[-1].clone())
        return torch.stack(frames)

    def extract(self, video_path, max_duration_sec):
        video_path = Path(video_path)
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise InvalidExperimentalVideoError("Uploaded video is missing or empty.")
        frame_count, fps, duration = self.inspect_video(video_path, max_duration_sec)
        model, transform = self._load_model()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise InvalidExperimentalVideoError("Uploaded video could not be reopened for feature extraction.")
        features = []
        try:
            for segment_index in range(SEGMENT_COUNT):
                sampled = self._read_segment(capture, segment_index, frame_count, fps, duration)
                # Official R3D weights transform handles resize, center crop,
                # [0,1] scaling, and Kinetics normalization.
                clip = transform(sampled).unsqueeze(0)
                with torch.inference_mode():
                    vector = model(clip).reshape(-1)
                if vector.numel() != FEATURE_DIM:
                    raise RuntimeError(f"R3D-18 returned {vector.numel()} features; expected {FEATURE_DIM}.")
                values = vector.cpu().numpy().astype(np.float32, copy=False)
                if not np.isfinite(values).all():
                    raise RuntimeError(f"R3D-18 produced non-finite features for segment {segment_index}.")
                features.append(values)
        finally:
            capture.release()
        matrix = np.stack(features)
        if matrix.shape != (SEGMENT_COUNT, FEATURE_DIM) or not np.isfinite(matrix).all():
            raise RuntimeError("Extracted R3D-18 features failed shape or finite-value validation.")
        return matrix, self.map_segment_timestamps(duration), duration, frame_count, fps
