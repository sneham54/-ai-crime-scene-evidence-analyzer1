"""Streaming OpenCV clip reader with stable source-frame and timestamp mapping."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import PreprocessConfig
from app.services.video_timing import probe_video_timeline


@dataclass
class ClipWindow:
    tensor: torch.Tensor  # T,C,H,W, float32 normalized
    frame_indices: list[int]
    timestamps_sec: list[float | None]
    start_timestamp_sec: float | None
    end_timestamp_sec: float | None
    timestamp_mapping_status: str = "unavailable"
    timestamps_verified: bool = False


def _prepare(frame, config):
    rgb = cv2.cvtColor(cv2.resize(frame, (config.image_size, config.image_size)), cv2.COLOR_BGR2RGB)
    values = rgb.astype(np.float32) / 255.0
    values = (values - np.asarray(config.mean, np.float32)) / np.asarray(config.std, np.float32)
    return np.transpose(values, (2, 0, 1))


def iter_video_clips(video_path, config: PreprocessConfig | None = None):
    """Yield fixed clips without retaining the entire video in memory."""
    config = config or PreprocessConfig()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Video cannot be opened: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    timeline = probe_video_timeline(video_path, frame_count, fps)
    max_frames = None
    if config.max_duration_sec is not None:
        if timeline.timestamps_sec:
            max_frames = max(1, sum(value <= config.max_duration_sec for value in timeline.timestamps_sec))
        elif timeline.fps:
            max_frames = max(1, int(config.max_duration_sec * timeline.fps) + 1)
    window = deque(maxlen=config.clip_length)
    next_start = 0
    decoded = 0
    emitted = False
    last_emitted_end = -1
    try:
        while max_frames is None or decoded < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            index = decoded
            window.append((index, timeline.timestamp(index), _prepare(frame, config)))
            decoded += 1
            if len(window) == config.clip_length and index >= next_start + config.clip_length - 1:
                entries = list(window)
                yield _make_clip(entries, timeline, config.max_duration_sec)
                emitted = True
                last_emitted_end = entries[-1][0]
                next_start = entries[0][0] + config.clip_stride
        if window and (not emitted or window[-1][0] > last_emitted_end):
            entries = list(window)
            while len(entries) < config.clip_length:
                entries.append(entries[-1])
            yield _make_clip(entries, timeline, config.max_duration_sec)
        if decoded == 0:
            raise ValueError(f"Video has no decodable frames: {video_path}")
    finally:
        capture.release()


def _make_clip(entries, timeline, max_duration_sec=None):
    indices = [int(index) for index, _, _ in entries]
    times = [timestamp for _, timestamp, _ in entries]
    end_time = timeline.frame_end(indices[-1]) if indices else None
    if end_time is not None and max_duration_sec is not None:
        end_time = min(end_time, float(max_duration_sec))
    return ClipWindow(torch.from_numpy(np.stack([frame for _, _, frame in entries])), indices, times,
                      times[0] if times else None, end_time, timeline.mapping_status,
                      timeline.timestamps_verified)


def probe_video(video_path):
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Video cannot be opened: {video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or frame_count <= 0:
            raise ValueError("Video metadata is empty or invalid")
        timeline = probe_video_timeline(video_path, frame_count, fps)
        return {"width": width, "height": height, "fps": timeline.fps,
                "frame_count": frame_count, "duration_sec": timeline.duration_sec,
                "timestamp_mapping_status": timeline.mapping_status,
                "timestamp_basis": timeline.timestamp_basis,
                "timestamps_verified": timeline.timestamps_verified,
                "constant_frame_rate_verified": timeline.constant_frame_rate_verified,
                "timestamp_warning": (None if timeline.timestamps_verified else
                    "Timestamp mapping is estimated from frame index and reported FPS; VFR timing is unverified.")}
    finally:
        capture.release()
