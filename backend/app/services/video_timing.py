"""Shared mapping from decoded source-frame indices to playback time.

ffprobe is optional. With complete, frame-aligned timestamps, verified CFR
uses zero-based frame_index/FPS and other rates use normalized frame PTS. If
probing is unavailable or alignment cannot be proven, FPS conversion is an
explicit unverified fallback; callers must preserve and expose its status.
"""
from __future__ import annotations

import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class VideoTimeline:
    fps: float | None
    frame_count: int
    duration_sec: float | None
    timestamps_sec: tuple[float, ...]
    presentation_timestamps: tuple[int, ...]
    time_base: str | None
    mapping_status: str
    timestamp_basis: str
    constant_frame_rate_verified: bool

    @property
    def timestamps_verified(self) -> bool:
        return self.mapping_status in {
            "verified_cfr_frame_index_fps",
            "verified_vfr_presentation_timestamps",
        }

    @property
    def presentation_timestamps_available(self) -> bool:
        return bool(self.timestamps_sec)

    def timestamp(self, frame_index: int) -> float | None:
        if frame_index < 0 or frame_index >= self.frame_count:
            return None
        if self.timestamps_sec:
            return self.timestamps_sec[frame_index]
        if self.fps and math.isfinite(self.fps) and self.fps > 0:
            return frame_index / self.fps
        return None

    def frame_end(self, frame_index: int) -> float | None:
        """Return the exclusive playback-time boundary for one source frame."""
        if frame_index < 0 or frame_index >= self.frame_count:
            return None
        if frame_index + 1 < len(self.timestamps_sec):
            return self.timestamps_sec[frame_index + 1]
        if self.duration_sec is not None:
            return self.duration_sec
        start = self.timestamp(frame_index)
        return start + 1.0 / self.fps if start is not None and self.fps else start

    def frame_at_or_after(self, timestamp_sec: float) -> int:
        """Return the first source frame at/after a playback time, clamped."""
        if not self.frame_count:
            return 0
        if self.timestamps_sec:
            import bisect
            return min(self.frame_count - 1, bisect.bisect_left(self.timestamps_sec, timestamp_sec))
        if not self.fps or self.fps <= 0:
            return 0
        return min(self.frame_count - 1, max(0, math.ceil(timestamp_sec * self.fps)))

    def metadata(self) -> dict:
        return {
            "timestamp_mapping_status": self.mapping_status,
            "timestamp_basis": self.timestamp_basis,
            "timestamps_verified": self.timestamps_verified,
            "constant_frame_rate_verified": self.constant_frame_rate_verified,
            "presentation_timestamps_available": self.presentation_timestamps_available,
            "presentation_time_base": self.time_base,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "duration_sec": self.duration_sec,
        }


def probe_video_timeline(path, expected_frames: int, fps: float | None) -> VideoTimeline:
    """Probe aligned frame PTS and classify CFR/VFR; fall back explicitly."""
    valid_fps = fps is not None and math.isfinite(float(fps)) and float(fps) > 0
    fps_value = float(fps) if valid_fps else None
    fallback_status = "unverified_frame_index_fps_fallback" if valid_fps else "unavailable"
    fallback_basis = "unverified_frame_index_divided_by_reported_fps" if valid_fps else "unavailable"
    fallback_duration = expected_frames / fps_value if valid_fps and expected_frames > 0 else None
    base = dict(fps=fps_value, frame_count=max(0, int(expected_frames)),
                duration_sec=fallback_duration, timestamps_sec=(), presentation_timestamps=(),
                time_base=None, mapping_status=fallback_status, timestamp_basis=fallback_basis,
                constant_frame_rate_verified=False)
    if expected_frames <= 0 or not Path(path).is_file():
        return VideoTimeline(**base)

    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams",
        "-show_entries",
        "stream=time_base,avg_frame_rate,duration:frame=best_effort_timestamp,pkt_duration",
        "-of", "json", str(path),
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        payload = json.loads(output.stdout)
        stream = payload["streams"][0]
        time_base = Fraction(stream["time_base"])
        rows = payload.get("frames", [])
        pts = [int(row["best_effort_timestamp"])
               if row.get("best_effort_timestamp") not in (None, "N/A") else None for row in rows]
        aligned = (len(pts) == int(expected_frames) and bool(pts)
                   and all(value is not None for value in pts)
                   and all(right > left for left, right in zip(pts, pts[1:])))
        raw_rate = stream.get("avg_frame_rate")
        stream_rate = Fraction(raw_rate) if raw_rate not in (None, "0/0", "N/A") else None
        seconds = tuple((value - pts[0]) * float(time_base) for value in pts) if aligned else ()
        deltas = [right - left for left, right in zip(seconds, seconds[1:])]
        nominal_rate = fps_value or (float(stream_rate) if stream_rate else None)
        effective_fps = fps_value or nominal_rate
        nominal_interval = 1.0 / nominal_rate if nominal_rate and nominal_rate > 0 else None
        tolerance = max(float(time_base), (nominal_interval or 0) * .01)
        cfr = bool(aligned and nominal_interval and deltas and
                   all(abs(delta - nominal_interval) <= tolerance for delta in deltas))
        stream_duration = stream.get("duration")
        duration = (float(stream_duration) if stream_duration not in (None, "N/A") else None)
        if duration is None or not math.isfinite(duration) or duration <= 0:
            if aligned and len(seconds) > 1:
                duration = seconds[-1] + statistics.median(deltas)
            else:
                duration = fallback_duration
        if aligned:
            status = "verified_cfr_frame_index_fps" if cfr and effective_fps else "verified_vfr_presentation_timestamps"
            basis = "zero_based_frame_index_divided_by_source_fps" if status == "verified_cfr_frame_index_fps" else "frame_pts_relative_to_first_presentation_timestamp"
            return VideoTimeline(
                fps=effective_fps, frame_count=int(expected_frames), duration_sec=duration,
                timestamps_sec=tuple(index / effective_fps for index in range(expected_frames))
                    if status == "verified_cfr_frame_index_fps" else seconds,
                presentation_timestamps=tuple(pts), time_base=f"{time_base.numerator}/{time_base.denominator}",
                mapping_status=status, timestamp_basis=basis, constant_frame_rate_verified=bool(cfr),
            )
        return VideoTimeline(**base)
    except (OSError, subprocess.SubprocessError, KeyError, ValueError, IndexError,
            TypeError, ZeroDivisionError, json.JSONDecodeError, OverflowError):
        return VideoTimeline(**base)


def timestamp_record(timeline: VideoTimeline, frame_index: int) -> dict:
    """Return public timestamp and provenance fields for one source frame."""
    value = timeline.timestamp(frame_index)
    pts = (timeline.presentation_timestamps[frame_index]
           if timeline.presentation_timestamps and frame_index < len(timeline.presentation_timestamps)
           else None)
    return {
        "timestamp_sec": value,
        "presentation_timestamp": pts,
        "time_base": timeline.time_base,
        "timestamp_mapping_status": timeline.mapping_status,
        "timestamp_basis": timeline.timestamp_basis,
        "timestamps_verified": timeline.timestamps_verified,
    }
