import json
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.video_timing import probe_video_timeline


class VideoTimelineTests(unittest.TestCase):
    def probe_payload(self, pts, time_base="1/30000", fps="30/1", duration="1.0"):
        return {"streams": [{"time_base": time_base, "avg_frame_rate": fps,
                             "duration": duration}],
                "frames": [{"best_effort_timestamp": str(value)} for value in pts]}

    def test_verified_cfr_uses_source_frame_index_over_fps(self):
        with patch.object(Path, "is_file", return_value=True), patch(
                "app.services.video_timing.subprocess.run") as run:
            run.return_value.stdout = json.dumps(self.probe_payload([0, 1000, 2000], duration="0.1"))
            timeline = probe_video_timeline("sample.mp4", 3, 30)
        self.assertEqual(timeline.mapping_status, "verified_cfr_frame_index_fps")
        self.assertTrue(timeline.timestamps_verified)
        self.assertEqual(timeline.timestamps_sec, (0.0, 1 / 30, 2 / 30))

    def test_verified_vfr_uses_normalized_per_frame_pts(self):
        payload = self.probe_payload([9000, 10000, 13000], fps="20/1", duration="0.2")
        with patch.object(Path, "is_file", return_value=True), patch(
                "app.services.video_timing.subprocess.run") as run:
            run.return_value.stdout = json.dumps(payload)
            timeline = probe_video_timeline("sample.mp4", 3, 20)
        self.assertEqual(timeline.mapping_status, "verified_vfr_presentation_timestamps")
        self.assertTrue(timeline.timestamps_verified)
        self.assertEqual(timeline.timestamps_sec, (0.0, 1 / 30, 4 / 30))
        self.assertEqual(timeline.timestamp(2), timeline.timestamps_sec[2])

    def test_missing_ffprobe_is_explicit_unverified_fps_fallback(self):
        with patch.object(Path, "is_file", return_value=True), patch(
                "app.services.video_timing.subprocess.run", side_effect=FileNotFoundError):
            timeline = probe_video_timeline("sample.mp4", 31, 30)
        self.assertEqual(timeline.mapping_status, "unverified_frame_index_fps_fallback")
        self.assertFalse(timeline.timestamps_verified)
        self.assertEqual(timeline.timestamp(30), 1.0)


if __name__ == "__main__":
    unittest.main()
