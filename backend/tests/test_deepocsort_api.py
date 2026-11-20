"""Synthetic API checks for the isolated Ultralytics Deep OC-SORT runtime."""
import unittest
from pathlib import Path

import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import IterableSimpleNamespace, YAML


BACKEND = Path(__file__).resolve().parents[1]


class DeepOCSORTAPITest(unittest.TestCase):
    def make_tracker(self):
        config = YAML.load(str(BACKEND / "config" / "deepocsort.yaml"))
        self.assertEqual(config["tracker_type"], "deepocsort")
        self.assertFalse(config["use_byte"])
        return TRACKER_MAP["deepocsort"](args=IterableSimpleNamespace(**config))

    def detections(self, frame_index):
        # Ultralytics Boxes layout: xyxy, confidence, class ID.
        values = [[40 + frame_index * 2, 40, 100 + frame_index * 2, 140, .95, 5]]
        return Boxes(torch.tensor(values, dtype=torch.float32), (240, 320))

    def test_ids_persist_through_consecutive_frames_and_empty_frame(self):
        tracker = self.make_tracker()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        first = tracker.update(self.detections(0), frame)
        self.assertEqual(len(first), 1)
        track_id = int(first[0, 4])
        self.assertEqual(int(first[0, 6]), 5)

        empty = tracker.update(Boxes(torch.empty((0, 6)), (240, 320)), frame)
        self.assertEqual(len(empty), 0)
        returned = tracker.update(self.detections(2), frame)
        self.assertEqual(len(returned), 1)
        self.assertEqual(int(returned[0, 4]), track_id)
        self.assertEqual(int(returned[0, 6]), 5)

    def test_reset_clears_per_video_state(self):
        tracker = self.make_tracker()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tracker.update(self.detections(0), frame)
        tracker.update(self.detections(1), frame)
        self.assertGreater(tracker.frame_id, 0)
        tracker.reset()
        self.assertEqual(tracker.frame_id, 0)
        self.assertEqual(tracker.tracked_stracks, [])
        self.assertEqual(tracker.lost_stracks, [])
        new_video = tracker.update(self.detections(0), frame)
        self.assertEqual(len(new_video), 1)
        self.assertEqual(int(new_video[0, 6]), 5)


if __name__ == "__main__":
    unittest.main()
