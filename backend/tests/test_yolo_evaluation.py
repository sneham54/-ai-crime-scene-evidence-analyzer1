import unittest
from unittest.mock import patch

import numpy as np

from evaluate_yolo_cctv import evaluate, parse_thresholds
from app.services.yolo_evaluation import (
    filter_predictions,
    map_class_id,
    match_frame_detections,
)


class YoloEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.class_names = {
            0: "gun", 1: "hammer", 2: "helmet", 3: "knife", 4: "mask",
            5: "person", 6: "pistol", 7: "rifle", 8: "weapon",
        }

    def test_checkpoint_class_ids_map_without_semantic_aliasing(self):
        self.assertEqual(map_class_id(self.class_names, 8), "weapon")
        self.assertEqual(map_class_id(self.class_names, 0), "gun")
        self.assertEqual(map_class_id(self.class_names, 6), "pistol")
        self.assertNotEqual(map_class_id(self.class_names, 8), map_class_id(self.class_names, 0))
        with self.assertRaisesRegex(ValueError, "Unknown YOLO class id"):
            map_class_id(self.class_names, 9)

    def test_confidence_filter_is_inclusive_and_preserves_high_score_predictions(self):
        detections = [
            {"class": "weapon", "confidence": 0.24},
            {"class": "weapon", "confidence": 0.25},
            {"class": "gun", "confidence": 0.48},
        ]
        filtered = filter_predictions(detections, 0.25)
        self.assertEqual([item["confidence"] for item in filtered], [0.25, 0.48])
        self.assertEqual(len(filter_predictions(detections, 0.50)), 0)
        with self.assertRaises(ValueError):
            filter_predictions(detections, 1.1)

    def test_duplicate_or_overlapping_cross_class_boxes_do_not_double_match_truth(self):
        truth = [{"class": "weapon", "bbox": [10, 10, 50, 50]}]
        predictions = [
            {"class": "weapon", "bbox": [10, 10, 50, 50], "confidence": 0.8},
            {"class": "weapon", "bbox": [12, 12, 48, 48], "confidence": 0.7},
            {"class": "gun", "bbox": [10, 10, 50, 50], "confidence": 0.6},
        ]
        matches, false_positives, false_negatives = match_frame_detections(truth, predictions)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(false_positives), 2)
        self.assertEqual(len(false_negatives), 0)

    def test_wrong_class_prediction_is_false_positive_and_truth_is_missed(self):
        truth = [{"class": "weapon", "bbox": [10, 10, 50, 50]}]
        predictions = [{"class": "pistol", "bbox": [10, 10, 50, 50], "confidence": 0.9}]
        matches, false_positives, false_negatives = match_frame_detections(truth, predictions)
        self.assertEqual(matches, [])
        self.assertEqual(len(false_positives), 1)
        self.assertEqual(len(false_negatives), 1)

    def test_threshold_sweep_reports_false_positives_and_missed_detections(self):
        raw = [
            {"class": "weapon", "confidence": 0.48, "bbox": [10, 10, 50, 50]},
            {"class": "knife", "confidence": 0.35, "bbox": [60, 60, 90, 90]},
        ]
        with patch("evaluate_yolo_cctv.cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)), \
             patch("evaluate_yolo_cctv.predict_frame", return_value=(raw, None)):
            report = evaluate(None, {"frame.jpg": [{"class": "weapon", "bbox": [10, 10, 50, 50]}]},
                              self.class_names, [0.25, 0.50], 832)
        at_current = report["thresholds"]["0.25"]
        self.assertEqual(at_current["per_class"]["weapon"]["true_positives"], 1)
        self.assertEqual(at_current["per_class"]["knife"]["false_positives"], 1)
        at_candidate = report["thresholds"]["0.5"]
        self.assertEqual(at_candidate["per_class"]["weapon"]["missed_detections"], 1)
        self.assertEqual(at_candidate["per_class"]["knife"]["false_positives"], 0)

    def test_sweep_always_includes_configured_current_threshold(self):
        values = parse_thresholds("0.1,0.5")
        self.assertIn(0.25, values)

if __name__ == "__main__":
    unittest.main()
