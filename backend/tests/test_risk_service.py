"""Scenario checks for the uncalibrated risk-prioritization rubric."""
import unittest

from app.services.risk_service import RiskService


def legacy_score(detections):
    """Frozen old formula, retained here only for identical-case comparison."""
    weights = {"weapon": 70.0, "blood": 35.0, "person": 12.0, "vehicle": 6.0, "other": 0.0}
    weapon_words = {"weapon", "gun", "pistol", "rifle", "knife", "handgun", "shotgun", "revolver", "firearm", "blade", "dagger", "machete"}
    best = {key: 0.0 for key in weights}
    seen = set()
    for item in detections:
        label = str(item.get("class") or item.get("class_name") or "").casefold()
        category = str(item.get("category") or "").casefold()
        tokens = set(label.replace("/", " ").replace("_", " ").replace("-", " ").split())
        kind = ("weapon" if category == "weapons" or tokens & weapon_words else
                "blood" if category in {"blood / biological", "blood", "biological"} or "blood" in label else
                "person" if category == "persons" or "person" in label else
                "vehicle" if category == "vehicles" or any(word in label for word in ("vehicle", "car", "truck", "bus")) else "other")
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0
        confidence = max(0.0, min(1.0, confidence / 100 if confidence > 1 else confidence))
        best[kind] = max(best[kind], weights[kind] * confidence)
        seen.add(kind)
    if "weapon" in seen and "person" in seen:
        best["interaction"] = min(10.0, weights["weapon"] * .14)
    return int(round(min(100, sum(best.values()))))


class RiskServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = RiskService()

    def test_same_scenarios_compare_old_and_new_rubrics(self):
        scenarios = [
            ("empty", [], {}, 0, 0),
            ("isolated_weapon", [{"class": "gun", "confidence": .9}], {}, 63, 16),
            ("repeated_same_track", [
                {"class": "gun", "confidence": .9, "track_id": 3, "frame_number": 60, "timestamp_sec": 2},
                {"class": "gun", "confidence": .8, "track_id": 3, "frame_number": 120, "timestamp_sec": 4},
            ], {"tracking": {"used": True}}, 63, 21),
            ("unrelated_weapon_and_person", [
                {"class": "gun", "confidence": .7, "timestamp_sec": 5},
                {"class": "person", "confidence": .9, "timestamp_sec": 80},
            ], {}, 70, 13),
            ("timestamped_shooting_with_object_support", [
                {"class": "gun", "confidence": .9, "track_id": 1, "frame_number": 60, "timestamp_sec": 2},
                {"class": "gun", "confidence": .85, "track_id": 1, "frame_number": 120, "timestamp_sec": 4},
            ], {"classification": {"label": "Shooting", "confidence": .9},
                "activity_events": [{"category": "Shooting", "confidence": .9,
                                     "start_timestamp_sec": 0, "end_timestamp_sec": 5}],
                "tracking": {"method": "saved_tracking_csv", "used": True}, "verification_status": "unverified"}, 63, 79),
        ]
        self.assertEqual([(name, legacy_score(detections), self.service.calculate(detections, context)["overall_risk"])
                          for name, detections, context, _old, _new in scenarios],
                         [(name, old, new) for name, _detections, _context, old, new in scenarios])

    def test_weapon_alone_is_a_capped_review_signal_not_crime_proof(self):
        result = self.service.calculate([{"class": "weapon", "confidence": 1.0}])
        self.assertLess(result["overall_risk"], result["thresholds"]["moderate"])
        self.assertIn("not proof", result["score_disclaimer"])
        self.assertEqual(result["weapon_count"], 1)

    def test_repeated_detections_do_not_stack_object_points(self):
        one = self.service.calculate([{"class": "gun", "confidence": .7}])
        repeated = self.service.calculate([{"class": "gun", "confidence": .7} for _ in range(20)])
        self.assertEqual(one["score_components"]["objects"], repeated["score_components"]["objects"])

    def test_person_does_not_add_weapon_bonus_without_same_event_overlap(self):
        result = self.service.calculate([
            {"class": "gun", "confidence": .7, "timestamp_sec": 80},
            {"class": "person", "confidence": .9, "timestamp_sec": 50},
        ], {"activity_events": [{"category": "Assault", "confidence": .8,
                                 "start_timestamp_sec": 0, "end_timestamp_sec": 5}]})
        self.assertEqual(result["score_components"]["cross_source_corroboration"], 0)

    def test_identical_temporal_event_is_counted_once(self):
        event = {"category": "Assault", "confidence": .8,
                 "start_timestamp_sec": 4, "end_timestamp_sec": 8}
        one = self.service.calculate([], {"activity_events": [event]})
        duplicate = self.service.calculate([], {"activity_events": [event, dict(event)]})
        self.assertEqual(one["score_components"], duplicate["score_components"])

    def test_missing_sources_only_produce_warnings_and_completeness(self):
        result = self.service.calculate([], {"video_info": {}, "activity_events": [], "scene_changes": []})
        self.assertEqual(result["overall_risk"], 0)
        self.assertTrue(result["warnings"])
        self.assertEqual(result["score_components"]["activity"], 0)

    def test_levels_have_four_operational_bands(self):
        self.assertEqual([self.service._level(value) for value in (0, 25, 50, 75)],
                         ["Low", "Moderate", "High", "Critical"])


if __name__ == "__main__":
    unittest.main()
