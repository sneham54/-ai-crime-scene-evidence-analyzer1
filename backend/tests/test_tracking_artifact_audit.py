import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_tracking_artifacts.py"
SPEC = importlib.util.spec_from_file_location("tracking_artifact_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class TrackingArtifactAuditTests(unittest.TestCase):
    def test_csv_schema_tracks_ids_and_valid_empty_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "detections.csv"
            path.write_text("frame,timestamp_seconds,class,track_id\n1,0.5,person,7\n", encoding="utf-8")
            schema, rows, has_ids, frame_count, first, last, _, error = AUDIT.csv_details(path)
            self.assertEqual(schema, "frame|timestamp_seconds|class|track_id")
            self.assertEqual((rows, has_ids, frame_count, first, last, error), (1, True, 1, "0.5", "0.5", None))
            path.write_text("frame,timestamp_seconds,class,track_id\n", encoding="utf-8")
            self.assertEqual(AUDIT.csv_details(path)[1:3], (0, False))

    def test_malformed_csv_and_missing_mp4_are_not_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "Assault001.mp4"
            source.touch()
            outputs = root / "results"
            outputs.mkdir()
            with patch.object(AUDIT, "RESULT_ROOTS", {"Assault": [outputs]}), \
                 patch.object(AUDIT, "readable_video", return_value=(True, 10, None)):
                self.assertEqual(AUDIT.audit_row("Assault", source)["disposition"], "missing")
                folder = outputs / source.stem
                folder.mkdir()
                (folder / "detections.csv").write_text("not,a,valid\n1,2\n", encoding="utf-8")
                row = AUDIT.audit_row("Assault", source)
                self.assertEqual(row["disposition"], "invalid")
                self.assertIn("CSV unreadable", row["csv_errors"])

    def test_processing_status_and_stale_frame_count_require_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "Robbery001.mp4"
            source.touch()
            outputs = root / "results"
            folder = outputs / source.stem
            folder.mkdir(parents=True)
            (folder / "annotated.mp4").touch()
            (folder / "detections.csv").write_text("frame,timestamp_seconds,class,track_id\n1,0.5,person,1\n", encoding="utf-8")
            status = outputs / "tracking_status.csv"
            status.write_text("video_name,status,error\nRobbery001.mp4,processing,\n", encoding="utf-8")
            with patch.object(AUDIT, "RESULT_ROOTS", {"Robbery": [outputs]}), \
                 patch.object(AUDIT, "readable_video", side_effect=[(True, 10, None), (True, 8, None)]):
                row = AUDIT.audit_row("Robbery", source)
            self.assertEqual(row["tracking_status"], "processing")
            self.assertEqual(row["disposition"], "partial")

    def test_duplicate_artifact_matches_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "Assault001.mp4"
            source.touch()
            outputs = root / "results"
            for suffix in ("a", "b"):
                folder = outputs / suffix / source.stem
                folder.mkdir(parents=True)
                (folder / "Assault001_tracked.mp4").touch()
                (folder / "detections.csv").write_text("frame,track_id\n1,2\n", encoding="utf-8")
            with patch.object(AUDIT, "RESULT_ROOTS", {"Assault": [outputs]}), \
                 patch.object(AUDIT, "readable_video", return_value=(True, 10, None)):
                row = AUDIT.audit_row("Assault", source)
            self.assertTrue(row["ambiguous_match"])
            self.assertEqual(row["disposition"], "needs_human_review")


if __name__ == "__main__":
    unittest.main()
