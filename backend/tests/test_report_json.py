import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from flask import Flask
from flask_jwt_extended import JWTManager

from app.routes.report import bp
from app.services.report_service import ReportService


class JsonReportRouteTests(unittest.TestCase):
    def setUp(self):
        self.uploads = Path(__file__).resolve().parents[1] / "uploads" / f"test_artifacts_{uuid.uuid4().hex}"
        self.uploads.mkdir(parents=True)
        self.reports = self.uploads / "reports"
        self.reports.mkdir()
        app = Flask(__name__)
        app.config.update(UPLOAD_FOLDER=str(self.uploads), JWT_SECRET_KEY="test-key")
        app.db = None
        JWTManager(app)
        app.register_blueprint(bp, url_prefix="/api/report")
        self.client = app.test_client()

    def tearDown(self):
        shutil.rmtree(self.uploads)

    def test_json_report_can_be_saved_and_downloaded_unverified(self):
        response = self.client.post("/api/report/generate-json", json={
            "file_name": "camera.mp4",
            "candidate_intervals": [],
            "verification_status": "verified",
            "experimental_anomaly": {
                "analysis_mode": "experimental_precomputed_features",
                "segment_scores": [{"segment_index": 0, "raw_logit": -0.25}],
            },
        })
        self.assertEqual(response.status_code, 200)
        metadata = response.get_json()
        report = self.client.get(metadata["download_url"])
        self.assertEqual(report.status_code, 200)
        payload = json.loads(report.get_data(as_text=True))
        report.close()
        self.assertEqual(payload["verification_status"], "unverified")
        self.assertTrue(payload["human_review_required"])
        self.assertEqual(payload["experimental_anomaly"]["segment_scores"][0]["raw_logit"], -0.25)

    def test_json_report_rejects_non_object_payload(self):
        response = self.client.post("/api/report/generate-json", data="[]", content_type="application/json")
        self.assertEqual(response.status_code, 400)

    def test_pdf_report_renders_unverified_object_detection_events(self):
        app = self.client.application
        with app.app_context():
            report_path, _ = ReportService().generate(
                case_data={"case_id": "CASE-TEST"}, detections=[], risk={},
                video_result={
                    "classification": {"label": "Normal", "confidence": .8},
                    "video_info": {"duration_sec": 12, "total_frames": 360, "fps": 30,
                                   "analysis_coverage": "portion", "processed_frames": 12,
                                   "sampling_interval_frames": 30, "first_analyzed_timestamp_sec": 0,
                                   "last_analyzed_timestamp_sec": 11,
                                   "classifier_sampling_coverage": {"sampled_frame_count": 16,
                                       "first_sample_timestamp_sec": 0, "last_sample_timestamp_sec": 11,
                                       "covered_entire_video": True}},
                    "evidence_events": [{
                        "event_id": "object-event-000001", "class": "pistol",
                        "start_timestamp_sec": 2, "end_timestamp_sec": 4,
                        "duration_sec": 2, "observation_count": 3,
                        "confidence_summary": {"min": .6, "mean": .7, "max": .8},
                        "representative_frame_number": 120,
                        "representative_evidence_image": "/api/video/evidence/abc/frame_000120.jpg",
                    }],
                    "timeline": [],
                },
            )
        self.assertTrue(Path(report_path).is_file())
        self.assertGreater(Path(report_path).stat().st_size, 0)

    def test_pdf_report_list_exposes_saved_file_metadata_and_inline_view(self):
        report_path = self.reports / "forensic_report_saved001.pdf"
        report_path.write_bytes(b"%PDF-1.4\nfixture\n")
        (self.reports / "forensic_report_saved001.pdf.meta.json").write_text(json.dumps({
            "report_id": "forensic_report_saved001",
            "report_name": report_path.name,
            "generated_at": "2026-10-31T10:00:00Z",
            "status": "generated",
            "investigation_id": "CASE-REAL-1",
            "evidence_id": "evidence-1",
            "evidence_filename": "camera.mp4",
            "category": "Robbery",
            "risk_score": 65,
        }), encoding="utf-8")

        payload = self.client.get("/api/report/list").get_json()

        self.assertEqual(payload["total_reports"], 1)
        self.assertEqual(payload["successful_reports"], 1)
        self.assertEqual(payload["failed_reports"], None)
        self.assertEqual(payload["reports_associated_with_investigations"], 1)
        self.assertEqual(payload["reports"][0]["evidence_filename"], "camera.mp4")
        view_response = self.client.get(payload["reports"][0]["view_url"])
        download_response = self.client.get(payload["reports"][0]["download_url"])
        self.assertEqual(view_response.status_code, 200)
        self.assertIn("inline", view_response.headers["Content-Disposition"])
        self.assertEqual(download_response.status_code, 200)
        self.assertIn("attachment", download_response.headers["Content-Disposition"])

    def test_pdf_generation_persists_submitted_investigation_metadata(self):
        report_name = "forensic_report_newrun.pdf"
        report_path = self.reports / report_name
        report_path.write_bytes(b"%PDF-1.4\nfixture\n")
        with patch("app.routes.report.ReportService.generate", return_value=(str(report_path), report_name)):
            response = self.client.post("/api/report/generate", json={
                "case": {"case_id": "CASE-2026-1"},
                "evidence_id": "evidence-42", "file_name": "camera.mp4",
                "risk": {"overall_risk": 55},
                "video_result": {"analysis_id": "analysis-1", "classification": {"label": "Assault"}},
            })
        self.assertEqual(response.status_code, 200)
        metadata = json.loads(Path(str(report_path) + ".meta.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["evidence_id"], "evidence-42")
        self.assertEqual(metadata["investigation_id"], "CASE-2026-1")
        self.assertEqual(metadata["category"], "Assault")
        self.assertEqual(metadata["risk_score"], 55)


if __name__ == "__main__":
    unittest.main()
