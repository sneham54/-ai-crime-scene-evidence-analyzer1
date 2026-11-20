from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from bson import ObjectId

from flask import Flask
from flask_jwt_extended import JWTManager

from app.routes.dashboard import bp
from app.models.case import update_case_risk


class Cursor(list):
    def sort(self, key, direction):
        return Cursor(sorted(self, key=lambda row: row.get(key) or datetime.min, reverse=direction < 0))

    def limit(self, count):
        return Cursor(self[:count])

class Collection:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def find(self, query=None, projection=None):
        return Cursor(self.rows)

    def count_documents(self, query):
        return len(self.rows)

    def distinct(self, key, query=None):
        return list(dict.fromkeys(row.get(key) for row in self.rows if row.get(key) is not None))

    def insert_one(self, document):
        stored = dict(document)
        stored["_id"] = ObjectId()
        self.rows.append(stored)
        return SimpleNamespace(inserted_id=stored["_id"])

    def update_one(self, query, update):
        for row in self.rows:
            if row.get("_id") == query.get("_id"):
                row.update(update.get("$set", {}))
                return SimpleNamespace(matched_count=1)
        return SimpleNamespace(matched_count=0)


class FakeDB:
    def __init__(self, rows):
        self.evidence = Collection(rows)
        self.cases = Collection()
        self.custody_logs = Collection()

    def __getitem__(self, name):
        return getattr(self, name, Collection())

    def list_collection_names(self):
        return ["cases", "custody_logs", "evidence"]


def video(identifier, filename, category, stamp, **extra):
    row = {
        "_id": identifier,
        "filename": filename,
        "file_type": "video",
        "status": "analyzed",
        "uploaded_at": stamp,
        "detections": [{"class": "person", "track_id": 3}, {"class": "car", "track_id": 4}],
        "risk_score": {"overall_risk": 10, "risk_level": "Low", "contributing_factors": []},
        "event_classification": ({"label": category, "classification_status": "provisional"} if category else None),
        "video_analysis": {"tracking": {"tracking_id_count": 2, "used": True}},
    }
    row.update(extra)
    return row


def client_for(rows, tmp_path):
    app = Flask(__name__)
    app.config.update(TESTING=True, JWT_SECRET_KEY="dashboard-test-secret", UPLOAD_FOLDER=str(tmp_path))
    JWTManager(app)
    app.register_blueprint(bp, url_prefix="/api/dashboard")
    app.db = FakeDB(rows)
    return app.test_client()


def test_dashboard_empty_database_returns_genuine_zeroes(tmp_path):
    response = client_for([], tmp_path).get("/api/dashboard/summary")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["total_evidence"] == 0
    assert payload["videos_analyzed"] == 0
    assert payload["crime_types"] == {"Assault": 0, "Robbery": 0, "Shooting": 0, "Normal": 0}
    assert payload["risk_distribution"] == {"low": 0, "moderate": 0, "high": 0, "critical": 0}
    assert payload["recent_investigations"] == []
    assert payload["failed_analyses"] == 0
    assert payload["total_custody_entries"] == 0
    assert payload["unique_evidence_with_custody"] == 0


def test_database_health_endpoint_reports_connection_state(tmp_path):
    client = client_for([], tmp_path)
    client.application.config["DB_NAME"] = "real-project-database"
    response = client.get("/api/dashboard/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "connected"
    assert response.get_json()["database"] == "real-project-database"

    client.application.db = None
    response = client.get("/api/dashboard/health")
    assert response.status_code == 503
    assert response.get_json()["status"] == "unavailable"
    assert response.get_json()["database"] is None


def test_database_health_and_summary_surface_safe_connection_reason():
    client = client_for([], Path("backend/uploads"))
    client.application.db = None
    client.application.extensions["mongo_connection_state"] = {
        "error": "MongoDB at localhost:27017 refused the connection; start MongoDB or check that the configured host and port are reachable.",
        "error_type": "ServerSelectionTimeoutError",
    }
    health = client.get("/api/dashboard/health")
    summary = client.get("/api/dashboard/summary")
    assert health.status_code == 503
    assert health.get_json()["message"].startswith("MongoDB at localhost:27017 refused")
    assert summary.status_code == 503
    assert summary.get_json()["message"] == health.get_json()["message"]
    assert summary.get_json()["total_cases"] is None


def test_mongodb_manager_retries_ping_and_masks_credentials(monkeypatch):
    from app import refresh_database_connection
    from pymongo.errors import ServerSelectionTimeoutError

    class Client:
        admin = SimpleNamespace(command=lambda _command: (_ for _ in ()).throw(
            ServerSelectionTimeoutError("localhost:27017: connection refused")))

    app = Flask(__name__)
    app.config.update(MONGO_URI="mongodb://private-user:private-pass@localhost:27017/crime_evidence_db",
                      MONGO_SERVER_SELECTION_TIMEOUT_MS=100,
                      MONGO_SOCKET_TIMEOUT_MS=1000,
                      MONGO_RECONNECT_INTERVAL_SECONDS=1,
                      DB_NAME="crime_evidence_db")
    app.mongo_client = Client()
    app.db = None
    monkeypatch.setattr("app.MongoClient", lambda *args, **kwargs: Client())
    assert refresh_database_connection(app, force=True) is False
    state = app.extensions["mongo_connection_state"]
    assert "localhost:27017 refused" in state["error"]
    assert "private-pass" not in state["error"]
    assert state["error_type"] == "ServerSelectionTimeoutError"


def test_case_registration_and_list_use_persisted_case_collection(tmp_path):
    client = client_for([], tmp_path)
    created = client.post("/api/dashboard/cases", json={"title": "Warehouse incident", "description": "Night shift"})
    assert created.status_code == 201
    case = created.get_json()["case"]
    assert case["title"] == "Warehouse incident"
    assert case["risk_level"] == "unknown"
    assert client.application.db.cases.rows[0]["description"] == "Night shift"
    listed = client.get("/api/dashboard/cases")
    assert listed.status_code == 200
    assert listed.get_json()["cases"][0]["id"] == case["id"]


def test_case_registration_reports_database_unavailable(tmp_path):
    client = client_for([], tmp_path)
    client.application.db = None
    response = client.post("/api/dashboard/cases", json={"title": "Cannot save offline"})
    assert response.status_code == 503
    assert "unavailable" in response.get_json()["error"].casefold()


def test_existing_risk_assessment_updates_only_the_linked_case():
    case_id = ObjectId()
    database = FakeDB([])
    database.cases.rows.append({"_id": case_id, "risk_level": "unknown"})
    assert update_case_risk(database, str(case_id), {"overall_risk": 62, "risk_level": "high"})
    assert database.cases.rows[0]["risk_level"] == "high"
    assert database.cases.rows[0]["latest_risk_score"] == 62
    assert not update_case_risk(database, "not-an-object-id", {"overall_risk": 70})


def test_dashboard_counts_latest_video_classification_once_and_ignores_missing(tmp_path):
    now = datetime.now(timezone.utc)
    rows = [
        video("a1", "a.mp4", "Assault", now - timedelta(minutes=2)),
        video("a2", "a.mp4", "Robbery", now,
              event_classification={"label": "Robbery", "classification_status": "uncertain_prediction_disagreement"}),
        video("b", "b.mp4", "Robbery", now - timedelta(minutes=1), event_classification=None),
        video("c", "c.mp4", "Shooting", now - timedelta(minutes=3)),
        video("d", "d.mp4", "Normal", now - timedelta(minutes=4)),
        video("e", "e.mp4", "Assault", now - timedelta(minutes=5)),
        video("failed", "failed.mp4", "Assault", now, status="failed"),
        video("a-failed", "a.mp4", "Assault", now + timedelta(minutes=1), status="failed"),
    ]
    with patch("app.routes.dashboard.CustodyService.get_logs", return_value=[]):
        payload = client_for(rows, tmp_path).get("/api/dashboard/summary").get_json()
    assert payload["videos_analyzed"] == 5
    assert payload["crime_types"] == {"Assault": 1, "Robbery": 0, "Shooting": 1, "Normal": 1}
    assert payload["reference_video_categories"]["Shooting"] == 0
    assert payload["reference_category_unknown_count"] == 5
    assert payload["total_detections"] == 10
    assert sum(payload["object_distribution"].values()) == payload["total_detections"]
    assert payload["object_distribution"] == {"person": 5, "car": 5}
    assert payload["valid_tracking_ids"] == 10
    assert len(payload["recent_investigations"]) == 5
    latest_a = next(row for row in payload["recent_investigations"] if row["filename"] == "a.mp4")
    assert latest_a["classification_status"] == "uncertain_prediction_disagreement"
    assert latest_a["predicted_category"] is None
    assert latest_a["model_candidate_category"] == "Robbery"
    assert latest_a["analyzed_at"].endswith("Z")
    assert payload["failed_analyses"] == 2


def test_dashboard_keeps_reference_category_separate_from_unresolved_prediction(tmp_path):
    row = video("ref-shooting", "Shooting001_x264.mp4", "Assault", datetime.now(timezone.utc),
                reference_category="Shooting",
                event_classification={"label": "Unresolved", "classification_status": "uncertain_prediction_disagreement",
                                      "whole_video_prediction": {"label": "Assault"}})
    payload = client_for([row], tmp_path).get("/api/dashboard/summary").get_json()
    assert payload["crime_types"]["Assault"] == 0
    assert payload["reference_video_categories"]["Shooting"] == 1
    assert payload["reference_category_unknown_count"] == 0
    item = payload["recent_investigations"][0]
    assert item["reference_category"] == "Shooting"
    assert item["predicted_category"] is None
    assert item["model_candidate_category"] == "Assault"


def test_dashboard_distinguishes_unavailable_database_from_empty_database(tmp_path):
    client = client_for([], tmp_path)
    client.application.db = None
    payload = client.get("/api/dashboard/summary").get_json()
    assert payload["status"] == "unavailable"
    assert payload["total_evidence"] is None
    assert payload["crime_types"] is None
    assert payload["risk_distribution"] is None


def test_custody_history_endpoint_reports_database_unavailable_not_empty(tmp_path):
    client = client_for([], tmp_path)
    client.application.db = None
    response = client.get("/api/dashboard/history?evidence_id=evidence-1")
    assert response.status_code == 503
    assert response.get_json()["status"] == "unavailable"
    assert response.get_json()["logs"] == []


def test_dashboard_does_not_use_local_video_artifacts_as_database_fallback(tmp_path):
    artifact_dir = tmp_path / "evidence" / "analysis-run-001"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "analysis.json").write_text(json.dumps({
        "schema_version": 2,
        "analysis_id": "analysis-run-001",
        "status": "completed",
        "original_filename": "Assault001_x264.mp4",
        "uploaded_at": "2026-10-31T09:00:00Z",
        "analyzed_at": "2026-10-31T09:05:00Z",
        "event_classification": {"label": "Assault", "confidence": 0.94},
        "risk": {"overall_risk": 42, "risk_level": "Moderate", "contributing_factors": []},
        "frames": [{"detections": [
            {"class": "person", "track_id": 3},
            {"class": "person", "track_id": 3},
            {"class": "knife", "track_id": 8},
        ]}],
        "tracking": {"used": True, "method": "ByteTrack"},
    }), encoding="utf-8")

    app = Flask(__name__)
    app.config.update(TESTING=True, JWT_SECRET_KEY="dashboard-test-secret", UPLOAD_FOLDER=str(tmp_path))
    JWTManager(app)
    app.register_blueprint(bp, url_prefix="/api/dashboard")
    app.db = None
    payload = app.test_client().get("/api/dashboard/summary").get_json()

    assert payload["status"] == "unavailable"
    assert payload["total_evidence"] is None
    assert payload["videos_analyzed"] is None
    assert payload["completed_analyses"] is None
    assert payload["failed_analyses"] is None
    assert payload["total_detections"] is None
    assert payload["crime_types"] is None
    assert payload["risk_distribution"] is None
    assert payload["monthly_evidence"] is None
    assert payload["recent_investigations"] is None


def test_dashboard_does_not_count_saved_image_outputs_when_database_is_unavailable(tmp_path):
    annotated = tmp_path / "annotated"
    annotated.mkdir()
    (annotated / "hybrid_001.jpg").write_bytes(b"saved output")
    (annotated / "hybrid_002.PNG").write_bytes(b"saved output")
    (annotated / ".gitkeep").write_bytes(b"")
    (annotated / "empty.jpg").write_bytes(b"")

    artifact_dir = tmp_path / "evidence" / "analysis-run-002"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "analysis.json").write_text(json.dumps({
        "analysis_id": "analysis-run-002", "status": "completed",
        "event_classification": {"label": "Normal"}, "risk": {"overall_risk": 12},
    }), encoding="utf-8")

    app = Flask(__name__)
    app.config.update(TESTING=True, JWT_SECRET_KEY="dashboard-test-secret", UPLOAD_FOLDER=str(tmp_path))
    JWTManager(app)
    app.register_blueprint(bp, url_prefix="/api/dashboard")
    app.db = None
    payload = app.test_client().get("/api/dashboard/summary").get_json()

    assert payload["status"] == "unavailable"
    assert payload["images_analyzed"] is None
    assert payload["high_risk_cases"] is None
    assert payload["monthly_evidence"] is None
    assert payload["custody_status"] == "unavailable"


def test_dashboard_does_not_count_saved_reports_as_database_video_records(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    report = {
        "report_type": "AI-assisted investigation evidence",
        "file_name": "Robbery005_x264.mp4",
        "analyzed_at": "2026-10-30T18:00:00Z",
        "risk": {"overall_risk": 65, "risk_level": "Elevated", "thresholds": {
            "moderate": 20, "elevated": 45, "critical": 75}},
        "video_result": {"analysis_id": "saved-report-run", "file_name": "Robbery005_x264.mp4",
                         "analyzed_at": "2026-10-30T18:00:00Z",
                         "event_classification": {"label": "Robbery"},
                         "detections": [{"class": "person", "track_id": 7}]},
    }
    (reports / "first.json").write_text(json.dumps(report), encoding="utf-8")
    (reports / "duplicate.json").write_text(json.dumps(report), encoding="utf-8")

    app = Flask(__name__)
    app.config.update(TESTING=True, JWT_SECRET_KEY="dashboard-test-secret", UPLOAD_FOLDER=str(tmp_path))
    JWTManager(app)
    app.register_blueprint(bp, url_prefix="/api/dashboard")
    app.db = None
    payload = app.test_client().get("/api/dashboard/summary").get_json()

    assert payload["status"] == "unavailable"
    assert payload["total_evidence"] is None
    assert payload["videos_analyzed"] is None
    assert payload["crime_types"] is None
    assert payload["total_detections"] is None
    assert payload["risk_assessments"] is None
    assert payload["risk_distribution"] is None
    assert payload["recent_investigations"] is None


def test_dashboard_deduplicates_mongo_and_matching_artifact(tmp_path):
    now = datetime.now(timezone.utc)
    artifact_dir = tmp_path / "evidence" / "run-db"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "analysis.json").write_text(json.dumps({
        "analysis_id": "run-db", "status": "completed", "filename": "a.mp4",
        "analyzed_at": now.isoformat(), "event_classification": {"label": "Robbery"},
    }), encoding="utf-8")
    row = video("mongo-a", "a.mp4", "Robbery", now,
                analysis_id="run-db",
                video_analysis={"analysis_id": "run-db", "tracking": {"used": True}})
    payload = client_for([row], tmp_path).get("/api/dashboard/summary").get_json()
    assert payload["total_evidence"] == 1
    assert payload["videos_analyzed"] == 1
    assert payload["completed_analyses"] == 1
    assert payload["crime_types"]["Robbery"] == 1


def test_dashboard_counts_persisted_failures_without_treating_them_as_completed(tmp_path):
    artifact_dir = tmp_path / "evidence" / "failed-run"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "analysis.json").write_text(json.dumps({
        "analysis_id": "failed-run", "status": "failed",
        "filename": "Shooting002_x264.mp4", "failure_code": "video_decode_failed",
        "uploaded_at": "2026-10-31T09:00:00Z", "analyzed_at": "2026-10-31T09:01:00Z",
    }), encoding="utf-8")
    app = Flask(__name__)
    app.config.update(TESTING=True, JWT_SECRET_KEY="dashboard-test-secret", UPLOAD_FOLDER=str(tmp_path))
    JWTManager(app)
    app.register_blueprint(bp, url_prefix="/api/dashboard")
    app.db = None

    payload = app.test_client().get("/api/dashboard/summary").get_json()

    assert payload["status"] == "unavailable"
    assert payload["total_evidence"] is None
    assert payload["videos_analyzed"] is None
    assert payload["completed_analyses"] is None
    assert payload["failed_analyses"] is None
    assert payload["analysis_attempts"] is None
    assert payload["recent_investigations"] is None


def test_dashboard_uses_stored_risk_levels_and_legacy_thresholds():
    from app.routes.dashboard import _risk_level, _risk_factors
    from app.services.risk_service import RiskService

    service = RiskService()
    assert _risk_level({"overall_risk": 0, "risk_level": "Critical"}, service) == "critical"
    assert _risk_level({"overall_risk": 50, "thresholds": {"moderate": 20, "elevated": 45, "critical": 75}}, service) == "high"
    assert _risk_level({}, service) is None
    assert _risk_factors({"contributing_factors": [{"name": "Weapon", "points": 10}, {"name": "Empty", "points": 0}]}) == ["Weapon"]


def test_dashboard_handles_missing_tracking_as_zero_valid_ids(tmp_path):
    row = video("untracked", "untracked.mp4", "Normal", datetime.now(timezone.utc),
                detections=[{"class": "person", "track_id": None}],
                video_analysis={"tracking": {"used": False}})
    payload = client_for([row], tmp_path).get("/api/dashboard/summary").get_json()
    assert payload["videos_analyzed"] == 1
    assert payload["valid_tracking_ids"] == 0
    assert payload["recent_investigations"][0]["tracking_status"] == "unavailable"


def test_dashboard_risk_distribution_counts_saved_assessments_once(tmp_path):
    now = datetime.now(timezone.utc)
    levels = [("low", 0), ("moderate", 25), ("high", 50), ("critical", 75)]
    rows = [video(str(index), f"{index}.mp4", "Normal", now - timedelta(minutes=index),
                  risk_score={"overall_risk": score, "risk_level": level, "contributing_factors": []})
            for index, (level, score) in enumerate(levels)]
    rows.append(video("repeat", "0.mp4", "Normal", now - timedelta(minutes=10),
                      risk_score={"overall_risk": 99, "risk_level": "Critical", "contributing_factors": []}))
    client = client_for(rows, tmp_path)
    client.application.db.cases.rows.append({"_id": ObjectId(), "risk_level": "Critical"})
    payload = client.get("/api/dashboard/summary").get_json()
    assert payload["risk_distribution"] == {"low": 1, "moderate": 1, "high": 1, "critical": 1}
    assert payload["risk_assessments"] == 4
    assert payload["average_risk_score"] == 37.5
    assert payload["high_risk_cases"] == 1


def test_dashboard_reports_real_custody_counts_and_links(tmp_path):
    stamp = datetime.now(timezone.utc)
    client = client_for([video("mongo-ev1", "camera.mp4", "Assault", stamp, evidence_id="ev1")], tmp_path)
    client.application.db.custody_logs.rows = [{
        "_id": "custody-1", "evidence_id": "ev1", "action": "upload_video",
        "user_id": "investigator", "timestamp": stamp.isoformat(), "created_at": stamp,
        "details": {"original_name": "camera.mp4"}, "integrity_hash": "sha256-real-record",
    }]

    payload = client.get("/api/dashboard/summary").get_json()

    assert payload["custody_status"] == "available"
    assert payload["total_custody_entries"] == 1
    assert payload["unique_evidence_with_custody"] == 1
    assert payload["evidence_with_complete_custody"] == 1
    assert payload["evidence_missing_custody"] == 0
    assert payload["recent_custody"][0]["filename"] == "camera.mp4"
    assert payload["recent_custody"][0]["user_id"] == "investigator"
