import sys
from pathlib import Path

from flask import Flask
from flask_jwt_extended import JWTManager, create_access_token

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.routes.evidence import bp  # noqa: E402
from app.services.evidence_catalog_service import EvidenceCatalogService  # noqa: E402


def _client():
    app = Flask(__name__)
    app.config.update(
        TESTING=True,
        JWT_SECRET_KEY="evidence-catalog-test-secret-for-endpoint-tests",
        EVIDENCE_SOURCE_DIR=str(Path.home() / "OneDrive" / "Documents" / "crime_video_dataset"),
        ASSAULT_RESULTS_DIR=str(PROJECT_ROOT / "backend" / "tracked_results_400" / "Assault"),
        ROBBERY_RESULTS_DIR=str(PROJECT_ROOT / "backend" / "robbery_videomae"),
        SHOOTING_RESULTS_DIR=str(PROJECT_ROOT / "backend" / "shooting_evaluation"),
        ASSAULT_LEGACY_JSON_PATH=str(PROJECT_ROOT / "backend" / "assault_77_video_detections.json"),
    )
    JWTManager(app)
    app.register_blueprint(bp, url_prefix="/api/evidence")
    with app.app_context():
        token = create_access_token(identity="test-investigator")
    return app.test_client(), {"Authorization": f"Bearer {token}"}


def test_category_catalog_requires_auth_and_reports_real_local_counts():
    client, auth = _client()
    assert client.get("/api/evidence/categories").status_code == 401
    response = client.get("/api/evidence/categories", headers=auth)
    assert response.status_code == 200
    categories = {item["category"]: item for item in response.json["categories"]}
    assert categories["assault"]["source_video_count"] == 100
    assert categories["assault"]["videos_with_results"] >= 30
    assert categories["robbery"]["result_record_count"] is None
    assert categories["robbery"]["videos_with_results"] is None
    assert "evaluation_report" not in categories["robbery"]
    assert categories["shooting"]["source_video_count"] == 100
    assert categories["shooting"]["results_directory_available"] is False
    normal_source = Path.home() / "OneDrive" / "Documents" / "crime_video_dataset" / "Normal"
    expected_normal_count = (sum(path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
                                 for path in normal_source.iterdir()) if normal_source.is_dir() else 0)
    assert categories["normal"]["source_available"] is normal_source.is_dir()
    assert categories["normal"]["source_video_count"] == expected_normal_count
    assert categories["normal"]["results_directory_available"] is False
    assert categories["normal"]["processing_status"] == "unavailable"


def test_assault_detail_preserves_detection_metadata_and_serves_annotated_video():
    client, auth = _client()
    response = client.get("/api/evidence/assault/videos/Assault001_x264.mp4", headers=auth)
    assert response.status_code == 200
    artifact = next(item for item in response.json["result_sets"] if item["source_file"] == "detections.csv")
    assert len(artifact["records"]) == 942
    record = artifact["records"][0]
    assert record["frame"] == 156
    assert record["timestamp_seconds"] == 5.167
    assert record["class_name"] == "person"
    assert record["bbox"] is not None
    report = client.get(
        "/api/evidence/assault/videos/Assault001_x264.mp4/report-payload",
        query_string={"artifact_id": artifact["artifact_id"]},
        headers=auth,
    )
    assert report.status_code == 200
    assert report.json["report"]["verification_status"] == "unverified_model_output"
    assert len(report.json["report"]["results"]) == 942
    media = client.head(
        "/api/evidence/assault/videos/Assault001_x264.mp4/media",
        query_string={"kind": "annotated", "artifact_id": artifact["artifact_id"]},
        headers=auth,
    )
    assert media.status_code == 200
    assert int(media.headers["Content-Length"]) > 0


def test_all_local_assault_final_pairs_are_discoverable_and_streamable():
    client, auth = _client()
    results = PROJECT_ROOT / "backend" / "tracked_results_400" / "Assault"
    pairs = [folder for folder in results.iterdir() if folder.is_dir()
             and (folder / "detections.csv").is_file()
             and list(folder.glob("*_tracked.mp4"))]
    assert len(pairs) == 30
    listing = client.get("/api/evidence/assault/videos", headers=auth)
    assert listing.status_code == 200
    listed_ids = {item["video_id"].casefold() for item in listing.json["videos"]}
    for folder in pairs:
        video_name = folder.name + ".mp4"
        assert video_name.casefold() in listed_ids
        detail = client.get(f"/api/evidence/assault/videos/{video_name}", headers=auth)
        assert detail.status_code == 200, folder.name
        artifact = next(item for item in detail.json["result_sets"] if item["source_file"] == "detections.csv")
        assert artifact["records"], folder.name
        media = client.head(
            f"/api/evidence/assault/videos/{video_name}/media",
            query_string={"kind": "annotated", "artifact_id": artifact["artifact_id"]},
            headers=auth,
        )
        assert media.status_code == 200, folder.name
        assert int(media.headers["Content-Length"]) > 0


def test_model_evaluation_surface_is_removed_and_report_api_remains_available():
    client, auth = _client()
    report = client.get("/api/evidence/robbery/reports", headers=auth)
    assert report.status_code == 200
    assert "historical_metrics" not in report.json
    downloaded = client.get("/api/evidence/robbery/reports/historical-cnn-lstm-metrics", headers=auth)
    assert downloaded.status_code == 404
    videos = client.get("/api/evidence/robbery/videos", headers=auth)
    assert videos.status_code == 200
    assert videos.json["result_record_count"] is None


def test_path_and_category_validation():
    client, auth = _client()
    assert client.get("/api/evidence/unlisted/videos", headers=auth).status_code == 404
    assert client.get("/api/evidence/assault/videos/..%2Fsecret", headers=auth).status_code in {400, 404}
    assert client.get(
        "/api/evidence/assault/videos/Assault001_x264.mp4/media?kind=remote",
        headers=auth,
    ).status_code == 400


def test_catalog_counts_unique_videos_separately_from_artifact_versions_and_statuses(tmp_path):
    source = tmp_path / "source"
    for category in ("Assault", "Shooting", "Robbery", "Normal"):
        (source / category).mkdir(parents=True)
    (source / "Assault" / "Assault001_x264.mp4").write_bytes(b"source")
    (source / "Assault" / "Assault002_x264.mp4").write_bytes(b"source")
    results = tmp_path / "assault-results"
    first = results / "run-a" / "detections.csv"
    second = results / "run-b" / "detections.csv"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    header = "video_name,frame,class_id,class_name,confidence,x1,y1,x2,y2\n"
    (first).write_text(header + "Assault001_x264.mp4,0,5,person,0.8,1,2,3,4\n", encoding="utf-8")
    (second).write_text(header + "Assault001_x264.mp4,1,5,person,0.7,1,2,3,4\n", encoding="utf-8")
    (results / "tracking_status.csv").write_text(
        "video_name,status\nAssault001_x264.mp4,completed\nAssault002_x264.mp4,partial\n",
        encoding="utf-8",
    )
    config = {
        "EVIDENCE_SOURCE_DIR": str(source),
        "ASSAULT_RESULTS_DIR": str(results),
        "ROBBERY_RESULTS_DIR": str(tmp_path / "robbery-results"),
        "SHOOTING_RESULTS_DIR": str(tmp_path / "shooting-results"),
        "NORMAL_RESULTS_DIR": str(tmp_path / "normal-results"),
        "ASSAULT_LEGACY_JSON_PATH": str(tmp_path / "missing.json"),
    }
    assault = next(item for item in EvidenceCatalogService(config).list_categories() if item["category"] == "assault")
    assert assault["source_video_count"] == 2
    assert assault["videos_with_valid_catalog_artifacts"] == 1
    assert assault["artifact_set_count"] == 2
    assert assault["duplicate_artifact_version_count"] == 1
    assert assault["videos_with_completed_status"] == 1
    assert assault["videos_with_partial_status"] == 1
    assert assault["videos_with_failed_or_invalid_status"] == 0
    assert assault["videos_not_yet_processed"] == 0


def test_catalog_reports_missing_source_inventory_as_unavailable(tmp_path):
    config = {
        "EVIDENCE_SOURCE_DIR": str(tmp_path / "missing-source"),
        **{item["results_config"]: str(tmp_path / f"{category}-results")
           for category, item in __import__("app.services.evidence_catalog_service", fromlist=["CATEGORIES"]).CATEGORIES.items()},
    }
    summary = EvidenceCatalogService(config).list_categories()
    assert all(item["source_available"] is False for item in summary)
    assert all(item["source_video_count"] is None for item in summary)
    assert all(item["source_inventory_status"] == "unavailable" for item in summary)
