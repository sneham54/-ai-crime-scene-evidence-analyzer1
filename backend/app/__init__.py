"""Flask application factory and initialization."""
from pathlib import Path
import logging
import importlib.metadata
import sys

from dotenv import load_dotenv

# Load project-root settings before importing Config, whose values are evaluated
# at module import time. Missing .env files are normal and use config defaults.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from pymongo import MongoClient
from app.config import Config
from app.models.user import create_user_doc
from app.utils.security import hash_password
import os
import threading
import time
import atexit
from urllib.parse import urlsplit
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError


def _mongo_connection_error(exc, uri):
    """Return actionable MongoDB diagnostics without exposing URI credentials."""
    try:
        parsed = urlsplit(uri)
        host = parsed.hostname or "configured host"
        port = parsed.port or (27017 if parsed.scheme == "mongodb" else "default port")
        endpoint = f"{host}:{port}"
    except (TypeError, ValueError):
        endpoint = "the configured MongoDB endpoint"

    if isinstance(exc, OperationFailure) and getattr(exc, "code", None) in {18, 11}:
        return f"MongoDB authentication failed at {endpoint}; check the configured username, password, and authentication database."
    if isinstance(exc, ServerSelectionTimeoutError):
        detail = str(exc).casefold()
        if "10061" in detail or "refused" in detail:
            return f"MongoDB at {endpoint} refused the connection; start MongoDB or check that the configured host and port are reachable."
        return f"MongoDB at {endpoint} did not respond before the connection timeout; check service availability and Docker networking."
    return f"MongoDB connection failed at {endpoint} ({type(exc).__name__}); check the URI, service, and authentication settings."


def refresh_database_connection(app, force=False):
    """Ping MongoDB periodically and reconnect after an outage without replacing data."""
    state = app.extensions.setdefault("mongo_connection_state", {
        "lock": threading.Lock(), "last_check": 0.0, "connected": False,
        "error": None, "error_type": None, "indexes_ensured": False,
    })
    now = time.monotonic()
    if not force and now - state["last_check"] < app.config["MONGO_RECONNECT_INTERVAL_SECONDS"]:
        return state["connected"]
    if not state["lock"].acquire(blocking=False):
        return state["connected"]
    try:
        now = time.monotonic()
        if not force and now - state["last_check"] < app.config["MONGO_RECONNECT_INTERVAL_SECONDS"]:
            return state["connected"]
        state["last_check"] = now
        client = getattr(app, "mongo_client", None)
        if client is None:
            client = MongoClient(
                app.config["MONGO_URI"],
                serverSelectionTimeoutMS=app.config["MONGO_SERVER_SELECTION_TIMEOUT_MS"],
                connectTimeoutMS=app.config["MONGO_SERVER_SELECTION_TIMEOUT_MS"],
                socketTimeoutMS=app.config["MONGO_SOCKET_TIMEOUT_MS"],
            )
            app.mongo_client = client
        client.admin.command("ping")
        app.db = client[app.config["DB_NAME"]]
        if not state["indexes_ensured"]:
            # Index checks run once per process. They are persistent in MongoDB,
            # so reconnect pings on later requests must not re-run index DDL.
            for collection, keys in (
                ("evidence", [("uploaded_at", -1)]),
                ("evidence", [("evidence_id", 1)]),
                ("evidence", [("case_id", 1), ("file_type", 1), ("status", 1)]),
                ("cases", [("created_at", -1)]),
                ("custody_logs", [("evidence_id", 1), ("created_at", -1)]),
            ):
                try:
                    app.db[collection].create_index(keys)
                except Exception:
                    app.logger.exception("Could not create MongoDB index on %s", collection)
            try:
                app.db.evidence.create_index([("shooting_event.evidence_id", 1)], unique=True,
                                             sparse=True, name="shooting_event_evidence_id_unique")
                app.db.evidence.create_index([("event_evidence.evidence_id", 1)], unique=True,
                                             sparse=True, name="event_evidence_id_unique")
            except Exception:
                app.logger.exception("Could not create unique Shooting event evidence index")
            state["indexes_ensured"] = True
        if not state["connected"]:
            app.logger.info("MongoDB connection established for database %s", app.config["DB_NAME"])
        state.update(connected=True, error=None, error_type=None)
        return True
    except Exception as exc:
        app.db = None
        message = _mongo_connection_error(exc, app.config["MONGO_URI"])
        state.update(connected=False, error=message, error_type=type(exc).__name__)
        app.logger.warning("%s", message)
        return False
    finally:
        state["lock"].release()


def seed_default_users(app):
    db = app.db
    if db is None:
        return

    default_users = {
        "investigator": "investigator",
        "officer": "officer",
    }

    for username, role in default_users.items():
        if db.users.find_one({"username": username}) is None:
            db.users.insert_one(create_user_doc(username, hash_password("demo123"), role))
            print(f"Seeded demo user: {username}")


def create_app():
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)
    app.logger.info(
        "Backend runtime python=%s prefix=%s flask=%s pymongo=%s ultralytics=%s torch=%s",
        sys.executable,
        sys.prefix,
        importlib.metadata.version("flask"),
        importlib.metadata.version("pymongo"),
        importlib.metadata.version("ultralytics"),
        importlib.metadata.version("torch"),
    )
    app.config.from_object(Config)
    CORS(app, resources={r"/api/*": {"origins": r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"}}, supports_credentials=True)
    jwt = JWTManager(app)

    @jwt.unauthorized_loader
    def missing_jwt(_reason):
        return {"error": "Authentication required. Please sign in.", "code": "authentication_required"}, 401

    @jwt.expired_token_loader
    def expired_jwt(_jwt_header, _jwt_payload):
        return {"error": "Your session has expired. Please sign in again.", "code": "token_expired"}, 401

    @jwt.invalid_token_loader
    def invalid_jwt(_reason):
        return {"error": "Your session token is invalid. Please sign in again.", "code": "invalid_token"}, 401

    # Ensure upload folders exist (matches synopsis modules)
    for sub in ("images", "videos", "reports", "annotated", "annotated_videos", "scene_change"):
        os.makedirs(os.path.join(app.config["UPLOAD_FOLDER"], sub), exist_ok=True)

    # MongoDB remains optional for local analysis artifacts. Keep a reusable
    # client so requests can recover automatically when the service returns.
    app.mongo_client = None
    app.db = None
    refresh_database_connection(app, force=True)
    # PyMongo clients own background monitor threads; close them on process exit.
    atexit.register(lambda: app.mongo_client.close() if app.mongo_client is not None else None)
    @app.before_request
    def check_database_connection():
        refresh_database_connection(app)

    seed_default_users(app)

    # Register blueprints – aligned with synopsis objectives
    from app.routes.auth import bp as auth_bp
    from app.routes.upload import bp as upload_bp
    from app.routes.detection import bp as detection_bp
    from app.routes.video import bp as video_bp
    from app.routes.scene_change import bp as scene_change_bp
    from app.routes.report import bp as report_bp
    from app.routes.chatbot import bp as chatbot_bp
    from app.routes.dashboard import bp as dashboard_bp
    from app.routes.evidence import bp as evidence_bp

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(upload_bp, url_prefix="/api/upload")
    app.register_blueprint(detection_bp, url_prefix="/api/detection")
    app.register_blueprint(video_bp, url_prefix="/api/video")
    app.register_blueprint(scene_change_bp, url_prefix="/api/scene-change")
    app.register_blueprint(report_bp, url_prefix="/api/report")
    app.register_blueprint(chatbot_bp, url_prefix="/api/chatbot")
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")
    app.register_blueprint(evidence_bp, url_prefix="/api/evidence")

    # Initialize available image models once per process. Missing weights leave
    # the app usable for its other features and are reported by model status.
    with app.app_context():
        if os.path.isfile(app.config["YOLO_MODEL_PATH"]):
            try:
                from app.services.yolo_service import YOLOService
                YOLOService()
            except Exception:
                app.logger.exception("YOLO model initialization failed")
        if os.path.isfile(app.config["CNN_MODEL_PATH"]):
            try:
                from app.services.hybrid_detection_service import HybridDetectionService
                HybridDetectionService.load_model()
            except Exception:
                app.logger.exception("CNN model initialization failed")

    @app.route("/")
    def index():
        return {
            "project": "AI Crime Scene Evidence Analyzer",
            "status": "running",
            "version": "1.1.0",
            "institute": "Don Bosco Institute of Technology, Bengaluru",
            "endpoints": [
                "/api/auth/login",
                "/api/upload/image",
                "/api/detection/analyze",
                "/api/video/analyze",
                "/api/scene-change/compare",
                "/api/report/generate",
                "/api/chatbot/query",
                "/api/dashboard/summary",
                "/api/dashboard/health",
                "/api/dashboard/cases",
                "/api/evidence/categories",
            ],
        }

    @app.errorhandler(404)
    def json_not_found(_error):
        return {"success": False, "error": "API endpoint not found."}, 404

    @app.errorhandler(413)
    def json_too_large(_error):
        return {"success": False, "error": "Uploaded file exceeds the 100 MB limit."}, 413

    @app.errorhandler(500)
    def json_server_error(_error):
        return {"success": False, "error": "Internal server error."}, 500

    return app
