from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from pymongo import MongoClient
from app.config import Config
from app.models.user import create_user_doc
from app.utils.security import hash_password
import os


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
    app.config.from_object(Config)
    CORS(app, resources={r"/api/*": {"origins": r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"}}, supports_credentials=True)
    JWTManager(app)

    # Ensure upload folders exist (matches synopsis modules)
    for sub in ("images", "videos", "reports", "annotated", "annotated_videos", "scene_change"):
        os.makedirs(os.path.join(app.config["UPLOAD_FOLDER"], sub), exist_ok=True)

    # MongoDB (optional – demo mode works without it)
    try:
        client = MongoClient(app.config["MONGO_URI"], serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        app.db = client[app.config["DB_NAME"]]
        print("Connected to MongoDB")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        print("  Continuing without DB (demo mode)")
        app.db = None

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

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(upload_bp, url_prefix="/api/upload")
    app.register_blueprint(detection_bp, url_prefix="/api/detection")
    app.register_blueprint(video_bp, url_prefix="/api/video")
    app.register_blueprint(scene_change_bp, url_prefix="/api/scene-change")
    app.register_blueprint(report_bp, url_prefix="/api/report")
    app.register_blueprint(chatbot_bp, url_prefix="/api/chatbot")
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")

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
