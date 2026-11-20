from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from app.utils.security import hash_password, verify_password
from app.models.user import create_user_doc, serialize_user
from datetime import datetime, timedelta
from pymongo.errors import PyMongoError
import re

bp = Blueprint("auth", __name__)


def _database_unavailable():
    state = current_app.extensions.get("mongo_connection_state", {})
    return jsonify({
        "error": "Authentication is temporarily unavailable because the user database cannot be reached.",
        "code": "database_unavailable",
        "error_type": state.get("error_type") or "DatabaseUnavailable",
    }), 503

@bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")
    username = username.strip() if isinstance(username, str) else ""

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    db = current_app.db
    if db is None:
        return _database_unavailable()

    try:
        if db.users.find_one({"username": username}):
            return jsonify({"error": "Username already exists"}), 409

        user_doc = create_user_doc(username, hash_password(password))
        result = db.users.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
    except PyMongoError:
        current_app.logger.exception("User registration database operation failed")
        return _database_unavailable()

    return jsonify({
        "message": "User registered successfully",
        "user": serialize_user(user_doc)
    }), 201


@bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")
    username = username.strip() if isinstance(username, str) else ""

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    db = current_app.db
    if db is None:
        return _database_unavailable()

    # The UI accepts either a badge/username or an email address. Existing
    # accounts may store either identifier, so check both fields without
    # creating or migrating user records. Email lookup is case-insensitive.
    identifier_match = {"$or": [{"username": username}, {"email": username}]}
    if "@" in username:
        identifier_match["$or"][1] = {
            "email": {"$regex": f"^{re.escape(username)}$", "$options": "i"}
        }

    try:
        matching_users = list(db.users.find(identifier_match).limit(2))
    except PyMongoError:
        current_app.logger.exception("User lookup database operation failed")
        return _database_unavailable()

    if not matching_users:
        return jsonify({"error": "Invalid credentials"}), 401
    if len(matching_users) > 1:
        current_app.logger.warning("Login identifier matched multiple user records")
        return jsonify({"error": "Login identifier is ambiguous", "code": "ambiguous_login_identifier"}), 409
    user = matching_users[0]

    password_hash = user.get("password_hash")
    try:
        password_valid = bool(password_hash) and verify_password(password_hash, password)
    except (TypeError, ValueError):
        password_valid = False
    if not password_valid:
        return jsonify({"error": "Invalid credentials"}), 401

    try:
        db.users.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.utcnow()}})
    except PyMongoError:
        current_app.logger.exception("User login timestamp update failed")
        return _database_unavailable()
    token = create_access_token(identity=str(user["_id"]), expires_delta=timedelta(hours=12))

    return jsonify({
        "access_token": token,
        "user": serialize_user(user)
    })


@bp.route("/biometric", methods=["POST"])
@jwt_required(optional=True)
def biometric_auth():
    """
    Simulated biometric gate.
    In real system this would verify fingerprint / face.
    Here we accept a simple flag or dummy fingerprint file.
    """
    data = request.get_json() or {}
    verified = data.get("verified", False)

    if verified or data.get("biometric_token") == "demo-biometric-ok":
        return jsonify({
            "status": "success",
            "message": "Biometric authentication successful",
            "level": "high"
        })
    return jsonify({"status": "failed", "message": "Biometric verification failed"}), 401


@bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    identity = get_jwt_identity()
    return jsonify({"user_id": identity})
